use std::collections::HashMap;

use rand::RngExt;

use crate::brain::ppo::{PpoBrain, PpoLayerHealth, PpoTrainingStats};
use crate::brain::ppo::buffer::TrainerRolloutBuffer;
use crate::brain::common::math::{normal_entropy, normal_log_prob};
use crate::brain::common::mlp::Linear;


/// Accumulated diagnostics for the current epoch, persisted across chunk ticks.
#[derive(Default)]
pub struct EpochAccumulator {
    pub policy_loss_sum: f32,
    pub value_loss_sum: f32,
    pub entropy_sum: f32,
    pub action_sum: [f32; 2],
    pub action_sumsq: [f32; 2],
    pub clamped_count: usize,
    pub clip_count: usize,
    pub approx_kl_sum: f32,
    pub actor_dead: [usize; 2],
    pub critic_dead: [usize; 2],
    pub actor_seen: [usize; 2],
    pub critic_seen: [usize; 2],
}

/// Pre-computed data for a PPO update that is amortised across multiple frames.
pub struct PreparedUpdate {
    pub frozen_buffer: TrainerRolloutBuffer,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
    pub epochs_remaining: usize,
    pub sample_offset: usize,
    pub accum: EpochAccumulator,
    pub shuffled_indices: Vec<usize>,
}

impl PreparedUpdate {
    pub fn is_active(&self) -> bool {
        self.epochs_remaining > 0
    }
}

/// Validates the buffer, computes per-env GAE, and freezes the data for staged
/// epoch processing. Returns `None` if the buffer is empty or misaligned.
///
/// Takes ownership of the buffer contents via `std::mem::take` to avoid
/// a deep clone. The caller's buffer is left empty.
pub fn ppo_prepare_update(
    brain: &mut PpoBrain,
    buffer: &mut TrainerRolloutBuffer,
    bootstrap_values: &HashMap<u32, f32>,
) -> Option<PreparedUpdate> {
    if buffer.len() == 0 {
        return None;
    }

    if !buffer.is_aligned() {
        bevy::log::warn!(
            "Trainer rollout misalignment detected (s={}, a={}, z={}, v={}, r={}, d={}, c={}, e={}); skipping update.",
            buffer.pre_step_count(),
            buffer.actions.len() / buffer.act_dim,
            buffer.latent_actions.len() / buffer.act_dim,
            buffer.values.len(),
            buffer.rewards.len(),
            buffer.dones.len(),
            buffer.safety_clamp_hits.len(),
            buffer.env_ids.len(),
        );
        return None;
    }

    let (advantages, returns) =
        buffer.compute_gae_per_env(bootstrap_values, brain.config.gamma, brain.config.gae_lambda);
    let len = buffer.len();

    // Fisher-Yates shuffle for minibatch sample ordering
    let mut indices: Vec<usize> = (0..len).collect();
    for i in (1..indices.len()).rev() {
        let j = brain.rng.random_range(0..=i);
        indices.swap(i, j);
    }

    // Take ownership of buffer contents — avoids deep clone.
    let frozen_buffer = std::mem::take(buffer);

    Some(PreparedUpdate {
        frozen_buffer,
        advantages,
        returns,
        epochs_remaining: brain.config.ppo_epochs,
        sample_offset: 0,
        accum: EpochAccumulator::default(),
        shuffled_indices: indices,
    })
}

/// Processes up to `max_samples` from the current epoch using **batched**
/// forward and backward passes. Gradients accumulate across chunks.
/// Returns `true` when the epoch's samples are exhausted.
pub fn ppo_process_chunk(
    brain: &mut PpoBrain,
    prepared: &mut PreparedUpdate,
    max_samples: usize,
) -> bool {
    // Start-of-epoch housekeeping
    if prepared.sample_offset == 0 {
        brain.model.zero_grad();
        prepared.accum = EpochAccumulator::default();
        // Re-shuffle indices for this epoch
        for i in (1..prepared.shuffled_indices.len()).rev() {
            let j = brain.rng.random_range(0..=i);
            prepared.shuffled_indices.swap(i, j);
        }
    }

    let buffer = &prepared.frozen_buffer;
    let advantages = &prepared.advantages;
    let returns = &prepared.returns;
    let clip_eps = brain.config.clip_epsilon;
    let value_huber_delta = brain.config.value_huber_delta;
    let entropy_coef = brain.config.entropy_coef;
    let batch_size = buffer.len();
    let batch_size_f32 = batch_size as f32;
    let end = (prepared.sample_offset + max_samples).min(batch_size);
    let chunk_size = end - prepared.sample_offset;
    let acc = &mut prepared.accum;

    if chunk_size == 0 {
        return true;
    }

    let chunk_indices = &prepared.shuffled_indices[prepared.sample_offset..end];

    // Per-chunk advantage normalisation
    let chunk_size_f = chunk_size as f32;
    let chunk_adv_mean = chunk_indices.iter().map(|&idx| advantages[idx]).sum::<f32>() / chunk_size_f.max(1.0);
    let chunk_adv_var = chunk_indices.iter().map(|&idx| (advantages[idx] - chunk_adv_mean).powi(2)).sum::<f32>() / chunk_size_f.max(1.0);
    let chunk_adv_std = (chunk_adv_var + 1e-8).sqrt();

    let obs_dim = brain.model.scratch.obs_dim;
    let act_dim = brain.model.scratch.act_dim;

    // ── Stack observations into the pre-allocated input scratch ────
    // `batch_io` is a sibling field of `scratch` on `ActorCritic`, so Rust's
    // disjoint-field borrow inference lets `forward_batch` mutably borrow
    // `scratch` while simultaneously reading `batch_io.obs_batch`. No
    // raw-pointer aliasing is required.
    {
        let obs_batch = &mut brain.model.batch_io.obs_batch;
        for (s, &idx) in chunk_indices.iter().enumerate() {
            let src = &buffer.states[idx * obs_dim..(idx + 1) * obs_dim];
            obs_batch[s * obs_dim..(s + 1) * obs_dim].copy_from_slice(src);
        }
    }

    // ── Batched forward pass ────────────────────────────────────────
    brain.model.forward_batch(chunk_size);

    // ── Collect tanh saturation stats from batch caches ─────────────
    collect_saturated_slice(
        brain.model.a_tanh1.batch_cache(),
        &mut acc.actor_dead[0],
        &mut acc.actor_seen[0],
    );
    collect_saturated_slice(
        brain.model.a_tanh2.batch_cache(),
        &mut acc.actor_dead[1],
        &mut acc.actor_seen[1],
    );
    collect_saturated_slice(
        brain.model.c_tanh1.batch_cache(),
        &mut acc.critic_dead[0],
        &mut acc.critic_seen[0],
    );
    collect_saturated_slice(
        brain.model.c_tanh2.batch_cache(),
        &mut acc.critic_dead[1],
        &mut acc.critic_seen[1],
    );

    // ── Per-sample PPO loss computation + gradient seeds ────────────
    // Critic: grad_values[s] = Huber gradient / batch_size
    // Actor: grad_means[s * act_dim + j] = (-policy_weight * adv * d_lp/d_mean_j) / batch_size
    let grad_values = &mut brain.model.scratch.gc_out;
    let grad_means = &mut brain.model.scratch.ga_out;

    let a_out = &brain.model.scratch.a_out;
    let c_out = &brain.model.scratch.c_out;
    let std_vals: [f32; 2] = [
        brain.model.a_log_std[0].exp(),
        brain.model.a_log_std[1].exp(),
    ];

    for (s, &idx) in chunk_indices.iter().enumerate() {
        let action = &buffer.actions[idx * act_dim..(idx + 1) * act_dim];
        let latent_action = &buffer.latent_actions[idx * act_dim..(idx + 1) * act_dim];
        let old_log_prob = buffer.old_log_probs[idx];
        let adv = (advantages[idx] - chunk_adv_mean) / chunk_adv_std;
        let ret = returns[idx];

        // Read forward pass results for this sample
        let value = c_out[s];
        let means: [f32; 2] = [
            a_out[s * act_dim],
            a_out[s * act_dim + 1],
        ];

        // ── Value loss (Huber) ──
        let value_error = value - ret;
        let value_grad = if value_error.abs() <= value_huber_delta {
            value_error
        } else {
            value_huber_delta * value_error.signum()
        };
        acc.value_loss_sum += if value_error.abs() <= value_huber_delta {
            0.5 * value_error.powi(2)
        } else {
            value_huber_delta * (value_error.abs() - 0.5 * value_huber_delta)
        };
        grad_values[s] = value_grad / batch_size_f32;

        // ── PPO clipped policy loss ──
        let mut new_log_prob = 0.0;
        let mut d_lp_d_means = [0.0f32; 2];
        let mut d_lp_d_log_stds = [0.0f32; 2];

        for j in 0..2 {
            let latent = latent_action[j];
            let a = action[j];
            let squashed = if j == 0 { a } else { 2.0 * a - 1.0 };
            let mean = means[j];
            let std = std_vals[j];

            let lp = squashed_gaussian_log_prob(latent, squashed, mean, std, j);
            new_log_prob += lp;
            d_lp_d_means[j] = (latent - mean) / (std * std + 1e-8);
            d_lp_d_log_stds[j] = ((latent - mean).powi(2) / (std * std + 1e-8)) - 1.0;

            acc.entropy_sum += normal_entropy(std);
            acc.action_sum[j] += a;
            acc.action_sumsq[j] += a * a;
            if buffer.safety_clamp_hits[idx][j] {
                acc.clamped_count += 1;
            }
        }

        let ratio = (new_log_prob - old_log_prob).exp();
        let clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        let surr_unclipped = ratio * adv;
        let surr_clipped = clipped_ratio * adv;
        let use_unclipped = surr_unclipped <= surr_clipped;
        let policy_weight = if use_unclipped { ratio } else { 0.0 };

        if (ratio - 1.0).abs() > clip_eps {
            acc.clip_count += 1;
        }
        acc.approx_kl_sum += old_log_prob - new_log_prob;
        acc.policy_loss_sum += -surr_unclipped.min(surr_clipped);

        // Gradient seeds for actor backward
        for j in 0..2 {
            grad_means[s * act_dim + j] =
                (-policy_weight * adv * d_lp_d_means[j]) / batch_size_f32;
            brain.model.a_log_std_grad[j] +=
                (-policy_weight * adv * d_lp_d_log_stds[j] - entropy_coef) / batch_size_f32;
        }
    }

    // ── Batched backward passes ─────────────────────────────────────
    // Copy gradient seeds from the forward `scratch` buffers into the
    // sibling `batch_io` seed buffers. After this copy, `batch_io` owns
    // the inputs for the backward pass, and `scratch` can be borrowed
    // mutably for the intermediate writes — no raw-pointer aliasing.
    brain.model.batch_io.grad_seed_values[..chunk_size]
        .copy_from_slice(&brain.model.scratch.gc_out[..chunk_size]);
    brain.model.batch_io.grad_seed_means[..chunk_size * act_dim]
        .copy_from_slice(&brain.model.scratch.ga_out[..chunk_size * act_dim]);

    brain.model.backward_batch_critic(chunk_size);
    brain.model.backward_batch_actor(chunk_size);

    prepared.sample_offset = end;
    end >= batch_size
}

/// Clips gradients, steps the optimiser, and updates log-std after all samples
/// in the current epoch have been processed. On the final epoch, writes stats.
pub fn ppo_finish_epoch(
    brain: &mut PpoBrain,
    prepared: &PreparedUpdate,
    stats: &mut PpoTrainingStats,
    is_final_epoch: bool,
) {
    clip_linear_gradients(
        &mut [
            &mut brain.model.a_fc1,
            &mut brain.model.a_fc2,
            &mut brain.model.a_mean,
        ],
        brain.config.actor_grad_clip,
    );
    clip_linear_gradients(
        &mut [
            &mut brain.model.c_fc1,
            &mut brain.model.c_fc2,
            &mut brain.model.c_value,
        ],
        brain.config.critic_grad_clip,
    );

    brain.model.a_opt.step(&mut [
        &mut brain.model.a_fc1,
        &mut brain.model.a_fc2,
        &mut brain.model.a_mean,
    ]);
    brain.model.c_opt.step(&mut [
        &mut brain.model.c_fc1,
        &mut brain.model.c_fc2,
        &mut brain.model.c_value,
    ]);

    brain.model.opt_t += 1.0;
    for j in 0..2 {
        let g = brain.model.a_log_std_grad[j];
        brain.model.log_std_opt_m[j] = 0.9 * brain.model.log_std_opt_m[j] + 0.1 * g;
        brain.model.log_std_opt_v[j] =
            0.999 * brain.model.log_std_opt_v[j] + 0.001 * g * g;

        let m_hat =
            brain.model.log_std_opt_m[j] / (1.0 - 0.9f32.powf(brain.model.opt_t));
        let v_hat =
            brain.model.log_std_opt_v[j] / (1.0 - 0.999f32.powf(brain.model.opt_t));

        brain.model.a_log_std[j] -= brain.config.log_std_lr * m_hat / (v_hat.sqrt() + 1e-8);
        brain.model.a_log_std[j] = brain.model.a_log_std[j].clamp(brain.config.log_std_floor, brain.config.log_std_ceil);
    }

    if is_final_epoch {
        let acc = &prepared.accum;
        let batch_size = prepared.frozen_buffer.len();
        let batch_size_f32 = batch_size as f32;

        stats.last_completed_update = stats.last_completed_update.saturating_add(1);
        stats.batch_size = batch_size;
        stats.policy_loss = acc.policy_loss_sum / batch_size_f32.max(1.0);
        stats.value_loss = acc.value_loss_sum / batch_size_f32.max(1.0);
        stats.policy_entropy = acc.entropy_sum / (batch_size_f32 * 2.0).max(1.0);
        stats.explained_variance =
            explained_variance(&prepared.returns, &prepared.frozen_buffer.values);
        stats.steering_mean = acc.action_sum[0] / batch_size_f32.max(1.0);
        stats.steering_std = std_from_sums(acc.action_sum[0], acc.action_sumsq[0], batch_size);
        stats.throttle_mean = acc.action_sum[1] / batch_size_f32.max(1.0);
        stats.throttle_std = std_from_sums(acc.action_sum[1], acc.action_sumsq[1], batch_size);
        stats.clamped_action_fraction =
            acc.clamped_count as f32 / (batch_size.saturating_mul(2) as f32).max(1.0);
        stats.clip_fraction = acc.clip_count as f32 / batch_size_f32.max(1.0);
        stats.approx_kl = acc.approx_kl_sum / batch_size_f32.max(1.0);
        stats.layer_health = vec![
            PpoLayerHealth {
                layer_name: "actor_fc1".to_string(),
                weight_l2_norm: brain.model.a_fc1.weight_l2_norm(),
                gradient_l2_norm: brain.model.a_fc1.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.actor_dead[0], acc.actor_seen[0])),
            },
            PpoLayerHealth {
                layer_name: "actor_fc2".to_string(),
                weight_l2_norm: brain.model.a_fc2.weight_l2_norm(),
                gradient_l2_norm: brain.model.a_fc2.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.actor_dead[1], acc.actor_seen[1])),
            },
            PpoLayerHealth {
                layer_name: "actor_mean".to_string(),
                weight_l2_norm: brain.model.a_mean.weight_l2_norm(),
                gradient_l2_norm: brain.model.a_mean.grad_l2_norm(),
                saturated_fraction: None,
            },
            PpoLayerHealth {
                layer_name: "critic_fc1".to_string(),
                weight_l2_norm: brain.model.c_fc1.weight_l2_norm(),
                gradient_l2_norm: brain.model.c_fc1.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.critic_dead[0], acc.critic_seen[0])),
            },
            PpoLayerHealth {
                layer_name: "critic_fc2".to_string(),
                weight_l2_norm: brain.model.c_fc2.weight_l2_norm(),
                gradient_l2_norm: brain.model.c_fc2.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.critic_dead[1], acc.critic_seen[1])),
            },
            PpoLayerHealth {
                layer_name: "critic_value".to_string(),
                weight_l2_norm: brain.model.c_value.weight_l2_norm(),
                gradient_l2_norm: brain.model.c_value.grad_l2_norm(),
                saturated_fraction: None,
            },
        ];

        bevy::log::info!(
            "PPO update #{}: batch={} epochs={} policy_loss={:.4} value_loss={:.4} entropy={:.4} ev={:.4} clip={:.2}% kl={:.5}",
            stats.last_completed_update,
            batch_size,
            brain.config.ppo_epochs,
            stats.policy_loss,
            stats.value_loss,
            stats.policy_entropy,
            stats.explained_variance,
            stats.clip_fraction * 100.0,
            stats.approx_kl,
        );
    }
}

/// Blocking PPO update — runs all epochs synchronously. Used only for the
/// on-exit flush where frame budget does not matter.
pub fn ppo_update_blocking(
    brain: &mut PpoBrain,
    buffer: &mut TrainerRolloutBuffer,
    stats: &mut PpoTrainingStats,
    bootstrap_values: &HashMap<u32, f32>,
) {
    let Some(mut prepared) = ppo_prepare_update(brain, buffer, bootstrap_values) else {
        return;
    };
    let batch_size = prepared.frozen_buffer.len();
    while prepared.is_active() {
        let is_final = prepared.epochs_remaining == 1;
        ppo_process_chunk(brain, &mut prepared, batch_size);
        ppo_finish_epoch(brain, &prepared, stats, is_final);
        prepared.epochs_remaining -= 1;
        prepared.sample_offset = 0;
    }
}

fn std_from_sums(sum: f32, sumsq: f32, count: usize) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let n = count as f32;
    let mean = sum / n;
    ((sumsq / n) - mean * mean).max(0.0).sqrt()
}

fn explained_variance(targets: &[f32], predictions: &[f32]) -> f32 {
    if targets.len() != predictions.len() || targets.is_empty() {
        return 0.0;
    }

    let mean_target = targets.iter().sum::<f32>() / targets.len() as f32;
    let variance_target = targets
        .iter()
        .map(|target| (target - mean_target).powi(2))
        .sum::<f32>()
        / targets.len() as f32;
    if variance_target <= 1e-8 {
        return 0.0;
    }

    let error_variance = targets
        .iter()
        .zip(predictions.iter())
        .map(|(target, prediction)| (target - prediction).powi(2))
        .sum::<f32>()
        / targets.len() as f32;

    1.0 - (error_variance / variance_target)
}

fn collect_saturated_slice(values: &[f32], saturated: &mut usize, seen: &mut usize) {
    *seen += values.len();
    *saturated += values.iter().filter(|value| value.abs() > 0.99).count();
}

fn fraction(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

pub(crate) fn squashed_gaussian_log_prob(
    latent: f32,
    squashed: f32,
    mean: f32,
    std: f32,
    component_idx: usize,
) -> f32 {
    let gaussian_log_prob = normal_log_prob(latent, mean, std);
    let log_det_jacobian = (1.0 - squashed * squashed + 1e-6).ln();
    let affine_log_det = if component_idx == 1 {
        (2.0f32).ln()
    } else {
        0.0
    };
    gaussian_log_prob - log_det_jacobian + affine_log_det
}

fn clip_linear_gradients(layers: &mut [&mut Linear], max_norm: f32) {
    if max_norm <= 0.0 {
        return;
    }

    let mut sumsq = 0.0f32;
    for layer in layers.iter() {
        sumsq += layer.grad_weights.iter().map(|g| g * g).sum::<f32>();
        sumsq += layer.grad_biases.iter().map(|g| g * g).sum::<f32>();
    }

    let norm = sumsq.sqrt();
    if norm <= max_norm || norm <= 1e-8 {
        return;
    }

    let scale = max_norm / norm;
    for layer in layers.iter_mut() {
        layer.grad_weights.iter_mut().for_each(|g| *g *= scale);
        layer.grad_biases.iter_mut().for_each(|g| *g *= scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── squashed_gaussian_log_prob ───────────────────────────────────────

    #[test]
    fn squashed_gaussian_log_prob_is_finite_for_normal_inputs() {
        // Steering component (idx 0): squashed in [-1, 1]
        let lp = squashed_gaussian_log_prob(0.1, 0.09, 0.0, 1.0, 0);
        assert!(lp.is_finite(), "got {}", lp);
        // Throttle component (idx 1): squashed in [0, 1], scaled back to [-1, 1]
        let lp = squashed_gaussian_log_prob(0.1, 0.1, 0.0, 1.0, 1);
        assert!(lp.is_finite(), "got {}", lp);
    }

    #[test]
    fn squashed_gaussian_log_prob_symmetric_around_mean_for_steering() {
        // Steering path is symmetric: log_prob(latent, squashed, mean) should
        // equal log_prob(-latent, -squashed, -mean) for component 0.
        let a = squashed_gaussian_log_prob(0.5, 0.3, 0.0, 1.0, 0);
        let b = squashed_gaussian_log_prob(-0.5, -0.3, 0.0, 1.0, 0);
        assert!(approx(a, b, 1e-5), "a={}, b={}", a, b);
    }

    // ── clip_linear_gradients ────────────────────────────────────────────

    #[test]
    fn clip_linear_gradients_scales_when_norm_exceeds_threshold() {
        use crate::brain::common::mlp::Linear;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(1);
        let mut layer = Linear::new_orthogonal(2, 2, 1.0, &mut rng);
        // Pre-existing grads with L2 norm ~sqrt(16)=4
        layer.grad_weights = vec![2.0, 2.0, 2.0, 2.0];
        layer.grad_biases = vec![0.0, 0.0];

        let max_norm = 1.0;
        clip_linear_gradients(&mut [&mut layer], max_norm);

        // New norm should be ~max_norm
        let new_norm = layer.grad_l2_norm();
        assert!(approx(new_norm, max_norm, 1e-4), "new norm: {}", new_norm);
    }

    #[test]
    fn clip_linear_gradients_noop_below_threshold() {
        use crate::brain::common::mlp::Linear;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(2);
        let mut layer = Linear::new_orthogonal(2, 2, 1.0, &mut rng);
        // Small grads
        layer.grad_weights = vec![0.1, 0.1, 0.1, 0.1];
        layer.grad_biases = vec![0.0, 0.0];
        let before = layer.grad_weights.clone();

        clip_linear_gradients(&mut [&mut layer], 10.0);

        // Unchanged
        for (a, b) in layer.grad_weights.iter().zip(before.iter()) {
            assert!(approx(*a, *b, 1e-8));
        }
    }

    #[test]
    fn clip_linear_gradients_zero_max_norm_is_noop() {
        use crate::brain::common::mlp::Linear;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(3);
        let mut layer = Linear::new_orthogonal(2, 2, 1.0, &mut rng);
        layer.grad_weights = vec![5.0, 5.0, 5.0, 5.0];
        let before = layer.grad_weights.clone();

        // max_norm <= 0 means "no clipping" — explicit early return.
        clip_linear_gradients(&mut [&mut layer], 0.0);

        for (a, b) in layer.grad_weights.iter().zip(before.iter()) {
            assert!(approx(*a, *b, 1e-8));
        }
    }

    // ── PPO ratio / clip semantics (inline logic verification) ───────────

    #[test]
    fn ppo_ratio_is_one_when_log_probs_equal() {
        let ratio = (0.3f32 - 0.3).exp();
        assert!(approx(ratio, 1.0, 1e-6));
    }

    #[test]
    fn ppo_ratio_clips_at_upper_bound_when_log_prob_increases_sharply() {
        let clip_eps = 0.2f32;
        let ratio = (2.0f32 - 0.0).exp(); // e^2 ≈ 7.389
        let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        assert!(approx(clipped, 1.0 + clip_eps, 1e-6));
    }

    #[test]
    fn ppo_ratio_clips_at_lower_bound_when_log_prob_decreases_sharply() {
        let clip_eps = 0.2f32;
        let ratio = (-2.0f32 - 0.0).exp(); // e^-2 ≈ 0.135
        let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        assert!(approx(clipped, 1.0 - clip_eps, 1e-6));
    }

    // ── Huber value loss (inline logic verification) ─────────────────────

    #[test]
    fn huber_value_loss_is_quadratic_near_zero_error() {
        let delta = 1.0f32;
        let error = 0.5f32;
        let loss = if error.abs() <= delta {
            0.5 * error.powi(2)
        } else {
            delta * (error.abs() - 0.5 * delta)
        };
        // 0.5 * 0.25 = 0.125
        assert!(approx(loss, 0.125, 1e-6));
    }

    #[test]
    fn huber_value_loss_is_linear_past_threshold() {
        let delta = 1.0f32;
        let error = 3.0f32;
        let loss = if error.abs() <= delta {
            0.5 * error.powi(2)
        } else {
            delta * (error.abs() - 0.5 * delta)
        };
        // 1.0 * (3.0 - 0.5) = 2.5
        assert!(approx(loss, 2.5, 1e-6));
    }
}
