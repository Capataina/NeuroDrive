use std::collections::HashMap;

use rand::RngExt;

use crate::brain::a2c::{A2cBrain, A2cLayerHealth, A2cTrainingStats};
use crate::brain::a2c::buffer::TrainerRolloutBuffer;
use crate::brain::common::math::{normal_entropy, normal_log_prob};
use crate::brain::common::mlp::Linear;

const VALUE_HUBER_DELTA: f32 = 1.0;
const ACTOR_GRAD_CLIP_NORM: f32 = 0.5;
const CRITIC_GRAD_CLIP_NORM: f32 = 0.5;
const ENTROPY_COEF: f32 = 0.01;

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
/// Samples are processed in chunks (samples_per_tick), and epochs advance once
/// all samples are done.
pub struct PreparedUpdate {
    pub frozen_buffer: TrainerRolloutBuffer,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
    pub value_predictions: Vec<f32>,
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
pub fn ppo_prepare_update(
    brain: &mut A2cBrain,
    buffer: &TrainerRolloutBuffer,
    bootstrap_values: &HashMap<u32, f32>,
) -> Option<PreparedUpdate> {
    if buffer.len() == 0 {
        return None;
    }

    if !buffer.is_aligned() {
        bevy::log::warn!(
            "Trainer rollout misalignment detected (s={}, a={}, z={}, v={}, r={}, d={}, c={}, e={}); skipping update.",
            buffer.states.len(),
            buffer.actions.len(),
            buffer.latent_actions.len(),
            buffer.values.len(),
            buffer.rewards.len(),
            buffer.dones.len(),
            buffer.safety_clamp_hits.len(),
            buffer.env_ids.len(),
        );
        return None;
    }

    let (advantages, returns) =
        buffer.compute_gae_per_env(bootstrap_values, brain.gamma, brain.gae_lambda);
    let value_predictions = buffer.values.clone();

    // Fisher-Yates shuffle for minibatch sample ordering
    let mut indices: Vec<usize> = (0..buffer.len()).collect();
    for i in (1..indices.len()).rev() {
        let j = brain.rng.random_range(0..=i);
        indices.swap(i, j);
    }

    Some(PreparedUpdate {
        frozen_buffer: buffer.clone(),
        advantages,
        returns,
        value_predictions,
        epochs_remaining: brain.ppo_epochs,
        sample_offset: 0,
        accum: EpochAccumulator::default(),
        shuffled_indices: indices,
    })
}

/// Processes up to `max_samples` from the current epoch. Gradients accumulate
/// across chunks. Returns `true` when the epoch's samples are exhausted and the
/// caller should run `ppo_finish_epoch`.
pub fn ppo_process_chunk(
    brain: &mut A2cBrain,
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
    let clip_eps = brain.clip_epsilon;
    let batch_size = buffer.len();
    let batch_size_f32 = batch_size as f32;
    let end = (prepared.sample_offset + max_samples).min(batch_size);
    let acc = &mut prepared.accum;

    // Per-chunk advantage normalisation
    let chunk_indices = &prepared.shuffled_indices[prepared.sample_offset..end];
    let chunk_size_f = (end - prepared.sample_offset) as f32;
    let chunk_adv_mean = chunk_indices.iter().map(|&idx| advantages[idx]).sum::<f32>() / chunk_size_f.max(1.0);
    let chunk_adv_var = chunk_indices.iter().map(|&idx| (advantages[idx] - chunk_adv_mean).powi(2)).sum::<f32>() / chunk_size_f.max(1.0);
    let chunk_adv_std = (chunk_adv_var + 1e-8).sqrt();

    for i in prepared.sample_offset..end {
        let idx = prepared.shuffled_indices[i];
        let state = &buffer.states[idx];
        let action = &buffer.actions[idx];
        let latent_action = &buffer.latent_actions[idx];
        let old_log_prob = buffer.old_log_probs[idx];
        let adv = (advantages[idx] - chunk_adv_mean) / chunk_adv_std;
        let ret = returns[idx];

        let (action_dist, value) = brain.model.forward(state);
        collect_saturated_tanh(
            brain.model.a_tanh1.output_cache.as_deref(),
            &mut acc.actor_dead[0],
            &mut acc.actor_seen[0],
        );
        collect_saturated_tanh(
            brain.model.a_tanh2.output_cache.as_deref(),
            &mut acc.actor_dead[1],
            &mut acc.actor_seen[1],
        );
        collect_saturated_tanh(
            brain.model.c_tanh1.output_cache.as_deref(),
            &mut acc.critic_dead[0],
            &mut acc.critic_seen[0],
        );
        collect_saturated_tanh(
            brain.model.c_tanh2.output_cache.as_deref(),
            &mut acc.critic_dead[1],
            &mut acc.critic_seen[1],
        );

        // --- Value loss (Huber) ---
        let value_error = value - ret;
        let value_grad = if value_error.abs() <= VALUE_HUBER_DELTA {
            value_error
        } else {
            VALUE_HUBER_DELTA * value_error.signum()
        };
        acc.value_loss_sum += if value_error.abs() <= VALUE_HUBER_DELTA {
            0.5 * value_error.powi(2)
        } else {
            VALUE_HUBER_DELTA * (value_error.abs() - 0.5 * VALUE_HUBER_DELTA)
        };
        let d_value = vec![value_grad / batch_size_f32];

        let c2_g = brain.model.c_value.backward(&d_value);
        let c2_r_g = brain.model.c_tanh2.backward(&c2_g);
        let c1_g = brain.model.c_fc2.backward(&c2_r_g);
        let c1_r_g = brain.model.c_tanh1.backward(&c1_g);
        brain.model.c_fc1.backward(&c1_r_g);

        // --- PPO clipped policy loss ---
        let mut new_log_prob = 0.0;
        let mut d_lp_d_means = [0.0f32; 2];
        let mut d_lp_d_log_stds = [0.0f32; 2];

        for j in 0..2 {
            let latent = latent_action[j];
            let a = action[j];
            let squashed = if j == 0 { a } else { 2.0 * a - 1.0 };
            let mean = action_dist.mean[j];
            let std = action_dist.std[j];

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

        let mut d_mean = vec![0.0; 2];
        for j in 0..2 {
            d_mean[j] = (-policy_weight * adv * d_lp_d_means[j]) / batch_size_f32;
            brain.model.a_log_std_grad[j] +=
                (-policy_weight * adv * d_lp_d_log_stds[j] - ENTROPY_COEF) / batch_size_f32;
        }

        let a2_g = brain.model.a_mean.backward(&d_mean);
        let a2_r_g = brain.model.a_tanh2.backward(&a2_g);
        let a1_g = brain.model.a_fc2.backward(&a2_r_g);
        let a1_r_g = brain.model.a_tanh1.backward(&a1_g);
        brain.model.a_fc1.backward(&a1_r_g);
    }

    prepared.sample_offset = end;
    end >= batch_size
}

/// Clips gradients, steps the optimiser, and updates log-std after all samples
/// in the current epoch have been processed. On the final epoch, writes stats.
pub fn ppo_finish_epoch(
    brain: &mut A2cBrain,
    prepared: &PreparedUpdate,
    stats: &mut A2cTrainingStats,
    is_final_epoch: bool,
) {
    clip_linear_gradients(
        &mut [
            &mut brain.model.a_fc1,
            &mut brain.model.a_fc2,
            &mut brain.model.a_mean,
        ],
        ACTOR_GRAD_CLIP_NORM,
    );
    clip_linear_gradients(
        &mut [
            &mut brain.model.c_fc1,
            &mut brain.model.c_fc2,
            &mut brain.model.c_value,
        ],
        CRITIC_GRAD_CLIP_NORM,
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

        brain.model.a_log_std[j] -= 3e-4 * m_hat / (v_hat.sqrt() + 1e-8);
        brain.model.a_log_std[j] = brain.model.a_log_std[j].clamp(-2.0, 0.5);
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
            explained_variance(&prepared.returns, &prepared.value_predictions);
        stats.steering_mean = acc.action_sum[0] / batch_size_f32.max(1.0);
        stats.steering_std = std_from_sums(acc.action_sum[0], acc.action_sumsq[0], batch_size);
        stats.throttle_mean = acc.action_sum[1] / batch_size_f32.max(1.0);
        stats.throttle_std = std_from_sums(acc.action_sum[1], acc.action_sumsq[1], batch_size);
        stats.clamped_action_fraction =
            acc.clamped_count as f32 / (batch_size.saturating_mul(2) as f32).max(1.0);
        stats.clip_fraction = acc.clip_count as f32 / batch_size_f32.max(1.0);
        stats.approx_kl = acc.approx_kl_sum / batch_size_f32.max(1.0);
        stats.layer_health = vec![
            A2cLayerHealth {
                layer_name: "actor_fc1".to_string(),
                weight_l2_norm: brain.model.a_fc1.weight_l2_norm(),
                gradient_l2_norm: brain.model.a_fc1.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.actor_dead[0], acc.actor_seen[0])),
            },
            A2cLayerHealth {
                layer_name: "actor_fc2".to_string(),
                weight_l2_norm: brain.model.a_fc2.weight_l2_norm(),
                gradient_l2_norm: brain.model.a_fc2.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.actor_dead[1], acc.actor_seen[1])),
            },
            A2cLayerHealth {
                layer_name: "actor_mean".to_string(),
                weight_l2_norm: brain.model.a_mean.weight_l2_norm(),
                gradient_l2_norm: brain.model.a_mean.grad_l2_norm(),
                saturated_fraction: None,
            },
            A2cLayerHealth {
                layer_name: "critic_fc1".to_string(),
                weight_l2_norm: brain.model.c_fc1.weight_l2_norm(),
                gradient_l2_norm: brain.model.c_fc1.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.critic_dead[0], acc.critic_seen[0])),
            },
            A2cLayerHealth {
                layer_name: "critic_fc2".to_string(),
                weight_l2_norm: brain.model.c_fc2.weight_l2_norm(),
                gradient_l2_norm: brain.model.c_fc2.grad_l2_norm(),
                saturated_fraction: Some(fraction(acc.critic_dead[1], acc.critic_seen[1])),
            },
            A2cLayerHealth {
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
            brain.ppo_epochs,
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
    brain: &mut A2cBrain,
    buffer: &TrainerRolloutBuffer,
    stats: &mut A2cTrainingStats,
    bootstrap_values: &HashMap<u32, f32>,
) {
    let Some(mut prepared) = ppo_prepare_update(brain, buffer, bootstrap_values) else {
        return;
    };
    let batch_size = buffer.len();
    while prepared.is_active() {
        let is_final = prepared.epochs_remaining == 1;
        // Process entire epoch in one go
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

fn collect_saturated_tanh(cache: Option<&[f32]>, saturated: &mut usize, seen: &mut usize) {
    let Some(values) = cache else {
        return;
    };
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

    let mut sumsq = 0.0;
    for layer in layers.iter() {
        sumsq += layer
            .grad_weights
            .iter()
            .flat_map(|row| row.iter())
            .map(|grad| grad * grad)
            .sum::<f32>();
        sumsq += layer
            .grad_biases
            .iter()
            .map(|grad| grad * grad)
            .sum::<f32>();
    }

    let norm = sumsq.sqrt();
    if norm <= max_norm || norm <= 1e-8 {
        return;
    }

    let scale = max_norm / norm;
    for layer in layers.iter_mut() {
        for row in &mut layer.grad_weights {
            for grad in row {
                *grad *= scale;
            }
        }
        for grad in &mut layer.grad_biases {
            *grad *= scale;
        }
    }
}
