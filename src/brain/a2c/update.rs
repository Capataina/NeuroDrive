use crate::brain::a2c::{A2cBrain, A2cLayerHealth, A2cTrainingStats};
use crate::brain::common::math::{normal_entropy, normal_log_prob};
use crate::brain::common::mlp::Linear;

const VALUE_HUBER_DELTA: f32 = 1.0;
const ACTOR_GRAD_CLIP_NORM: f32 = 0.5;
const CRITIC_GRAD_CLIP_NORM: f32 = 0.5;

/// Runs one full A2C update from the currently buffered rollout and records
/// learning and network-health metrics for analytics export.
pub fn a2c_update(
    brain: &mut A2cBrain,
    stats: &mut A2cTrainingStats,
    bootstrap_state: Option<&[f32]>,
) {
    if brain.buffer.rewards.is_empty() {
        return;
    }

    let batch_size = brain.buffer.rewards.len();
    if brain.buffer.states.len() != batch_size
        || brain.buffer.actions.len() != batch_size
        || brain.buffer.latent_actions.len() != batch_size
        || brain.buffer.values.len() != batch_size
        || brain.buffer.dones.len() != batch_size
        || brain.buffer.safety_clamp_hits.len() != batch_size
    {
        bevy::log::warn!(
            "A2C rollout misalignment detected (s={}, a={}, z={}, v={}, r={}, d={}, c={}); skipping update.",
            brain.buffer.states.len(),
            brain.buffer.actions.len(),
            brain.buffer.latent_actions.len(),
            brain.buffer.values.len(),
            brain.buffer.rewards.len(),
            brain.buffer.dones.len(),
            brain.buffer.safety_clamp_hits.len(),
        );
        brain.buffer.clear();
        return;
    }

    let mut next_value = 0.0;
    if !brain.buffer.dones.last().unwrap_or(&false) {
        if let Some(bootstrap_state) = bootstrap_state {
            let (_, val) = brain.model.forward(bootstrap_state);
            next_value = val;
        } else {
            bevy::log::warn!("Missing bootstrap observation for non-terminal A2C rollout; using zero bootstrap value.");
        }
    }

    let (advantages, returns) = brain
        .buffer
        .compute_gae(next_value, brain.gamma, brain.gae_lambda);
    let entropy_coef = 0.01;

    brain.model.zero_grad();

    let batch_size_f32 = batch_size as f32;
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut entropy_sum = 0.0;
    let mut action_sum = [0.0; 2];
    let mut action_sumsq = [0.0; 2];
    let mut clamped_action_count = 0usize;
    let mut actor_dead = [0usize; 2];
    let mut critic_dead = [0usize; 2];
    let mut actor_seen = [0usize; 2];
    let mut critic_seen = [0usize; 2];

    for i in 0..batch_size {
        let state = &brain.buffer.states[i];
        let action = &brain.buffer.actions[i];
        let latent_action = &brain.buffer.latent_actions[i];
        let adv = advantages[i];
        let ret = returns[i];

        let (action_dist, value) = brain.model.forward(state);
        collect_dead_relu(
            brain.model.a_relu1.input_cache.as_deref(),
            &mut actor_dead[0],
            &mut actor_seen[0],
        );
        collect_dead_relu(
            brain.model.a_relu2.input_cache.as_deref(),
            &mut actor_dead[1],
            &mut actor_seen[1],
        );
        collect_dead_relu(
            brain.model.c_relu1.input_cache.as_deref(),
            &mut critic_dead[0],
            &mut critic_seen[0],
        );
        collect_dead_relu(
            brain.model.c_relu2.input_cache.as_deref(),
            &mut critic_dead[1],
            &mut critic_seen[1],
        );

        let value_error = value - ret;
        let value_grad = if value_error.abs() <= VALUE_HUBER_DELTA {
            value_error
        } else {
            VALUE_HUBER_DELTA * value_error.signum()
        };
        value_loss_sum += if value_error.abs() <= VALUE_HUBER_DELTA {
            0.5 * value_error.powi(2)
        } else {
            VALUE_HUBER_DELTA * (value_error.abs() - 0.5 * VALUE_HUBER_DELTA)
        };
        let d_value = vec![value_grad / batch_size_f32];

        let c2_g = brain.model.c_value.backward(&d_value);
        let c2_r_g = brain.model.c_relu2.backward(&c2_g);
        let c1_g = brain.model.c_fc2.backward(&c2_r_g);
        let c1_r_g = brain.model.c_relu1.backward(&c1_g);
        brain.model.c_fc1.backward(&c1_r_g);

        let mut d_mean = vec![0.0; 2];

        for j in 0..2 {
            let latent = latent_action[j];
            let a = action[j];
            let squashed = if j == 0 { a } else { 2.0 * a - 1.0 };
            let mean = action_dist.mean[j];
            let std = action_dist.std[j];
            let log_prob = squashed_gaussian_log_prob(latent, squashed, mean, std, j);
            let entropy = normal_entropy(std);

            let d_lp_d_mean = (latent - mean) / (std * std + 1e-8);
            let d_lp_d_log_std = ((latent - mean).powi(2) / (std * std + 1e-8)) - 1.0;

            let d_loss_d_mean = -adv * d_lp_d_mean;
            let d_loss_d_log_std = -adv * d_lp_d_log_std - entropy_coef;

            d_mean[j] = d_loss_d_mean / batch_size_f32;
            brain.model.a_log_std_grad[j] += d_loss_d_log_std / batch_size_f32;

            policy_loss_sum += -adv * log_prob;
            entropy_sum += entropy;
            action_sum[j] += a;
            action_sumsq[j] += a * a;
            if brain.buffer.safety_clamp_hits[i][j] {
                clamped_action_count += 1;
            }
        }

        let a2_g = brain.model.a_mean.backward(&d_mean);
        let a2_r_g = brain.model.a_relu2.backward(&a2_g);
        let a1_g = brain.model.a_fc2.backward(&a2_r_g);
        let a1_r_g = brain.model.a_relu1.backward(&a1_g);
        brain.model.a_fc1.backward(&a1_r_g);
    }

    clip_linear_gradients(
        &mut [&mut brain.model.a_fc1, &mut brain.model.a_fc2, &mut brain.model.a_mean],
        ACTOR_GRAD_CLIP_NORM,
    );
    clip_linear_gradients(
        &mut [&mut brain.model.c_fc1, &mut brain.model.c_fc2, &mut brain.model.c_value],
        CRITIC_GRAD_CLIP_NORM,
    );

    brain
        .model
        .a_opt
        .step(&mut [&mut brain.model.a_fc1, &mut brain.model.a_fc2, &mut brain.model.a_mean]);
    brain
        .model
        .c_opt
        .step(&mut [&mut brain.model.c_fc1, &mut brain.model.c_fc2, &mut brain.model.c_value]);

    brain.model.opt_t += 1.0;
    for j in 0..2 {
        let g = brain.model.a_log_std_grad[j];
        brain.model.log_std_opt_m[j] = 0.9 * brain.model.log_std_opt_m[j] + 0.1 * g;
        brain.model.log_std_opt_v[j] = 0.999 * brain.model.log_std_opt_v[j] + 0.001 * g * g;

        let m_hat = brain.model.log_std_opt_m[j] / (1.0 - 0.9f32.powf(brain.model.opt_t));
        let v_hat = brain.model.log_std_opt_v[j] / (1.0 - 0.999f32.powf(brain.model.opt_t));

        brain.model.a_log_std[j] -= 3e-4 * m_hat / (v_hat.sqrt() + 1e-8);
        brain.model.a_log_std[j] = brain.model.a_log_std[j].clamp(-2.0, 0.5);
    }

    let value_predictions = brain.buffer.values.clone();
    stats.last_completed_update = stats.last_completed_update.saturating_add(1);
    stats.batch_size = batch_size;
    stats.policy_loss = policy_loss_sum / batch_size_f32.max(1.0);
    stats.value_loss = value_loss_sum / batch_size_f32.max(1.0);
    stats.policy_entropy = entropy_sum / (batch_size_f32 * 2.0).max(1.0);
    stats.explained_variance = explained_variance(&returns, &value_predictions);
    stats.steering_mean = action_sum[0] / batch_size_f32.max(1.0);
    stats.steering_std = std_from_sums(action_sum[0], action_sumsq[0], batch_size);
    stats.throttle_mean = action_sum[1] / batch_size_f32.max(1.0);
    stats.throttle_std = std_from_sums(action_sum[1], action_sumsq[1], batch_size);
    stats.clamped_action_fraction =
        clamped_action_count as f32 / (batch_size.saturating_mul(2) as f32).max(1.0);
    stats.layer_health = vec![
        A2cLayerHealth {
            layer_name: "actor_fc1".to_string(),
            weight_l2_norm: brain.model.a_fc1.weight_l2_norm(),
            gradient_l2_norm: brain.model.a_fc1.grad_l2_norm(),
            dead_relu_fraction: Some(fraction(actor_dead[0], actor_seen[0])),
        },
        A2cLayerHealth {
            layer_name: "actor_fc2".to_string(),
            weight_l2_norm: brain.model.a_fc2.weight_l2_norm(),
            gradient_l2_norm: brain.model.a_fc2.grad_l2_norm(),
            dead_relu_fraction: Some(fraction(actor_dead[1], actor_seen[1])),
        },
        A2cLayerHealth {
            layer_name: "actor_mean".to_string(),
            weight_l2_norm: brain.model.a_mean.weight_l2_norm(),
            gradient_l2_norm: brain.model.a_mean.grad_l2_norm(),
            dead_relu_fraction: None,
        },
        A2cLayerHealth {
            layer_name: "critic_fc1".to_string(),
            weight_l2_norm: brain.model.c_fc1.weight_l2_norm(),
            gradient_l2_norm: brain.model.c_fc1.grad_l2_norm(),
            dead_relu_fraction: Some(fraction(critic_dead[0], critic_seen[0])),
        },
        A2cLayerHealth {
            layer_name: "critic_fc2".to_string(),
            weight_l2_norm: brain.model.c_fc2.weight_l2_norm(),
            gradient_l2_norm: brain.model.c_fc2.grad_l2_norm(),
            dead_relu_fraction: Some(fraction(critic_dead[1], critic_seen[1])),
        },
        A2cLayerHealth {
            layer_name: "critic_value".to_string(),
            weight_l2_norm: brain.model.c_value.weight_l2_norm(),
            gradient_l2_norm: brain.model.c_value.grad_l2_norm(),
            dead_relu_fraction: None,
        },
    ];

    brain.buffer.clear();
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

fn collect_dead_relu(cache: Option<&[f32]>, dead: &mut usize, seen: &mut usize) {
    let Some(values) = cache else {
        return;
    };
    *seen += values.len();
    *dead += values.iter().filter(|value| **value <= 0.0).count();
}

fn fraction(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn squashed_gaussian_log_prob(
    latent: f32,
    squashed: f32,
    mean: f32,
    std: f32,
    component_idx: usize,
) -> f32 {
    let gaussian_log_prob = normal_log_prob(latent, mean, std);
    let log_det_jacobian = (1.0 - squashed * squashed + 1e-6).ln();
    let affine_log_det = if component_idx == 1 { (2.0f32).ln() } else { 0.0 };
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
