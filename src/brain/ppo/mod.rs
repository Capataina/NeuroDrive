pub mod buffer;
pub mod model;
pub mod update;

use std::collections::HashMap;

use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::agent::action::{
    ActionState, CarAction, action_smoothing_system, keyboard_action_input_system,
};
use crate::agent::observation::{OBSERVATION_DIM, ObservationVector};
use crate::brain::types::{AgentMode, PolicyOutput};
use crate::game::car::{Car, EnvInstanceId};
use crate::game::episode::EpisodeState;

use self::buffer::TrainerRolloutBuffer;
use self::model::ActorCritic;
use self::update::{
    PreparedUpdate, ppo_finish_epoch, ppo_prepare_update, ppo_process_chunk,
    ppo_update_blocking, squashed_gaussian_log_prob,
};

/// All PPO hyperparameters in a single canonical location.
#[derive(Clone, Debug)]
pub struct PpoConfig {
    // Rollout
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub min_update_steps: usize,
    pub ppo_epochs: usize,
    pub clip_epsilon: f32,
    pub samples_per_tick: usize,
    // Network
    pub actor_hidden_dim: usize,
    pub critic_hidden_dim: usize,
    // Optimiser
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub critic_weight_decay: f32,
    pub entropy_coef: f32,
    pub actor_grad_clip: f32,
    pub critic_grad_clip: f32,
    pub value_huber_delta: f32,
    // Exploration
    pub log_std_floor: f32,
    pub log_std_ceil: f32,
    pub log_std_lr: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            gae_lambda: 0.95,
            max_steps: 512,
            min_update_steps: 128,
            ppo_epochs: 4,
            clip_epsilon: 0.2,
            samples_per_tick: 64,
            actor_hidden_dim: 64,
            critic_hidden_dim: 128,
            actor_lr: 3e-4,
            critic_lr: 5e-4,
            critic_weight_decay: 3e-4,
            entropy_coef: 0.01,
            actor_grad_clip: 0.5,
            critic_grad_clip: 0.5,
            value_huber_delta: 1.0,
            log_std_floor: -1.0,
            log_std_ceil: 0.5,
            log_std_lr: 3e-4,
        }
    }
}

/// Shared PPO brain resource. Owns the policy/value network and hyperparameters.
/// The rollout buffer is now a separate `TrainerRolloutBuffer` resource.
#[derive(Resource)]
pub struct PpoBrain {
    pub model: ActorCritic,
    pub rng: StdRng,
    pub config: PpoConfig,
    pub step_counter: usize,
}

impl Default for PpoBrain {
    fn default() -> Self {
        let config = PpoConfig::default();
        let mut init_rng = rand::rng();
        Self {
            model: ActorCritic::new(OBSERVATION_DIM, config.actor_hidden_dim, config.critic_hidden_dim, 2, config.actor_lr, config.critic_lr, config.critic_weight_decay, &mut init_rng),
            rng: StdRng::from_rng(&mut init_rng),
            config,
            step_counter: 0,
        }
    }
}

/// Snapshot of one layer's parameter and activation health after a PPO update.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PpoLayerHealth {
    pub layer_name: String,
    pub weight_l2_norm: f32,
    pub gradient_l2_norm: f32,
    pub saturated_fraction: Option<f32>,
}

/// Aggregated learning-health metrics for the most recent completed PPO update.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct PpoTrainingStats {
    pub last_completed_update: u64,
    pub batch_size: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub policy_entropy: f32,
    pub explained_variance: f32,
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
    pub clamped_action_fraction: f32,
    pub clip_fraction: f32,
    pub approx_kl: f32,
    pub layer_health: Vec<PpoLayerHealth>,
}

/// Holds an in-progress PPO update that is amortised across frames.
/// One epoch runs per `FixedUpdate` tick to keep the simulation smooth.
#[derive(Resource, Default)]
pub struct PpoUpdateState {
    prepared: Option<PreparedUpdate>,
}

pub struct PpoPlugin;

impl Plugin for PpoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PpoBrain>()
            .init_resource::<PpoTrainingStats>()
            .init_resource::<TrainerRolloutBuffer>()
            .init_resource::<PpoUpdateState>()
            .add_systems(
                FixedUpdate,
                ppo_act_all_cars_system
                    .after(keyboard_action_input_system)
                    .before(action_smoothing_system)
                    .in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                (
                    ppo_collect_rewards_all_cars_system,
                    ppo_epoch_system.after(ppo_collect_rewards_all_cars_system),
                )
                    .after(crate::game::episode::episode_loop_system)
                    .after(crate::agent::observation::build_observation_vector_system)
                    .in_set(crate::sim::sets::SimSet::Measurement),
            )
            .add_systems(Last, ppo_flush_on_exit_system);
    }
}

/// Per-car intermediate results from the actor pass, held briefly between
/// the actor and batched-critic passes within `ppo_act_all_cars_system`.
struct CarActResult {
    entity: bevy::ecs::entity::Entity,
    env_id: u32,
    actions: [f32; 2],
    latent_actions: [f32; 2],
    safety_clamp_hits: [bool; 2],
    old_log_prob: f32,
    steering_mean: f32,
    steering_std: f32,
    throttle_mean: f32,
    throttle_std: f32,
}

/// Runs the shared policy for all cars, writes per-car actions, and pushes
/// all transitions to the TrainerRolloutBuffer with env_id tagging.
///
/// Structured as two passes to batch critic evaluation:
/// 1. Per-car actor forward + action sampling (sequential mat-vec — unavoidable)
/// 2. Single batched critic forward for all cars (one mat-mat instead of N mat-vec)
/// 3. Distribute value predictions and push to buffer
pub fn ppo_act_all_cars_system(
    mode: Res<AgentMode>,
    mut car_query: Query<(bevy::ecs::entity::Entity, &EnvInstanceId, &ObservationVector, &mut ActionState, &mut PolicyOutput), With<Car>>,
    mut brain: ResMut<PpoBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    // ── Pass 1: actor forward + action sampling, collect obs for critic ──
    let mut results: Vec<CarActResult> = Vec::new();
    let mut obs_stack: Vec<f32> = Vec::new();

    for (entity, env_id, obs, mut action_state, _policy_output) in car_query.iter_mut() {
        let action_dist = brain.model.forward_actor(&obs.values);

        let mut actions = [0.0f32; 2];
        let mut latent_actions = [0.0f32; 2];

        for i in 0..2 {
            let latent = crate::brain::common::math::sample_normal(
                action_dist.mean[i],
                action_dist.std[i],
                &mut brain.rng,
            );
            latent_actions[i] = latent;

            let squashed = latent.tanh();
            actions[i] = if i == 1 {
                0.5 * (squashed + 1.0)
            } else {
                squashed
            };
        }

        let raw_action = CarAction {
            steering: actions[0],
            throttle: actions[1],
        };
        let applied_action = raw_action.clamped();
        let safety_clamp_hits = [
            (applied_action.steering - raw_action.steering).abs() > 1e-6,
            (applied_action.throttle - raw_action.throttle).abs() > 1e-6,
        ];

        actions[0] = applied_action.steering;
        actions[1] = applied_action.throttle;

        action_state.desired = applied_action;

        let mut old_log_prob = 0.0;
        for j in 0..2 {
            let squashed = if j == 0 {
                actions[j]
            } else {
                2.0 * actions[j] - 1.0
            };
            old_log_prob += squashed_gaussian_log_prob(
                latent_actions[j],
                squashed,
                action_dist.mean[j],
                action_dist.std[j],
                j,
            );
        }

        obs_stack.extend_from_slice(&obs.values);

        results.push(CarActResult {
            entity,
            env_id: env_id.0,
            actions,
            latent_actions,
            safety_clamp_hits,
            old_log_prob,
            steering_mean: action_dist.mean[0],
            steering_std: action_dist.std[0],
            throttle_mean: action_dist.mean[1],
            throttle_std: action_dist.std[1],
        });
    }

    let car_count = results.len();
    if car_count == 0 {
        return;
    }

    // ── Pass 2: single batched critic forward ───────────────────────
    brain.model.forward_critic_batch(&obs_stack, car_count);

    // ── Pass 3: distribute values, write PolicyOutput, push to buffer ─
    for (i, res) in results.iter().enumerate() {
        let value = brain.model.scratch.c_out[i];

        if let Ok((_entity, _env_id, obs, _action_state, mut policy_output)) = car_query.get_mut(res.entity) {
            policy_output.value_prediction = value;
            policy_output.steering_mean = res.steering_mean;
            policy_output.steering_std = res.steering_std;
            policy_output.throttle_mean = res.throttle_mean;
            policy_output.throttle_std = res.throttle_std;

            buffer.push_pre_step(
                res.env_id,
                &obs.values,
                &res.actions,
                &res.latent_actions,
                res.safety_clamp_hits,
                value,
                res.old_log_prob,
            );
        }
    }

    brain.step_counter += car_count;
}

/// Collects per-car rewards and done flags. When the buffer reaches the
/// horizon, prepares a PPO update (GAE + frozen buffer) for the epoch system
/// to process one epoch per tick.
pub fn ppo_collect_rewards_all_cars_system(
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<PpoBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut update_state: ResMut<PpoUpdateState>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    if buffer.pending_rewards() == 0 {
        return;
    }

    let mut any_done = false;
    for (_, _, episode_state) in car_query.iter() {
        let done = episode_state.tick.end_reason.is_some();
        buffer.push_reward(episode_state.tick.reward, done);
        if done {
            any_done = true;
        }
    }

    debug_assert!(
        buffer.is_aligned(),
        "Trainer rollout buffer misaligned: states={}, rewards={}, env_ids={}",
        buffer.pre_step_count(),
        buffer.len(),
        buffer.env_ids.len(),
    );

    let reached_horizon = buffer.len() >= brain.config.max_steps;
    let reached_terminal_batch = any_done && buffer.len() >= brain.config.min_update_steps;

    // Only start a new update if there is no epoch-spread update in progress
    if (reached_horizon || reached_terminal_batch) && update_state.prepared.is_none() {
        let mut bootstrap_values: HashMap<u32, f32> = HashMap::new();
        for (env_id, obs, episode_state) in car_query.iter() {
            let done = episode_state.tick.end_reason.is_some();
            if done {
                bootstrap_values.insert(env_id.0, 0.0);
            } else {
                let value = brain.model.forward_critic(&obs.values);
                bootstrap_values.insert(env_id.0, value);
            }
        }

        if let Some(prepared) = ppo_prepare_update(&mut brain, &mut buffer, &bootstrap_values) {
            update_state.prepared = Some(prepared);
            // buffer is already empty — ppo_prepare_update took its contents.
        } else {
            buffer.clear();
        }
    }
}

/// Processes a chunk of samples per tick from the in-progress PPO update.
/// A full epoch is split across multiple ticks (samples_per_tick samples each)
/// so the simulation stays smooth.
pub fn ppo_epoch_system(
    mode: Res<AgentMode>,
    mut brain: ResMut<PpoBrain>,
    mut update_state: ResMut<PpoUpdateState>,
    mut stats: ResMut<PpoTrainingStats>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    let Some(prepared) = update_state.prepared.as_mut() else {
        return;
    };

    if !prepared.is_active() {
        update_state.prepared = None;
        return;
    }

    let chunk_size = brain.config.samples_per_tick;
    let epoch_complete = ppo_process_chunk(&mut brain, prepared, chunk_size);

    if epoch_complete {
        let is_final = prepared.epochs_remaining == 1;
        ppo_finish_epoch(&mut brain, prepared, &mut stats, is_final);
        prepared.epochs_remaining -= 1;
        prepared.sample_offset = 0;

        if prepared.epochs_remaining == 0 {
            update_state.prepared = None;
        }
    }
}

/// Flushes any in-progress PPO epochs and remaining rollout data on exit.
/// Runs synchronously since frame budget does not matter at shutdown.
pub fn ppo_flush_on_exit_system(
    mut exit_events: MessageReader<AppExit>,
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<PpoBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut update_state: ResMut<PpoUpdateState>,
    mut stats: ResMut<PpoTrainingStats>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    if *mode != AgentMode::Ai {
        buffer.clear();
        update_state.prepared = None;
        return;
    }

    // Finish any in-progress staged update
    if let Some(prepared) = update_state.prepared.as_mut() {
        let batch_size = prepared.frozen_buffer.len();
        // Finish current epoch if partially processed
        if prepared.sample_offset > 0 {
            ppo_process_chunk(&mut brain, prepared, batch_size);
            let is_final = prepared.epochs_remaining == 1;
            ppo_finish_epoch(&mut brain, prepared, &mut stats, is_final);
            prepared.epochs_remaining -= 1;
            prepared.sample_offset = 0;
        }
        // Run any remaining full epochs
        while prepared.is_active() {
            let is_final = prepared.epochs_remaining == 1;
            ppo_process_chunk(&mut brain, prepared, batch_size);
            ppo_finish_epoch(&mut brain, prepared, &mut stats, is_final);
            prepared.epochs_remaining -= 1;
            prepared.sample_offset = 0;
        }
        update_state.prepared = None;
    }

    // Flush any remaining buffer data
    if buffer.len() == 0 {
        return;
    }

    let mut bootstrap_values: HashMap<u32, f32> = HashMap::new();
    for (env_id, obs, episode_state) in car_query.iter() {
        let done = episode_state.tick.end_reason.is_some();
        if done {
            bootstrap_values.insert(env_id.0, 0.0);
        } else {
            let (_, value) = brain.model.forward(&obs.values);
            bootstrap_values.insert(env_id.0, value);
        }
    }

    ppo_update_blocking(&mut brain, &mut buffer, &mut stats, &bootstrap_values);
    // buffer is already empty — ppo_update_blocking took its contents.
}
