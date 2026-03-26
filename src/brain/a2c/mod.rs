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
use crate::brain::types::AgentMode;
use crate::game::car::{Car, EnvInstanceId};
use crate::game::episode::EpisodeState;

use self::buffer::TrainerRolloutBuffer;
use self::model::ActorCritic;
use self::update::{
    PreparedUpdate, ppo_finish_epoch, ppo_prepare_update, ppo_process_chunk,
    ppo_update_blocking, squashed_gaussian_log_prob,
};

/// Shared A2C brain resource. Owns the policy/value network and hyperparameters.
/// The rollout buffer is now a separate `TrainerRolloutBuffer` resource.
#[derive(Resource)]
pub struct A2cBrain {
    pub model: ActorCritic,
    pub rng: StdRng,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub min_update_steps: usize,
    pub step_counter: usize,
    pub ppo_epochs: usize,
    pub clip_epsilon: f32,
    pub samples_per_tick: usize,
}

impl Default for A2cBrain {
    fn default() -> Self {
        let mut init_rng = rand::rng();
        Self {
            model: ActorCritic::new(OBSERVATION_DIM, 64, 2, &mut init_rng),
            rng: StdRng::from_rng(&mut init_rng),
            gamma: 0.99,
            gae_lambda: 0.95,
            max_steps: 512,
            min_update_steps: 128,
            step_counter: 0,
            ppo_epochs: 4,
            clip_epsilon: 0.2,
            samples_per_tick: 128,
        }
    }
}

/// Snapshot of one layer's parameter and activation health after an A2C update.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct A2cLayerHealth {
    pub layer_name: String,
    pub weight_l2_norm: f32,
    pub gradient_l2_norm: f32,
    pub dead_relu_fraction: Option<f32>,
}

/// Aggregated learning-health metrics for the most recent completed A2C update.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct A2cTrainingStats {
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
    pub layer_health: Vec<A2cLayerHealth>,
}

/// Holds an in-progress PPO update that is amortised across frames.
/// One epoch runs per `FixedUpdate` tick to keep the simulation smooth.
#[derive(Resource, Default)]
pub struct PpoUpdateState {
    prepared: Option<PreparedUpdate>,
}

pub struct A2cPlugin;

impl Plugin for A2cPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<A2cBrain>()
            .init_resource::<A2cTrainingStats>()
            .init_resource::<TrainerRolloutBuffer>()
            .init_resource::<PpoUpdateState>()
            .add_systems(
                FixedUpdate,
                a2c_act_all_cars_system
                    .after(keyboard_action_input_system)
                    .before(action_smoothing_system)
                    .in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                (
                    a2c_collect_rewards_all_cars_system,
                    ppo_epoch_system.after(a2c_collect_rewards_all_cars_system),
                )
                    .after(crate::game::episode::episode_loop_system)
                    .after(crate::agent::observation::build_observation_vector_system)
                    .in_set(crate::sim::sets::SimSet::Measurement),
            )
            .add_systems(Last, a2c_flush_on_exit_system);
    }
}

/// Runs the shared policy for all cars, writes per-car actions, and pushes
/// all transitions to the TrainerRolloutBuffer with env_id tagging.
pub fn a2c_act_all_cars_system(
    mode: Res<AgentMode>,
    mut car_query: Query<(&EnvInstanceId, &ObservationVector, &mut ActionState), With<Car>>,
    mut brain: ResMut<A2cBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    for (env_id, obs, mut action_state) in car_query.iter_mut() {
        let (action_dist, value) = brain.model.forward(&obs.values);

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
            actions[i] = if i == 0 {
                squashed
            } else {
                0.5 * (squashed + 1.0)
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

        buffer.push_pre_step(
            env_id.0,
            obs.values.to_vec(),
            actions.to_vec(),
            latent_actions.to_vec(),
            safety_clamp_hits,
            value,
            old_log_prob,
        );
    }

    brain.step_counter += car_query.iter().count();
}

/// Collects per-car rewards and done flags. When the buffer reaches the
/// horizon, prepares a PPO update (GAE + frozen buffer) for the epoch system
/// to process one epoch per tick.
pub fn a2c_collect_rewards_all_cars_system(
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<A2cBrain>,
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
        let done = episode_state.current_tick_end_reason.is_some();
        buffer.push_reward(episode_state.current_tick_reward, done);
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

    let reached_horizon = buffer.len() >= brain.max_steps;
    let reached_terminal_batch = any_done && buffer.len() >= brain.min_update_steps;

    // Only start a new update if there is no epoch-spread update in progress
    if (reached_horizon || reached_terminal_batch) && update_state.prepared.is_none() {
        let mut bootstrap_values: HashMap<u32, f32> = HashMap::new();
        for (env_id, obs, episode_state) in car_query.iter() {
            let done = episode_state.current_tick_end_reason.is_some();
            if done {
                bootstrap_values.insert(env_id.0, 0.0);
            } else {
                let (_, value) = brain.model.forward(&obs.values);
                bootstrap_values.insert(env_id.0, value);
            }
        }

        if let Some(prepared) = ppo_prepare_update(&brain, &buffer, &bootstrap_values) {
            update_state.prepared = Some(prepared);
            buffer.clear();
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
    mut brain: ResMut<A2cBrain>,
    mut update_state: ResMut<PpoUpdateState>,
    mut stats: ResMut<A2cTrainingStats>,
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

    let chunk_size = brain.samples_per_tick;
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
pub fn a2c_flush_on_exit_system(
    mut exit_events: MessageReader<AppExit>,
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<A2cBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut update_state: ResMut<PpoUpdateState>,
    mut stats: ResMut<A2cTrainingStats>,
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
        let done = episode_state.current_tick_end_reason.is_some();
        if done {
            bootstrap_values.insert(env_id.0, 0.0);
        } else {
            let (_, value) = brain.model.forward(&obs.values);
            bootstrap_values.insert(env_id.0, value);
        }
    }

    ppo_update_blocking(&mut brain, &buffer, &mut stats, &bootstrap_values);
    buffer.clear();
}
