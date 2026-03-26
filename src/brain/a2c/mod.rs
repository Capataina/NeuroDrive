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
use self::update::a2c_update;

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
    pub layer_health: Vec<A2cLayerHealth>,
}

pub struct A2cPlugin;

impl Plugin for A2cPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<A2cBrain>()
            .init_resource::<A2cTrainingStats>()
            .init_resource::<TrainerRolloutBuffer>()
            .add_systems(
                FixedUpdate,
                a2c_act_all_cars_system
                    .after(keyboard_action_input_system)
                    .before(action_smoothing_system)
                    .in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                a2c_collect_rewards_all_cars_system
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

        buffer.push_pre_step(
            env_id.0,
            obs.values.to_vec(),
            actions.to_vec(),
            latent_actions.to_vec(),
            safety_clamp_hits,
            value,
        );
    }

    brain.step_counter += car_query.iter().count();
}

/// Collects per-car rewards and done flags, then triggers a shared A2C update
/// when the total buffer reaches the horizon.
pub fn a2c_collect_rewards_all_cars_system(
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<A2cBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut stats: ResMut<A2cTrainingStats>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    // Only push rewards if we have pending pre-step entries
    if buffer.pending_rewards() == 0 {
        return;
    }

    // Push reward/done for each car, matching the order they were acted on
    // The pre-step entries were pushed in query iteration order, so we push
    // rewards in the same order.
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

    if reached_horizon || reached_terminal_batch {
        // Compute per-env bootstrap values for non-terminal envs
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

        a2c_update(&mut brain, &mut buffer, &mut stats, &bootstrap_values);
    }
}

/// Flushes any remaining rollout data on application exit.
pub fn a2c_flush_on_exit_system(
    mut exit_events: MessageReader<AppExit>,
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<A2cBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut stats: ResMut<A2cTrainingStats>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    if *mode != AgentMode::Ai {
        buffer.clear();
        return;
    }

    if buffer.len() == 0 {
        return;
    }

    // Compute per-env bootstrap values
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

    a2c_update(&mut brain, &mut buffer, &mut stats, &bootstrap_values);
}
