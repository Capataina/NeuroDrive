pub mod buffer;
pub mod model;
pub mod update;

use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent::action::{
    ActionState, CarAction, action_smoothing_system, keyboard_action_input_system,
};
use crate::agent::observation::{OBSERVATION_DIM, ObservationVector};
use crate::brain::types::{AgentMode, Brain};
use crate::game::episode::EpisodeState;

use self::buffer::RolloutBuffer;
use self::model::ActorCritic;
use self::update::a2c_update;

#[derive(Resource)]
pub struct A2cBrain {
    pub model: ActorCritic,
    pub buffer: RolloutBuffer,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub min_update_steps: usize,
    pub step_counter: usize,
}

impl Default for A2cBrain {
    fn default() -> Self {
        let mut rng = rand::rng();
        Self {
            model: ActorCritic::new(OBSERVATION_DIM, 64, 2, &mut rng),
            buffer: RolloutBuffer::new(),
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

impl Brain for A2cBrain {
    fn act(&mut self, obs: &ObservationVector) -> CarAction {
        let mut rng = rand::rng();
        let (action_dist, value) = self.model.forward(&obs.values);

        let mut actions = vec![0.0; 2];
        let mut latent_actions = vec![0.0; 2];

        for i in 0..2 {
            let mean = action_dist.mean[i];
            let std = action_dist.std[i];
            let latent = crate::brain::common::math::sample_normal(mean, std, &mut rng);
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

        self.buffer.states.push(obs.values.to_vec());
        self.buffer.actions.push(actions.clone());
        self.buffer.latent_actions.push(latent_actions);
        self.buffer.safety_clamp_hits.push(safety_clamp_hits);
        self.buffer.values.push(value);

        applied_action
    }
}

pub struct A2cPlugin;

impl Plugin for A2cPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<A2cBrain>()
            .init_resource::<A2cTrainingStats>()
            .add_systems(
                FixedUpdate,
                a2c_act_system
                    .after(keyboard_action_input_system)
                    .before(action_smoothing_system)
                    .in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                a2c_collect_reward_system
                    .after(crate::game::episode::episode_loop_system)
                    .after(crate::agent::observation::build_observation_vector_system)
                    .in_set(crate::sim::sets::SimSet::Measurement),
            )
            .add_systems(Last, a2c_flush_on_exit_system);
    }
}

pub fn a2c_act_system(
    mode: Res<AgentMode>,
    obs_query: Query<&ObservationVector>,
    mut action_state: ResMut<ActionState>,
    mut brain: ResMut<A2cBrain>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    if let Ok(obs) = obs_query.single() {
        let action = brain.act(obs);
        action_state.desired = action;
        brain.step_counter += 1;
    }
}

pub fn a2c_collect_reward_system(
    mode: Res<AgentMode>,
    obs_query: Query<&ObservationVector>,
    episode_state: Res<EpisodeState>,
    mut brain: ResMut<A2cBrain>,
    mut stats: ResMut<A2cTrainingStats>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    if brain.buffer.states.len() > brain.buffer.rewards.len() {
        brain.buffer.rewards.push(episode_state.current_tick_reward);

        let done = episode_state.current_tick_end_reason.is_some();
        brain.buffer.dones.push(done);

        debug_assert_eq!(brain.buffer.states.len(), brain.buffer.actions.len());
        debug_assert_eq!(brain.buffer.states.len(), brain.buffer.latent_actions.len());
        debug_assert_eq!(brain.buffer.states.len(), brain.buffer.values.len());
        debug_assert_eq!(
            brain.buffer.states.len(),
            brain.buffer.safety_clamp_hits.len()
        );
        debug_assert_eq!(brain.buffer.rewards.len(), brain.buffer.dones.len());

        let reached_horizon = brain.buffer.states.len() >= brain.max_steps;
        let reached_terminal_batch = done && brain.buffer.states.len() >= brain.min_update_steps;

        if reached_horizon || reached_terminal_batch {
            let bootstrap_state = if done {
                None
            } else {
                obs_query.single().ok().map(|obs| obs.values.to_vec())
            };
            a2c_update(&mut brain, &mut stats, bootstrap_state.as_deref());
        }
    }
}

pub fn a2c_flush_on_exit_system(
    mut exit_events: MessageReader<AppExit>,
    mode: Res<AgentMode>,
    obs_query: Query<&ObservationVector>,
    mut brain: ResMut<A2cBrain>,
    mut stats: ResMut<A2cTrainingStats>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    if *mode != AgentMode::Ai {
        brain.buffer.clear();
        return;
    }

    if brain.buffer.rewards.is_empty() {
        return;
    }

    let bootstrap_state = if brain.buffer.dones.last().copied().unwrap_or(true) {
        None
    } else {
        obs_query.single().ok().map(|obs| obs.values.to_vec())
    };
    a2c_update(&mut brain, &mut stats, bootstrap_state.as_deref());
}
