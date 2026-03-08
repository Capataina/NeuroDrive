pub mod buffer;
pub mod model;
pub mod update;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent::action::{ActionState, CarAction};
use crate::agent::observation::ObservationVector;
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
    pub step_counter: usize,
}

impl Default for A2cBrain {
    fn default() -> Self {
        let mut rng = rand::rng();
        Self {
            model: ActorCritic::new(14, 64, 2, &mut rng),
            buffer: RolloutBuffer::new(),
            gamma: 0.99,
            gae_lambda: 0.95,
            max_steps: 2048,
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
        let mut log_probs = vec![0.0; 2];

        for i in 0..2 {
            let mean = action_dist.mean[i];
            let std = action_dist.std[i];
            let sampled_action = crate::brain::common::math::sample_normal(mean, std, &mut rng);
            actions[i] = sampled_action;
            log_probs[i] = crate::brain::common::math::normal_log_prob(sampled_action, mean, std);
        }

        let applied_action = CarAction {
            steering: actions[0],
            throttle: actions[1],
        }
        .clamped();
        actions[0] = applied_action.steering;
        actions[1] = applied_action.throttle;
        log_probs[0] =
            crate::brain::common::math::normal_log_prob(actions[0], action_dist.mean[0], action_dist.std[0]);
        log_probs[1] =
            crate::brain::common::math::normal_log_prob(actions[1], action_dist.mean[1], action_dist.std[1]);

        self.buffer.states.push(obs.values.to_vec());
        self.buffer.actions.push(actions.clone());
        self.buffer.log_probs.push(log_probs);
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
                a2c_act_system.in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                a2c_collect_reward_system
                    .after(crate::game::episode::episode_loop_system)
                    .in_set(crate::sim::sets::SimSet::Measurement),
            );
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

        if brain.buffer.states.len() >= brain.max_steps {
            a2c_update(&mut brain, &mut stats);
        }
    }
}
