use bevy::prelude::*;

use crate::agent::action::CarAction;
use crate::agent::observation::ObservationVector;

/// The active mode of the agent.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentMode {
    Keyboard,
    Ai,
}

impl Default for AgentMode {
    fn default() -> Self {
        Self::Ai // Default to AI for Milestone 1
    }
}

/// Per-car component exposing the brain's internal predictions for analytics.
/// Written by the act system each tick when in AI mode.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PolicyOutput {
    /// Critic's estimate of the current state value.
    pub value_prediction: f32,
    /// Policy distribution mean for steering.
    pub steering_mean: f32,
    /// Policy distribution std for steering.
    pub steering_std: f32,
    /// Policy distribution mean for throttle.
    pub throttle_mean: f32,
    /// Policy distribution std for throttle.
    pub throttle_std: f32,
}

/// Interface for any Brain algorithm.
/// TODO: remove in Stage 5 cleanup — vectorised path calls model.forward() directly.
#[allow(dead_code)]
pub trait Brain: Send + Sync {
    /// Given an observation, returns the chosen action and any algorithm-specific state.
    fn act(&mut self, obs: &ObservationVector) -> CarAction;
}
