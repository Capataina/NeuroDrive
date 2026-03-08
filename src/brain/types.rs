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

/// Interface for any Brain algorithm.
pub trait Brain: Send + Sync {
    /// Given an observation, returns the chosen action and any algorithm-specific state.
    fn act(&mut self, obs: &ObservationVector) -> CarAction;
}
