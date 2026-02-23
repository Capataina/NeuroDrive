use bevy::prelude::*;

use crate::agent::action::{
    ActionSmoothing,
    ActionState,
    action_smoothing_system,
    keyboard_action_input_system,
};
use crate::agent::observation::{
    ObservationConfig,
    build_observation_vector_system,
    update_sensor_readings_system,
};
use crate::game::progress::update_track_progress_system;
use crate::sim::sets::SimSet;

/// Plugin providing agent-facing interfaces (actions now; sensors later).
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActionState>()
            .init_resource::<ActionSmoothing>()
            .init_resource::<ObservationConfig>()
            // Actions must be updated on the fixed simulation tick.
            .add_systems(
                FixedUpdate,
                (
                    keyboard_action_input_system,
                    action_smoothing_system,
                )
                    .chain()
                    .in_set(SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                (
                    update_sensor_readings_system.after(update_track_progress_system),
                    build_observation_vector_system,
                )
                    .chain()
                    .in_set(SimSet::Measurement),
            );
    }
}
