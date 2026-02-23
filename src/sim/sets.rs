use bevy::prelude::*;

/// Fixed-timestep simulation pipeline ordering.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimSet {
    /// Collect and smooth actions for this tick.
    Input,
    /// Apply actions to vehicle dynamics.
    Physics,
    /// Detect and resolve collisions/resets.
    Collision,
    /// Compute derived measurements (progress, sensors, telemetry sources).
    Measurement,
}

