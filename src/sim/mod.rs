//! Simulation scheduling primitives.
//!
//! This module defines shared system sets for the fixed-timestep simulation
//! pipeline, keeping ordering explicit without creating cross-module
//! dependencies (e.g. agent code depending on game code).

use std::f32::consts::PI;

use bevy::prelude::Vec2;

pub mod sets;

/// Wraps an angle to the range `[-PI, PI]`.
pub fn wrap_angle(mut angle: f32) -> f32 {
    while angle > PI {
        angle -= 2.0 * PI;
    }
    while angle < -PI {
        angle += 2.0 * PI;
    }
    angle
}

/// Returns the signed angle from `from` to `to` in radians, in `[-PI, PI]`.
pub fn signed_angle_between(from: Vec2, to: Vec2) -> f32 {
    let from_n = from.normalize_or_zero();
    let to_n = to.normalize_or_zero();
    if from_n == Vec2::ZERO || to_n == Vec2::ZERO {
        return 0.0;
    }
    wrap_angle(to_n.to_angle() - from_n.to_angle())
}
