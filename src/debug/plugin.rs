use bevy::prelude::*;

use crate::debug::hud::{
    DrivingHudStats,
    spawn_driving_hud_system,
    update_driving_hud_stats_system,
    update_driving_hud_text_system,
    update_driving_hud_visibility_system,
};
use crate::debug::overlays::{
    DebugOverlayState,
    debug_overlay_toggle_system,
    draw_geometry_overlay_system,
    draw_sensor_overlay_system,
};
use crate::sim::sets::SimSet;

/// Plugin for debug/observability features (overlays, gizmos, telemetry).
pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugOverlayState>()
            .init_resource::<DrivingHudStats>()
            .add_systems(Startup, spawn_driving_hud_system)
            .add_systems(FixedUpdate, update_driving_hud_stats_system.in_set(SimSet::Measurement))
            .add_systems(
                Update,
                (
                    debug_overlay_toggle_system,
                    draw_geometry_overlay_system,
                    draw_sensor_overlay_system,
                    update_driving_hud_visibility_system,
                    update_driving_hud_text_system,
                ),
            );
    }
}
