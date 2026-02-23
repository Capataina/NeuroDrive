use bevy::math::Isometry2d;
use bevy::prelude::*;

use crate::agent::observation::{ObservationConfig, SensorReadings};
use crate::game::car::Car;
use crate::game::progress::TrackProgress;
use crate::maps::track::Track;

/// Debug overlay toggles.
#[derive(Resource, Clone, Copy, Debug)]
pub struct DebugOverlayState {
    /// Geometry overlays (centreline, projections, tangents).
    pub geometry: bool,
    /// Sensor overlays (raycasts and hit points).
    pub sensors: bool,
    /// Telemetry overlay (minimal: window-title update for now).
    pub telemetry: bool,
}

impl Default for DebugOverlayState {
    fn default() -> Self {
        Self {
            geometry: false,
            sensors: false,
            telemetry: false,
        }
    }
}

/// Handles overlay toggle keybindings.
///
/// Milestone 0 convention (per `README.md`):
/// - F1: geometry overlays
/// - F2: sensor overlays
/// - F3: telemetry overlay
pub fn debug_overlay_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut overlay: ResMut<DebugOverlayState>,
) {
    if keyboard.just_pressed(KeyCode::F1) {
        overlay.geometry = !overlay.geometry;
        info!("Debug overlay F1 (geometry): {}", overlay.geometry);
    }
    if keyboard.just_pressed(KeyCode::F2) {
        overlay.sensors = !overlay.sensors;
        info!("Debug overlay F2 (sensors): {}", overlay.sensors);
    }
    if keyboard.just_pressed(KeyCode::F3) {
        overlay.telemetry = !overlay.telemetry;
        info!("Debug overlay F3 (telemetry): {}", overlay.telemetry);
    }
}

/// Draws centreline and projection debug geometry using gizmos.
pub fn draw_geometry_overlay_system(
    overlay: Res<DebugOverlayState>,
    track_query: Query<&Track>,
    car_query: Query<(&Transform, &TrackProgress, &Car), With<Car>>,
    mut gizmos: Gizmos,
) {
    if !overlay.geometry {
        return;
    }

    let Ok(track) = track_query.single() else {
        return;
    };

    // Track centreline.
    let pts = &track.centerline.points;
    if pts.len() >= 2 {
        let line_color = Color::srgb(0.1, 0.9, 0.1);
        for i in 0..pts.len() {
            let a = pts[i];
            let b = pts[(i + 1) % pts.len()];
            gizmos.line_2d(a, b, line_color);
        }
    }

    // Car projection.
    for (transform, progress, car) in car_query.iter() {
        let car_pos = Vec2::new(transform.translation.x, transform.translation.y);

        gizmos.line_2d(car_pos, progress.closest_point, Color::srgb(0.9, 0.9, 0.1));

        gizmos.circle_2d(
            Isometry2d::from_translation(progress.closest_point),
            6.0,
            Color::srgb(0.2, 0.6, 1.0),
        );

        let tangent_end = progress.closest_point + progress.tangent * 40.0;
        gizmos.arrow_2d(progress.closest_point, tangent_end, Color::srgb(0.2, 0.6, 1.0));

        let forward = (transform.rotation * Vec3::X).truncate().normalize_or_zero();
        gizmos.arrow_2d(car_pos, car_pos + forward * 45.0, Color::srgb(0.2, 0.8, 1.0));

        let velocity = car.velocity;
        if velocity.length_squared() > 1e-3 {
            gizmos.arrow_2d(
                car_pos,
                car_pos + velocity.normalize() * velocity.length().min(70.0),
                Color::srgb(1.0, 0.2, 0.7),
            );
        }
    }
}

/// Draws raycast sensor lines and hit points.
pub fn draw_sensor_overlay_system(
    overlay: Res<DebugOverlayState>,
    observation_config: Res<ObservationConfig>,
    car_query: Query<(&Transform, &SensorReadings), With<Car>>,
    mut gizmos: Gizmos,
) {
    if !overlay.sensors {
        return;
    }

    for (transform, sensors) in &car_query {
        let origin = transform.translation.truncate();

        for index in 0..sensors.ray_distances.len() {
            let hit = sensors.ray_hits[index];
            let distance = sensors.ray_distances[index];
            let maxed = distance >= observation_config.ray_max_range - 1e-3;
            let line_color = if maxed {
                Color::srgb(0.6, 0.6, 0.6)
            } else {
                Color::srgb(1.0, 0.5, 0.1)
            };
            let hit_color = if maxed {
                Color::srgb(0.5, 0.5, 0.5)
            } else {
                Color::srgb(1.0, 0.2, 0.2)
            };

            gizmos.line_2d(origin, hit, line_color);
            gizmos.circle_2d(Isometry2d::from_translation(hit), 2.0, hit_color);
        }
    }
}
