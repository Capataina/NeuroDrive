use bevy::prelude::*;

use crate::game::car::Car;
use crate::maps::track::Track;

/// Continuous progress state along the track centreline.
///
/// This is an environment measurement (not an observation): it is used for
/// telemetry, lap logic, and reward shaping, but should not be included in the
/// agent observation vector.
#[derive(Component, Clone, Copy, Debug)]
pub struct TrackProgress {
    /// Arc-length distance along the centreline from the start point.
    pub s: f32,
    /// `s / track_length` in `[0, 1]`.
    pub fraction: f32,
    /// Closest world-space point on the centreline.
    pub closest_point: Vec2,
    /// Unit tangent direction at `closest_point`.
    pub tangent: Vec2,
    /// Euclidean distance from the car position to the centreline.
    pub distance: f32,
}

impl Default for TrackProgress {
    fn default() -> Self {
        Self {
            s: 0.0,
            fraction: 0.0,
            closest_point: Vec2::ZERO,
            tangent: Vec2::X,
            distance: 0.0,
        }
    }
}

/// Updates the car's centreline projection and progress on the fixed tick.
pub fn update_track_progress_system(
    track_query: Query<&Track>,
    mut car_query: Query<(&Transform, &mut TrackProgress), With<Car>>,
) {
    let Ok(track) = track_query.single() else {
        return;
    };

    for (transform, mut progress) in car_query.iter_mut() {
        let pos = Vec2::new(transform.translation.x, transform.translation.y);
        let projection = track.centerline.project(pos);

        progress.s = projection.s;
        progress.fraction = projection.fraction;
        progress.closest_point = projection.closest_point;
        progress.tangent = projection.tangent;
        progress.distance = projection.distance;
    }
}

