use bevy::prelude::*;

use crate::game::car::{Car, CAR_HEIGHT, CAR_WIDTH};
use crate::maps::track::Track;

/// Event emitted when the car leaves the driveable road surface.
#[derive(Event)]
pub struct CollisionEvent;

/// Checks each frame whether any corner of the car's bounding rectangle lies
/// off the driveable road surface.
///
/// The four corners of the car sprite (defined by [`CAR_WIDTH`] × [`CAR_HEIGHT`])
/// are rotated into world space and tested individually against
/// `track.grid.is_road_at()`. A collision is triggered as soon as any corner
/// leaves the road, giving accurate edge-level detection rather than
/// centre-only checking.
pub fn collision_detection_system(
    car_query: Query<&Transform, With<Car>>,
    track_query: Query<&Track>,
    mut commands: Commands,
) {
    let Ok(car_transform) = car_query.single() else {
        return;
    };
    let Ok(track) = track_query.single() else {
        return;
    };

    let car_pos = Vec2::new(car_transform.translation.x, car_transform.translation.y);
    let half_w = CAR_WIDTH * 0.5;
    let half_h = CAR_HEIGHT * 0.5;

    let local_corners = [
        Vec2::new( half_w,  half_h),
        Vec2::new( half_w, -half_h),
        Vec2::new(-half_w,  half_h),
        Vec2::new(-half_w, -half_h),
    ];

    let (_, _, angle) = car_transform.rotation.to_euler(EulerRot::ZYX);
    let (sin, cos) = angle.sin_cos();

    for local in &local_corners {
        let rotated = Vec2::new(
            local.x * cos - local.y * sin,
            local.x * sin + local.y * cos,
        );
        if !track.grid.is_road_at(car_pos + rotated) {
            commands.trigger(CollisionEvent);
            return;
        }
    }
}

/// Handles a `CollisionEvent` by despawning the car and respawning it at the
/// track's designated spawn position and rotation.
pub fn handle_collision_system(
    _trigger: On<CollisionEvent>,
    mut commands: Commands,
    car_query: Query<Entity, With<Car>>,
    track_query: Query<&Track>,
) {
    info!("Car off-track — resetting to spawn.");

    for entity in car_query.iter() {
        commands.entity(entity).despawn();
    }

    if let Ok(track) = track_query.single() {
        crate::game::car::spawn_car(
            &mut commands,
            track.spawn_position,
            track.spawn_rotation,
        );
    }
}
