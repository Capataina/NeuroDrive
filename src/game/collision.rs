use bevy::prelude::*;
use bevy::ecs::message::MessageWriter;

use crate::game::car::{Car, CAR_HEIGHT, CAR_WIDTH};
use crate::maps::track::Track;

/// Message emitted when the car leaves the driveable road surface.
#[derive(Message)]
pub struct CollisionEvent;

/// Checks each fixed tick whether any corner of the car's bounding rectangle lies
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
    mut collision_events: MessageWriter<CollisionEvent>,
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

    for local in &local_corners {
        let rotated = (car_transform.rotation * Vec3::new(local.x, local.y, 0.0)).truncate();
        if !track.grid.is_road_at(car_pos + rotated) {
            collision_events.write(CollisionEvent);
            return;
        }
    }
}
