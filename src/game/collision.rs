use bevy::prelude::*;

use crate::game::car::{CAR_HEIGHT, CAR_WIDTH, Car};
use crate::maps::track::Track;

/// Marker component added to car entities that are off the driveable road surface
/// this tick. Removed when the car is back on road.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct Collided;

/// Checks each fixed tick whether any corner of each car's bounding rectangle lies
/// off the driveable road surface.
///
/// Adds the `Collided` marker to cars with an off-road corner; removes it from
/// cars that are fully on-road. This replaces the old global `CollisionEvent`
/// message so each car's collision state is independent.
pub fn collision_detection_system(
    car_query: Query<(Entity, &Transform), With<Car>>,
    track_query: Query<&Track>,
    mut commands: Commands,
) {
    let Ok(track) = track_query.single() else {
        return;
    };

    const LOCAL_CORNERS: [Vec2; 4] = {
        let hw = CAR_WIDTH * 0.5;
        let hh = CAR_HEIGHT * 0.5;
        [
            Vec2::new(hw, hh),
            Vec2::new(hw, -hh),
            Vec2::new(-hw, hh),
            Vec2::new(-hw, -hh),
        ]
    };

    for (entity, car_transform) in car_query.iter() {
        let car_pos = Vec2::new(car_transform.translation.x, car_transform.translation.y);
        let mut off_road = false;

        for local in &LOCAL_CORNERS {
            let rotated =
                (car_transform.rotation * Vec3::new(local.x, local.y, 0.0)).truncate();
            if !track.grid.is_road_at(car_pos + rotated) {
                off_road = true;
                break;
            }
        }

        if off_road {
            commands.entity(entity).insert(Collided);
        } else {
            commands.entity(entity).remove::<Collided>();
        }
    }
}
