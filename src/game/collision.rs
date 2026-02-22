use bevy::prelude::*;
use crate::game::car::Car;
use crate::maps::track::Track;

/// Event emitted when the car collides with a barrier.
#[derive(Event)]
pub struct CollisionEvent;

/// Checks if the car is outside track boundaries and marks it for reset.
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

    // Car is off-track if outside outer boundary OR inside inner boundary
    let inside_outer = point_in_polygon(car_pos, &track.outer_boundary);
    let inside_inner = point_in_polygon(car_pos, &track.inner_boundary);

    if !inside_outer || inside_inner {
        commands.trigger(CollisionEvent);
    }
}

/// Handles collision events by resetting the game.
pub fn handle_collision_system(
    _trigger: On<CollisionEvent>,
    mut commands: Commands,
    car_query: Query<Entity, With<Car>>,
    track_query: Query<&Track>,
) {
    info!("Car died (collision detected).");

    // Despawn the car
    for entity in car_query.iter() {
        commands.entity(entity).despawn();
    }

    // Respawn at track start
    if let Ok(track) = track_query.single() {
        crate::game::car::spawn_car(
            &mut commands,
            track.spawn_position,
            track.spawn_rotation,
        );
    }
}

/// Ray-casting algorithm to determine if a point is inside a polygon.
fn point_in_polygon(point: Vec2, polygon: &[Vec2]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];

        if ((pi.y > point.y) != (pj.y > point.y))
            && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }
        j = i;
    }

    inside
}
