use std::f32::consts::PI;

use bevy::prelude::*;

use crate::game::car::Car;
use crate::game::progress::TrackProgress;
use crate::maps::grid::TrackGrid;
use crate::maps::track::Track;

/// Number of ray sensors in the observation model.
pub const NUM_RAYS: usize = 11;

/// Raycast sensor readings and derived kinematics for one car.
#[derive(Component, Clone, Debug)]
pub struct SensorReadings {
    /// Ray distances in world units, one per configured ray angle.
    pub ray_distances: [f32; NUM_RAYS],
    /// World-space hit points for debug rendering.
    pub ray_hits: [Vec2; NUM_RAYS],
    /// Ray directions in world-space for debug rendering.
    pub ray_directions: [Vec2; NUM_RAYS],
    /// Current scalar speed in world units / second.
    pub speed: f32,
    /// Signed heading error to centerline tangent in radians.
    pub heading_error: f32,
    /// Estimated yaw rate in radians / second.
    pub angular_velocity: f32,
    /// Last heading sample used for yaw rate estimation.
    pub previous_heading: f32,
}

impl Default for SensorReadings {
    fn default() -> Self {
        Self {
            ray_distances: [0.0; NUM_RAYS],
            ray_hits: [Vec2::ZERO; NUM_RAYS],
            ray_directions: [Vec2::X; NUM_RAYS],
            speed: 0.0,
            heading_error: 0.0,
            angular_velocity: 0.0,
            previous_heading: 0.0,
        }
    }
}

/// Fixed-size, normalised observation vector consumed by controllers.
#[derive(Component, Clone, Debug)]
pub struct ObservationVector {
    /// Feature vector in stable order:
    /// [ray distances..., speed, heading_error, angular_velocity]
    pub values: [f32; NUM_RAYS + 3],
}

impl Default for ObservationVector {
    fn default() -> Self {
        Self {
            values: [0.0; NUM_RAYS + 3],
        }
    }
}

/// Sensor and observation configuration.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ObservationConfig {
    /// Raycast max range in world units.
    pub ray_max_range: f32,
    /// Raycast march step in world units.
    pub ray_step: f32,
    /// Speed normalisation scale in world units / second.
    pub speed_norm_max: f32,
    /// Angular-velocity normalisation scale in radians / second.
    pub angular_velocity_norm_max: f32,
    /// Relative ray angles around the car forward vector, in radians.
    pub ray_angles: [f32; NUM_RAYS],
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            ray_max_range: 375.0,
            ray_step: 3.0,
            speed_norm_max: 900.0,
            angular_velocity_norm_max: 8.0,
            ray_angles: [
                -150f32.to_radians(),
                -90f32.to_radians(),
                -60f32.to_radians(),
                -35f32.to_radians(),
                -15f32.to_radians(),
                0.0,
                15f32.to_radians(),
                35f32.to_radians(),
                60f32.to_radians(),
                90f32.to_radians(),
                150f32.to_radians(),
            ],
        }
    }
}

/// Updates raycasts and derived kinematics on the fixed simulation tick.
pub fn update_sensor_readings_system(
    time: Res<Time<bevy::time::Fixed>>,
    config: Res<ObservationConfig>,
    track_query: Query<&Track>,
    mut car_query: Query<(&Transform, &Car, &TrackProgress, &mut SensorReadings)>,
) {
    let Ok(track) = track_query.single() else {
        return;
    };
    let dt = time.delta_secs().max(1e-6);

    for (transform, car, progress, mut sensors) in &mut car_query {
        let position = transform.translation.truncate();
        let forward = (transform.rotation * Vec3::X).truncate().normalize_or_zero();
        let heading = forward.y.atan2(forward.x);

        sensors.speed = car.velocity.length();
        sensors.heading_error = signed_angle_between(forward, progress.tangent);
        sensors.angular_velocity = wrap_angle(heading - sensors.previous_heading) / dt;
        sensors.previous_heading = heading;

        for (index, relative_angle) in config.ray_angles.iter().enumerate() {
            let world_angle = heading + *relative_angle;
            let dir = Vec2::new(world_angle.cos(), world_angle.sin());
            let (distance, hit) = raycast_to_road_boundary(
                &track.grid,
                position,
                dir,
                config.ray_max_range,
                config.ray_step,
            );
            sensors.ray_distances[index] = distance;
            sensors.ray_hits[index] = hit;
            sensors.ray_directions[index] = dir;
        }
    }
}

/// Converts sensor readings into a stable, normalised observation vector.
pub fn build_observation_vector_system(
    config: Res<ObservationConfig>,
    mut query: Query<(&SensorReadings, &mut ObservationVector)>,
) {
    for (sensors, mut observation) in &mut query {
        let mut values = [0.0; NUM_RAYS + 3];

        for (index, distance) in sensors.ray_distances.iter().enumerate() {
            values[index] = (*distance / config.ray_max_range).clamp(0.0, 1.0);
        }

        values[NUM_RAYS] = (sensors.speed / config.speed_norm_max).clamp(0.0, 1.0);
        values[NUM_RAYS + 1] = (sensors.heading_error / PI).clamp(-1.0, 1.0);
        values[NUM_RAYS + 2] =
            (sensors.angular_velocity / config.angular_velocity_norm_max).clamp(-1.0, 1.0);

        observation.values = values;
    }
}

fn raycast_to_road_boundary(
    grid: &TrackGrid,
    origin: Vec2,
    direction: Vec2,
    max_range: f32,
    step: f32,
) -> (f32, Vec2) {
    let dir = direction.normalize_or_zero();
    if dir == Vec2::ZERO {
        return (0.0, origin);
    }

    let step = step.max(0.5);
    let mut previous_distance = 0.0;
    let mut distance = step;

    while distance <= max_range {
        let point = origin + dir * distance;
        if !grid.is_road_at(point) {
            let refined = refine_boundary_distance(grid, origin, dir, previous_distance, distance);
            return (refined, origin + dir * refined);
        }
        previous_distance = distance;
        distance += step;
    }

    (max_range, origin + dir * max_range)
}

fn refine_boundary_distance(
    grid: &TrackGrid,
    origin: Vec2,
    direction: Vec2,
    mut inside: f32,
    mut outside: f32,
) -> f32 {
    for _ in 0..8 {
        let mid = 0.5 * (inside + outside);
        let point = origin + direction * mid;
        if grid.is_road_at(point) {
            inside = mid;
        } else {
            outside = mid;
        }
    }
    inside
}

fn wrap_angle(mut angle: f32) -> f32 {
    while angle > PI {
        angle -= 2.0 * PI;
    }
    while angle < -PI {
        angle += 2.0 * PI;
    }
    angle
}

fn signed_angle_between(from: Vec2, to: Vec2) -> f32 {
    let from_n = from.normalize_or_zero();
    let to_n = to.normalize_or_zero();
    if from_n == Vec2::ZERO || to_n == Vec2::ZERO {
        return 0.0;
    }
    wrap_angle(to_n.to_angle() - from_n.to_angle())
}
