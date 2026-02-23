use bevy::prelude::*;

use crate::agent::action::ActionState;
use crate::game::car::Car;

/// Minimal deterministic car state used by the pure replay stepper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarKinematicState {
    pub position: Vec2,
    pub velocity: Vec2,
    pub heading: f32,
}

/// Immutable car dynamics parameters consumed by the pure stepper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarDynamicsParams {
    pub rotation_speed: f32,
    pub thrust: f32,
    pub drag: f32,
}

/// Applies the current action to the car on the fixed simulation tick.
///
/// This system is the only place where actions become state mutation:
/// it updates the car transform and velocity deterministically given the fixed
/// timestep and the fixed-tick `ActionState`.
pub fn car_physics_system(
    time: Res<Time<bevy::time::Fixed>>,
    action_state: Res<ActionState>,
    mut query: Query<(&mut Transform, &mut Car)>,
) {
    let dt = time.delta_secs();
    let action = action_state.applied;

    for (mut transform, mut car) in query.iter_mut() {
        let forward = (transform.rotation * Vec3::X).truncate();
        let heading = forward.y.atan2(forward.x);
        let mut state = CarKinematicState {
            position: transform.translation.truncate(),
            velocity: car.velocity,
            heading,
        };
        let params = CarDynamicsParams {
            rotation_speed: car.rotation_speed,
            thrust: car.thrust,
            drag: car.drag,
        };

        step_car_dynamics(&mut state, action.steering, action.throttle, dt, params);

        transform.translation.x = state.position.x;
        transform.translation.y = state.position.y;
        transform.rotation = Quat::from_rotation_z(state.heading);
        car.velocity = state.velocity;
    }
}

/// Pure deterministic car step used by runtime physics and replay tests.
pub fn step_car_dynamics(
    state: &mut CarKinematicState,
    steering: f32,
    throttle: f32,
    dt: f32,
    params: CarDynamicsParams,
) {
    state.heading += -steering.clamp(-1.0, 1.0) * params.rotation_speed * dt;

    if throttle > 0.0 {
        let forward = Vec2::new(state.heading.cos(), state.heading.sin());
        state.velocity += forward * (params.thrust * throttle.clamp(0.0, 1.0)) * dt;
    }

    state.velocity *= params.drag;
    state.position += state.velocity * dt;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = ((*seed >> 32) as u32) as f32 / u32::MAX as f32;
        value
    }

    #[test]
    fn deterministic_replay_same_seed_same_actions_identical_trajectory() {
        let dt = 1.0 / 60.0;
        let steps = 1200;
        let seed = 0xDEADBEEFCAFEBABEu64;
        let params = CarDynamicsParams {
            rotation_speed: 4.0,
            thrust: 1500.0,
            drag: 0.985,
        };

        let mut first_run_state = CarKinematicState {
            position: Vec2::ZERO,
            velocity: Vec2::ZERO,
            heading: 0.0,
        };
        let mut second_run_state = first_run_state;

        let mut first_seed = seed;
        for _ in 0..steps {
            let steering = lcg_next(&mut first_seed) * 2.0 - 1.0;
            let throttle = if lcg_next(&mut first_seed) > 0.35 { 1.0 } else { 0.0 };
            step_car_dynamics(&mut first_run_state, steering, throttle, dt, params);
        }

        let mut second_seed = seed;
        for _ in 0..steps {
            let steering = lcg_next(&mut second_seed) * 2.0 - 1.0;
            let throttle = if lcg_next(&mut second_seed) > 0.35 { 1.0 } else { 0.0 };
            step_car_dynamics(&mut second_run_state, steering, throttle, dt, params);
        }

        assert_eq!(first_run_state.position, second_run_state.position);
        assert_eq!(first_run_state.velocity, second_run_state.velocity);
        assert_eq!(first_run_state.heading, second_run_state.heading);
    }
}
