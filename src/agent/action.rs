use bevy::prelude::*;

/// Continuous action interface for the car.
///
/// This is the stable control surface used by all controllers (keyboard,
/// heuristic, replay, learning).
///
/// ## Invariants
/// - `steering` is clamped to `[-1, 1]` (left negative, right positive).
/// - `throttle` is clamped to `[0, 1]` (0 = coast, 1 = full throttle).
#[derive(Clone, Copy, Debug, Default)]
pub struct CarAction {
    pub steering: f32,
    pub throttle: f32,
}

impl CarAction {
    /// Returns this action clamped to its allowed ranges.
    pub fn clamped(self) -> Self {
        Self {
            steering: self.steering.clamp(-1.0, 1.0),
            throttle: self.throttle.clamp(0.0, 1.0),
        }
    }
}

/// Per-car component holding the current desired and applied actions.
///
/// Controllers should write `desired` once per fixed tick. Vehicle dynamics
/// should consume `applied`, which may differ if smoothing is enabled.
#[derive(Component, Clone, Copy, Debug)]
pub struct ActionState {
    pub desired: CarAction,
    pub applied: CarAction,
}

impl Default for ActionState {
    fn default() -> Self {
        Self {
            desired: CarAction::default(),
            applied: CarAction::default(),
        }
    }
}

/// Optional action smoothing configuration.
///
/// When enabled, `applied` is low-pass filtered towards `desired` each tick.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ActionSmoothing {
    pub enabled: bool,
    /// Time constant in seconds for first-order smoothing.
    pub time_constant_s: f32,
}

impl Default for ActionSmoothing {
    fn default() -> Self {
        Self {
            enabled: false,
            time_constant_s: 0.12,
        }
    }
}

/// Latches keyboard input into the fixed-tick `ActionState.desired` for every
/// car marked with the `KeyboardCar` component.
///
/// Keyboard mode spawns a single car, so in practice this writes one desired
/// action per tick. In theory (e.g. for debugging) multiple `KeyboardCar`s
/// all receive the same WASD input.
pub fn keyboard_action_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut car_query: Query<&mut ActionState, With<crate::brain::types::KeyboardCar>>,
) {
    let mut steering = 0.0;
    if keyboard.pressed(KeyCode::KeyA) {
        steering -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        steering += 1.0;
    }

    let throttle = if keyboard.pressed(KeyCode::KeyW) {
        1.0
    } else {
        0.0
    };

    let desired = CarAction { steering, throttle }.clamped();
    for mut action_state in car_query.iter_mut() {
        action_state.desired = desired;
    }
}

/// Updates `ActionState.applied` from `ActionState.desired` for all cars.
///
/// When smoothing is disabled, this is a direct copy.
pub fn action_smoothing_system(
    time: Res<Time<bevy::time::Fixed>>,
    smoothing: Res<ActionSmoothing>,
    mut car_query: Query<&mut ActionState>,
) {
    let dt = time.delta_secs();
    let tau = smoothing.time_constant_s.max(1e-4);
    let alpha = 1.0 - (-dt / tau).exp();

    for mut action_state in car_query.iter_mut() {
        let desired = action_state.desired.clamped();

        if !smoothing.enabled {
            action_state.applied = desired;
            continue;
        }

        let applied = action_state.applied;
        action_state.applied = CarAction {
            steering: applied.steering + (desired.steering - applied.steering) * alpha,
            throttle: applied.throttle + (desired.throttle - applied.throttle) * alpha,
        }
        .clamped();
    }
}
