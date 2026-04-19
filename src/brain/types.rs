use bevy::prelude::*;

/// Identifies which controller drives a given car.
///
/// Carried as a per-car component so the runtime can mix controllers in the
/// same simulation (the side-by-side layout in Milestone 6 runs 8 PPO cars
/// and 8 brain-inspired cars at once). Also used by analytics and HUD to
/// discriminate which fields apply per car.
///
/// ZST marker components (`PpoCar`, `BrainCar`, `KeyboardCar`) are attached
/// alongside this enum so systems can filter via `With<PpoCar>` — idiomatic
/// Bevy and faster than matching on the enum value in every system.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Controller {
    Keyboard,
    Ppo,
    Brain,
}

/// Marker component: PPO drives this car.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PpoCar;

/// Marker component: the brain-inspired learner drives this car.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct BrainCar;

/// Marker component: WASD input drives this car.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct KeyboardCar;

/// Per-car component exposing controller internals for analytics and HUD.
///
/// Field semantics depend on the active controller on this car:
///
/// - In PPO (`PpoCar`): fields carry the Gaussian-policy mean and std for
///   each action dimension and the critic's value prediction.
/// - In brain-inspired (`BrainCar`): `steering_mean` / `throttle_mean` carry
///   the raw output-neuron activations, `*_std` are 0.0 (the brain is
///   deterministic given observations), and `value_prediction` carries the
///   per-tick modulator M starting in Stage 2.
///
/// Analytics reads this component unconditionally; semantics are
/// discriminated downstream by the per-car `PpoCar` / `BrainCar` marker.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PolicyOutput {
    pub value_prediction: f32,
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
}
