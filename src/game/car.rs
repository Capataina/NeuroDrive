use bevy::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::agent::action::ActionState;
use crate::agent::observation::{ObservationVector, SensorReadings};
use crate::brain::types::PolicyOutput;
use crate::game::episode::{EpisodeMovingAverages, EpisodeState};
use crate::game::progress::TrackProgress;

/// Stable environment instance identity for each training car.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EnvInstanceId(pub u32);

/// Per-car spawn configuration used for deterministic reset.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpawnConfig {
    pub position: Vec2,
    pub rotation: f32,
}

/// Trainer-wide configuration for multi-car vectorised training.
#[derive(Resource, Clone, Copy, Debug)]
pub struct TrainerConfig {
    /// Number of concurrent environment instances.
    pub num_envs: usize,
    /// Sprite alpha for non-best cars.
    pub default_car_alpha: f32,
    /// Sprite alpha for the best-performing car.
    pub best_car_alpha: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_envs: 8,
            default_car_alpha: 0.35,
            best_car_alpha: 1.0,
        }
    }
}

/// Global RNG resource used for generating random spawn positions.
#[derive(Resource)]
pub struct SpawnRng(pub StdRng);

impl Default for SpawnRng {
    fn default() -> Self {
        Self(StdRng::from_rng(&mut rand::rng()))
    }
}

/// Marker component identifying a car entity.
#[derive(Component)]
pub struct Car {
    pub velocity: Vec2,
    pub rotation_speed: f32,
    pub thrust: f32,
    pub drag: f32,
}

impl Default for Car {
    fn default() -> Self {
        Self {
            velocity: Vec2::ZERO,
            rotation_speed: 8.0,
            thrust: 750.0,
            drag: 0.985,
        }
    }
}

/// Car dimensions for collision detection and rendering.
pub const CAR_WIDTH: f32 = 12.0;
pub const CAR_HEIGHT: f32 = 6.0;

/// Per-car assigned colour for visual identification.
#[derive(Component, Clone, Copy, Debug)]
pub struct CarColour {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// Distinct colour palette for car identification.
/// Designed to be visually distinguishable on a dark track.
const CAR_PALETTE: &[(f32, f32, f32)] = &[
    (0.95, 0.25, 0.21), // red
    (0.25, 0.65, 0.96), // blue
    (0.30, 0.87, 0.47), // green
    (0.98, 0.74, 0.18), // amber
    (0.73, 0.33, 0.83), // purple
    (0.00, 0.84, 0.76), // teal
    (0.96, 0.49, 0.13), // orange
    (0.92, 0.26, 0.56), // pink
    (0.55, 0.76, 0.29), // lime
    (0.40, 0.58, 0.93), // indigo
    (0.94, 0.82, 0.09), // yellow
    (0.47, 0.33, 0.28), // brown
    (0.62, 0.62, 0.62), // grey
    (0.00, 0.59, 0.53), // dark teal
    (0.83, 0.18, 0.18), // dark red
    (0.12, 0.47, 0.71), // dark blue
    (0.18, 0.54, 0.34), // dark green
    (0.69, 0.56, 0.00), // dark amber
    (0.48, 0.19, 0.57), // dark purple
    (0.00, 0.60, 0.57), // cyan
    (0.85, 0.37, 0.10), // deep orange
    (0.76, 0.09, 0.36), // deep pink
    (0.41, 0.60, 0.14), // dark lime
    (0.25, 0.40, 0.70), // steel blue
    (0.72, 0.53, 0.04), // dark gold
];

/// Returns the palette colour for a given env index, wrapping if needed.
pub fn car_colour_for_env(env_id: u32) -> CarColour {
    let (r, g, b) = CAR_PALETTE[env_id as usize % CAR_PALETTE.len()];
    CarColour { r, g, b }
}

/// Spawns a car entity with all per-car components.
/// `spawn_s` is the arc-length position on the centreline where this car starts,
/// used to seed distance tracking so the first-tick delta is correct.
pub fn spawn_car(commands: &mut Commands, env_id: u32, spawn_config: SpawnConfig, alpha: f32, spawn_s: f32) {
    let colour = car_colour_for_env(env_id);
    info!(
        "Spawn car env#{} at ({:.1}, {:.1}) rot {:.2}.",
        env_id, spawn_config.position.x, spawn_config.position.y, spawn_config.rotation
    );
    let mut sensor_readings = SensorReadings::default();
    sensor_readings.previous_heading = spawn_config.rotation;

    let mut episode_state = EpisodeState::default();
    episode_state.previous_s = spawn_s;
    episode_state.spawn_s = spawn_s;

    commands.spawn((
        Sprite {
            color: Color::srgba(colour.r, colour.g, colour.b, alpha),
            custom_size: Some(Vec2::new(CAR_WIDTH, CAR_HEIGHT)),
            ..default()
        },
        Transform::from_xyz(spawn_config.position.x, spawn_config.position.y, 10.0)
            .with_rotation(Quat::from_rotation_z(spawn_config.rotation)),
        Car::default(),
        EnvInstanceId(env_id),
        colour,
        spawn_config,
        ActionState::default(),
        PolicyOutput::default(),
        episode_state,
        EpisodeMovingAverages::default(),
        TrackProgress::default(),
        sensor_readings,
        ObservationVector::default(),
    ));
}
