use crate::brain::types::Controller;
use crate::game::car::{
    Car, CarColour, SpawnConfig, SpawnRng, TrainerConfig, TrainerLayout, car_colour_cool,
    car_colour_for_env, car_colour_warm, spawn_car,
};
use crate::game::collision::collision_detection_system;
use crate::game::episode::EpisodeConfig;
use crate::game::physics::car_physics_system;
use crate::game::progress::update_track_progress_system;
use crate::maps::track::Track;
use crate::sim::sets::SimSet;
use bevy::prelude::*;
use rand::RngExt;

/// Main game plugin that bundles all game systems.
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EpisodeConfig>()
            .init_resource::<TrainerConfig>()
            .init_resource::<SpawnRng>()
            .add_systems(PostStartup, setup_game)
            .configure_sets(
                FixedUpdate,
                (
                    SimSet::Input,
                    SimSet::Physics,
                    SimSet::Collision,
                    SimSet::Measurement,
                )
                    .chain(),
            )
            // Core simulation loop: runs on the fixed timestep.
            .add_systems(FixedUpdate, car_physics_system.in_set(SimSet::Physics))
            .add_systems(
                FixedUpdate,
                collision_detection_system.in_set(SimSet::Collision),
            )
            .add_systems(
                FixedUpdate,
                (
                    update_track_progress_system,
                    crate::game::episode::episode_loop_system
                        .after(update_track_progress_system),
                )
                    .chain()
                    .in_set(SimSet::Measurement),
            );
    }
}

/// Initial game setup: camera and multi-car spawn for the default layout.
/// All cars spawn at random centreline positions — no privileged car 0.
fn setup_game(
    mut commands: Commands,
    track_query: Query<&Track>,
    trainer_config: Res<TrainerConfig>,
    mut spawn_rng: ResMut<SpawnRng>,
) {
    // Spawn 2D camera
    commands.spawn(Camera2d::default());

    let Ok(track) = track_query.single() else {
        warn!("No track found at startup. Cars were not spawned.");
        return;
    };

    info!(
        "Track ready. Spawning cars for layout {:?}.",
        trainer_config.layout
    );

    spawn_cars_for_layout(
        &mut commands,
        &trainer_config,
        track,
        &mut spawn_rng,
    );
}

/// Spawns the set of cars described by `trainer_config.layout`. Idempotent
/// on an empty world — existing cars should be despawned by the caller
/// beforehand (see the F4 toggle in `brain::plugin`).
pub fn spawn_cars_for_layout(
    commands: &mut Commands,
    trainer_config: &TrainerConfig,
    track: &Track,
    spawn_rng: &mut SpawnRng,
) {
    let layout = trainer_config.layout;
    let total = layout.total_cars();

    let mut env_id_counter: u32 = 0;
    let mut emit_car = |commands: &mut Commands,
                        rng: &mut SpawnRng,
                        controller: Controller,
                        palette_index: u32| {
        let s = rng.0.random::<f32>() * track.centerline.total_length();
        let position = track.centerline.point_at_s(s);
        let tangent = track.centerline.tangent_at_s(s);
        let rotation = tangent.y.atan2(tangent.x);
        let spawn_config = SpawnConfig { position, rotation };

        let colour = match (layout, controller) {
            (TrainerLayout::SideBySide { .. }, Controller::Ppo) => car_colour_warm(palette_index),
            (TrainerLayout::SideBySide { .. }, Controller::Brain) => car_colour_cool(palette_index),
            _ => car_colour_for_env(env_id_counter),
        };

        let alpha = if env_id_counter == 0 {
            trainer_config.best_car_alpha
        } else {
            trainer_config.default_car_alpha
        };

        spawn_car(
            commands,
            env_id_counter,
            spawn_config,
            alpha,
            s,
            colour,
            controller,
        );
        env_id_counter += 1;
    };

    match layout {
        TrainerLayout::Keyboard => {
            emit_car(commands, spawn_rng, Controller::Keyboard, 0);
        }
        TrainerLayout::AllPpo { count } => {
            for i in 0..count {
                emit_car(commands, spawn_rng, Controller::Ppo, i as u32);
            }
        }
        TrainerLayout::AllBrain { count } => {
            for i in 0..count {
                emit_car(commands, spawn_rng, Controller::Brain, i as u32);
            }
        }
        TrainerLayout::SideBySide { ppo, brain } => {
            for i in 0..ppo {
                emit_car(commands, spawn_rng, Controller::Ppo, i as u32);
            }
            for i in 0..brain {
                emit_car(commands, spawn_rng, Controller::Brain, i as u32);
            }
        }
    }

    debug_assert_eq!(env_id_counter as usize, total);
    let _ = (Car::default(), CarColour { r: 0.0, g: 0.0, b: 0.0 }); // keep imports live
}
