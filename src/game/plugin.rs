use crate::game::car::{SpawnConfig, TrainerConfig, spawn_car};
use crate::game::collision::collision_detection_system;
use crate::game::episode::EpisodeConfig;
use crate::game::physics::car_physics_system;
use crate::game::progress::update_track_progress_system;
use crate::maps::track::Track;
use crate::sim::sets::SimSet;
use bevy::prelude::*;

/// Main game plugin that bundles all game systems.
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EpisodeConfig>()
            .init_resource::<TrainerConfig>()
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

/// Initial game setup: camera and multi-car spawn.
fn setup_game(
    mut commands: Commands,
    track_query: Query<&Track>,
    trainer_config: Res<TrainerConfig>,
) {
    // Spawn 2D camera
    commands.spawn(Camera2d::default());

    // Spawn training cars at track start with lateral offsets
    let Ok(track) = track_query.single() else {
        warn!("No track found at startup. Cars were not spawned.");
        return;
    };

    let num_envs = trainer_config.num_envs;
    info!(
        "Track ready. Spawning {} cars at ({:.1}, {:.1}) rot {:.2}.",
        num_envs, track.spawn_position.x, track.spawn_position.y, track.spawn_rotation
    );

    // Compute perpendicular direction to spawn heading for lateral offsets
    let heading = track.spawn_rotation;
    let perp = Vec2::new(-heading.sin(), heading.cos());

    for i in 0..num_envs {
        // Distribute cars evenly across the lateral spread
        let t = if num_envs <= 1 {
            0.0
        } else {
            (i as f32 / (num_envs - 1) as f32) * 2.0 - 1.0 // range [-1, 1]
        };
        let lateral_offset = t * trainer_config.spawn_lateral_spread * 0.5;
        let offset_position = track.spawn_position + perp * lateral_offset;

        let spawn_config = SpawnConfig {
            position: offset_position,
            rotation: track.spawn_rotation,
        };

        // First car gets full alpha (acts as default best until ranking kicks in)
        let alpha = if i == 0 {
            trainer_config.best_car_alpha
        } else {
            trainer_config.default_car_alpha
        };

        spawn_car(&mut commands, i as u32, spawn_config, alpha);
    }
}
