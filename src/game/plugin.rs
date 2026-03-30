use crate::game::car::{SpawnConfig, SpawnRng, TrainerConfig, spawn_car};
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

/// Initial game setup: camera and multi-car spawn.
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

    let num_envs = trainer_config.num_envs;
    info!(
        "Track ready. Spawning {} cars at random centreline positions.",
        num_envs
    );

    for i in 0..num_envs {
        let s = spawn_rng.0.random::<f32>() * track.centerline.total_length();
        let position = track.centerline.point_at_s(s);
        let tangent = track.centerline.tangent_at_s(s);
        let rotation = tangent.y.atan2(tangent.x);
        let spawn_config = SpawnConfig { position, rotation };

        // First car gets full alpha as default best until ranking kicks in.
        let alpha = if i == 0 {
            trainer_config.best_car_alpha
        } else {
            trainer_config.default_car_alpha
        };

        spawn_car(&mut commands, i as u32, spawn_config, alpha, s);
    }
}
