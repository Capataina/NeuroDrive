use bevy::prelude::*;
use crate::game::car::spawn_car;
use crate::game::collision::{CollisionEvent, collision_detection_system, handle_collision_system};
use crate::game::episode::{EpisodeConfig, EpisodeMovingAverages, EpisodeState, episode_loop_system};
use crate::game::physics::car_physics_system;
use crate::game::progress::update_track_progress_system;
use crate::maps::track::Track;
use crate::sim::sets::SimSet;

/// Main game plugin that bundles all game systems.
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<CollisionEvent>()
            .init_resource::<EpisodeConfig>()
            .init_resource::<EpisodeState>()
            .init_resource::<EpisodeMovingAverages>()
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
                (
                    collision_detection_system,
                    handle_collision_system,
                )
                    .chain()
                    .in_set(SimSet::Collision),
            )
            .add_systems(
                FixedUpdate,
                (
                    update_track_progress_system,
                    episode_loop_system.after(update_track_progress_system),
                )
                    .chain()
                    .in_set(SimSet::Measurement),
            );
    }
}

/// Initial game setup: camera and car spawn.
fn setup_game(mut commands: Commands, track_query: Query<&Track>) {
    // Spawn 2D camera
    commands.spawn(Camera2d::default());

    // Spawn car at track start position
    if let Ok(track) = track_query.single() {
        info!(
            "Track ready. Spawning car at ({:.1}, {:.1}) rot {:.2}.",
            track.spawn_position.x, track.spawn_position.y, track.spawn_rotation
        );
        spawn_car(&mut commands, track.spawn_position, track.spawn_rotation);
    } else {
        warn!("No track found at startup. Car was not spawned.");
    }
}
