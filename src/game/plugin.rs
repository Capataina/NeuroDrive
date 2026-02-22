use bevy::prelude::*;
use crate::game::car::{car_control_system, spawn_car};
use crate::game::collision::{collision_detection_system, handle_collision_system};
use crate::maps::track::Track;

/// Main game plugin that bundles all game systems.
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(handle_collision_system)
            .add_systems(PostStartup, setup_game)
            .add_systems(
                Update,
                (
                    car_control_system,
                    collision_detection_system,
                ),
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
