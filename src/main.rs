mod game;
mod maps;
mod agent;
mod debug;
mod sim;

use bevy::prelude::*;
use bevy::time::Fixed;
use game::GamePlugin;
use maps::MonacoPlugin;
use agent::AgentPlugin;
use debug::DebugPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "NeuroDrive".to_string(),
                resolution: (1600, 900).into(),
                ..default()
            }),
            ..default()
        }))
        // Fixed timestep: required for determinism, replay, and stable metrics.
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        // Track must be spawned before game systems query it
        .add_plugins(MonacoPlugin)
        .add_plugins(AgentPlugin)
        .add_plugins(GamePlugin)
        .add_plugins(DebugPlugin)
        .run();
}
