mod agent;
mod analytics;
mod brain;
mod debug;
mod game;
mod maps;
mod sim;

use agent::AgentPlugin;
use analytics::plugin::AnalyticsPlugin;
use bevy::prelude::*;
use bevy::time::Fixed;
use brain::plugin::BrainPlugin;
use debug::DebugPlugin;
use game::GamePlugin;
use maps::MonacoPlugin;

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
        .add_plugins(BrainPlugin)
        .add_plugins(AnalyticsPlugin)
        .add_plugins(GamePlugin)
        .add_plugins(DebugPlugin)
        .run();
}
