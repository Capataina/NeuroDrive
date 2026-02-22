mod game;
mod maps;

use bevy::prelude::*;
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
        // Track must be spawned before game systems query it
        .add_plugins(MonacoPlugin)
        .add_plugins(GamePlugin)
        .run();
}
