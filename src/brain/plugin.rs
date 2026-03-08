use bevy::prelude::*;

use crate::brain::types::AgentMode;

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AgentMode>();
        
        // Add specific brain plugins
        app.add_plugins(crate::brain::a2c::A2cPlugin);

        app.add_systems(Update, toggle_agent_mode_system);
    }
}

fn toggle_agent_mode_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<AgentMode>,
) {
    if keyboard.just_pressed(KeyCode::F4) {
        *mode = match *mode {
            AgentMode::Keyboard => {
                info!("Agent Mode: AI");
                AgentMode::Ai
            }
            AgentMode::Ai => {
                info!("Agent Mode: Keyboard");
                AgentMode::Keyboard
            }
        };
    }
}
