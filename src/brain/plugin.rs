use bevy::prelude::*;

use crate::brain::ppo::PpoBrain;
use crate::brain::ppo::buffer::TrainerRolloutBuffer;
use crate::brain::ranking::{
    TrainerLiveRanking, update_car_visual_roles_system, update_trainer_ranking_system,
};
use crate::brain::types::AgentMode;

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AgentMode>()
            .init_resource::<TrainerLiveRanking>();

        // Add specific brain plugins
        app.add_plugins(crate::brain::ppo::PpoPlugin);

        app.add_systems(
            Update,
            (
                toggle_agent_mode_system,
                update_trainer_ranking_system,
                update_car_visual_roles_system.after(update_trainer_ranking_system),
            ),
        );
    }
}

fn toggle_agent_mode_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<AgentMode>,
    mut ppo_brain: Option<ResMut<PpoBrain>>,
    mut buffer: Option<ResMut<TrainerRolloutBuffer>>,
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

        if let Some(ref mut brain) = ppo_brain {
            brain.step_counter = 0;
        }
        if let Some(ref mut buf) = buffer {
            buf.clear();
            info!("Trainer rollout buffer reset after mode switch.");
        }
    }
}
