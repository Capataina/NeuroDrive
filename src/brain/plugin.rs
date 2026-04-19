use bevy::prelude::*;

use crate::brain::inspired::{BrainBrain, BrainInspiredPlugin};
use crate::brain::ppo::PpoBrain;
use crate::brain::ppo::buffer::TrainerRolloutBuffer;
use crate::brain::ranking::{
    TrainerLiveRanking, update_car_visual_roles_system, update_trainer_ranking_system,
};
use crate::game::car::{Car, SpawnRng, TrainerConfig, TrainerLayout};
use crate::game::plugin::spawn_cars_for_layout;
use crate::maps::track::Track;

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TrainerLiveRanking>();

        // Sub-plugins: PPO and the brain-inspired learner both register their
        // own per-tick systems and resources.
        app.add_plugins(crate::brain::ppo::PpoPlugin);
        app.add_plugins(BrainInspiredPlugin);

        app.add_systems(
            Update,
            (
                cycle_trainer_layout_system,
                update_trainer_ranking_system,
                update_car_visual_roles_system.after(update_trainer_ranking_system),
            ),
        );
    }
}

/// F4 cycles the `TrainerLayout`.
///
/// Order: Keyboard → AllPpo{8} → AllBrain{8} → SideBySide{8,8} → Keyboard …
///
/// On each press, we despawn every existing car, reset both controllers'
/// internal state (PPO rollout buffer + step counter, brain graph), and spawn
/// the set of cars the new layout demands. Rebuilding the fleet is safer
/// than migrating controller markers on existing entities because it avoids
/// subtle contamination between runs (stale eligibility traces, stale rollout
/// values).
pub fn cycle_trainer_layout_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    existing_cars: Query<Entity, With<Car>>,
    track_query: Query<&Track>,
    mut trainer_config: ResMut<TrainerConfig>,
    mut spawn_rng: ResMut<SpawnRng>,
    mut ppo_brain: Option<ResMut<PpoBrain>>,
    mut ppo_buffer: Option<ResMut<TrainerRolloutBuffer>>,
    mut brain_brain: Option<ResMut<BrainBrain>>,
) {
    if !keyboard.just_pressed(KeyCode::F4) {
        return;
    }

    let next_layout: TrainerLayout = trainer_config.layout.next();
    info!(
        "Trainer layout: {:?} → {:?}",
        trainer_config.layout, next_layout
    );
    trainer_config.set_layout(next_layout);

    // Despawn all existing cars. They will be re-spawned from scratch below.
    for entity in existing_cars.iter() {
        commands.entity(entity).despawn();
    }

    // Reset controllers' internal state so the new layout starts clean.
    if let Some(ref mut brain) = ppo_brain {
        brain.step_counter = 0;
    }
    if let Some(ref mut buf) = ppo_buffer {
        buf.clear();
    }
    if let Some(ref mut bb) = brain_brain {
        bb.reset_to_seed(next_layout.total_cars().max(1));
    }

    // Respawn cars for the new layout.
    let Ok(track) = track_query.single() else {
        warn!("No track found at layout cycle. Cars were not respawned.");
        return;
    };

    spawn_cars_for_layout(&mut commands, &trainer_config, track, &mut spawn_rng);
}
