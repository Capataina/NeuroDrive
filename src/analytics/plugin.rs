use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;

use crate::agent::observation::build_observation_vector_system;
use crate::analytics::exporters::json::{export_compact_json, export_full_json};
use crate::analytics::exporters::markdown::export_to_markdown;
use crate::analytics::models::{AnalyticsConfig, EpisodeTracker, RunMetadata};
use crate::analytics::trackers::action::{
    PerCarActionAccumulators, capture_episode_action_stats_system,
    snapshot_completed_episode_action_stats_system,
};
use crate::analytics::trackers::episode::episode_tracker_system;
use crate::analytics::trackers::trace::{
    PerCarTraceAccumulators, capture_episode_tick_trace_system,
    snapshot_completed_episode_trace_system,
};
use crate::brain::a2c::a2c_collect_rewards_all_cars_system;
use crate::brain::a2c::A2cBrain;
use crate::game::car::TrainerConfig;
use crate::game::episode::episode_loop_system;
use crate::sim::sets::SimSet;

pub struct AnalyticsPlugin;

impl Plugin for AnalyticsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EpisodeTracker>()
            .init_resource::<PerCarActionAccumulators>()
            .init_resource::<PerCarTraceAccumulators>()
            .init_resource::<AnalyticsConfig>()
            .add_systems(
                FixedUpdate,
                capture_episode_action_stats_system.in_set(SimSet::Physics),
            )
            .add_systems(
                FixedUpdate,
                capture_episode_tick_trace_system
                    .after(build_observation_vector_system)
                    .after(episode_loop_system)
                    .before(a2c_collect_rewards_all_cars_system)
                    .in_set(SimSet::Measurement),
            )
            .add_systems(
                FixedUpdate,
                snapshot_completed_episode_trace_system
                    .after(capture_episode_tick_trace_system)
                    .in_set(SimSet::Measurement),
            )
            .add_systems(
                FixedUpdate,
                snapshot_completed_episode_action_stats_system
                    .after(episode_loop_system)
                    .in_set(SimSet::Measurement),
            )
            .add_systems(Update, episode_tracker_system)
            .add_systems(Last, on_exit_system);
    }
}

fn on_exit_system(
    mut exit_events: MessageReader<AppExit>,
    tracker: Res<EpisodeTracker>,
    config: Res<AnalyticsConfig>,
    trainer_config: Res<TrainerConfig>,
    brain: Res<A2cBrain>,
) {
    for exit_event in exit_events.read() {
        info!("Game exit event detected: {:?}", exit_event);

        if tracker.episodes.is_empty() && tracker.a2c_updates.is_empty() {
            info!("No analytics data to export.");
            return;
        }

        info!(
            "Starting analytics export for {} episodes and {} A2C updates...",
            tracker.episodes.len(),
            tracker.a2c_updates.len()
        );
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let metadata = RunMetadata {
            car_count: trainer_config.num_envs,
            track_name: "monaco".to_string(),
            session_timestamp: timestamp,
            ppo_epochs: brain.ppo_epochs,
            clip_epsilon: brain.clip_epsilon,
            gamma: brain.gamma,
            gae_lambda: brain.gae_lambda,
            max_steps: brain.max_steps,
            samples_per_tick: brain.samples_per_tick,
        };

        // Always write compact JSON (no traces).
        let compact_path = format!("reports/run_{}.json", timestamp);
        info!("Exporting compact JSON to: {}", compact_path);
        export_compact_json(&tracker, &metadata, &compact_path);

        // Opt-in: write full trace JSON when configured.
        if config.full_trace_export {
            let traces_path = format!("reports/run_{}_traces.json", timestamp);
            info!("Exporting full trace JSON to: {}", traces_path);
            export_full_json(&tracker, &traces_path);
        }

        // Always write the markdown report from full in-memory data.
        let md_path = format!("reports/run_{}.md", timestamp);
        info!("Exporting Markdown to: {}", md_path);
        export_to_markdown(&tracker, &md_path);

        info!("Analytics successfully exported.");
    }
}
