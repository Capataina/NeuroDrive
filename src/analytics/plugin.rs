use std::path::Path;

use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;

use crate::agent::observation::{ObservationConfig, build_observation_vector_system};
use crate::analytics::exporters::cleanup::enforce_retention;
use crate::analytics::exporters::context::RunContext;
use crate::analytics::exporters::json::{export_compact_json, export_full_json};
use crate::analytics::exporters::markdown::export_to_markdown;
use crate::analytics::models::{AnalyticsConfig, EpisodeTracker, RunMetadata};
use crate::game::episode::EpisodeConfig;
use crate::analytics::trackers::action::{
    PerCarActionAccumulators, capture_episode_action_stats_system,
    snapshot_completed_episode_action_stats_system,
};
use crate::analytics::trackers::episode::episode_tracker_system;
use crate::analytics::trackers::trace::{
    PerCarTraceAccumulators, capture_episode_tick_trace_system,
    snapshot_completed_episode_trace_system,
};
use crate::brain::ppo::ppo_collect_rewards_all_cars_system;
use crate::brain::ppo::PpoBrain;
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
                    .before(ppo_collect_rewards_all_cars_system)
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
    episode_config: Res<EpisodeConfig>,
    obs_config: Res<ObservationConfig>,
    brain: Res<PpoBrain>,
) {
    for exit_event in exit_events.read() {
        info!("Game exit event detected: {:?}", exit_event);

        if tracker.episodes.is_empty() && tracker.ppo_updates.is_empty() {
            info!("No analytics data to export.");
            return;
        }

        info!(
            "Starting analytics export for {} episodes and {} PPO updates...",
            tracker.episodes.len(),
            tracker.ppo_updates.len()
        );
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let metadata = RunMetadata {
            car_count: trainer_config.num_envs,
            track_name: "monaco".to_string(),
            session_timestamp: timestamp,
            ppo_epochs: brain.config.ppo_epochs,
            clip_epsilon: brain.config.clip_epsilon,
            gamma: brain.config.gamma,
            gae_lambda: brain.config.gae_lambda,
            max_steps: brain.config.max_steps,
            samples_per_tick: brain.config.samples_per_tick,
            layout: trainer_config.layout.label().to_string(),
            ppo_cars: trainer_config.layout.ppo_count(),
            brain_cars: trainer_config.layout.brain_count(),
        };

        // Capture run context for the markdown header.
        let run_context = RunContext::capture(
            &trainer_config,
            &episode_config,
            &obs_config,
            &brain,
            tracker.episodes.len(),
            tracker.ppo_updates.len(),
        );
        let context_header = run_context.to_markdown_header();

        // Always write compact JSON (no traces).
        let json_dir = Path::new("reports/json/analytics");
        let compact_path = format!("reports/json/analytics/run_{}.json", timestamp);
        info!("Exporting compact JSON to: {}", compact_path);
        export_compact_json(&tracker, &metadata, &compact_path);

        // Opt-in: write full trace JSON when configured.
        if config.full_trace_export {
            let traces_path = format!("reports/json/analytics/run_{}_traces.json", timestamp);
            info!("Exporting full trace JSON to: {}", traces_path);
            export_full_json(&tracker, &traces_path);
        }

        // Enforce retention on JSON directory.
        enforce_retention(json_dir, 3);

        // Always write the markdown report from full in-memory data.
        let analytics_dir = Path::new("reports/analytics");
        let md_path = format!("reports/analytics/run_{}.md", timestamp);
        info!("Exporting Markdown to: {}", md_path);
        export_to_markdown(&tracker, &md_path, &context_header);

        // Enforce retention on analytics markdown directory.
        enforce_retention(analytics_dir, 5);

        info!("Analytics successfully exported.");
    }
}
