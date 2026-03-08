use bevy::prelude::*;
use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;

use crate::analytics::trackers::episode::{
    EpisodeActionAccumulator,
    EpisodeTracker,
    capture_episode_action_stats_system,
    episode_tracker_system,
    snapshot_completed_episode_action_stats_system,
};
use crate::analytics::exporters::json::export_to_json;
use crate::analytics::exporters::markdown::export_to_markdown;
use crate::game::episode::episode_loop_system;
use crate::sim::sets::SimSet;

pub struct AnalyticsPlugin;

impl Plugin for AnalyticsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EpisodeTracker>()
           .init_resource::<EpisodeActionAccumulator>()
           .add_systems(
               FixedUpdate,
               capture_episode_action_stats_system.in_set(SimSet::Physics),
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
        
        let json_path = format!("reports/run_{}.json", timestamp);
        info!("Exporting JSON to: {}", json_path);
        export_to_json(&tracker, &json_path);
        
        let md_path = format!("reports/run_{}.md", timestamp);
        info!("Exporting Markdown to: {}", md_path);
        export_to_markdown(&tracker, &md_path);
        
        info!("Analytics successfully exported.");
    }
}
