use crate::analytics::models::{CompactRunExport, EpisodeTracker, RunMetadata};
use std::fs;
use std::path::Path;

/// Writes the full `EpisodeTracker` (including per-tick traces) to a JSON file.
/// This is the opt-in heavyweight export.
pub fn export_full_json(tracker: &EpisodeTracker, filepath: &str) {
    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    if let Ok(json) = serde_json::to_string_pretty(tracker) {
        let _ = fs::write(filepath, json);
    }
}

/// Writes a compact JSON export — metadata, episode records, and PPO update
/// records, but no per-tick trace data. This is always written on exit.
pub fn export_compact_json(tracker: &EpisodeTracker, metadata: &RunMetadata, filepath: &str) {
    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let export = CompactRunExport {
        metadata: metadata.clone(),
        episodes: tracker.episodes.clone(),
        ppo_updates: tracker.ppo_updates.clone(),
        brain_records: tracker.brain_records.clone(),
    };

    if let Ok(json) = serde_json::to_string_pretty(&export) {
        let _ = fs::write(filepath, json);
    }
}
