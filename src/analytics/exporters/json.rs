use std::fs;
use std::path::Path;
use crate::analytics::trackers::episode::EpisodeTracker;

pub fn export_to_json(tracker: &EpisodeTracker, filepath: &str) {
    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    if let Ok(json) = serde_json::to_string_pretty(tracker) {
        let _ = fs::write(filepath, json);
    }
}
