use std::fs;
use std::path::Path;

/// Enforces a retention limit on files in a directory.
/// Keeps only the `keep` most recent files (sorted by filename, which contains timestamps).
/// Deletes the oldest files beyond the limit.
pub(crate) fn enforce_retention(dir: &Path, keep: usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();

    // Sort by filename ascending (timestamps sort lexicographically)
    files.sort_by_key(|e| e.file_name());

    if files.len() > keep {
        let to_remove = files.len() - keep;
        for entry in files.into_iter().take(to_remove) {
            let path = entry.path();
            if fs::remove_file(&path).is_ok() {
                bevy::log::info!("Auto-cleanup: removed old report {}", path.display());
            }
        }
    }
}
