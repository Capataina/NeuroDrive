use crate::analytics::models::EpisodeTrace;

#[derive(Clone, Debug)]
pub struct TrajectorySnapshotRow {
    pub selection: &'static str,
    pub episode_id: u32,
    pub end_reason: String,
    pub best_progress: f32,
    pub ticks: usize,
    pub mean_speed: f32,
    pub peak_speed: f32,
    pub mean_abs_heading_deg: f32,
    pub max_abs_heading_deg: f32,
}

pub fn select_trajectory_snapshots(traces: &[EpisodeTrace]) -> Vec<TrajectorySnapshotRow> {
    let mut rows = Vec::new();
    let latest = traces.last();
    let best = traces
        .iter()
        .max_by(|a, b| a.best_progress.total_cmp(&b.best_progress));

    if let Some(trace) = latest {
        rows.push(snapshot_row("Latest", trace));
    }
    if let Some(trace) = best {
        if rows.iter().all(|row| row.episode_id != trace.episode_id) {
            rows.push(snapshot_row("Best progress", trace));
        } else if let Some(row) = rows
            .iter_mut()
            .find(|row| row.episode_id == trace.episode_id)
        {
            row.selection = "Latest / Best progress";
        }
    }
    if let Some(trace) = traces
        .iter()
        .rev()
        .find(|trace| trace.end_reason == "Crash")
        .filter(|trace| rows.iter().all(|row| row.episode_id != trace.episode_id))
    {
        rows.push(snapshot_row("Latest crash", trace));
    }

    rows
}

fn snapshot_row(selection: &'static str, trace: &EpisodeTrace) -> TrajectorySnapshotRow {
    let ticks = trace.ticks.len();
    let mean_speed = if ticks == 0 {
        0.0
    } else {
        trace.ticks.iter().map(|tick| tick.speed).sum::<f32>() / ticks as f32
    };
    let peak_speed = trace
        .ticks
        .iter()
        .map(|tick| tick.speed)
        .fold(0.0, f32::max);
    let mean_abs_heading_deg = if ticks == 0 {
        0.0
    } else {
        trace
            .ticks
            .iter()
            .map(|tick| tick.heading_error.abs().to_degrees())
            .sum::<f32>()
            / ticks as f32
    };
    let max_abs_heading_deg = trace
        .ticks
        .iter()
        .map(|tick| tick.heading_error.abs().to_degrees())
        .fold(0.0, f32::max);

    TrajectorySnapshotRow {
        selection,
        episode_id: trace.episode_id,
        end_reason: trace.end_reason.clone(),
        best_progress: trace.best_progress,
        ticks,
        mean_speed,
        peak_speed,
        mean_abs_heading_deg,
        max_abs_heading_deg,
    }
}
