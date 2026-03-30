use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use serde::Serialize;

use crate::agent::observation::ObservationConfig;
use crate::analytics::exporters::cleanup::enforce_retention;
use crate::analytics::exporters::context::RunContext;
use crate::analytics::models::EpisodeTracker;
use crate::brain::ppo::PpoBrain;
use crate::game::car::TrainerConfig;
use crate::game::episode::EpisodeConfig;
use crate::profiling::config::ProfilingConfig;
use crate::profiling::timers::{FrameRecord, FrameTimings};

use super::markdown::export_performance_markdown;

/// Summary statistics computed from the frame ring buffer.
#[derive(Clone, Debug, Serialize)]
struct PerformanceSummary {
    total_ticks: u64,
    duration_seconds: f32,

    // Frame time (microseconds)
    frame_mean_us: f64,
    frame_max_us: u64,
    frame_min_us: u64,
    frame_p99_us: u64,
    frame_p95_us: u64,

    // Per-set means (microseconds)
    input_mean_us: f64,
    physics_mean_us: f64,
    collision_mean_us: f64,
    measurement_mean_us: f64,

    // Per-set max (microseconds)
    input_max_us: u64,
    physics_max_us: u64,
    collision_max_us: u64,
    measurement_max_us: u64,

    // Budget
    frame_budget_us: u64,
    frames_over_budget: u64,
    over_budget_fraction: f64,

    // Stutter detection (frames > 2x budget)
    stutter_count: u64,
    worst_stutter_us: u64,
    worst_stutter_tick: u64,
}

/// Top-level export schema.
#[derive(Clone, Debug, Serialize)]
struct PerformanceReport {
    config: ProfilingConfigSnapshot,
    summary: PerformanceSummary,
    frames: Vec<FrameRecord>,
}

/// Serialisable snapshot of profiling config.
#[derive(Clone, Debug, Serialize)]
struct ProfilingConfigSnapshot {
    duration_seconds: f32,
    ring_buffer_frames: usize,
    track_set_timings: bool,
    track_buffer_sizes: bool,
}

/// Exports the performance report (JSON + Markdown) on application exit.
pub fn on_exit_export_system(
    mut exit_events: MessageReader<AppExit>,
    config: Res<ProfilingConfig>,
    timings: Res<FrameTimings>,
    trainer_config: Res<TrainerConfig>,
    episode_config: Res<EpisodeConfig>,
    obs_config: Res<ObservationConfig>,
    brain: Res<PpoBrain>,
    tracker: Res<EpisodeTracker>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    let frames: Vec<FrameRecord> = timings.frames.iter().cloned().collect();
    if frames.is_empty() {
        info!("Profiling: no frames captured, skipping export.");
        return;
    }

    let summary = compute_summary(&frames, &config);
    let report = PerformanceReport {
        config: ProfilingConfigSnapshot {
            duration_seconds: config.duration_seconds,
            ring_buffer_frames: config.ring_buffer_frames,
            track_set_timings: config.track_set_timings,
            track_buffer_sizes: config.track_buffer_sizes,
        },
        summary,
        frames: frames.clone(),
    };

    let json_dir = Path::new("reports/json/performance");
    let md_dir = Path::new("reports/performance");
    let _ = fs::create_dir_all(json_dir);
    let _ = fs::create_dir_all(md_dir);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // ── JSON export ──────────────────────────────────────────────────
    let json_path = json_dir.join(format!("perf_{timestamp}.json"));
    match serde_json::to_string_pretty(&report) {
        Ok(json) => match fs::write(&json_path, json) {
            Ok(()) => info!("Performance report (JSON) exported to {}", json_path.display()),
            Err(e) => warn!("Failed to write performance report: {e}"),
        },
        Err(e) => warn!("Failed to serialise performance report: {e}"),
    }

    // ── Markdown export ──────────────────────────────────────────────
    let run_context = RunContext::capture(
        &trainer_config,
        &episode_config,
        &obs_config,
        &brain,
        tracker.episodes.len(),
        tracker.ppo_updates.len(),
    );
    let context_header = run_context.to_markdown_header();

    let md_path = md_dir.join(format!("perf_{timestamp}.md"));
    export_performance_markdown(
        &frames,
        &config,
        &context_header,
        md_path.to_str().unwrap_or("reports/performance/perf.md"),
    );
    info!("Performance report (Markdown) exported to {}", md_path.display());

    // ── Retention cleanup ────────────────────────────────────────────
    enforce_retention(json_dir, 3);
    enforce_retention(md_dir, 3);
}

fn compute_summary(frames: &[FrameRecord], config: &ProfilingConfig) -> PerformanceSummary {
    let n = frames.len() as f64;
    let budget_us: u64 = 16_667; // 60 Hz = 16.667ms

    let frame_mean_us = frames.iter().map(|f| f.total_us as f64).sum::<f64>() / n;
    let frame_max_us = frames.iter().map(|f| f.total_us).max().unwrap_or(0);
    let frame_min_us = frames.iter().map(|f| f.total_us).min().unwrap_or(0);

    let mut sorted_totals: Vec<u64> = frames.iter().map(|f| f.total_us).collect();
    sorted_totals.sort_unstable();
    let frame_p99_us = percentile_u64(&sorted_totals, 99);
    let frame_p95_us = percentile_u64(&sorted_totals, 95);

    let input_mean_us = frames.iter().map(|f| f.input_us as f64).sum::<f64>() / n;
    let physics_mean_us = frames.iter().map(|f| f.physics_us as f64).sum::<f64>() / n;
    let collision_mean_us = frames.iter().map(|f| f.collision_us as f64).sum::<f64>() / n;
    let measurement_mean_us = frames.iter().map(|f| f.measurement_us as f64).sum::<f64>() / n;

    let input_max_us = frames.iter().map(|f| f.input_us).max().unwrap_or(0);
    let physics_max_us = frames.iter().map(|f| f.physics_us).max().unwrap_or(0);
    let collision_max_us = frames.iter().map(|f| f.collision_us).max().unwrap_or(0);
    let measurement_max_us = frames.iter().map(|f| f.measurement_us).max().unwrap_or(0);

    let frames_over_budget = frames.iter().filter(|f| f.total_us > budget_us).count() as u64;
    let over_budget_fraction = frames_over_budget as f64 / n;

    let stutter_threshold = budget_us * 2;
    let stutter_count = frames
        .iter()
        .filter(|f| f.total_us > stutter_threshold)
        .count() as u64;
    let worst_frame = frames
        .iter()
        .max_by_key(|f| f.total_us)
        .cloned()
        .unwrap_or_default();

    PerformanceSummary {
        total_ticks: frames.last().map(|f| f.tick + 1).unwrap_or(0),
        duration_seconds: config.duration_seconds,
        frame_mean_us,
        frame_max_us,
        frame_min_us,
        frame_p99_us,
        frame_p95_us,
        input_mean_us,
        physics_mean_us,
        collision_mean_us,
        measurement_mean_us,
        input_max_us,
        physics_max_us,
        collision_max_us,
        measurement_max_us,
        frame_budget_us: budget_us,
        frames_over_budget,
        over_budget_fraction,
        stutter_count,
        worst_stutter_us: worst_frame.total_us,
        worst_stutter_tick: worst_frame.tick,
    }
}

fn percentile_u64(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() * pct / 100).min(sorted.len() - 1);
    sorted[idx]
}
