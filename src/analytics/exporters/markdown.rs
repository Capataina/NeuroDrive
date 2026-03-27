use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analytics::metrics::chunking::{DEFAULT_CHUNK_COUNT, calculate_chunks};
use crate::analytics::metrics::consistency::{
    compute_sector_consistency, overall_consistency_score,
};
use crate::analytics::metrics::diagnostics::compute_diagnostic_flags;
use crate::analytics::metrics::phases::detect_learning_phase;
use crate::analytics::metrics::sectors::compute_sector_diagnostics;
use crate::analytics::metrics::sparkline::{ascii_bar, heatmap_row, sparkline};
use crate::analytics::metrics::stats::mean;
use crate::analytics::metrics::timeseries::{
    extract_episode_series, extract_update_series, rolling_mean,
};
use crate::analytics::metrics::trajectory::select_trajectory_snapshots;
use crate::analytics::metrics::turns::summarize_failure_modes;
use crate::analytics::models::{EpisodeRecord, EpisodeTracker, NUM_PROGRESS_SECTORS};

const SPARKLINE_WIDTH: usize = 40;
const BAR_WIDTH: usize = 20;

pub fn export_to_markdown(tracker: &EpisodeTracker, filepath: &str) {
    if tracker.episodes.is_empty() {
        return;
    }

    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut md = String::with_capacity(8192);

    // ── Pre-compute all analytics ──────────────────────────────────────
    let ep_series = extract_episode_series(tracker);
    let upd_series = extract_update_series(tracker);
    let chunks = calculate_chunks(&tracker.episodes, DEFAULT_CHUNK_COUNT);
    let diag_flags = compute_diagnostic_flags(tracker);
    let phase = detect_learning_phase(tracker);
    let sector_rows = compute_sector_diagnostics(&tracker.episode_traces);
    let consistency_profiles = compute_sector_consistency(tracker, 50);
    let consistency_score = overall_consistency_score(&consistency_profiles);
    let failure_modes = summarize_failure_modes(&tracker.episodes);
    let trajectory_rows = select_trajectory_snapshots(&tracker.episode_traces);

    // Progress/reward rolling means for sparklines.
    let progress_pct: Vec<f32> = ep_series.progress.iter().map(|p| p * 100.0).collect();
    let reward_vals: Vec<f32> = ep_series.reward.clone();
    let progress_smooth = rolling_mean(&progress_pct, 20);
    let reward_smooth = rolling_mean(&reward_vals, 20);
    let crash_flags: Vec<f32> = ep_series
        .is_crash
        .iter()
        .map(|&c| if c { 1.0 } else { 0.0 })
        .collect();
    let crash_smooth = rolling_mean(&crash_flags, 20);

    // ════════════════════════════════════════════════════════════════════
    // 1. Run Summary
    // ════════════════════════════════════════════════════════════════════
    md.push_str("# NeuroDrive Analytics Report\n\n");
    md.push_str("## 1. Run Summary\n\n");

    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Episodes | {} |\n", tracker.episodes.len()));

    // Per-car count.
    let mut car_set: Vec<u32> = ep_series.env_ids.iter().copied().collect();
    car_set.sort_unstable();
    car_set.dedup();
    md.push_str(&format!("| Cars | {} |\n", car_set.len()));

    md.push_str(&format!(
        "| PPO updates | {} |\n",
        tracker.a2c_updates.len()
    ));
    md.push_str(&format!("| Learning phase | **{}** |\n", phase));
    md.push('\n');

    if !diag_flags.is_empty() {
        md.push_str("**Diagnostics:**\n");
        for flag in &diag_flags {
            let icon = match flag.severity {
                crate::analytics::metrics::diagnostics::Severity::Warning => "[!]",
                crate::analytics::metrics::diagnostics::Severity::Info => "[i]",
            };
            md.push_str(&format!("- {icon} {}\n", flag.message));
        }
        md.push('\n');
    }

    // ════════════════════════════════════════════════════════════════════
    // 2. Is the Policy Learning?
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 2. Is the Policy Learning?\n\n");

    md.push_str(&format!(
        "**Progress:** `{}` trend: {}\n\n",
        sparkline(&progress_smooth, SPARKLINE_WIDTH),
        trend_word(&progress_smooth),
    ));
    md.push_str(&format!(
        "**Reward:** `{}` trend: {}\n\n",
        sparkline(&reward_smooth, SPARKLINE_WIDTH),
        trend_word(&reward_smooth),
    ));
    md.push_str(&format!(
        "**Crash rate:** `{}`\n\n",
        sparkline(&crash_smooth, SPARKLINE_WIDTH),
    ));

    // 10-chunk trend table.
    if !chunks.is_empty() {
        md.push_str("| Chunk | Episodes | Avg Progress | Max | Crash % | Avg Reward | Ticks |\n");
        md.push_str("|------:|---------:|-------------:|----:|--------:|-----------:|------:|\n");
        for c in &chunks {
            md.push_str(&format!(
                "| {} | {}-{} | {:.1}% | {:.1}% | {:.0}% | {:.2} | {:.0} |\n",
                c.chunk_index + 1,
                c.start_episode,
                c.end_episode,
                c.avg_progress * 100.0,
                c.max_progress * 100.0,
                c.crash_rate * 100.0,
                c.avg_reward,
                c.avg_ticks,
            ));
        }
        md.push('\n');
    }

    // ════════════════════════════════════════════════════════════════════
    // 3. Has It Found a Route?
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 3. Has It Found a Route?\n\n");
    md.push_str(&format!(
        "**Overall consistency score:** {:.3}\n\n",
        consistency_score
    ));

    // Speed profile bar chart by sector.
    if !sector_rows.is_empty() {
        let max_speed = sector_rows
            .iter()
            .map(|r| r.speed_mean)
            .fold(0.0_f32, f32::max);

        md.push_str("**Speed profile by sector:**\n\n```text\n");
        for row in &sector_rows {
            md.push_str(&format!(
                "S{:02} | {}\n",
                row.sector_index + 1,
                ascii_bar(row.speed_mean, max_speed, BAR_WIDTH),
            ));
        }
        md.push_str("```\n\n");
    }

    // Per-sector consistency — show top 5 sectors by total variance.
    let mut sorted_profiles: Vec<(usize, f32)> = consistency_profiles
        .iter()
        .filter(|p| p.sample_count > 0)
        .map(|p| {
            (
                p.sector,
                p.speed_var + p.steering_var + p.throttle_var + p.centerline_dist_var,
            )
        })
        .collect();
    sorted_profiles.sort_by(|a, b| b.1.total_cmp(&a.1));

    if !sorted_profiles.is_empty() {
        md.push_str("**Highest-variance sectors (least consistent):**\n\n");
        md.push_str("| Sector | Speed Var | Steer Var | Throttle Var | CL Dist Var | Samples |\n");
        md.push_str("|-------:|----------:|----------:|-------------:|------------:|--------:|\n");
        for &(sector, _) in sorted_profiles.iter().take(5) {
            let p = &consistency_profiles[sector];
            md.push_str(&format!(
                "| {} | {:.1} | {:.4} | {:.4} | {:.2} | {} |\n",
                sector + 1,
                p.speed_var,
                p.steering_var,
                p.throttle_var,
                p.centerline_dist_var,
                p.sample_count,
            ));
        }
        md.push('\n');
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. Per-Car Performance
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 4. Per-Car Performance\n\n");
    let car_stats = per_car_stats(&tracker.episodes);
    if car_stats.len() > 1 {
        md.push_str("| Car | Episodes | Avg Progress | Avg Reward | Crash % |\n");
        md.push_str("|----:|---------:|-------------:|-----------:|--------:|\n");
        for cs in &car_stats {
            md.push_str(&format!(
                "| {} | {} | {:.1}% | {:.2} | {:.0}% |\n",
                cs.env_id,
                cs.count,
                cs.avg_progress * 100.0,
                cs.avg_reward,
                cs.crash_rate * 100.0,
            ));
        }
        md.push('\n');

        if let (Some(best), Some(worst)) = (
            car_stats
                .iter()
                .max_by(|a, b| a.avg_progress.total_cmp(&b.avg_progress)),
            car_stats
                .iter()
                .min_by(|a, b| a.avg_progress.total_cmp(&b.avg_progress)),
        ) {
            if best.env_id != worst.env_id {
                md.push_str(&format!(
                    "Best car {} ({:.1}% avg progress) vs worst car {} ({:.1}% avg progress).\n\n",
                    best.env_id,
                    best.avg_progress * 100.0,
                    worst.env_id,
                    worst.avg_progress * 100.0,
                ));
            }
        }
    } else {
        md.push_str("Single-car run — no comparison available.\n\n");
    }

    // ════════════════════════════════════════════════════════════════════
    // 5. Where Does It Fail?
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 5. Where Does It Fail?\n\n");

    // Crash heatmap by sector.
    let crash_by_sector = crash_counts_by_sector(&tracker.episodes);
    let crash_sector_floats: Vec<f32> = crash_by_sector.iter().map(|&c| c as f32).collect();
    let total_sector_crashes: usize = crash_by_sector.iter().sum();

    if total_sector_crashes > 0 {
        md.push_str(&format!(
            "**Crash heatmap by sector:** `{}`\n\n",
            heatmap_row(&crash_sector_floats),
        ));

        // Crash sector table — top 5.
        let mut sector_crash_list: Vec<(usize, usize)> = crash_by_sector
            .iter()
            .enumerate()
            .filter(|&(_, c)| *c > 0)
            .map(|(i, &c)| (i, c))
            .collect();
        sector_crash_list.sort_by(|a, b| b.1.cmp(&a.1));

        md.push_str("| Sector | Crashes | Share |\n");
        md.push_str("|-------:|--------:|------:|\n");
        for &(sector, count) in sector_crash_list.iter().take(5) {
            md.push_str(&format!(
                "| {} | {} | {:.0}% |\n",
                sector + 1,
                count,
                count as f32 / total_sector_crashes.max(1) as f32 * 100.0,
            ));
        }
        md.push('\n');
    } else {
        md.push_str("No sector-attributed crashes recorded.\n\n");
    }

    // Failure mode table.
    if !failure_modes.is_empty() {
        md.push_str("**Failure modes:**\n\n");
        md.push_str("| Mode | Count | Share |\n");
        md.push_str("|------|------:|------:|\n");
        for mode in &failure_modes {
            md.push_str(&format!(
                "| {} | {} | {:.0}% |\n",
                mode.label,
                mode.count,
                mode.share * 100.0,
            ));
        }
        md.push('\n');
    }

    // Corner vs straight crash comparison.
    if !sector_rows.is_empty() {
        let (corner_crashes, corner_total, straight_crashes, straight_total) =
            corner_vs_straight_crashes(&sector_rows);
        if corner_total > 0 || straight_total > 0 {
            let corner_rate = if corner_total > 0 {
                corner_crashes as f32 / corner_total as f32 * 100.0
            } else {
                0.0
            };
            let straight_rate = if straight_total > 0 {
                straight_crashes as f32 / straight_total as f32 * 100.0
            } else {
                0.0
            };
            md.push_str(&format!(
                "Corner crash rate: {:.0}% ({}/{}) | Straight crash rate: {:.0}% ({}/{})\n\n",
                corner_rate, corner_crashes, corner_total, straight_rate, straight_crashes,
                straight_total,
            ));
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 6. Training Health
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 6. Training Health\n\n");

    if !upd_series.entropy.is_empty() {
        md.push_str(&format!(
            "**Entropy:** `{}`\n\n",
            sparkline(&upd_series.entropy, SPARKLINE_WIDTH),
        ));
        md.push_str(&format!(
            "**Clip %:** `{}`\n\n",
            sparkline(&upd_series.clip_fraction, SPARKLINE_WIDTH),
        ));
        md.push_str(&format!(
            "**Approx KL:** `{}`\n\n",
            sparkline(&upd_series.approx_kl, SPARKLINE_WIDTH),
        ));
        md.push_str(&format!(
            "**Explained Var:** `{}`\n\n",
            sparkline(&upd_series.explained_variance, SPARKLINE_WIDTH),
        ));
    }

    // Latest update snapshot.
    if let Some(latest) = tracker.a2c_updates.last() {
        md.push_str("**Latest update:**\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Policy entropy | {:.4} |\n",
            latest.policy_entropy
        ));
        md.push_str(&format!(
            "| Value loss | {:.4} |\n",
            latest.value_loss
        ));
        md.push_str(&format!(
            "| Explained variance | {:.4} |\n",
            latest.explained_variance
        ));
        md.push_str(&format!(
            "| Clip fraction | {:.2}% |\n",
            latest.clip_fraction * 100.0
        ));
        md.push_str(&format!(
            "| Approx KL | {:.5} |\n",
            latest.approx_kl
        ));
        md.push_str(&format!(
            "| Steering mean / std | {:.4} / {:.4} |\n",
            latest.steering_mean, latest.steering_std
        ));
        md.push_str(&format!(
            "| Throttle mean / std | {:.4} / {:.4} |\n",
            latest.throttle_mean, latest.throttle_std
        ));
        md.push('\n');

        // Layer health table.
        if !latest.layer_health.is_empty() {
            md.push_str("**Layer health:**\n\n");
            md.push_str("| Layer | Weight Norm | Grad Norm | Saturated % |\n");
            md.push_str("|-------|------------:|----------:|------------:|\n");
            for layer in &latest.layer_health {
                let dead = layer
                    .saturated_fraction
                    .map(|v| format!("{:.1}%", v * 100.0))
                    .unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!(
                    "| {} | {:.4} | {:.4} | {} |\n",
                    layer.layer_name, layer.weight_l2_norm, layer.gradient_l2_norm, dead,
                ));
            }
            md.push('\n');
        }
    }

    // Reward decomposition trends by chunk.
    if !chunks.is_empty() {
        md.push_str("**Reward decomposition by chunk:**\n\n");
        md.push_str("| Chunk | Progress R | Time Pen | Crash Pen | Lap Bonus |\n");
        md.push_str("|------:|-----------:|---------:|----------:|----------:|\n");
        for c in &chunks {
            md.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
                c.chunk_index + 1,
                c.avg_progress_reward,
                c.avg_time_penalty,
                c.avg_crash_penalty,
                c.avg_lap_bonus,
            ));
        }
        md.push('\n');
    }

    // ════════════════════════════════════════════════════════════════════
    // 7. Trajectory Snapshots
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 7. Trajectory Snapshots\n\n");

    if trajectory_rows.is_empty() {
        md.push_str("No per-tick traces were captured.\n");
    } else {
        md.push_str("| Selection | Episode | End | Progress | Ticks | Mean Speed | Peak Speed |\n");
        md.push_str("|-----------|--------:|-----|--------:|------:|-----------:|-----------:|\n");
        for row in &trajectory_rows {
            md.push_str(&format!(
                "| {} | {} | {} | {:.1}% | {} | {:.1} | {:.1} |\n",
                row.selection,
                row.episode_id,
                row.end_reason,
                row.best_progress * 100.0,
                row.ticks,
                row.mean_speed,
                row.peak_speed,
            ));
        }
    }

    let _ = fs::write(filepath, md);
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Summarises the direction of a smoothed series in a single word.
fn trend_word(smoothed: &[f32]) -> &'static str {
    if smoothed.len() < 4 {
        return "insufficient data";
    }
    let n = smoothed.len();
    let recent = mean(&smoothed[n.saturating_sub(10)..].to_vec());
    let early = mean(&smoothed[..10.min(n)].to_vec());
    let delta = recent - early;
    if delta > 3.0 {
        "rising"
    } else if delta < -3.0 {
        "falling"
    } else {
        "flat"
    }
}

struct CarStats {
    env_id: u32,
    count: usize,
    avg_progress: f32,
    avg_reward: f32,
    crash_rate: f32,
}

fn per_car_stats(episodes: &[EpisodeRecord]) -> Vec<CarStats> {
    let mut map: HashMap<u32, (usize, f32, f32, usize)> = HashMap::new();
    for ep in episodes {
        let entry = map.entry(ep.env_id).or_insert((0, 0.0, 0.0, 0));
        entry.0 += 1;
        entry.1 += ep.progress;
        entry.2 += ep.reward;
        if ep.end_reason == "crash" {
            entry.3 += 1;
        }
    }

    let mut stats: Vec<CarStats> = map
        .into_iter()
        .map(|(env_id, (count, progress_sum, reward_sum, crashes))| {
            let c = count as f32;
            CarStats {
                env_id,
                count,
                avg_progress: progress_sum / c,
                avg_reward: reward_sum / c,
                crash_rate: crashes as f32 / c,
            }
        })
        .collect();
    stats.sort_by_key(|s| s.env_id);
    stats
}

/// Counts crashes per progress sector. Episodes without `crash_position`
/// contribute nothing. We estimate the sector from `progress` as a fraction
/// of the track.
fn crash_counts_by_sector(episodes: &[EpisodeRecord]) -> Vec<usize> {
    let mut counts = vec![0usize; NUM_PROGRESS_SECTORS];
    for ep in episodes {
        if ep.end_reason == "crash" || ep.end_reason.contains("Crash") {
            // Use the episode progress to estimate sector.
            let sector = (ep.progress * NUM_PROGRESS_SECTORS as f32).floor() as usize;
            let sector = sector.min(NUM_PROGRESS_SECTORS - 1);
            counts[sector] += 1;
        }
    }
    counts
}

/// Classifies sectors as corners (high mean steering magnitude in traces) or
/// straights, then computes crash rates for each category.
///
/// Returns (corner_crashes, corner_terminals, straight_crashes, straight_terminals).
fn corner_vs_straight_crashes(
    sector_rows: &[crate::analytics::metrics::sectors::SectorDiagnosticsRow],
) -> (usize, usize, usize, usize) {
    // A sector is a "corner" if the mean absolute heading error exceeds 0.05 rad (~3 deg).
    let heading_threshold = 0.05_f32;

    let mut corner_crashes = 0usize;
    let mut corner_total = 0usize;
    let mut straight_crashes = 0usize;
    let mut straight_total = 0usize;

    for row in sector_rows {
        if row.terminal_count == 0 {
            continue;
        }
        if row.heading_abs_mean_rad > heading_threshold {
            corner_crashes += row.crash_terminal_count;
            corner_total += row.terminal_count;
        } else {
            straight_crashes += row.crash_terminal_count;
            straight_total += row.terminal_count;
        }
    }

    (corner_crashes, corner_total, straight_crashes, straight_total)
}
