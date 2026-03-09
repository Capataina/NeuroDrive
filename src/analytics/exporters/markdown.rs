use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analytics::metrics::chunking::calculate_chunks;
use crate::analytics::trackers::episode::{
    EpisodeRecord,
    EpisodeTrace,
    EpisodeTracker,
    TickTraceRecord,
    NUM_PROGRESS_SECTORS,
};

const CRITIC_CURVATURE_THRESHOLD: f32 = 0.015;
const FIRST_TURN_PROGRESS_THRESHOLD: f32 = 0.25;
const LATE_TURN_IN_THRESHOLD: f32 = 0.015;
const HIGH_CURVATURE_THROTTLE_THRESHOLD: f32 = 0.70;
const LOW_STEERING_STD_THRESHOLD: f32 = 0.08;

pub fn export_to_markdown(tracker: &EpisodeTracker, filepath: &str) {
    if tracker.episodes.is_empty() {
        return;
    }

    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let total_episodes = tracker.episodes.len();
    let chunk_size = (total_episodes / 10).max(1);
    let chunks = calculate_chunks(&tracker.episodes, chunk_size);

    let reward_values: Vec<f32> = tracker.episodes.iter().map(|r| r.reward).collect();
    let pre_terminal_values: Vec<f32> = tracker
        .episodes
        .iter()
        .map(|r| r.pre_terminal_return)
        .collect();
    let progress_reward_values: Vec<f32> = tracker
        .episodes
        .iter()
        .map(|r| r.progress_reward_sum)
        .collect();
    let time_penalty_values: Vec<f32> = tracker
        .episodes
        .iter()
        .map(|r| r.time_penalty_sum)
        .collect();
    let terminal_reward_values: Vec<f32> = tracker
        .episodes
        .iter()
        .map(|r| r.terminal_reward_sum)
        .collect();
    let crash_penalty_values: Vec<f32> = tracker
        .episodes
        .iter()
        .map(|r| r.crash_penalty_sum)
        .collect();
    let lap_bonus_values: Vec<f32> = tracker.episodes.iter().map(|r| r.lap_bonus_sum).collect();
    let progress_values: Vec<f32> = tracker.episodes.iter().map(|r| r.progress).collect();
    let max_progress_ever = progress_values.iter().copied().fold(0.0, f32::max);
    let lap_completion_rate = tracker
        .episodes
        .iter()
        .filter(|record| record.lap_completed)
        .count() as f32
        / total_episodes as f32;
    let total_crashes: u32 = tracker.episodes.iter().map(|record| record.crashes).sum();

    let mut md = String::new();
    md.push_str("# NeuroDrive Analytics Report\n\n");
    md.push_str(&format!("**Total Episodes:** {}\n\n", total_episodes));
    md.push_str(&format!("**Max Progress Achieved:** {:.2}%\n\n", max_progress_ever * 100.0));

    md.push_str("## Performance Summary\n\n");
    md.push_str(&format!(
        "- Return mean/std: {:.2} / {:.2}\n- Progress mean/std: {:.2}% / {:.2}%\n- Progress median/p90: {:.2}% / {:.2}%\n- Lap completion rate: {:.2}%\n- Total crashes: {}\n\n",
        mean(&reward_values),
        std_dev(&reward_values),
        mean(&progress_values) * 100.0,
        std_dev(&progress_values) * 100.0,
        percentile(progress_values.clone(), 0.50) * 100.0,
        percentile(progress_values.clone(), 0.90) * 100.0,
        lap_completion_rate * 100.0,
        total_crashes,
    ));

    md.push_str("## Reward Dynamics\n\n");
    md.push_str(&format!(
        "- Pre-terminal return mean/std: {:.4} / {:.4}\n- Progress reward mean/std: {:.4} / {:.4}\n- Time penalty mean/std: {:.4} / {:.4}\n- Terminal reward mean/std: {:.4} / {:.4}\n- Crash penalty mean: {:.4}\n- Lap bonus mean: {:.4}\n\n",
        mean(&pre_terminal_values),
        std_dev(&pre_terminal_values),
        mean(&progress_reward_values),
        std_dev(&progress_reward_values),
        mean(&time_penalty_values),
        std_dev(&time_penalty_values),
        mean(&terminal_reward_values),
        std_dev(&terminal_reward_values),
        mean(&crash_penalty_values),
        mean(&lap_bonus_values),
    ));

    md.push_str("## Crash Location Summary\n\n");
    let crash_bins = crash_bins(&tracker.episodes);
    if crash_bins.is_empty() {
        md.push_str("No crash positions were recorded.\n\n");
    } else {
        md.push_str("| Bin Center (x, y) | Crash Count |\n");
        md.push_str("|-------------------|-------------|\n");
        for ((x, y), count) in crash_bins.iter().take(5) {
            md.push_str(&format!("| ({x:.0}, {y:.0}) | {} |\n", count));
        }
        md.push('\n');
    }

    md.push_str("## Progress Over Time (Chunks)\n\n");
    md.push_str("| Chunk | Episodes | Avg Progress | Max Progress | Progress Std | Median | P90 | Avg Reward | Reward Std | Avg Ticks | Crash Rate | Lap Rate |\n");
    md.push_str("|-------|----------|--------------|--------------|--------------|--------|-----|------------|------------|-----------|------------|----------|\n");

    for chunk in &chunks {
        md.push_str(&format!(
            "| {} | {}-{} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2} | {:.2} | {:.1} | {:.2}% | {:.2}% |\n",
            chunk.chunk_index + 1,
            chunk.start_episode,
            chunk.end_episode,
            chunk.avg_progress * 100.0,
            chunk.max_progress * 100.0,
            chunk.progress_std * 100.0,
            chunk.progress_median * 100.0,
            chunk.progress_p90 * 100.0,
            chunk.avg_reward,
            chunk.reward_std,
            chunk.avg_ticks,
            chunk.crash_rate * 100.0,
            chunk.lap_completion_rate * 100.0,
        ));
    }

    md.push_str("\n## ASCII Chart: Average Progress\n\n```text\n");
    for chunk in &chunks {
        let bar_len = (chunk.avg_progress * 50.0).clamp(0.0, 50.0) as usize;
        let bar = "█".repeat(bar_len);
        md.push_str(&format!("Chunk {:02} | {}\n", chunk.chunk_index + 1, bar));
    }
    md.push_str("```\n");

    md.push_str("\n## Reward Dynamics (Chunks)\n\n");
    md.push_str("| Chunk | Avg Return | Avg Pre-Terminal | Avg Progress Reward | Avg Time Penalty | Avg Terminal Reward | Avg Crash Penalty | Avg Lap Bonus |\n");
    md.push_str("|-------|------------|------------------|---------------------|------------------|---------------------|-------------------|---------------|\n");
    for chunk in &chunks {
        md.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            chunk.chunk_index + 1,
            chunk.avg_reward,
            chunk.avg_pre_terminal_return,
            chunk.avg_progress_reward,
            chunk.avg_time_penalty,
            chunk.avg_terminal_reward,
            chunk.avg_crash_penalty,
            chunk.avg_lap_bonus,
        ));
    }

    md.push_str("\n## Control Summary (Chunks)\n\n");
    md.push_str("| Chunk | Steering Mean | Steering Std | Throttle Mean | Throttle Std |\n");
    md.push_str("|-------|---------------|--------------|---------------|--------------|\n");
    for chunk in &chunks {
        md.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            chunk.chunk_index + 1,
            chunk.steering_mean,
            chunk.steering_std,
            chunk.throttle_mean,
            chunk.throttle_std,
        ));
    }

    append_policy_mismatch_section(&mut md, &tracker.episodes);

    let sector_rows = compute_sector_diagnostics(&tracker.episode_traces);
    append_sector_diagnostics_section(&mut md, &sector_rows);

    let critic_diagnostics = compute_critic_diagnostics(&tracker.episode_traces);
    append_critic_context_section(&mut md, &critic_diagnostics);

    append_failure_signature_section(
        &mut md,
        &tracker.episodes,
        &tracker.episode_traces,
        &sector_rows,
        &critic_diagnostics,
    );

    append_trajectory_snapshot_section(&mut md, &tracker.episode_traces);

    if !tracker.a2c_updates.is_empty() {
        let latest = tracker.a2c_updates.last().unwrap();
        md.push_str("\n## A2C Learning Health\n\n");
        md.push_str(&format!(
            "- Total updates: {}\n- Latest policy entropy: {:.4}\n- Latest value loss: {:.4}\n- Latest explained variance: {:.4}\n- Latest steering mean/std: {:.4} / {:.4}\n- Latest throttle mean/std: {:.4} / {:.4}\n- Latest clamped action fraction: {:.2}%\n\n",
            tracker.a2c_updates.len(),
            latest.policy_entropy,
            latest.value_loss,
            latest.explained_variance,
            latest.steering_mean,
            latest.steering_std,
            latest.throttle_mean,
            latest.throttle_std,
            latest.clamped_action_fraction * 100.0,
        ));

        md.push_str("### Recent A2C Updates\n\n");
        md.push_str("| Update | Batch | Policy Loss | Value Loss | Entropy | Explained Var | Clamp % |\n");
        md.push_str("|--------|-------|-------------|------------|---------|---------------|---------|\n");
        for update in tracker.a2c_updates.iter().rev().take(5).rev() {
            md.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2}% |\n",
                update.update_index,
                update.batch_size,
                update.policy_loss,
                update.value_loss,
                update.policy_entropy,
                update.explained_variance,
                update.clamped_action_fraction * 100.0,
            ));
        }

        md.push_str("\n### Latest Layer Health\n\n");
        md.push_str("| Layer | Weight Norm | Grad Norm | Dead ReLU % |\n");
        md.push_str("|-------|-------------|-----------|-------------|\n");
        for layer in &latest.layer_health {
            let dead = layer
                .dead_relu_fraction
                .map(|value| format!("{:.2}%", value * 100.0))
                .unwrap_or_else(|| "N/A".to_string());
            md.push_str(&format!(
                "| {} | {:.4} | {:.4} | {} |\n",
                layer.layer_name, layer.weight_l2_norm, layer.gradient_l2_norm, dead
            ));
        }
    }

    let _ = fs::write(filepath, md);
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

fn percentile(mut values: Vec<f32>, q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn crash_bins(records: &[EpisodeRecord]) -> Vec<((f32, f32), usize)> {
    let mut bins = HashMap::<(i32, i32), usize>::new();
    for record in records {
        if let Some([x, y]) = record.crash_position {
            let key = ((x / 100.0).round() as i32, (y / 100.0).round() as i32);
            *bins.entry(key).or_insert(0) += 1;
        }
    }

    let mut sorted: Vec<((f32, f32), usize)> = bins
        .into_iter()
        .map(|((x, y), count)| ((x as f32 * 100.0, y as f32 * 100.0), count))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted
}

#[derive(Clone, Copy, Debug, Default)]
struct SectorDiagnosticsRow {
    sector_index: usize,
    samples: usize,
    tick_share: f32,
    speed_mean: f32,
    heading_abs_mean_rad: f32,
    throttle_mean: f32,
    terminal_count: usize,
    crash_terminal_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct SectorAccumulator {
    samples: usize,
    speed_sum: f32,
    heading_abs_sum: f32,
    throttle_sum: f32,
    terminal_count: usize,
    crash_terminal_count: usize,
}

fn compute_sector_diagnostics(traces: &[EpisodeTrace]) -> Vec<SectorDiagnosticsRow> {
    if traces.is_empty() {
        return Vec::new();
    }

    let mut accumulators = vec![SectorAccumulator::default(); NUM_PROGRESS_SECTORS];
    let mut total_ticks = 0usize;

    for trace in traces {
        for tick in &trace.ticks {
            let sector = (tick.sector_index as usize).min(NUM_PROGRESS_SECTORS.saturating_sub(1));
            let accumulator = &mut accumulators[sector];
            accumulator.samples += 1;
            accumulator.speed_sum += tick.speed;
            accumulator.heading_abs_sum += tick.heading_error.abs();
            accumulator.throttle_sum += tick.throttle;
            if tick.done {
                accumulator.terminal_count += 1;
                if tick.done_reason.as_deref() == Some("Crash") {
                    accumulator.crash_terminal_count += 1;
                }
            }
            total_ticks += 1;
        }
    }

    let mut rows = Vec::new();
    for (sector_index, accumulator) in accumulators.iter().enumerate() {
        if accumulator.samples == 0 {
            continue;
        }
        let count = accumulator.samples as f32;
        rows.push(SectorDiagnosticsRow {
            sector_index,
            samples: accumulator.samples,
            tick_share: accumulator.samples as f32 / total_ticks.max(1) as f32,
            speed_mean: accumulator.speed_sum / count,
            heading_abs_mean_rad: accumulator.heading_abs_sum / count,
            throttle_mean: accumulator.throttle_sum / count,
            terminal_count: accumulator.terminal_count,
            crash_terminal_count: accumulator.crash_terminal_count,
        });
    }

    rows
}

fn append_sector_diagnostics_section(md: &mut String, sector_rows: &[SectorDiagnosticsRow]) {
    md.push_str("\n## Turn and Sector Diagnostics\n\n");
    if sector_rows.is_empty() {
        md.push_str("No trajectory traces were recorded for sector diagnostics.\n\n");
        return;
    }

    md.push_str("| Sector | Progress Range | Ticks | Tick Share | Mean Speed | Mean Abs Heading | Mean Throttle | Terminal Ticks | Crash Terminals |\n");
    md.push_str("|--------|----------------|-------|------------|------------|------------------|---------------|----------------|-----------------|\n");
    for row in sector_rows {
        let start = row.sector_index as f32 / NUM_PROGRESS_SECTORS as f32;
        let end = (row.sector_index + 1) as f32 / NUM_PROGRESS_SECTORS as f32;
        md.push_str(&format!(
            "| {} | {:.1}%–{:.1}% | {} | {:.2}% | {:.1} | {:.2}° | {:.3} | {} | {} |\n",
            row.sector_index + 1,
            start * 100.0,
            end * 100.0,
            row.samples,
            row.tick_share * 100.0,
            row.speed_mean,
            row.heading_abs_mean_rad.to_degrees(),
            row.throttle_mean,
            row.terminal_count,
            row.crash_terminal_count,
        ));
    }
    md.push('\n');

    let mut risky = sector_rows.to_vec();
    risky.sort_by(|a, b| {
        b.crash_terminal_count
            .cmp(&a.crash_terminal_count)
            .then_with(|| b.heading_abs_mean_rad.total_cmp(&a.heading_abs_mean_rad))
    });
    md.push_str("Top risk sectors (by crash terminals then heading error): ");
    for row in risky.iter().take(3) {
        md.push_str(&format!(
            "S{} (crash {}, |heading| {:.1}°), ",
            row.sector_index + 1,
            row.crash_terminal_count,
            row.heading_abs_mean_rad.to_degrees(),
        ));
    }
    md.push_str("\n\n");
}

fn append_policy_mismatch_section(md: &mut String, episodes: &[EpisodeRecord]) {
    md.push_str("\n## Policy vs Required Action Mismatch\n\n");
    if episodes.is_empty() {
        md.push_str("No episode records were found.\n\n");
        return;
    }

    let turn_in_latencies: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.turn_in_latency_fraction)
        .collect();
    let throttle_release_latencies: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.throttle_release_latency_fraction)
        .collect();
    let steering_adequacy_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.steering_adequacy)
        .collect();
    let high_curvature_throttle_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.high_curvature_throttle_mean)
        .collect();

    md.push_str("| Metric | Mean | Std | Median | P90 | Samples |\n");
    md.push_str("|--------|------|-----|--------|-----|---------|\n");
    md.push_str(&format!(
        "| Turn-in latency (% track) | {} | {} | {} | {} | {} |\n",
        format_optional_stat_pct(&turn_in_latencies, mean),
        format_optional_stat_pct(&turn_in_latencies, std_dev),
        format_optional_stat_pct(&turn_in_latencies, |values| percentile(values.to_vec(), 0.50)),
        format_optional_stat_pct(&turn_in_latencies, |values| percentile(values.to_vec(), 0.90)),
        turn_in_latencies.len(),
    ));
    md.push_str(&format!(
        "| Throttle-release latency (% track) | {} | {} | {} | {} | {} |\n",
        format_optional_stat_pct(&throttle_release_latencies, mean),
        format_optional_stat_pct(&throttle_release_latencies, std_dev),
        format_optional_stat_pct(&throttle_release_latencies, |values| percentile(values.to_vec(), 0.50)),
        format_optional_stat_pct(&throttle_release_latencies, |values| percentile(values.to_vec(), 0.90)),
        throttle_release_latencies.len(),
    ));
    md.push_str(&format!(
        "| Steering adequacy (0-1) | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
        mean(&steering_adequacy_values),
        std_dev(&steering_adequacy_values),
        percentile(steering_adequacy_values.clone(), 0.50),
        percentile(steering_adequacy_values.clone(), 0.90),
        steering_adequacy_values.len(),
    ));
    md.push_str(&format!(
        "| High-curvature throttle | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
        mean(&high_curvature_throttle_values),
        std_dev(&high_curvature_throttle_values),
        percentile(high_curvature_throttle_values.clone(), 0.50),
        percentile(high_curvature_throttle_values.clone(), 0.90),
        high_curvature_throttle_values.len(),
    ));
    md.push('\n');
}

#[derive(Clone, Copy, Debug, Default)]
struct CriticStats {
    count: usize,
    target_sum: f32,
    target_sumsq: f32,
    error_sum: f32,
    error_sumsq: f32,
    abs_error_sum: f32,
}

impl CriticStats {
    fn push(&mut self, target: f32, prediction: f32) {
        let error = target - prediction;
        self.count += 1;
        self.target_sum += target;
        self.target_sumsq += target * target;
        self.error_sum += error;
        self.error_sumsq += error * error;
        self.abs_error_sum += error.abs();
    }

    fn mae(self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.abs_error_sum / self.count as f32
        }
    }

    fn bias(self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.error_sum / self.count as f32
        }
    }

    fn explained_variance(self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f32;
        let target_mean = self.target_sum / n;
        let error_mean = self.error_sum / n;
        let target_var = (self.target_sumsq / n) - target_mean * target_mean;
        let error_var = (self.error_sumsq / n) - error_mean * error_mean;
        if target_var <= 1e-8 {
            0.0
        } else {
            1.0 - (error_var / target_var)
        }
    }
}

#[derive(Clone, Debug)]
struct CriticDiagnostics {
    straight: CriticStats,
    high_curvature: CriticStats,
    terminal: CriticStats,
    by_sector: Vec<CriticStats>,
}

impl Default for CriticDiagnostics {
    fn default() -> Self {
        Self {
            straight: CriticStats::default(),
            high_curvature: CriticStats::default(),
            terminal: CriticStats::default(),
            by_sector: vec![CriticStats::default(); NUM_PROGRESS_SECTORS],
        }
    }
}

fn compute_critic_diagnostics(traces: &[EpisodeTrace]) -> CriticDiagnostics {
    let mut diagnostics = CriticDiagnostics::default();

    for trace in traces {
        let mut return_to_go = 0.0;
        for tick in trace.ticks.iter().rev() {
            return_to_go += tick.reward;
            let Some(prediction) = tick.value_prediction else {
                continue;
            };

            let sector = (tick.sector_index as usize).min(NUM_PROGRESS_SECTORS.saturating_sub(1));
            diagnostics.by_sector[sector].push(return_to_go, prediction);

            let max_curvature = max_abs_curvature(tick);
            if max_curvature >= CRITIC_CURVATURE_THRESHOLD {
                diagnostics.high_curvature.push(return_to_go, prediction);
            } else {
                diagnostics.straight.push(return_to_go, prediction);
            }
            if tick.done {
                diagnostics.terminal.push(return_to_go, prediction);
            }
        }
    }

    diagnostics
}

fn append_critic_context_section(md: &mut String, diagnostics: &CriticDiagnostics) {
    md.push_str("\n## Critic Quality by Context\n\n");
    md.push_str("| Context | Samples | MAE | Bias (Target-Pred) | Explained Var |\n");
    md.push_str("|---------|---------|-----|--------------------|---------------|\n");
    md.push_str(&format!(
        "| Straight-ish ticks | {} | {:.4} | {:.4} | {:.4} |\n",
        diagnostics.straight.count,
        diagnostics.straight.mae(),
        diagnostics.straight.bias(),
        diagnostics.straight.explained_variance(),
    ));
    md.push_str(&format!(
        "| High-curvature ticks | {} | {:.4} | {:.4} | {:.4} |\n",
        diagnostics.high_curvature.count,
        diagnostics.high_curvature.mae(),
        diagnostics.high_curvature.bias(),
        diagnostics.high_curvature.explained_variance(),
    ));
    md.push_str(&format!(
        "| Terminal ticks | {} | {:.4} | {:.4} | {:.4} |\n",
        diagnostics.terminal.count,
        diagnostics.terminal.mae(),
        diagnostics.terminal.bias(),
        diagnostics.terminal.explained_variance(),
    ));
    md.push('\n');

    let mut sector_rows: Vec<(usize, CriticStats)> = diagnostics
        .by_sector
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, stats)| stats.count > 0)
        .collect();
    sector_rows.sort_by(|a, b| b.1.mae().total_cmp(&a.1.mae()));

    if sector_rows.is_empty() {
        md.push_str("No per-sector critic diagnostics were available.\n\n");
        return;
    }

    md.push_str("Worst critic sectors by MAE:\n\n");
    md.push_str("| Sector | Samples | MAE | Bias | Explained Var |\n");
    md.push_str("|--------|---------|-----|------|---------------|\n");
    for (sector, stats) in sector_rows.iter().take(5) {
        md.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} |\n",
            sector + 1,
            stats.count,
            stats.mae(),
            stats.bias(),
            stats.explained_variance(),
        ));
    }
    md.push('\n');
}

fn append_failure_signature_section(
    md: &mut String,
    episodes: &[EpisodeRecord],
    traces: &[EpisodeTrace],
    sector_rows: &[SectorDiagnosticsRow],
    critic: &CriticDiagnostics,
) {
    md.push_str("\n## Failure Signature Flags\n\n");
    if episodes.is_empty() || traces.is_empty() {
        md.push_str("- Insufficient data for failure signature checks.\n\n");
        return;
    }

    let crash_end_ticks: Vec<&TickTraceRecord> = traces
        .iter()
        .filter_map(|trace| trace.ticks.last())
        .filter(|tick| tick.done_reason.as_deref() == Some("Crash"))
        .collect();
    let first_turn_crashes = crash_end_ticks
        .iter()
        .filter(|tick| tick.progress_fraction <= FIRST_TURN_PROGRESS_THRESHOLD)
        .count();
    let first_turn_crash_rate = if crash_end_ticks.is_empty() {
        0.0
    } else {
        first_turn_crashes as f32 / crash_end_ticks.len() as f32
    };

    let turn_in_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.turn_in_latency_fraction)
        .collect();
    let turn_in_mean = mean(&turn_in_values);
    let high_curvature_throttle_mean = mean(
        &episodes
            .iter()
            .map(|episode| episode.high_curvature_throttle_mean)
            .collect::<Vec<_>>(),
    );
    let steering_std_mean = mean(
        &episodes
            .iter()
            .map(|episode| episode.steering_std)
            .collect::<Vec<_>>(),
    );
    let straight_mae = critic.straight.mae();
    let high_curvature_mae = critic.high_curvature.mae();

    let mut flags = Vec::new();
    if crash_end_ticks.len() >= 10 && first_turn_crash_rate >= 0.55 {
        flags.push(format!(
            "First-turn bottleneck: {:.1}% of crashes happen before {:.0}% progress.",
            first_turn_crash_rate * 100.0,
            FIRST_TURN_PROGRESS_THRESHOLD * 100.0
        ));
    }
    if !turn_in_values.is_empty() && turn_in_mean >= LATE_TURN_IN_THRESHOLD {
        flags.push(format!(
            "Late turn-in behaviour: mean turn-in latency is {:.2}% of lap length.",
            turn_in_mean * 100.0
        ));
    }
    if high_curvature_throttle_mean >= HIGH_CURVATURE_THROTTLE_THRESHOLD {
        flags.push(format!(
            "Throttle remains high in curves: high-curvature throttle mean is {:.3}.",
            high_curvature_throttle_mean
        ));
    }
    if critic.high_curvature.count >= 20
        && critic.straight.count >= 20
        && high_curvature_mae > straight_mae * 1.35
    {
        flags.push(format!(
            "Critic curve blind spot: high-curvature MAE {:.4} vs straight MAE {:.4}.",
            high_curvature_mae, straight_mae
        ));
    }
    if steering_std_mean <= LOW_STEERING_STD_THRESHOLD {
        flags.push(format!(
            "Low steering variability: mean steering std is {:.4}.",
            steering_std_mean
        ));
    }
    if let Some(top_sector) = sector_rows
        .iter()
        .max_by_key(|row| row.crash_terminal_count)
        .filter(|row| row.crash_terminal_count > 0)
    {
        flags.push(format!(
            "Crash hotspot sector: S{} with {} crash terminals.",
            top_sector.sector_index + 1,
            top_sector.crash_terminal_count
        ));
    }

    if flags.is_empty() {
        md.push_str("- No strong failure signatures crossed threshold in this run.\n\n");
        return;
    }

    for flag in flags {
        md.push_str(&format!("- {}\n", flag));
    }
    md.push('\n');
}

fn append_trajectory_snapshot_section(md: &mut String, traces: &[EpisodeTrace]) {
    md.push_str("\n## Trajectory Snapshots\n\n");
    if traces.is_empty() {
        md.push_str("No per-tick traces were captured.\n\n");
        return;
    }

    let latest_trace = traces.last();
    let best_trace = traces
        .iter()
        .max_by(|a, b| a.best_progress.total_cmp(&b.best_progress));
    let latest_crash_trace = traces
        .iter()
        .rev()
        .find(|trace| trace.end_reason == "Crash");

    md.push_str("| Selection | Episode | End | Best Progress | Ticks | Mean Speed | Peak Speed | Mean Abs Heading | Max Abs Heading |\n");
    md.push_str("|-----------|---------|-----|---------------|-------|------------|------------|------------------|-----------------|\n");
    if let Some(trace) = latest_trace {
        md.push_str(&trajectory_row("Latest", trace));
    }
    if let Some(trace) = best_trace {
        md.push_str(&trajectory_row("Best progress", trace));
    }
    if let Some(trace) = latest_crash_trace {
        md.push_str(&trajectory_row("Latest crash", trace));
    }
    md.push('\n');
}

fn trajectory_row(label: &str, trace: &EpisodeTrace) -> String {
    let speeds: Vec<f32> = trace.ticks.iter().map(|tick| tick.speed).collect();
    let abs_headings: Vec<f32> = trace
        .ticks
        .iter()
        .map(|tick| tick.heading_error.abs())
        .collect();
    format!(
        "| {} | {} | {} | {:.2}% | {} | {:.1} | {:.1} | {:.2}° | {:.2}° |\n",
        label,
        trace.episode_id,
        trace.end_reason,
        trace.best_progress * 100.0,
        trace.ticks.len(),
        mean(&speeds),
        speeds.iter().copied().fold(0.0, f32::max),
        mean(&abs_headings).to_degrees(),
        abs_headings.iter().copied().fold(0.0, f32::max).to_degrees(),
    )
}

fn max_abs_curvature(tick: &TickTraceRecord) -> f32 {
    tick.lookahead_curvatures
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max)
}

fn format_optional_stat_pct<F>(values: &[f32], calculator: F) -> String
where
    F: Fn(&[f32]) -> f32,
{
    if values.is_empty() {
        "N/A".to_string()
    } else {
        format!("{:.2}%", calculator(values) * 100.0)
    }
}
