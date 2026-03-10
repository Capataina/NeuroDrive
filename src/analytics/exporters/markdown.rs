use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analytics::metrics::chunking::{DEFAULT_CHUNK_COUNT, calculate_chunks};
use crate::analytics::metrics::critic::compute_critic_diagnostics;
use crate::analytics::metrics::inputs::{
    calculate_input_learning_chunks, summarize_input_signal_trends,
};
use crate::analytics::metrics::insights::build_report_insights;
use crate::analytics::metrics::sectors::compute_sector_diagnostics;
use crate::analytics::metrics::stats::{mean, percentile, std_dev};
use crate::analytics::metrics::trajectory::select_trajectory_snapshots;
use crate::analytics::metrics::turns::{summarize_failure_modes, summarize_turn_execution};
use crate::analytics::models::{EpisodeRecord, EpisodeTracker, NUM_PROGRESS_SECTORS};

pub fn export_to_markdown(tracker: &EpisodeTracker, filepath: &str) {
    if tracker.episodes.is_empty() {
        return;
    }

    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let chunks = calculate_chunks(&tracker.episodes, DEFAULT_CHUNK_COUNT);
    let input_chunks = calculate_input_learning_chunks(&tracker.episodes);
    let input_trends = summarize_input_signal_trends(&input_chunks);
    let turn_summary = summarize_turn_execution(&tracker.episodes);
    let failure_modes = summarize_failure_modes(&tracker.episodes);
    let sector_rows = compute_sector_diagnostics(&tracker.episode_traces);
    let critic = compute_critic_diagnostics(&tracker.episode_traces);
    let trajectory_rows = select_trajectory_snapshots(&tracker.episode_traces);
    let insights = build_report_insights(
        &tracker.episodes,
        &chunks,
        &input_chunks,
        &input_trends,
        &turn_summary,
        &failure_modes,
        &sector_rows,
        &critic,
        &tracker.a2c_updates,
    );

    let reward_values: Vec<f32> = tracker.episodes.iter().map(|r| r.reward).collect();
    let progress_values: Vec<f32> = tracker.episodes.iter().map(|r| r.progress).collect();
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
    let max_progress_ever = progress_values.iter().copied().fold(0.0, f32::max);
    let lap_completion_rate = tracker
        .episodes
        .iter()
        .filter(|record| record.lap_completed)
        .count() as f32
        / tracker.episodes.len() as f32;
    let total_crashes: u32 = tracker.episodes.iter().map(|record| record.crashes).sum();

    let mut md = String::new();
    md.push_str("# NeuroDrive Analytics Report\n\n");
    md.push_str("> Auto-generated run summary with behaviour, turn-execution, and optimisation diagnostics.\n\n");

    md.push_str("## Executive Summary\n\n");
    md.push_str(&format!(
        "- Episodes: **{}**\n- Max progress: **{:.2}%**\n- Lap completion rate: **{:.2}%**\n- Total crashes: **{}**\n\n",
        tracker.episodes.len(),
        max_progress_ever * 100.0,
        lap_completion_rate * 100.0,
        total_crashes
    ));
    append_insights(&mut md, &insights.overview);

    md.push_str("## Performance Overview\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!(
        "| Return mean / std | {:.2} / {:.2} |\n",
        mean(&reward_values),
        std_dev(&reward_values)
    ));
    md.push_str(&format!(
        "| Progress mean / std | {:.2}% / {:.2}% |\n",
        mean(&progress_values) * 100.0,
        std_dev(&progress_values) * 100.0
    ));
    md.push_str(&format!(
        "| Progress median / p90 | {:.2}% / {:.2}% |\n",
        percentile(progress_values.clone(), 0.50) * 100.0,
        percentile(progress_values.clone(), 0.90) * 100.0
    ));
    md.push_str(&format!(
        "| Pre-terminal return mean / std | {:.4} / {:.4} |\n",
        mean(&pre_terminal_values),
        std_dev(&pre_terminal_values)
    ));
    md.push_str(&format!(
        "| Progress reward mean / std | {:.4} / {:.4} |\n",
        mean(&progress_reward_values),
        std_dev(&progress_reward_values)
    ));
    md.push_str(&format!(
        "| Time penalty mean / std | {:.4} / {:.4} |\n",
        mean(&time_penalty_values),
        std_dev(&time_penalty_values)
    ));
    md.push_str(&format!(
        "| Terminal reward mean / std | {:.4} / {:.4} |\n",
        mean(&terminal_reward_values),
        std_dev(&terminal_reward_values)
    ));
    md.push_str(&format!(
        "| Crash penalty mean | {:.4} |\n| Lap bonus mean | {:.4} |\n\n",
        mean(&crash_penalty_values),
        mean(&lap_bonus_values)
    ));
    append_insights(&mut md, &insights.performance);

    md.push_str("### Progress Trend (10 Chunks)\n\n");
    md.push_str("| Chunk | Episodes | Avg Progress | Max Progress | Std | Median | P90 | Avg Reward | Avg Ticks | Crash Rate | Lap Rate |\n");
    md.push_str("|------:|----------|-------------:|-------------:|----:|-------:|----:|-----------:|----------:|-----------:|---------:|\n");
    for chunk in &chunks {
        md.push_str(&format!(
            "| {} | {}-{} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2} | {:.1} | {:.2}% | {:.2}% |\n",
            chunk.chunk_index + 1,
            chunk.start_episode,
            chunk.end_episode,
            chunk.avg_progress * 100.0,
            chunk.max_progress * 100.0,
            chunk.progress_std * 100.0,
            chunk.progress_median * 100.0,
            chunk.progress_p90 * 100.0,
            chunk.avg_reward,
            chunk.avg_ticks,
            chunk.crash_rate * 100.0,
            chunk.lap_completion_rate * 100.0,
        ));
    }
    md.push('\n');
    append_ascii_chart(
        &mut md,
        "Average Progress by Chunk",
        &chunks
            .iter()
            .map(|chunk| (format!("C{:02}", chunk.chunk_index + 1), chunk.avg_progress))
            .collect::<Vec<_>>(),
        false,
    );
    md.push_str("### Reward Dynamics by Chunk\n\n");
    md.push_str("| Chunk | Avg Return | Reward Std | Avg Pre-Terminal | Avg Progress Reward | Avg Time Penalty | Avg Terminal Reward | Avg Crash Penalty | Avg Lap Bonus |\n");
    md.push_str("|------:|-----------:|-----------:|-----------------:|--------------------:|-----------------:|--------------------:|------------------:|--------------:|\n");
    for chunk in &chunks {
        md.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            chunk.chunk_index + 1,
            chunk.avg_reward,
            chunk.reward_std,
            chunk.avg_pre_terminal_return,
            chunk.avg_progress_reward,
            chunk.avg_time_penalty,
            chunk.avg_terminal_reward,
            chunk.avg_crash_penalty,
            chunk.avg_lap_bonus,
        ));
    }
    md.push('\n');
    md.push_str("### Control Summary by Chunk\n\n");
    md.push_str("| Chunk | Steering Mean | Steering Std | Throttle Mean | Throttle Std |\n");
    md.push_str("|------:|--------------:|-------------:|--------------:|-------------:|\n");
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
    md.push('\n');

    md.push_str("## Input Learning Diagnostics\n\n");
    md.push_str("| Chunk | Episodes | Mean Gap | Mean Abs Offset | Mean Abs Heading | Mean All Rays | Mean Front Rays | Mean Side Rays |\n");
    md.push_str("|------:|----------|---------:|---------------:|----------------:|--------------:|----------------:|---------------:|\n");
    for chunk in &input_chunks {
        md.push_str(&format!(
            "| {} | {}-{} | {:.2} | {:.2} | {:.2}° | {:.2} | {:.2} | {:.2} |\n",
            chunk.chunk_index + 1,
            chunk.start_episode,
            chunk.end_episode,
            chunk.mean_centerline_distance,
            chunk.mean_abs_lateral_offset,
            chunk.mean_abs_heading_error_deg,
            chunk.mean_all_ray_distance,
            chunk.mean_front_ray_distance,
            chunk.mean_side_ray_distance,
        ));
    }
    md.push('\n');
    md.push_str("| Signal | Desired | Early | Late | Delta |\n");
    md.push_str("|--------|---------|------:|-----:|------:|\n");
    for trend in &input_trends {
        md.push_str(&format!(
            "| {} | {} | {:.2} | {:.2} | {:+.2} |\n",
            trend.name, trend.desired_direction, trend.early_value, trend.late_value, trend.delta
        ));
    }
    md.push('\n');
    append_ascii_chart(
        &mut md,
        "Centreline Distance by Chunk (lower is better)",
        &input_chunks
            .iter()
            .map(|chunk| {
                (
                    format!("C{:02}", chunk.chunk_index + 1),
                    chunk.mean_centerline_distance,
                )
            })
            .collect::<Vec<_>>(),
        true,
    );
    append_ascii_chart(
        &mut md,
        "Front Ray Clearance by Chunk (higher is better)",
        &input_chunks
            .iter()
            .map(|chunk| {
                (
                    format!("C{:02}", chunk.chunk_index + 1),
                    chunk.mean_front_ray_distance,
                )
            })
            .collect::<Vec<_>>(),
        false,
    );
    append_insights(&mut md, &insights.inputs);

    md.push_str("## Turn Execution Diagnostics\n\n");
    md.push_str("### Curvature Demand vs Actual Steering\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!(
        "| Steering adequacy | {:.3} |\n| Curvature steering abs error mean | {:.3} |\n| Curvature steering bias mean | {:.3} |\n| Understeer rate | {:.1}% |\n| High-curvature throttle mean | {:.3} |\n\n",
        turn_summary.steering_adequacy_mean,
        turn_summary.curvature_steering_error_mean,
        turn_summary.curvature_steering_bias_mean,
        turn_summary.understeer_rate_mean * 100.0,
        turn_summary.high_curvature_throttle_mean,
    ));
    md.push_str("### Turn Preparation\n\n");
    md.push_str("| Metric | Mean | P90 |\n");
    md.push_str("|--------|-----:|----:|\n");
    md.push_str(&format!(
        "| Turn-in latency (% track) | {} | {} |\n",
        turn_summary
            .turn_in_latency_fraction_mean
            .map(|value| format!("{:.2}%", value * 100.0))
            .unwrap_or_else(|| "N/A".to_string()),
        turn_summary
            .turn_in_latency_fraction_p90
            .map(|value| format!("{:.2}%", value * 100.0))
            .unwrap_or_else(|| "N/A".to_string()),
    ));
    md.push_str(&format!(
        "| Turn-in latency (ticks) | {} | N/A |\n",
        turn_summary
            .turn_in_latency_ticks_mean
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "N/A".to_string()),
    ));
    md.push_str(&format!(
        "| Throttle-release latency (% track) | {} | N/A |\n",
        turn_summary
            .throttle_release_latency_fraction_mean
            .map(|value| format!("{:.2}%", value * 100.0))
            .unwrap_or_else(|| "N/A".to_string()),
    ));
    md.push_str(&format!(
        "| Throttle-release latency (ticks) | {} | N/A |\n\n",
        turn_summary
            .throttle_release_latency_ticks_mean
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "N/A".to_string()),
    ));

    md.push_str("### Speed and Lane Position Through Turn\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    append_optional_row(
        &mut md,
        "Turn-entry speed mean",
        turn_summary.turn_entry_speed_mean,
        "",
    );
    append_optional_row(
        &mut md,
        "Turn-entry speed p90",
        turn_summary.turn_entry_speed_p90,
        "",
    );
    append_optional_row(
        &mut md,
        "Peak-curvature speed mean",
        turn_summary.peak_curvature_speed_mean,
        "",
    );
    append_optional_row(
        &mut md,
        "Crash speed mean",
        turn_summary.crash_speed_mean,
        "",
    );
    append_optional_row(
        &mut md,
        "Entry lateral offset mean",
        turn_summary.entry_lateral_offset_mean,
        "",
    );
    append_optional_row(
        &mut md,
        "Peak lateral offset mean",
        turn_summary.peak_lateral_offset_mean,
        "",
    );
    append_optional_row(
        &mut md,
        "Peak centreline distance mean",
        turn_summary.peak_centerline_distance_mean,
        "",
    );
    md.push('\n');

    md.push_str("### Failure Mode Classification\n\n");
    if failure_modes.is_empty() {
        md.push_str("No classified crash modes were available.\n\n");
    } else {
        md.push_str("| Failure Mode | Count | Share |\n");
        md.push_str("|--------------|------:|------:|\n");
        for mode in &failure_modes {
            md.push_str(&format!(
                "| {} | {} | {:.1}% |\n",
                mode.label,
                mode.count,
                mode.share * 100.0
            ));
        }
        md.push('\n');
    }
    append_insights(&mut md, &insights.turns);

    md.push_str("## Sector and Crash Geography\n\n");
    append_crash_location_summary(&mut md, &tracker.episodes);
    if sector_rows.is_empty() {
        md.push_str("No trajectory traces were recorded for sector diagnostics.\n\n");
    } else {
        md.push_str("| Sector | Progress Range | Ticks | Tick Share | Mean Speed | Mean Abs Heading | Mean Gap | Mean Throttle | Terminal Ticks | Crash Terminals |\n");
        md.push_str("|------:|----------------|------:|-----------:|-----------:|---------------:|---------:|--------------:|---------------:|----------------:|\n");
        for row in &sector_rows {
            let start = row.sector_index as f32 / NUM_PROGRESS_SECTORS as f32;
            let end = (row.sector_index + 1) as f32 / NUM_PROGRESS_SECTORS as f32;
            md.push_str(&format!(
                "| {} | {:.1}%–{:.1}% | {} | {:.2}% | {:.1} | {:.2}° | {:.2} | {:.3} | {} | {} |\n",
                row.sector_index + 1,
                start * 100.0,
                end * 100.0,
                row.samples,
                row.tick_share * 100.0,
                row.speed_mean,
                row.heading_abs_mean_rad.to_degrees(),
                row.centerline_distance_mean,
                row.throttle_mean,
                row.terminal_count,
                row.crash_terminal_count,
            ));
        }
        md.push('\n');
    }
    append_insights(&mut md, &insights.sectors);

    md.push_str("## Critic and Optimisation Diagnostics\n\n");
    md.push_str("### Critic Quality by Context\n\n");
    md.push_str("| Context | Samples | MAE | Bias | Explained Var |\n");
    md.push_str("|---------|--------:|----:|-----:|--------------:|\n");
    md.push_str(&format!(
        "| Straight-ish ticks | {} | {:.4} | {:.4} | {:.4} |\n",
        critic.straight.count,
        critic.straight.mae(),
        critic.straight.bias(),
        critic.straight.explained_variance(),
    ));
    md.push_str(&format!(
        "| High-curvature ticks | {} | {:.4} | {:.4} | {:.4} |\n",
        critic.high_curvature.count,
        critic.high_curvature.mae(),
        critic.high_curvature.bias(),
        critic.high_curvature.explained_variance(),
    ));
    md.push_str(&format!(
        "| Terminal ticks | {} | {:.4} | {:.4} | {:.4} |\n\n",
        critic.terminal.count,
        critic.terminal.mae(),
        critic.terminal.bias(),
        critic.terminal.explained_variance(),
    ));
    append_insights(&mut md, &insights.critic);

    if let Some(latest) = tracker.a2c_updates.last() {
        md.push_str("### Latest A2C Snapshot\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total updates | {} |\n| Latest policy entropy | {:.4} |\n| Latest value loss | {:.4} |\n| Latest explained variance | {:.4} |\n| Latest steering mean / std | {:.4} / {:.4} |\n| Latest throttle mean / std | {:.4} / {:.4} |\n| Latest clamped action fraction | {:.2}% |\n\n",
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
        md.push_str(
            "| Update | Batch | Policy Loss | Value Loss | Entropy | Explained Var | Clamp % |\n",
        );
        md.push_str(
            "|------:|------:|------------:|-----------:|--------:|--------------:|--------:|\n",
        );
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
        md.push('\n');
        md.push_str("### Latest Layer Health\n\n");
        md.push_str("| Layer | Weight Norm | Grad Norm | Dead ReLU % |\n");
        md.push_str("|-------|------------:|----------:|------------:|\n");
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
        md.push('\n');
        append_insights(&mut md, &insights.optimisation);
    }

    md.push_str("## Trajectory Snapshots\n\n");
    if trajectory_rows.is_empty() {
        md.push_str("No per-tick traces were captured.\n");
    } else {
        md.push_str("| Selection | Episode | End | Best Progress | Ticks | Mean Speed | Peak Speed | Mean Abs Heading | Max Abs Heading |\n");
        md.push_str("|-----------|--------:|-----|--------------:|------:|-----------:|-----------:|---------------:|--------------:|\n");
        for row in &trajectory_rows {
            md.push_str(&format!(
                "| {} | {} | {} | {:.2}% | {} | {:.1} | {:.1} | {:.2}° | {:.2}° |\n",
                row.selection,
                row.episode_id,
                row.end_reason,
                row.best_progress * 100.0,
                row.ticks,
                row.mean_speed,
                row.peak_speed,
                row.mean_abs_heading_deg,
                row.max_abs_heading_deg,
            ));
        }
    }

    let _ = fs::write(filepath, md);
}

fn append_optional_row(md: &mut String, label: &str, value: Option<f32>, suffix: &str) {
    let value = value
        .map(|value| format!("{value:.2}{suffix}"))
        .unwrap_or_else(|| "N/A".to_string());
    md.push_str(&format!("| {label} | {value} |\n"));
}

fn append_insights(md: &mut String, insights: &[String]) {
    if insights.is_empty() {
        return;
    }
    md.push_str("> Insights\n");
    for insight in insights {
        md.push_str(&format!("> - {insight}\n"));
    }
    md.push('\n');
}

fn append_ascii_chart(md: &mut String, title: &str, rows: &[(String, f32)], invert: bool) {
    if rows.is_empty() {
        return;
    }
    let max_value = rows
        .iter()
        .map(|(_, value)| *value)
        .fold(0.0, f32::max)
        .max(1e-6);
    let min_value = rows
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::MAX, f32::min);
    md.push_str(&format!("### {title}\n\n```text\n"));
    for (label, value) in rows {
        let normalized = if invert {
            1.0 - ((*value - min_value) / (max_value - min_value).max(1e-6))
        } else {
            *value / max_value
        };
        let bar = "█".repeat((normalized.clamp(0.0, 1.0) * 32.0).round() as usize);
        md.push_str(&format!("{label:>4} | {bar} {value:.2}\n"));
    }
    md.push_str("```\n\n");
}

fn append_crash_location_summary(md: &mut String, records: &[EpisodeRecord]) {
    md.push_str("### Crash Location Summary\n\n");
    let bins = crash_bins(records);
    if bins.is_empty() {
        md.push_str("No crash positions were recorded.\n\n");
        return;
    }

    md.push_str("| Bin Center (x, y) | Crash Count |\n");
    md.push_str("|-------------------|------------:|\n");
    for ((x, y), count) in bins.iter().take(8) {
        md.push_str(&format!("| ({x:.0}, {y:.0}) | {} |\n", count));
    }
    md.push('\n');
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
