use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analytics::metrics::chunking::calculate_chunks;
use crate::analytics::trackers::episode::{EpisodeRecord, EpisodeTracker};

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
