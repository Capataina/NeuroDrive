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
use crate::analytics::models::{EpisodeRecord, EpisodeTracker, NUM_PROGRESS_SECTORS};

const SPARKLINE_WIDTH: usize = 40;
const BAR_WIDTH: usize = 20;

pub fn export_to_markdown(tracker: &EpisodeTracker, filepath: &str, context_header: &str) {
    if tracker.episodes.is_empty() {
        return;
    }

    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut md = String::with_capacity(16384);

    // ── Pre-compute all analytics ──────────────────────────────────────
    let ep_series = extract_episode_series(tracker);
    let upd_series = extract_update_series(tracker);
    let chunks = calculate_chunks(&tracker.episodes, DEFAULT_CHUNK_COUNT);
    let diag_flags = compute_diagnostic_flags(tracker);
    let phase = detect_learning_phase(tracker);
    let sector_rows = compute_sector_diagnostics(&tracker.episode_traces);
    let consistency_profiles = compute_sector_consistency(tracker, 50);
    let consistency_score = overall_consistency_score(&consistency_profiles);
    let trajectory_rows = select_trajectory_snapshots(&tracker.episode_traces);

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

    // Episode-level series for new sparklines.
    let speed_vals: Vec<f32> = tracker.episodes.iter().map(|e| e.mean_speed).collect();
    let speed_smooth = rolling_mean(&speed_vals, 20);
    let dist_vals: Vec<f32> = tracker.episodes.iter().map(|e| e.distance_driven).collect();
    let dist_smooth = rolling_mean(&dist_vals, 20);
    let life_vals: Vec<f32> = tracker.episodes.iter().map(|e| e.ticks as f32 / 60.0).collect();
    let life_smooth = rolling_mean(&life_vals, 20);
    let vp_vals: Vec<f32> = tracker.episodes.iter().map(|e| e.mean_velocity_projection).collect();
    let vp_smooth = rolling_mean(&vp_vals, 20);
    let brake_vals: Vec<f32> = tracker.episodes.iter().map(|e| e.braking_fraction * 100.0).collect();
    let brake_smooth = rolling_mean(&brake_vals, 20);
    let rps_vals: Vec<f32> = tracker.episodes.iter().map(|e| e.reward_per_second).collect();
    let rps_smooth = rolling_mean(&rps_vals, 20);

    // ════════════════════════════════════════════════════════════════════
    // 1. Run Summary
    // ════════════════════════════════════════════════════════════════════
    md.push_str("# NeuroDrive Analytics Report\n\n");

    if !context_header.is_empty() {
        md.push_str(context_header);
        md.push('\n');
    }

    md.push_str("## 1. Run Summary\n\n");

    let total_distance: f32 = tracker.episodes.iter().map(|e| e.distance_driven).sum();
    let mean_life = mean(&life_vals);
    let mean_rps = mean(&rps_vals);
    let max_sector = tracker.episodes.iter().map(|e| e.furthest_sector).max().unwrap_or(0);
    let mean_brake_pct = mean(&brake_vals);

    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Episodes | {} |\n", tracker.episodes.len()));
    let mut car_set: Vec<u32> = ep_series.env_ids.iter().copied().collect();
    car_set.sort_unstable();
    car_set.dedup();
    md.push_str(&format!("| Cars | {} |\n", car_set.len()));
    md.push_str(&format!("| PPO updates | {} |\n", tracker.ppo_updates.len()));
    md.push_str(&format!("| Learning phase | **{}** |\n", phase));
    md.push_str(&format!("| Total distance driven | {:.0} units |\n", total_distance));
    md.push_str(&format!("| Mean episode duration | {:.1}s |\n", mean_life));
    md.push_str(&format!("| Track coverage | Sector {} of {} |\n", max_sector + 1, NUM_PROGRESS_SECTORS));
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

    md.push_str(&format!(
        "> **Takeaway:** Cars average {:.1}s alive, reaching sector {}, spending {:.0}% of time braking, earning {:.2} reward/s.\n\n",
        mean_life, max_sector + 1, mean_brake_pct, mean_rps
    ));

    // ════════════════════════════════════════════════════════════════════
    // 2. Is the Policy Learning?
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 2. Is the Policy Learning?\n\n");

    md.push_str(&format!("**Progress:** `{}` trend: {}\n\n", sparkline(&progress_smooth, SPARKLINE_WIDTH), trend_word(&progress_smooth)));
    md.push_str(&format!("**Reward:** `{}` trend: {}\n\n", sparkline(&reward_smooth, SPARKLINE_WIDTH), trend_word(&reward_smooth)));
    md.push_str(&format!("**Distance driven:** `{}` trend: {}\n\n", sparkline(&dist_smooth, SPARKLINE_WIDTH), trend_word(&dist_smooth)));
    md.push_str(&format!("**Mean speed:** `{}` trend: {}\n\n", sparkline(&speed_smooth, SPARKLINE_WIDTH), trend_word(&speed_smooth)));
    md.push_str(&format!("**Life duration:** `{}` trend: {}\n\n", sparkline(&life_smooth, SPARKLINE_WIDTH), trend_word(&life_smooth)));
    md.push_str(&format!("**Crash rate:** `{}`\n\n", sparkline(&crash_smooth, SPARKLINE_WIDTH)));
    md.push_str(&format!("**Reward/s:** `{}` trend: {}\n\n", sparkline(&rps_smooth, SPARKLINE_WIDTH), trend_word(&rps_smooth)));

    if !chunks.is_empty() {
        md.push_str("| Chunk | Episodes | Distance | Avg Speed | Life(s) | Crash % | Reward | Reward/s |\n");
        md.push_str("|------:|---------:|---------:|----------:|--------:|--------:|-------:|---------:|\n");
        for c in &chunks {
            md.push_str(&format!(
                "| {} | {}-{} | {:.0} | {:.0} | {:.1} | {:.0}% | {:.2} | {:.2} |\n",
                c.chunk_index + 1, c.start_episode, c.end_episode,
                c.avg_distance_driven, c.avg_speed, c.avg_life_seconds,
                c.crash_rate * 100.0, c.avg_reward, c.avg_reward_per_second,
            ));
        }
        md.push('\n');
    }

    let dist_trend = trend_word(&dist_smooth);
    let speed_trend = trend_word(&speed_smooth);
    md.push_str(&format!(
        "> **Takeaway:** Distance driven is {}. Speed is {}. {}.\n\n",
        dist_trend, speed_trend,
        if dist_trend == "rising" && speed_trend == "rising" { "The car is learning to drive further and faster" }
        else if dist_trend == "flat" { "Learning has stagnated — the car is not making progress" }
        else { "Mixed signals — watch the next few chunks" }
    ));

    // ════════════════════════════════════════════════════════════════════
    // 3. Action Behaviour
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 3. Action Behaviour\n\n");

    let total_episodes = tracker.episodes.len() as f32;
    let global_brake = tracker.episodes.iter().map(|e| e.braking_fraction).sum::<f32>() / total_episodes * 100.0;
    let global_accel = tracker.episodes.iter().map(|e| e.acceleration_fraction).sum::<f32>() / total_episodes * 100.0;
    let global_coast = tracker.episodes.iter().map(|e| e.coasting_fraction).sum::<f32>() / total_episodes * 100.0;

    md.push_str("**Throttle distribution:**\n\n```text\n");
    md.push_str(&format!("Braking  (<-0.1)  {} {:.0}%\n", ascii_bar(global_brake, 100.0, BAR_WIDTH), global_brake));
    md.push_str(&format!("Coasting (±0.1)   {} {:.0}%\n", ascii_bar(global_coast, 100.0, BAR_WIDTH), global_coast));
    md.push_str(&format!("Throttle (>0.1)   {} {:.0}%\n", ascii_bar(global_accel, 100.0, BAR_WIDTH), global_accel));
    md.push_str("```\n\n");

    md.push_str(&format!("**Braking %:** `{}` trend: {}\n\n", sparkline(&brake_smooth, SPARKLINE_WIDTH), trend_word(&brake_smooth)));

    if !chunks.is_empty() {
        md.push_str("| Chunk | Brake % | Coast % | Accel % | Jitter | Steer σ | Throttle σ |\n");
        md.push_str("|------:|--------:|--------:|--------:|-------:|--------:|-----------:|\n");
        for c in &chunks {
            md.push_str(&format!(
                "| {} | {:.0}% | {:.0}% | {:.0}% | {:.3} | {:.3} | {:.3} |\n",
                c.chunk_index + 1,
                c.avg_braking_fraction * 100.0,
                c.avg_coasting_fraction * 100.0,
                c.avg_acceleration_fraction * 100.0,
                c.avg_action_change,
                c.avg_policy_steering_std,
                c.avg_policy_throttle_std,
            ));
        }
        md.push('\n');
    }

    let brake_discovered = global_brake > 5.0;
    let jitter_level = if !chunks.is_empty() {
        let last = chunks.last().unwrap();
        if last.avg_action_change > 0.5 { "high" } else if last.avg_action_change > 0.2 { "moderate" } else { "low" }
    } else { "unknown" };

    md.push_str(&format!(
        "> **Takeaway:** The car spends {:.0}% braking, {:.0}% accelerating, {:.0}% coasting. {}. Action jitter is {}.\n\n",
        global_brake, global_accel, global_coast,
        if brake_discovered { "Braking has been discovered" } else { "The car has NOT discovered braking yet" },
        jitter_level,
    ));

    // ════════════════════════════════════════════════════════════════════
    // 4. Speed and Momentum
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 4. Speed and Momentum\n\n");

    md.push_str(&format!("**Velocity projection:** `{}` trend: {}\n\n", sparkline(&vp_smooth, SPARKLINE_WIDTH), trend_word(&vp_smooth)));

    if !chunks.is_empty() {
        md.push_str("| Chunk | Avg Speed | Peak Speed | V-Projection | Drift° | Distance |\n");
        md.push_str("|------:|----------:|-----------:|-------------:|-------:|---------:|\n");
        for c in &chunks {
            md.push_str(&format!(
                "| {} | {:.0} | {:.0} | {:.1} | {:.1}° | {:.0} |\n",
                c.chunk_index + 1, c.avg_speed, c.avg_peak_speed,
                c.avg_velocity_projection, c.avg_drift_angle_deg, c.avg_distance_driven,
            ));
        }
        md.push('\n');
    }

    let avg_speed_global = mean(&speed_vals);
    let avg_vp_global = mean(&vp_vals);
    let vp_efficiency = if avg_speed_global > 1.0 { avg_vp_global / avg_speed_global * 100.0 } else { 0.0 };
    let avg_drift_global = mean(&tracker.episodes.iter().map(|e| e.mean_drift_angle_deg).collect::<Vec<_>>());

    md.push_str(&format!(
        "> **Takeaway:** Mean speed {:.0} u/s. Velocity projection efficiency {:.0}% (how much speed is useful). Mean drift {:.1}°. {}.\n\n",
        avg_speed_global, vp_efficiency, avg_drift_global,
        if avg_drift_global > 20.0 { "Heavy sliding — cars are losing control" }
        else if avg_drift_global > 8.0 { "Moderate drift — some sliding through corners" }
        else { "Minimal drift — cars are gripping well" }
    ));

    // ════════════════════════════════════════════════════════════════════
    // 5. Crash Forensics
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 5. Crash Forensics\n\n");

    let crash_episodes: Vec<&EpisodeRecord> = tracker.episodes.iter()
        .filter(|e| e.end_reason.contains("Crash"))
        .collect();
    let total_crashes = crash_episodes.len();

    if total_crashes > 0 {
        let mut type_counts: HashMap<&str, usize> = HashMap::new();
        for ep in &crash_episodes {
            let ct = ep.crash_type.as_deref().unwrap_or("Unknown");
            *type_counts.entry(ct).or_insert(0) += 1;
        }
        let mut type_list: Vec<(&&str, &usize)> = type_counts.iter().collect();
        type_list.sort_by(|a, b| b.1.cmp(a.1));

        md.push_str("**Crash type distribution:**\n\n```text\n");
        for (ct, count) in &type_list {
            let pct = **count as f32 / total_crashes as f32 * 100.0;
            md.push_str(&format!("{:<12} {} {:.0}%\n", ct, ascii_bar(pct, 100.0, BAR_WIDTH), pct));
        }
        md.push_str("```\n\n");

        let crash_speeds: Vec<f32> = crash_episodes.iter().filter_map(|e| e.crash_speed).collect();
        if !crash_speeds.is_empty() {
            let avg_crash_spd = crash_speeds.iter().sum::<f32>() / crash_speeds.len() as f32;
            md.push_str(&format!("**Mean crash speed:** {:.0} u/s\n\n", avg_crash_spd));
        }

        // Crash sector heatmap.
        let crash_by_sector = crash_counts_by_sector(&tracker.episodes);
        let crash_sector_floats: Vec<f32> = crash_by_sector.iter().map(|&c| c as f32).collect();
        let total_sector_crashes: usize = crash_by_sector.iter().sum();
        if total_sector_crashes > 0 {
            md.push_str(&format!("**Crash heatmap by sector:** `{}`\n\n", heatmap_row(&crash_sector_floats)));
        }

        // Crash chunk table.
        if !chunks.is_empty() {
            md.push_str("| Chunk | Crashes | Avg Crash Speed | Slide % | Overshoot % | Head-on % |\n");
            md.push_str("|------:|--------:|----------------:|--------:|------------:|----------:|\n");
            for c in &chunks {
                let episode_span = c.end_episode.saturating_sub(c.start_episode) + 1;
                let chunk_crashes = (c.crash_rate * episode_span as f32) as u32;
                md.push_str(&format!(
                    "| {} | ~{} | {:.0} | {:.0}% | {:.0}% | {:.0}% |\n",
                    c.chunk_index + 1, chunk_crashes, c.avg_crash_speed,
                    c.slide_crash_fraction * 100.0,
                    c.overshoot_crash_fraction * 100.0,
                    c.headon_crash_fraction * 100.0,
                ));
            }
            md.push('\n');
        }

        let dominant_type = type_list.first().map(|(t, _)| **t).unwrap_or("Unknown");
        let dominant_pct = type_list.first().map(|(_, c)| **c as f32 / total_crashes as f32 * 100.0).unwrap_or(0.0);

        md.push_str(&format!(
            "> **Takeaway:** {} crashes total. **{}** crashes dominate at {:.0}%. {}.\n\n",
            total_crashes, dominant_type, dominant_pct,
            match dominant_type {
                "Slide" => "Cars are sliding sideways into walls — momentum management is the issue",
                "Overshoot" => "Cars are missing corners — they need to brake earlier or steer more",
                "HeadOn" => "Cars are driving straight into walls — basic navigation is failing",
                "Spin" => "Cars are spinning out — steering is too aggressive",
                "Stall" => "Cars are crashing at very low speed — they may be stuck",
                _ => "",
            }
        ));
    } else {
        md.push_str("No crashes recorded (all episodes ended by timeout).\n\n");
        md.push_str("> **Takeaway:** No crashes — cars may be too passive.\n\n");
    }

    // ════════════════════════════════════════════════════════════════════
    // 6. What Does the Car Think?
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 6. What Does the Car Think?\n\n");

    let value_vals: Vec<f32> = tracker.episodes.iter()
        .filter_map(|e| e.mean_value_prediction)
        .collect();

    if !value_vals.is_empty() {
        let value_smooth = rolling_mean(&value_vals, 20);
        md.push_str(&format!("**Mean value prediction:** `{}` trend: {}\n\n", sparkline(&value_smooth, SPARKLINE_WIDTH), trend_word(&value_smooth)));

        let avg_val = mean(&value_vals);
        let crash_vals: Vec<f32> = crash_episodes.iter().filter_map(|e| e.value_at_crash).collect();
        let start_vals: Vec<f32> = tracker.episodes.iter().filter_map(|e| e.value_at_start).collect();

        md.push_str("| Situation | Mean value | Count |\n");
        md.push_str("|-----------|----------:|------:|\n");
        md.push_str(&format!("| Overall average | {:.3} | {} |\n", avg_val, value_vals.len()));
        if !start_vals.is_empty() {
            md.push_str(&format!("| Episode start | {:.3} | {} |\n", mean(&start_vals), start_vals.len()));
        }
        if !crash_vals.is_empty() {
            md.push_str(&format!("| At crash moment | {:.3} | {} |\n", mean(&crash_vals), crash_vals.len()));
        }
        md.push('\n');

        let crash_val_mean = if crash_vals.is_empty() { 0.0 } else { mean(&crash_vals) };
        let critic_sees_danger = crash_val_mean < avg_val * 0.5;

        md.push_str(&format!(
            "> **Takeaway:** Critic predicts {:.3} on average, {:.3} at crash moments. {}.\n\n",
            avg_val, crash_val_mean,
            if critic_sees_danger { "The critic IS distinguishing dangerous states (lower value at crashes)" }
            else { "The critic is NOT predicting crashes — dangerous and safe states look similar to it" }
        ));
    } else {
        md.push_str("No value predictions recorded (AI mode may not have been active).\n\n");
    }

    // Policy confidence.
    let steer_std_vals: Vec<f32> = tracker.episodes.iter()
        .filter_map(|e| e.mean_policy_steering_std)
        .collect();
    if !steer_std_vals.is_empty() {
        let steer_std_smooth = rolling_mean(&steer_std_vals, 20);
        let throttle_std_vals: Vec<f32> = tracker.episodes.iter()
            .filter_map(|e| e.mean_policy_throttle_std)
            .collect();
        let throttle_std_smooth = rolling_mean(&throttle_std_vals, 20);

        md.push_str(&format!("**Steering confidence (lower σ = more confident):** `{}`\n\n", sparkline(&steer_std_smooth, SPARKLINE_WIDTH)));
        md.push_str(&format!("**Throttle confidence:** `{}`\n\n", sparkline(&throttle_std_smooth, SPARKLINE_WIDTH)));
    }

    // ════════════════════════════════════════════════════════════════════
    // 7. Track Coverage and Exploration
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 7. Track Coverage and Exploration\n\n");

    let mut sector_reach_counts = vec![0u32; NUM_PROGRESS_SECTORS];
    for ep in &tracker.episodes {
        for s in 0..=(ep.furthest_sector as usize).min(NUM_PROGRESS_SECTORS - 1) {
            sector_reach_counts[s] += 1;
        }
    }
    let ep_count = tracker.episodes.len() as f32;

    md.push_str("**Sector reach frequency:**\n\n```text\n");
    for (i, &count) in sector_reach_counts.iter().enumerate() {
        let pct = count as f32 / ep_count * 100.0;
        if pct > 0.5 {
            md.push_str(&format!("S{:02} {} {:.0}%\n", i + 1, ascii_bar(pct, 100.0, BAR_WIDTH), pct));
        }
    }
    md.push_str("```\n\n");

    let sector_50_pct = sector_reach_counts.iter().filter(|&&c| c as f32 / ep_count > 0.5).count();
    let sector_any = sector_reach_counts.iter().filter(|&&c| c > 0).count();

    md.push_str(&format!(
        "> **Takeaway:** {} of {} sectors reached by >50% of episodes. {} sectors ever visited. {}.\n\n",
        sector_50_pct, NUM_PROGRESS_SECTORS, sector_any,
        if sector_50_pct <= 2 { "The car is stuck very early — first corner is the bottleneck" }
        else if sector_50_pct <= 5 { "Moderate exploration — the car gets past early sections but struggles further" }
        else { "Good track coverage — the car is exploring most of the circuit" }
    ));

    // ════════════════════════════════════════════════════════════════════
    // 8. Driving Quality
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 8. Driving Quality\n\n");

    if !chunks.is_empty() {
        md.push_str("| Chunk | CL Distance | Wall Prox % | Smoothness | Reward/s | Drift° |\n");
        md.push_str("|------:|------------:|------------:|-----------:|---------:|-------:|\n");
        for c in &chunks {
            let cl_dist = {
                let chunk_eps = &tracker.episodes[..];
                let start = (c.start_episode as usize).min(chunk_eps.len());
                let end = (c.end_episode as usize + 1).min(chunk_eps.len());
                if end > start {
                    chunk_eps[start..end].iter().map(|e| e.mean_centerline_distance).sum::<f32>() / (end - start) as f32
                } else { 0.0 }
            };
            md.push_str(&format!(
                "| {} | {:.1} | {:.0}% | {:.3} | {:.2} | {:.1}° |\n",
                c.chunk_index + 1, cl_dist,
                c.avg_wall_proximity_fraction * 100.0,
                c.avg_action_change, c.avg_reward_per_second, c.avg_drift_angle_deg,
            ));
        }
        md.push('\n');
    }

    // Has it found a route?
    md.push_str(&format!("**Route consistency score:** {:.3}\n\n", consistency_score));

    if !sector_rows.is_empty() {
        let max_speed = sector_rows.iter().map(|r| r.speed_mean).fold(0.0_f32, f32::max);
        md.push_str("**Speed profile by sector:**\n\n```text\n");
        for row in &sector_rows {
            md.push_str(&format!("S{:02} | {}\n", row.sector_index + 1, ascii_bar(row.speed_mean, max_speed, BAR_WIDTH)));
        }
        md.push_str("```\n\n");
    }

    let avg_wall_prox = mean(&tracker.episodes.iter().map(|e| e.wall_proximity_fraction * 100.0).collect::<Vec<_>>());
    md.push_str(&format!(
        "> **Takeaway:** {:.0}% of time spent near walls. Route consistency {:.3}. {}.\n\n",
        avg_wall_prox, consistency_score,
        if consistency_score > 0.5 { "The car has found a repeatable route" }
        else if consistency_score > 0.2 { "The car is developing a route but it's inconsistent" }
        else { "No consistent route — the car drives differently each episode" }
    ));

    // ════════════════════════════════════════════════════════════════════
    // 9. Training Health
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 9. Training Health\n\n");

    if !upd_series.entropy.is_empty() {
        md.push_str(&format!("**Entropy:** `{}`\n\n", sparkline(&upd_series.entropy, SPARKLINE_WIDTH)));
        md.push_str(&format!("**Clip %:** `{}`\n\n", sparkline(&upd_series.clip_fraction, SPARKLINE_WIDTH)));
        md.push_str(&format!("**Approx KL:** `{}`\n\n", sparkline(&upd_series.approx_kl, SPARKLINE_WIDTH)));
        md.push_str(&format!("**Explained Var:** `{}`\n\n", sparkline(&upd_series.explained_variance, SPARKLINE_WIDTH)));
    }

    if let Some(latest) = tracker.ppo_updates.last() {
        md.push_str("**Latest update:**\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Policy entropy | {:.4} |\n", latest.policy_entropy));
        md.push_str(&format!("| Value loss | {:.4} |\n", latest.value_loss));
        md.push_str(&format!("| Explained variance | {:.4} |\n", latest.explained_variance));
        md.push_str(&format!("| Clip fraction | {:.2}% |\n", latest.clip_fraction * 100.0));
        md.push_str(&format!("| Approx KL | {:.5} |\n", latest.approx_kl));
        md.push_str(&format!("| Steering mean / std | {:.4} / {:.4} |\n", latest.steering_mean, latest.steering_std));
        md.push_str(&format!("| Throttle mean / std | {:.4} / {:.4} |\n", latest.throttle_mean, latest.throttle_std));
        md.push('\n');

        if !latest.layer_health.is_empty() {
            md.push_str("**Layer health:**\n\n");
            md.push_str("| Layer | Weight Norm | Grad Norm | Saturated % |\n");
            md.push_str("|-------|------------:|----------:|------------:|\n");
            for layer in &latest.layer_health {
                let dead = layer.saturated_fraction.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "N/A".to_string());
                md.push_str(&format!("| {} | {:.4} | {:.4} | {} |\n", layer.layer_name, layer.weight_l2_norm, layer.gradient_l2_norm, dead));
            }
            md.push('\n');
        }
    }

    // Reward decomposition.
    if !chunks.is_empty() {
        md.push_str("**Reward decomposition by chunk:**\n\n");
        md.push_str("| Chunk | Progress R | Time Pen | Crash Pen |\n");
        md.push_str("|------:|-----------:|---------:|----------:|\n");
        for c in &chunks {
            md.push_str(&format!("| {} | {:.3} | {:.3} | {:.3} |\n", c.chunk_index + 1, c.avg_progress_reward, c.avg_time_penalty, c.avg_crash_penalty));
        }
        md.push('\n');
    }

    // ════════════════════════════════════════════════════════════════════
    // 10. Trajectory Snapshots
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 10. Trajectory Snapshots\n\n");

    if trajectory_rows.is_empty() {
        md.push_str("No per-tick traces were captured.\n");
    } else {
        md.push_str("| Selection | Episode | End | Progress | Ticks | Mean Speed | Peak Speed |\n");
        md.push_str("|-----------|--------:|-----|--------:|------:|-----------:|-----------:|\n");
        for row in &trajectory_rows {
            md.push_str(&format!(
                "| {} | {} | {} | {:.1}% | {} | {:.1} | {:.1} |\n",
                row.selection, row.episode_id, row.end_reason,
                row.best_progress * 100.0, row.ticks, row.mean_speed, row.peak_speed,
            ));
        }
    }

    let _ = fs::write(filepath, md);
}

// ── Helpers ────────────────────────────────────────────────────────────

fn trend_word(smoothed: &[f32]) -> &'static str {
    if smoothed.len() < 4 {
        return "insufficient data";
    }
    let n = smoothed.len();
    let recent = mean(&smoothed[n.saturating_sub(10)..].to_vec());
    let early = mean(&smoothed[..10.min(n)].to_vec());
    let delta = recent - early;
    if delta > 3.0 { "rising" } else if delta < -3.0 { "falling" } else { "flat" }
}

fn crash_counts_by_sector(episodes: &[EpisodeRecord]) -> Vec<usize> {
    let mut counts = vec![0usize; NUM_PROGRESS_SECTORS];
    for ep in episodes {
        if ep.end_reason.contains("Crash") {
            let sector = (ep.progress * NUM_PROGRESS_SECTORS as f32).floor() as usize;
            let sector = sector.min(NUM_PROGRESS_SECTORS - 1);
            counts[sector] += 1;
        }
    }
    counts
}
