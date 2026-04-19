use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analytics::metrics::chunking::{DEFAULT_CHUNK_COUNT, calculate_chunks};
use crate::analytics::metrics::consistency::{
    compute_sector_consistency, overall_consistency_score,
};
use crate::analytics::metrics::diagnostics::compute_diagnostic_flags;
use crate::analytics::metrics::phases::detect_learning_phase;
use crate::analytics::metrics::pre_crash::{
    PRE_CRASH_WINDOW, collect_pre_crash_profiles, summarise,
};
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
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for ep in &crash_episodes {
            let ct = ep.crash_type.map(|k| k.to_string()).unwrap_or_else(|| "Unknown".to_string());
            *type_counts.entry(ct).or_insert(0) += 1;
        }
        let mut type_list: Vec<(&String, &usize)> = type_counts.iter().collect();
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

        let dominant_type = type_list.first().map(|(t, _)| t.as_str()).unwrap_or("Unknown");
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
    // Section 9 is PPO-centric: entropy / clip / KL / explained-variance /
    // layer-health timeseries only make sense for a backprop-trained policy.
    // Skip the entire section for brain-only runs.
    if !tracker.ppo_updates.is_empty() {
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

        // Reward decomposition (universal, but kept under section 9's umbrella
        // since this is where readers look for chunk-level breakdowns).
        if !chunks.is_empty() {
            md.push_str("**Reward decomposition by chunk:**\n\n");
            md.push_str("| Chunk | Progress R | Time Pen | Crash Pen |\n");
            md.push_str("|------:|-----------:|---------:|----------:|\n");
            for c in &chunks {
                md.push_str(&format!("| {} | {:.3} | {:.3} | {:.3} |\n", c.chunk_index + 1, c.avg_progress_reward, c.avg_time_penalty, c.avg_crash_penalty));
            }
            md.push('\n');
        }
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

    // ════════════════════════════════════════════════════════════════════
    // 11. Pre-Crash Forensics (round-2 diagnostic)
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 11. Pre-Crash Forensics\n\n");
    md.push_str(&format!(
        "Analyses the last {} ticks ({:.2}s at 60Hz) before each crash to \
         distinguish **anticipation failures** (policy unaware) from \
         **reaction failures** (policy knew but couldn't respond in time).\n\n",
        PRE_CRASH_WINDOW,
        PRE_CRASH_WINDOW as f32 / 60.0,
    ));

    let pre_crash_profiles = collect_pre_crash_profiles(&tracker.episode_traces);
    if pre_crash_profiles.is_empty() {
        md.push_str("No crash traces available (episode_traces may be empty for this run).\n\n");
    } else {
        let summary = summarise(&pre_crash_profiles);
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Crashes analysed | {} |\n", summary.crash_count));
        md.push_str(&format!(
            "| Mean throttle delta (late − early half) | {:+.3} |\n",
            summary.mean_throttle_delta,
        ));
        md.push_str(&format!(
            "| Mean distance-to-wall at crash | {:.1} units |\n",
            summary.mean_distance_to_wall,
        ));
        md.push_str(&format!(
            "| Median distance-to-wall at crash | {:.1} units |\n",
            summary.median_distance_to_wall,
        ));
        md.push_str(&format!(
            "| Throttle released at least once in window | {:.0}% of crashes |\n",
            summary.release_any_fraction * 100.0,
        ));
        md.push_str(&format!(
            "| Throttle released > 0.25s before crash | {:.0}% of crashes |\n",
            summary.release_early_fraction * 100.0,
        ));
        if let Some(drop) = summary.mean_value_drop {
            md.push_str(&format!(
                "| Mean critic-value drop (window → crash) | {:+.2} |\n",
                drop,
            ));
        }
        md.push('\n');

        md.push_str("**Throttle-release latency histogram** (ticks-before-crash):\n\n");
        md.push_str("```text\n");
        let labels = [" 0- 5", " 5-10", "10-15", "15-20", "20-25", "25-30"];
        let release_max = summary.release_histogram.iter().max().copied().unwrap_or(1) as f32;
        for (label, &count) in labels.iter().zip(summary.release_histogram.iter()) {
            md.push_str(&format!(
                "{label}  {} {}\n",
                ascii_bar(count as f32, release_max, BAR_WIDTH),
                count,
            ));
        }
        md.push_str("```\n\n");

        md.push_str("**Distance-to-wall at crash histogram** (units):\n\n");
        md.push_str("```text\n");
        let dist_labels = [" 0- 5", " 5-10", "10-20", "20-40", "40-80", " 80+ "];
        let dist_max = summary.distance_histogram.iter().max().copied().unwrap_or(1) as f32;
        for (label, &count) in dist_labels.iter().zip(summary.distance_histogram.iter()) {
            md.push_str(&format!(
                "{label}  {} {}\n",
                ascii_bar(count as f32, dist_max, BAR_WIDTH),
                count,
            ));
        }
        md.push_str("```\n\n");

        let anticipatory = summary.release_early_fraction > 0.3
            && summary.mean_throttle_delta < -0.1
            && summary.mean_value_drop.map(|d| d > 0.0).unwrap_or(false);
        let reactive = summary.release_any_fraction < 0.2 && summary.mean_throttle_delta > -0.05;
        md.push_str(&format!(
            "> **Takeaway:** {}\n\n",
            if anticipatory {
                "Anticipation is emerging — policy releases throttle early and critic values drop before crashes."
            } else if reactive {
                "Purely reactive — policy doesn't release throttle, critic doesn't anticipate."
            } else {
                "Mixed picture — partial anticipation but not consistent."
            }
        ));
    }

    // ════════════════════════════════════════════════════════════════════
    // 12. Layer Health Over Training (round-2 diagnostic — PPO-only)
    // ════════════════════════════════════════════════════════════════════
    if !tracker.ppo_updates.is_empty() {
        md.push_str("## 12. Layer Health Over Training\n\n");
        // Collect per-layer timeseries: layer name → (sat %, weight L2, grad L2) per update.
        let mut layer_names: Vec<String> = tracker.ppo_updates.first()
            .map(|u| u.layer_health.iter().map(|l| l.layer_name.clone()).collect())
            .unwrap_or_default();
        layer_names.sort();
        layer_names.dedup();

        md.push_str("**Saturation % over training** (activation layers only; `.` = no data):\n\n");
        md.push_str("```text\n");
        for layer_name in &layer_names {
            let series: Vec<f32> = tracker.ppo_updates.iter()
                .filter_map(|u| u.layer_health.iter()
                    .find(|l| &l.layer_name == layer_name)
                    .and_then(|l| l.saturated_fraction)
                    .map(|f| f * 100.0))
                .collect();
            if !series.is_empty() {
                md.push_str(&format!(
                    "{:<14}  {}  latest {:.1}%\n",
                    layer_name,
                    sparkline(&series, SPARKLINE_WIDTH),
                    series.last().copied().unwrap_or(0.0),
                ));
            }
        }
        md.push_str("```\n\n");

        md.push_str("**Weight L2 norms over training:**\n\n");
        md.push_str("```text\n");
        for layer_name in &layer_names {
            let series: Vec<f32> = tracker.ppo_updates.iter()
                .filter_map(|u| u.layer_health.iter()
                    .find(|l| &l.layer_name == layer_name)
                    .map(|l| l.weight_l2_norm))
                .collect();
            if !series.is_empty() {
                md.push_str(&format!(
                    "{:<14}  {}  latest {:.2}\n",
                    layer_name,
                    sparkline(&series, SPARKLINE_WIDTH),
                    series.last().copied().unwrap_or(0.0),
                ));
            }
        }
        md.push_str("```\n\n");

        md.push_str("**Gradient L2 norms over training:**\n\n");
        md.push_str("```text\n");
        for layer_name in &layer_names {
            let series: Vec<f32> = tracker.ppo_updates.iter()
                .filter_map(|u| u.layer_health.iter()
                    .find(|l| &l.layer_name == layer_name)
                    .map(|l| l.gradient_l2_norm))
                .collect();
            if !series.is_empty() {
                md.push_str(&format!(
                    "{:<14}  {}  latest {:.4}\n",
                    layer_name,
                    sparkline(&series, SPARKLINE_WIDTH),
                    series.last().copied().unwrap_or(0.0),
                ));
            }
        }
        md.push_str("```\n\n");

        // Auto-generated takeaway focused on c_fc2 (the round-1 bottleneck).
        let c_fc2_series: Vec<f32> = tracker.ppo_updates.iter()
            .filter_map(|u| u.layer_health.iter()
                .find(|l| l.layer_name == "critic_fc2")
                .and_then(|l| l.saturated_fraction))
            .collect();
        if !c_fc2_series.is_empty() {
            let latest_c_fc2 = c_fc2_series.last().copied().unwrap_or(0.0);
            let trend_c_fc2 = {
                let n = c_fc2_series.len();
                if n < 4 { "stable" }
                else {
                    let early = mean(&c_fc2_series[..(n/3).max(1)].to_vec());
                    let late = mean(&c_fc2_series[(2*n/3)..].to_vec());
                    if late > early + 0.05 { "rising" }
                    else if late + 0.05 < early { "falling" }
                    else { "stable" }
                }
            };
            md.push_str(&format!(
                "> **Takeaway:** `critic_fc2` saturation latest {:.0}% (trend: {}). {}\n\n",
                latest_c_fc2 * 100.0,
                trend_c_fc2,
                if latest_c_fc2 > 0.5 {
                    "Critic hidden layer is saturated — target scale is likely the bottleneck."
                } else if latest_c_fc2 > 0.3 {
                    "Moderate saturation — worth watching."
                } else {
                    "Saturation within healthy range."
                },
            ));
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 13. Value Target Scale Tracker (PopArt, round-2 diagnostic — PPO-only)
    // ════════════════════════════════════════════════════════════════════
    if !tracker.ppo_updates.is_empty() {
        md.push_str("## 13. Value Target Scale Tracker\n\n");
        let return_means: Vec<f32> = tracker.ppo_updates.iter().map(|u| u.return_mean).collect();
        let return_stds: Vec<f32> = tracker.ppo_updates.iter().map(|u| u.return_std).collect();
        let return_maxes: Vec<f32> = tracker.ppo_updates.iter().map(|u| u.return_max).collect();
        let popart_mu_series: Vec<f32> = tracker.ppo_updates.iter().map(|u| u.value_norm_mu).collect();
        let popart_sigma_series: Vec<f32> = tracker.ppo_updates.iter().map(|u| u.value_norm_sigma).collect();

        md.push_str(&format!("**Return mean over updates:** `{}` latest {:.2}\n\n",
            sparkline(&return_means, SPARKLINE_WIDTH),
            return_means.last().copied().unwrap_or(0.0),
        ));
        md.push_str(&format!("**Return std over updates:** `{}` latest {:.2}\n\n",
            sparkline(&return_stds, SPARKLINE_WIDTH),
            return_stds.last().copied().unwrap_or(0.0),
        ));
        md.push_str(&format!("**Return max over updates:** `{}` latest {:.2}\n\n",
            sparkline(&return_maxes, SPARKLINE_WIDTH),
            return_maxes.last().copied().unwrap_or(0.0),
        ));

        let popart_active = popart_sigma_series.iter().any(|&s| (s - 1.0).abs() > 1e-4)
            || popart_mu_series.iter().any(|&m| m.abs() > 1e-4);
        if popart_active {
            md.push_str(&format!("**PopArt µ:** `{}` latest {:.2}\n\n",
                sparkline(&popart_mu_series, SPARKLINE_WIDTH),
                popart_mu_series.last().copied().unwrap_or(0.0),
            ));
            md.push_str(&format!("**PopArt σ:** `{}` latest {:.2}\n\n",
                sparkline(&popart_sigma_series, SPARKLINE_WIDTH),
                popart_sigma_series.last().copied().unwrap_or(0.0),
            ));

            // Tracking check: is PopArt's µ close to the current return mean?
            let latest_mu = popart_mu_series.last().copied().unwrap_or(0.0);
            let latest_return_mean = return_means.last().copied().unwrap_or(0.0);
            let mu_tracking_error = (latest_mu - latest_return_mean).abs();
            let tracking_ok = mu_tracking_error < return_stds.last().copied().unwrap_or(1.0) * 0.5;
            md.push_str(&format!(
                "> **Takeaway:** PopArt active. µ tracking error = {:.2} ({}).\n\n",
                mu_tracking_error,
                if tracking_ok { "within 0.5σ — healthy" } else { "> 0.5σ — β may be too low" },
            ));
        } else {
            let earliest_mean = return_means.first().copied().unwrap_or(0.0);
            let latest_mean = return_means.last().copied().unwrap_or(0.0);
            let growth = if earliest_mean.abs() > 1e-3 { latest_mean / earliest_mean } else { 0.0 };
            md.push_str(&format!(
                "> **Takeaway:** PopArt disabled (µ=0, σ=1 throughout). Returns grew from {:.1} to {:.1} ({:.1}× over training). Without PopArt, the critic must absorb this scale change via weight growth.\n\n",
                earliest_mean, latest_mean, growth,
            ));
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 14. Critic Prediction Quality (round-2 diagnostic — PPO-only)
    // ════════════════════════════════════════════════════════════════════
    if !tracker.ppo_updates.is_empty() && !tracker.episodes.is_empty() {
        md.push_str("## 14. Critic Prediction Quality\n\n");
        // Explained variance timeseries (already captured per update).
        let ev_series: Vec<f32> = tracker.ppo_updates.iter().map(|u| u.explained_variance).collect();
        md.push_str(&format!(
            "**Explained variance over updates:** `{}` latest {:.3}\n\n",
            sparkline(&ev_series, SPARKLINE_WIDTH),
            ev_series.last().copied().unwrap_or(0.0),
        ));

        // Standardised residual histogram from trace data — value_prediction
        // vs eventual return (approximated via the episode's total reward for
        // traces that reach the end of their episode).
        let mut residuals: Vec<f32> = Vec::new();
        for trace in &tracker.episode_traces {
            let total_return: f32 = trace.ticks.iter().map(|t| t.reward).sum();
            if let Some(first) = trace.ticks.first() {
                if let Some(predicted) = first.value_prediction {
                    // Predicted is for "from here on" → compare to full total return.
                    residuals.push(predicted - total_return);
                }
            }
        }
        if !residuals.is_empty() {
            let mean_resid = residuals.iter().sum::<f32>() / residuals.len() as f32;
            let var_resid = residuals.iter()
                .map(|r| (r - mean_resid).powi(2))
                .sum::<f32>() / residuals.len() as f32;
            let std_resid = var_resid.max(1e-6).sqrt();
            let mut buckets = [0u32; 7];
            for r in &residuals {
                let z = (r - mean_resid) / std_resid;
                let b = if z < -2.0 { 0 }
                    else if z < -1.0 { 1 }
                    else if z < -0.5 { 2 }
                    else if z < 0.5 { 3 }
                    else if z < 1.0 { 4 }
                    else if z < 2.0 { 5 }
                    else { 6 };
                buckets[b] = buckets[b].saturating_add(1);
            }
            let labels = [" < -2σ ", "-2 — -1", "-1 — -½", "-½ — +½", "+½ — +1 ", "+1 — +2 ", "  > +2σ"];
            let max_count = buckets.iter().max().copied().unwrap_or(1) as f32;

            md.push_str("**Standardised prediction residuals** (episode-start value vs actual return):\n\n");
            md.push_str("```text\n");
            for (label, &count) in labels.iter().zip(buckets.iter()) {
                md.push_str(&format!(
                    "{label}  {} {}\n",
                    ascii_bar(count as f32, max_count, BAR_WIDTH),
                    count,
                ));
            }
            md.push_str(&format!(
                "\nresidual μ = {:+.2}    σ = {:.2}    n = {}\n",
                mean_resid, std_resid, residuals.len(),
            ));
            md.push_str("```\n\n");

            let calibration_ok = mean_resid.abs() < 0.2 * std_resid;
            md.push_str(&format!(
                "> **Takeaway:** {}\n\n",
                if calibration_ok {
                    "Critic is well-calibrated — residual mean near zero."
                } else if mean_resid > 0.0 {
                    "Critic is biased HIGH — predicts higher values than actual returns deliver."
                } else {
                    "Critic is biased LOW — predicts lower values than actual returns deliver."
                },
            ));
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 15. Fleet Variance (round-2 diagnostic)
    // ════════════════════════════════════════════════════════════════════
    md.push_str("## 15. Fleet Variance\n\n");
    if tracker.episodes.is_empty() {
        md.push_str("No episodes recorded.\n\n");
    } else {
        // Aggregate per-car.
        use std::collections::HashMap;
        struct FleetStats {
            episodes: u32,
            max_progress: f32,
            mean_life_s: f32,
            mean_reward: f32,
            crash_count: u32,
        }
        let mut fleet: HashMap<u32, FleetStats> = HashMap::new();
        for ep in &tracker.episodes {
            let entry = fleet.entry(ep.env_id).or_insert(FleetStats {
                episodes: 0,
                max_progress: 0.0,
                mean_life_s: 0.0,
                mean_reward: 0.0,
                crash_count: 0,
            });
            entry.episodes = entry.episodes.saturating_add(1);
            if ep.progress > entry.max_progress {
                entry.max_progress = ep.progress;
            }
            entry.mean_life_s += ep.ticks as f32 / 60.0;
            entry.mean_reward += ep.reward;
            if ep.end_reason.contains("Crash") {
                entry.crash_count = entry.crash_count.saturating_add(1);
            }
        }
        // Finalise means.
        for entry in fleet.values_mut() {
            let n = entry.episodes as f32;
            if n > 0.0 {
                entry.mean_life_s /= n;
                entry.mean_reward /= n;
            }
        }
        let mut fleet_vec: Vec<(u32, FleetStats)> = fleet.into_iter().collect();
        fleet_vec.sort_by_key(|(env_id, _)| *env_id);

        md.push_str("| Car | Episodes | Max Progress | Mean Life (s) | Mean Reward | Crashes | Crash % |\n");
        md.push_str("|----:|---------:|-------------:|--------------:|------------:|--------:|--------:|\n");
        for (env_id, stats) in &fleet_vec {
            let crash_pct = if stats.episodes > 0 {
                stats.crash_count as f32 / stats.episodes as f32 * 100.0
            } else { 0.0 };
            md.push_str(&format!(
                "| {} | {} | {:.1}% | {:.2} | {:.2} | {} | {:.0}% |\n",
                env_id, stats.episodes, stats.max_progress * 100.0,
                stats.mean_life_s, stats.mean_reward, stats.crash_count, crash_pct,
            ));
        }
        md.push('\n');

        // Convergence check — is the fleet clustered on similar max-progress?
        let progresses: Vec<f32> = fleet_vec.iter().map(|(_, s)| s.max_progress).collect();
        let cars_over_50 = progresses.iter().filter(|&&p| p > 0.5).count();
        let cars_over_100 = progresses.iter().filter(|&&p| p >= 1.0).count();
        let mean_max_progress = if progresses.is_empty() { 0.0 }
            else { progresses.iter().sum::<f32>() / progresses.len() as f32 };
        let progress_spread = if progresses.len() >= 2 {
            let mut sorted = progresses.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted.last().copied().unwrap_or(0.0) - sorted.first().copied().unwrap_or(0.0)
        } else { 0.0 };

        md.push_str(&format!(
            "> **Takeaway:** {} of {} cars have reached >50% progress; {} reached full loop. Mean best progress = {:.1}%, fleet spread = {:.1}%. {}\n\n",
            cars_over_50, fleet_vec.len(), cars_over_100,
            mean_max_progress * 100.0,
            progress_spread * 100.0,
            if cars_over_50 == fleet_vec.len() {
                "Fleet has fully converged."
            } else if cars_over_50 >= fleet_vec.len() * 3 / 4 {
                "Fleet is mostly converging — stragglers remain."
            } else if cars_over_50 <= 1 {
                "Only one or no cars lead — the 'lucky car' pattern. Aggregate metrics are dominated by a small fraction of the fleet."
            } else {
                "Fleet is partially converging."
            },
        ));
    }

    // ════════════════════════════════════════════════════════════════════
    // 16–18. Brain-inspired learner diagnostics (populated when any brain
    // cars ran; otherwise these sections are skipped entirely).
    // ════════════════════════════════════════════════════════════════════
    if !tracker.brain_records.is_empty() {
        append_brain_sections(&mut md, tracker);
    }

    // ════════════════════════════════════════════════════════════════════
    // 19. Fleet Comparison — only when this run had both PPO and brain cars
    // (detected by scanning EpisodeRecord.controller values).
    // ════════════════════════════════════════════════════════════════════
    let has_ppo_eps = tracker.episodes.iter().any(|e| e.controller == "Ppo");
    let has_brain_eps = tracker.episodes.iter().any(|e| e.controller == "Brain");
    if has_ppo_eps && has_brain_eps {
        append_fleet_comparison(&mut md, tracker);
    }

    let _ = fs::write(filepath, md);
}

/// Appends sections 16 (structure), 17 (plasticity health), 18 (structural
/// events) for the brain-inspired learner. Only called when
/// `tracker.brain_records` has at least one entry.
fn append_brain_sections(md: &mut String, tracker: &EpisodeTracker) {
    // ─── Section 16: Brain Structure Over Time ──────────────────────────
    md.push_str("## 16. Brain Structure Over Time\n\n");
    let neuron_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.neuron_count as f32)
        .collect();
    let hidden_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.hidden_count as f32)
        .collect();
    let synapse_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.synapse_count as f32)
        .collect();
    md.push_str(&format!(
        "- Neurons: {} → {} ({} records)\n",
        neuron_series.first().copied().unwrap_or(0.0) as u32,
        neuron_series.last().copied().unwrap_or(0.0) as u32,
        tracker.brain_records.len()
    ));
    md.push_str(&format!(
        "- Hidden neurons: {} → {}\n",
        hidden_series.first().copied().unwrap_or(0.0) as u32,
        hidden_series.last().copied().unwrap_or(0.0) as u32
    ));
    md.push_str(&format!(
        "- Synapses: {} → {}\n\n",
        synapse_series.first().copied().unwrap_or(0.0) as u32,
        synapse_series.last().copied().unwrap_or(0.0) as u32
    ));
    md.push_str("Neuron count trajectory:\n```\n");
    md.push_str(&sparkline(&neuron_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");
    md.push_str("Synapse count trajectory:\n```\n");
    md.push_str(&sparkline(&synapse_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");

    // ─── Section 17: Plasticity Health ──────────────────────────────────
    md.push_str("## 17. Plasticity Health\n\n");
    let w_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.mean_abs_weight)
        .collect();
    let e_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.mean_abs_eligibility)
        .collect();
    let w_sigma_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.weight_sigma)
        .collect();
    let dead_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.dead_neuron_fraction * 100.0)
        .collect();
    let sat_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.saturation_fraction * 100.0)
        .collect();
    let m_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.mean_m)
        .collect();

    md.push_str(&format!(
        "- Mean |w| (final): {:.4}\n",
        w_series.last().copied().unwrap_or(0.0)
    ));
    md.push_str(&format!(
        "- Weight σ (final): {:.4}\n",
        w_sigma_series.last().copied().unwrap_or(0.0)
    ));
    md.push_str(&format!(
        "- Mean |eligibility| (final): {:.4}\n",
        e_series.last().copied().unwrap_or(0.0)
    ));
    md.push_str(&format!(
        "- Dead-neuron fraction (final): {:.1}%\n",
        dead_series.last().copied().unwrap_or(0.0)
    ));
    md.push_str(&format!(
        "- Saturation fraction (final): {:.1}%\n",
        sat_series.last().copied().unwrap_or(0.0)
    ));
    md.push_str(&format!(
        "- Mean modulator M (final): {:.4}\n\n",
        m_series.last().copied().unwrap_or(0.0)
    ));

    md.push_str("Mean |w| over cadence windows:\n```\n");
    md.push_str(&sparkline(&w_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");
    md.push_str("Mean |eligibility| over cadence windows:\n```\n");
    md.push_str(&sparkline(&e_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");
    md.push_str("Modulator M (per-car reward mean) over cadence windows:\n```\n");
    md.push_str(&sparkline(&m_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");

    // ─── Section 18: Structural Events ─────────────────────────────────
    md.push_str("## 18. Structural Events\n\n");
    let mut total_replace = 0u64;
    let mut total_neurogen = 0u64;
    let mut total_prune = 0u64;
    let mut total_sprout = 0u64;
    for r in &tracker.brain_records {
        total_replace += r.replacement_count as u64;
        total_neurogen += r.neurogenesis_count as u64;
        total_prune += r.prune_count as u64;
        total_sprout += r.sprout_count as u64;
    }
    md.push_str("| Event | Total |\n");
    md.push_str("|---|---|\n");
    md.push_str(&format!("| Neurons replaced (CBP) | {} |\n", total_replace));
    md.push_str(&format!(
        "| Neurogenesis (plateau-triggered) | {} |\n",
        total_neurogen
    ));
    md.push_str(&format!("| Synapses pruned | {} |\n", total_prune));
    md.push_str(&format!("| Synapses sprouted | {} |\n\n", total_sprout));

    let replace_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.replacement_count as f32)
        .collect();
    md.push_str("Replacements per cadence window:\n```\n");
    md.push_str(&sparkline(&replace_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");

    let prune_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.prune_count as f32)
        .collect();
    let sprout_series: Vec<f32> = tracker
        .brain_records
        .iter()
        .map(|r| r.sprout_count as f32)
        .collect();
    md.push_str("Prune count per cadence window:\n```\n");
    md.push_str(&sparkline(&prune_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");
    md.push_str("Sprout count per cadence window:\n```\n");
    md.push_str(&sparkline(&sprout_series, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");

    // Takeaways.
    md.push_str("### Takeaways\n\n");
    if total_replace == 0 {
        md.push_str(
            "- No neurons were replaced this run. Either the maturity gate \
             (config.maturity_ticks) is suppressing replacement, or utility \
             was above threshold everywhere.\n",
        );
    } else {
        md.push_str(&format!(
            "- {} neurons were replaced via continual-backprop utility \
             tracking. Low-utility neurons getting recycled is the project \
             working as designed.\n",
            total_replace
        ));
    }
    if total_neurogen == 0 {
        md.push_str(
            "- No plateau-triggered neurogenesis fired. Either learning was \
             not plateauing or the reward window had not filled.\n",
        );
    } else {
        md.push_str(&format!(
            "- Plateau detector triggered {} neurogenesis events, growing the \
             hidden layer as reward improvement stalled.\n",
            total_neurogen
        ));
    }
    md.push_str("\n");
}

/// Section 19 — head-to-head PPO vs brain-inspired fleet comparison.
/// Only emitted when the run had at least one of each controller kind in its
/// episodes. Segments episodes by `controller` and produces parallel sparklines,
/// means, loop-completion counts, and crash-rate deltas.
fn append_fleet_comparison(md: &mut String, tracker: &EpisodeTracker) {
    md.push_str("## 19. Fleet Comparison — PPO vs Brain-Inspired\n\n");
    md.push_str(
        "Side-by-side mode ran PPO (warm-palette cars) and the brain-inspired \
         learner (cool-palette cars) in the same simulation. Each fleet saw the \
         same track, same reward shaping, same observation contract — any \
         difference in learning trajectory is attributable to the controller.\n\n",
    );

    let ppo_episodes: Vec<&EpisodeRecord> = tracker
        .episodes
        .iter()
        .filter(|e| e.controller == "Ppo")
        .collect();
    let brain_episodes: Vec<&EpisodeRecord> = tracker
        .episodes
        .iter()
        .filter(|e| e.controller == "Brain")
        .collect();

    let ppo_rewards: Vec<f32> = ppo_episodes.iter().map(|e| e.reward).collect();
    let brain_rewards: Vec<f32> = brain_episodes.iter().map(|e| e.reward).collect();
    let ppo_progress: Vec<f32> = ppo_episodes.iter().map(|e| e.progress * 100.0).collect();
    let brain_progress: Vec<f32> =
        brain_episodes.iter().map(|e| e.progress * 100.0).collect();
    let ppo_crashes = ppo_episodes
        .iter()
        .filter(|e| e.end_reason == "Crash")
        .count();
    let brain_crashes = brain_episodes
        .iter()
        .filter(|e| e.end_reason == "Crash")
        .count();
    let ppo_loops = ppo_episodes.iter().filter(|e| e.progress >= 1.0).count();
    let brain_loops = brain_episodes.iter().filter(|e| e.progress >= 1.0).count();

    let ppo_mean_reward = mean(&ppo_rewards);
    let brain_mean_reward = mean(&brain_rewards);
    let ppo_mean_progress = mean(&ppo_progress);
    let brain_mean_progress = mean(&brain_progress);
    let ppo_crash_rate = if !ppo_episodes.is_empty() {
        100.0 * ppo_crashes as f32 / ppo_episodes.len() as f32
    } else {
        0.0
    };
    let brain_crash_rate = if !brain_episodes.is_empty() {
        100.0 * brain_crashes as f32 / brain_episodes.len() as f32
    } else {
        0.0
    };

    md.push_str("| Metric | PPO | Brain-Inspired |\n");
    md.push_str("|---|---|---|\n");
    md.push_str(&format!(
        "| Episodes | {} | {} |\n",
        ppo_episodes.len(),
        brain_episodes.len()
    ));
    md.push_str(&format!(
        "| Mean reward | {:.2} | {:.2} |\n",
        ppo_mean_reward, brain_mean_reward
    ));
    md.push_str(&format!(
        "| Mean progress (%) | {:.1} | {:.1} |\n",
        ppo_mean_progress, brain_mean_progress
    ));
    md.push_str(&format!(
        "| Loops completed | {} | {} |\n",
        ppo_loops, brain_loops
    ));
    md.push_str(&format!(
        "| Crash rate (%) | {:.1} | {:.1} |\n\n",
        ppo_crash_rate, brain_crash_rate
    ));

    let ppo_reward_smooth = rolling_mean(&ppo_rewards, 20);
    let brain_reward_smooth = rolling_mean(&brain_rewards, 20);
    md.push_str("PPO reward trajectory (rolling-20):\n```\n");
    md.push_str(&sparkline(&ppo_reward_smooth, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");
    md.push_str("Brain-inspired reward trajectory (rolling-20):\n```\n");
    md.push_str(&sparkline(&brain_reward_smooth, SPARKLINE_WIDTH));
    md.push_str("\n```\n\n");

    // Takeaway.
    md.push_str("### Verdict\n\n");
    let delta = brain_mean_reward - ppo_mean_reward;
    if delta.abs() < 0.05 * ppo_mean_reward.abs().max(1e-3) {
        md.push_str(
            "- Fleets are tracking each other closely (within ±5% on mean \
             reward). The brain-inspired learner is holding its own — if \
             progress rates are similar, this is strong evidence that pure \
             three-factor plasticity can match PPO on this task.\n",
        );
    } else if delta > 0.0 {
        md.push_str(&format!(
            "- Brain-inspired fleet is OUTPERFORMING PPO by {:.2} mean reward. \
             Unusual — double-check that PPO is actually training (check section 2 \
             of this report). If confirmed, this is a strong result for \
             biological plasticity.\n",
            delta
        ));
    } else {
        md.push_str(&format!(
            "- PPO fleet leads brain-inspired by {:.2} mean reward ({:.1}% \
             gap). Expected direction for v1 given the research synthesis; \
             what matters next is whether the brain-inspired trend is still \
             rising (sparkline above) or has flattened. A still-rising brain \
             trend with a gap is exactly the success bar defined in the M6 plan.\n",
            delta.abs(),
            100.0 * delta.abs() / ppo_mean_reward.abs().max(1e-3)
        ));
    }
    if brain_loops == 0 && ppo_loops > 0 {
        md.push_str(
            "- Brain-inspired fleet has completed zero loops while PPO has. \
             Expected for v1; the visible-learning success bar does not \
             require loop completion.\n",
        );
    }
    if brain_loops > 0 {
        md.push_str(&format!(
            "- Brain-inspired fleet completed {} loops — the biology-first \
             substrate has proven capable on this task.\n",
            brain_loops
        ));
    }
    md.push_str("\n");
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
