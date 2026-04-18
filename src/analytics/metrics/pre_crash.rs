//! Pre-crash forensics — reads the last ~30 ticks of each crash trace to
//! answer: was the crash an **anticipation failure** (policy unaware) or a
//! **reaction failure** (policy knew but couldn't respond in time)?
//!
//! This module is pure analysis over already-captured `TickTraceRecord`s. No
//! new tick-level data is stored; everything here is computed at export time.

use crate::analytics::models::{EpisodeTrace, TickTraceRecord};

/// Window size in ticks (60 Hz → 30 ticks = 0.5 s of pre-crash history).
pub const PRE_CRASH_WINDOW: usize = 30;

/// Anticipation signature for a single crash episode.
///
/// Not every field is consumed by the current Markdown exporter — some exist
/// for future round-3 analysis (per-trace deep-dive in the analytics TUI
/// plan). They are intentionally preserved so an extended exporter can read
/// them without re-running the pre-crash analysis pass.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct PreCrashProfile {
    /// Tick count in the window actually analysed (0 ≤ n ≤ PRE_CRASH_WINDOW).
    pub window_len: usize,
    /// Mean throttle in the early half of the window (further from crash).
    pub throttle_early: f32,
    /// Mean throttle in the late half of the window (closest to crash).
    pub throttle_late: f32,
    /// Throttle delta (late − early). Negative = policy released throttle as
    /// the wall approached.
    pub throttle_delta: f32,
    /// Ticks-before-crash at which throttle first dropped below 0.5.
    /// `None` if throttle stayed above 0.5 for the whole window.
    pub throttle_release_ticks: Option<u32>,
    /// Minimum ray distance at the crash tick — a proxy for "how close to a
    /// wall was the car when it actually crashed".
    pub distance_to_wall_at_crash: f32,
    /// Mean critic value-prediction across the window (denormalised).
    pub value_window_mean: Option<f32>,
    /// Critic value at the crash tick itself.
    pub value_at_crash: Option<f32>,
    /// Critic value drop from window mean to crash tick. Positive = value
    /// fell into the crash (critic saw it coming).
    pub value_drop: Option<f32>,
}

/// Collates every crash trace's pre-crash profile.
pub fn collect_pre_crash_profiles(traces: &[EpisodeTrace]) -> Vec<PreCrashProfile> {
    let mut out = Vec::new();
    for trace in traces {
        if !trace.end_reason.contains("Crash") {
            continue;
        }
        if let Some(profile) = analyse_trace(&trace.ticks) {
            out.push(profile);
        }
    }
    out
}

fn analyse_trace(ticks: &[TickTraceRecord]) -> Option<PreCrashProfile> {
    if ticks.is_empty() {
        return None;
    }
    let n = ticks.len();
    let window_start = n.saturating_sub(PRE_CRASH_WINDOW);
    let window = &ticks[window_start..n];
    let window_len = window.len();
    if window_len == 0 {
        return None;
    }
    let split = window_len / 2;
    let early = &window[..split];
    let late = &window[split..];

    let mean_throttle = |slice: &[TickTraceRecord]| -> f32 {
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().map(|t| t.throttle).sum::<f32>() / slice.len() as f32
    };
    let throttle_early = mean_throttle(early);
    let throttle_late = mean_throttle(late);
    let throttle_delta = throttle_late - throttle_early;

    // Find first tick-before-crash where throttle dropped below 0.5.
    let mut throttle_release_ticks = None;
    for (offset, t) in window.iter().enumerate().rev() {
        if t.throttle < 0.5 {
            let ticks_before_crash = (window_len - 1 - offset) as u32;
            throttle_release_ticks = Some(ticks_before_crash);
        }
    }
    // Reinterpret: we want the *earliest* tick (furthest before crash) where
    // the policy let off throttle. That's the *largest* `ticks_before_crash`.
    // The loop above captures all hits; keep the maximum.
    if let Some(_last) = throttle_release_ticks {
        let max_release = window
            .iter()
            .enumerate()
            .filter_map(|(offset, t)| {
                if t.throttle < 0.5 {
                    Some((window_len - 1 - offset) as u32)
                } else {
                    None
                }
            })
            .max();
        throttle_release_ticks = max_release;
    }

    let crash_tick = ticks.last()?;
    let distance_to_wall_at_crash = crash_tick.min_ray_distance;

    let value_values: Vec<f32> = window.iter().filter_map(|t| t.value_prediction).collect();
    let value_window_mean = if value_values.is_empty() {
        None
    } else {
        Some(value_values.iter().sum::<f32>() / value_values.len() as f32)
    };
    let value_at_crash = crash_tick.value_prediction;
    let value_drop = match (value_window_mean, value_at_crash) {
        (Some(mean), Some(end)) => Some(mean - end),
        _ => None,
    };

    Some(PreCrashProfile {
        window_len,
        throttle_early,
        throttle_late,
        throttle_delta,
        throttle_release_ticks,
        distance_to_wall_at_crash,
        value_window_mean,
        value_at_crash,
        value_drop,
    })
}

/// Aggregated statistics across all crashes.
#[derive(Clone, Debug, Default)]
pub struct PreCrashSummary {
    pub crash_count: usize,
    pub mean_throttle_delta: f32,
    pub mean_distance_to_wall: f32,
    pub median_distance_to_wall: f32,
    pub mean_value_drop: Option<f32>,
    /// Fraction of crashes where policy released throttle at least once in
    /// the window (throttle_release_ticks.is_some()).
    pub release_any_fraction: f32,
    /// Fraction where the release happened early (> 15 ticks before crash = > 0.25s).
    pub release_early_fraction: f32,
    /// Throttle-release latency histogram — 6 buckets over the window.
    pub release_histogram: [u32; 6],
    /// Distance-to-wall histogram at crash — 6 buckets: [0-5, 5-10, 10-20, 20-40, 40-80, 80+] units.
    pub distance_histogram: [u32; 6],
}

pub fn summarise(profiles: &[PreCrashProfile]) -> PreCrashSummary {
    let mut s = PreCrashSummary::default();
    if profiles.is_empty() {
        return s;
    }
    s.crash_count = profiles.len();
    s.mean_throttle_delta =
        profiles.iter().map(|p| p.throttle_delta).sum::<f32>() / profiles.len() as f32;
    let distances: Vec<f32> = profiles.iter().map(|p| p.distance_to_wall_at_crash).collect();
    s.mean_distance_to_wall = distances.iter().sum::<f32>() / profiles.len() as f32;
    let mut sorted = distances.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s.median_distance_to_wall = sorted[sorted.len() / 2];

    let value_drops: Vec<f32> = profiles.iter().filter_map(|p| p.value_drop).collect();
    if !value_drops.is_empty() {
        s.mean_value_drop = Some(value_drops.iter().sum::<f32>() / value_drops.len() as f32);
    }

    let released: Vec<u32> = profiles.iter().filter_map(|p| p.throttle_release_ticks).collect();
    s.release_any_fraction = released.len() as f32 / profiles.len() as f32;
    s.release_early_fraction =
        released.iter().filter(|&&t| t > 15).count() as f32 / profiles.len() as f32;

    // Release latency histogram (6 buckets across PRE_CRASH_WINDOW):
    // 0-5, 5-10, 10-15, 15-20, 20-25, 25-30 ticks-before-crash.
    for &t in &released {
        let bucket = ((t as usize) / 5).min(5);
        s.release_histogram[bucket] = s.release_histogram[bucket].saturating_add(1);
    }

    // Distance-to-wall histogram at crash:
    // 0-5, 5-10, 10-20, 20-40, 40-80, 80+ units.
    for d in &distances {
        let bucket = if *d < 5.0 {
            0
        } else if *d < 10.0 {
            1
        } else if *d < 20.0 {
            2
        } else if *d < 40.0 {
            3
        } else if *d < 80.0 {
            4
        } else {
            5
        };
        s.distance_histogram[bucket] = s.distance_histogram[bucket].saturating_add(1);
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tick(throttle: f32, min_ray: f32, value: Option<f32>, tick_index: u32) -> TickTraceRecord {
        TickTraceRecord {
            env_id: 0,
            tick_index,
            position_x: 0.0,
            position_y: 0.0,
            progress_fraction: 0.0,
            progress_s: 0.0,
            centerline_distance: 0.0,
            signed_lateral_offset: 0.0,
            speed: 100.0,
            v_forward: 100.0,
            v_lateral: 0.0,
            speed_delta: 0.0,
            drift_angle_deg: 0.0,
            heading_error: 0.0,
            min_ray_distance: min_ray,
            velocity_projection: 100.0,
            centreline_reward: 0.0,
            steering: 0.0,
            throttle,
            previous_steering: 0.0,
            previous_throttle: 0.0,
            reward: 0.1,
            progress_reward: 0.1,
            time_penalty: 0.0,
            terminal_reward: 0.0,
            done: false,
            done_reason: None,
            sector_index: 0,
            ray_distances: [0.0; 11],
            lookahead_heading_deltas: [0.0; 12],
            lookahead_curvatures: [0.0; 12],
            value_prediction: value,
            policy_steering_mean: None,
            policy_steering_std: None,
            policy_throttle_mean: None,
            policy_throttle_std: None,
        }
    }

    #[test]
    fn reactive_policy_has_zero_throttle_release() {
        // Full throttle throughout, no release.
        let ticks: Vec<TickTraceRecord> = (0..40)
            .map(|i| make_tick(1.0, 5.0, Some(50.0), i))
            .collect();
        let profile = analyse_trace(&ticks).unwrap();
        assert!(profile.throttle_release_ticks.is_none());
        assert!((profile.throttle_delta).abs() < 1e-4);
    }

    #[test]
    fn anticipatory_policy_shows_throttle_drop_early() {
        // Throttle 1.0 until tick 25, then 0.2 for last 15 ticks before crash.
        let mut ticks: Vec<TickTraceRecord> = Vec::new();
        for i in 0..25 {
            ticks.push(make_tick(1.0, 50.0, Some(80.0), i));
        }
        for i in 25..40 {
            ticks.push(make_tick(0.2, 15.0 - (i as f32 - 25.0), Some(40.0), i));
        }
        let profile = analyse_trace(&ticks).unwrap();
        assert!(profile.throttle_delta < 0.0, "throttle should decrease toward crash");
        // Release should be detected — window is last 30 ticks of 40, so ticks 10..40.
        // First tick with throttle<0.5 is tick 25 (= 15 ticks before crash).
        assert!(profile.throttle_release_ticks.is_some());
    }

    #[test]
    fn summary_handles_empty_input() {
        let s = summarise(&[]);
        assert_eq!(s.crash_count, 0);
    }
}
