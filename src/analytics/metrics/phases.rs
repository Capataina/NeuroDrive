use crate::analytics::metrics::timeseries::{extract_episode_series, rolling_mean};
use crate::analytics::models::EpisodeTracker;

/// Classifies the current phase of the learning process based on episode
/// progress trends, crash rates, and variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LearningPhase {
    /// Very early training: low progress, high crash rate.
    Exploration,
    /// Progress is actively increasing.
    Discovery,
    /// High progress with low variance — fine-tuning.
    Refinement,
    /// Progress has stalled.
    Plateau,
    /// Progress is declining.
    Regression,
}

impl std::fmt::Display for LearningPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exploration => write!(f, "Exploration"),
            Self::Discovery => write!(f, "Discovery"),
            Self::Refinement => write!(f, "Refinement"),
            Self::Plateau => write!(f, "Plateau"),
            Self::Regression => write!(f, "Regression"),
        }
    }
}

/// Detects the current learning phase from the tracker's episode history.
///
/// Uses rolling-mean progress, crash rate, and variance to classify the phase.
/// The thresholds are tuned for NeuroDrive's progress scale (0–100%).
pub fn detect_learning_phase(tracker: &EpisodeTracker) -> LearningPhase {
    let series = extract_episode_series(tracker);

    if series.progress.is_empty() || series.progress.len() < 10 {
        return LearningPhase::Exploration;
    }

    // Progress is stored as a fraction 0.0–1.0; convert to percentage for
    // threshold comparisons.
    let progress_pct: Vec<f32> = series.progress.iter().map(|p| p * 100.0).collect();
    let smoothed = rolling_mean(&progress_pct, 20);

    if smoothed.is_empty() {
        return LearningPhase::Exploration;
    }

    let n = smoothed.len();

    // Recent window: last 20 smoothed values (or all if fewer).
    let recent_start = n.saturating_sub(20);
    let recent_slice = &smoothed[recent_start..];
    let recent_mean = recent_slice.iter().sum::<f32>() / recent_slice.len() as f32;

    // Prior window: 20 values before the recent window (or first half).
    let prior_slice = if n >= 40 {
        &smoothed[n - 40..n - 20]
    } else {
        let mid = n / 2;
        &smoothed[..mid.max(1)]
    };
    let prior_mean = prior_slice.iter().sum::<f32>() / prior_slice.len().max(1) as f32;

    // Crash rate over the last 20 episodes.
    let crash_window_start = series.is_crash.len().saturating_sub(20);
    let crash_window = &series.is_crash[crash_window_start..];
    let crash_rate = crash_window.iter().filter(|&&c| c).count() as f32
        / crash_window.len().max(1) as f32;

    // Variance of the recent smoothed values.
    let recent_var = {
        let m = recent_mean;
        recent_slice.iter().map(|v| (v - m).powi(2)).sum::<f32>()
            / recent_slice.len().max(1) as f32
    };

    // Classification logic — order matters: check the most specific conditions
    // first.
    if recent_mean < 15.0 && crash_rate > 0.7 {
        return LearningPhase::Exploration;
    }
    if recent_mean > prior_mean + 3.0 {
        return LearningPhase::Discovery;
    }
    if recent_mean < prior_mean - 3.0 {
        return LearningPhase::Regression;
    }
    if recent_mean > 50.0 && recent_var < 100.0 {
        return LearningPhase::Refinement;
    }

    LearningPhase::Plateau
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::models::{EpisodeRecord, EpisodeTracker};

    fn episode(progress: f32, end_reason: &str) -> EpisodeRecord {
        EpisodeRecord {
            env_id: 0,
            episode_id: 0,
            progress,
            reward: 0.0,
            pre_terminal_return: 0.0,
            progress_reward_sum: 0.0,
            time_penalty_sum: 0.0,
            terminal_reward_sum: 0.0,
            crash_penalty_sum: 0.0,
            ticks: 100,
            crashes: if end_reason == "crash" { 1 } else { 0 },
            end_reason: end_reason.to_string(),
            distance_driven: progress * 1000.0,
            crash_position: None,
            steering_mean: 0.0,
            steering_std: 0.0,
            throttle_mean: 0.0,
            throttle_std: 0.0,
            braking_fraction: 0.0,
            acceleration_fraction: 0.0,
            coasting_fraction: 0.0,
            mean_action_change: 0.0,
            mean_speed: 0.0,
            peak_speed: 0.0,
            mean_v_forward: 0.0,
            mean_v_lateral_abs: 0.0,
            mean_velocity_projection: 0.0,
            mean_drift_angle_deg: 0.0,
            peak_drift_angle_deg: 0.0,
            crash_speed: None,
            crash_v_forward: None,
            crash_v_lateral: None,
            crash_drift_angle_deg: None,
            crash_heading_error_deg: None,
            crash_min_ray: None,
            crash_type: None,
            mean_value_prediction: None,
            value_at_crash: None,
            value_at_start: None,
            reward_per_second: 0.0,
            furthest_sector: 0,
            wall_proximity_fraction: 0.0,
            mean_policy_steering_std: None,
            mean_policy_throttle_std: None,
            turn_in_latency_fraction: None,
            turn_in_latency_ticks: None,
            throttle_release_latency_fraction: None,
            throttle_release_latency_ticks: None,
            steering_adequacy: 0.0,
            high_curvature_throttle_mean: 0.0,
            curvature_steering_error_mean: 0.0,
            curvature_steering_bias_mean: 0.0,
            understeer_rate: 0.0,
            turn_entry_speed: None,
            peak_curvature_speed: None,
            entry_lateral_offset: None,
            peak_lateral_offset: None,
            peak_centerline_distance: None,
            mean_centerline_distance: 0.0,
            mean_abs_lateral_offset: 0.0,
            mean_abs_heading_error_deg: 0.0,
            mean_all_ray_distance: 0.0,
            mean_front_ray_distance: 0.0,
            mean_side_ray_distance: 0.0,
            failure_mode: None,
        }
    }

    fn tracker_with(episodes: Vec<EpisodeRecord>) -> EpisodeTracker {
        EpisodeTracker {
            episodes,
            ppo_updates: Vec::new(),
            episode_traces: Vec::new(),
            last_recorded_update: 0,
        }
    }

    #[test]
    fn few_episodes_is_exploration() {
        let tracker = tracker_with(vec![episode(0.05, "crash"); 5]);
        assert_eq!(detect_learning_phase(&tracker), LearningPhase::Exploration);
    }

    #[test]
    fn low_progress_high_crash_is_exploration() {
        let mut eps: Vec<EpisodeRecord> = (0..40).map(|_| episode(0.05, "crash")).collect();
        // Add a few non-crashes to avoid 100% rate — still above 0.7.
        for ep in eps.iter_mut().take(5) {
            ep.end_reason = "timeout".to_string();
            ep.crashes = 0;
        }
        let tracker = tracker_with(eps);
        assert_eq!(detect_learning_phase(&tracker), LearningPhase::Exploration);
    }

    #[test]
    fn improving_progress_is_discovery() {
        let eps: Vec<EpisodeRecord> = (0..60)
            .map(|i| episode(0.10 + i as f32 * 0.005, "timeout"))
            .collect();
        let tracker = tracker_with(eps);
        assert_eq!(detect_learning_phase(&tracker), LearningPhase::Discovery);
    }

    #[test]
    fn declining_progress_is_regression() {
        let eps: Vec<EpisodeRecord> = (0..60)
            .map(|i| episode(0.60 - i as f32 * 0.005, "timeout"))
            .collect();
        let tracker = tracker_with(eps);
        assert_eq!(detect_learning_phase(&tracker), LearningPhase::Regression);
    }

    #[test]
    fn high_stable_progress_is_refinement() {
        let eps: Vec<EpisodeRecord> = (0..60)
            .map(|_| episode(0.75, "timeout"))
            .collect();
        let tracker = tracker_with(eps);
        assert_eq!(detect_learning_phase(&tracker), LearningPhase::Refinement);
    }

    #[test]
    fn flat_moderate_progress_is_plateau() {
        let eps: Vec<EpisodeRecord> = (0..60)
            .map(|_| episode(0.30, "timeout"))
            .collect();
        let tracker = tracker_with(eps);
        assert_eq!(detect_learning_phase(&tracker), LearningPhase::Plateau);
    }
}
