use crate::analytics::models::EpisodeTracker;

/// Ordered time-series vectors extracted from completed episodes.
pub struct EpisodeTimeSeries {
    pub progress: Vec<f32>,
    pub reward: Vec<f32>,
    pub life_ticks: Vec<u32>,
    pub is_crash: Vec<bool>,
    pub env_ids: Vec<u32>,
}

/// Ordered time-series vectors extracted from A2C update records.
pub struct UpdateTimeSeries {
    pub policy_loss: Vec<f32>,
    pub value_loss: Vec<f32>,
    pub entropy: Vec<f32>,
    pub clip_fraction: Vec<f32>,
    pub approx_kl: Vec<f32>,
    pub explained_variance: Vec<f32>,
}

/// Extracts per-episode scalar series from the tracker in recording order.
pub fn extract_episode_series(tracker: &EpisodeTracker) -> EpisodeTimeSeries {
    let mut progress = Vec::with_capacity(tracker.episodes.len());
    let mut reward = Vec::with_capacity(tracker.episodes.len());
    let mut life_ticks = Vec::with_capacity(tracker.episodes.len());
    let mut is_crash = Vec::with_capacity(tracker.episodes.len());
    let mut env_ids = Vec::with_capacity(tracker.episodes.len());

    for ep in &tracker.episodes {
        progress.push(ep.progress);
        reward.push(ep.reward);
        life_ticks.push(ep.ticks);
        is_crash.push(ep.end_reason == "crash");
        env_ids.push(ep.env_id);
    }

    EpisodeTimeSeries {
        progress,
        reward,
        life_ticks,
        is_crash,
        env_ids,
    }
}

/// Extracts per-update scalar series from the tracker in recording order.
pub fn extract_update_series(tracker: &EpisodeTracker) -> UpdateTimeSeries {
    let mut policy_loss = Vec::with_capacity(tracker.a2c_updates.len());
    let mut value_loss = Vec::with_capacity(tracker.a2c_updates.len());
    let mut entropy = Vec::with_capacity(tracker.a2c_updates.len());
    let mut clip_fraction = Vec::with_capacity(tracker.a2c_updates.len());
    let mut approx_kl = Vec::with_capacity(tracker.a2c_updates.len());
    let mut explained_variance = Vec::with_capacity(tracker.a2c_updates.len());

    for u in &tracker.a2c_updates {
        policy_loss.push(u.policy_loss);
        value_loss.push(u.value_loss);
        entropy.push(u.policy_entropy);
        clip_fraction.push(u.clip_fraction);
        approx_kl.push(u.approx_kl);
        explained_variance.push(u.explained_variance);
    }

    UpdateTimeSeries {
        policy_loss,
        value_loss,
        entropy,
        clip_fraction,
        approx_kl,
        explained_variance,
    }
}

/// Computes a simple moving average with an expanding window for the first
/// `window - 1` elements. The returned vector has the same length as `values`.
pub fn rolling_mean(values: &[f32], window: usize) -> Vec<f32> {
    if values.is_empty() || window == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(values.len());
    let mut running_sum: f32 = 0.0;

    for (i, &v) in values.iter().enumerate() {
        running_sum += v;

        if i >= window {
            running_sum -= values[i - window];
        }

        let current_window = (i + 1).min(window);
        result.push(running_sum / current_window as f32);
    }

    result
}

/// Detects whether the series has plateaued by comparing the maximum value in
/// the last `window` elements against the maximum in the preceding `window`
/// elements. If the difference is less than `threshold`, the series is
/// considered plateaued and the function returns the index where the final
/// window begins (i.e. the point where improvement stalled).
///
/// Returns `None` if the series is too short (fewer than `2 * window` elements)
/// or if the series is still improving.
pub fn detect_plateau(values: &[f32], window: usize, threshold: f32) -> Option<usize> {
    if values.len() < 2 * window || window == 0 {
        return None;
    }

    let smoothed = rolling_mean(values, window);

    let n = smoothed.len();
    let recent_start = n - window;
    let prior_start = recent_start - window;

    let recent_max = smoothed[recent_start..n]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let prior_max = smoothed[prior_start..recent_start]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    if (recent_max - prior_max).abs() < threshold {
        Some(recent_start)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_mean_basic() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&values, 3);
        let expected = [1.0, 1.5, 2.0, 3.0, 4.0];
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "got {a}, expected {b}",
            );
        }
    }

    #[test]
    fn rolling_mean_empty() {
        assert!(rolling_mean(&[], 3).is_empty());
    }

    #[test]
    fn detect_plateau_flat_series() {
        let flat: Vec<f32> = vec![1.0; 100];
        let result = detect_plateau(&flat, 20, 2.0);
        assert!(
            result.is_some(),
            "flat series should be detected as plateau",
        );
    }

    #[test]
    fn detect_plateau_rising_series() {
        let rising: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = detect_plateau(&rising, 20, 2.0);
        assert!(
            result.is_none(),
            "steadily rising series should not be detected as plateau",
        );
    }

    #[test]
    fn detect_plateau_too_short() {
        let short = vec![1.0; 10];
        assert!(detect_plateau(&short, 20, 2.0).is_none());
    }
}
