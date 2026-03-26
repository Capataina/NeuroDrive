use crate::analytics::models::EpisodeTracker;

use super::timeseries::{detect_plateau, extract_episode_series};

/// Severity level for a diagnostic flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
}

/// A single diagnostic observation about the training run.
#[derive(Clone, Debug)]
pub struct DiagnosticFlag {
    pub message: String,
    pub severity: Severity,
}

/// Scans the tracker for common training pathologies and returns any
/// diagnostic flags found, ordered by detection priority.
pub fn compute_diagnostic_flags(tracker: &EpisodeTracker) -> Vec<DiagnosticFlag> {
    let mut flags = Vec::new();

    check_entropy_collapse(tracker, &mut flags);
    check_clip_fraction_high(tracker, &mut flags);
    check_kl_spike(tracker, &mut flags);
    check_progress_plateau(tracker, &mut flags);
    check_action_collapse(tracker, &mut flags);
    check_crash_rate_spike(tracker, &mut flags);
    check_value_function_drift(tracker, &mut flags);

    flags
}

/// Entropy below 0.1 on the most recent update suggests exploration collapse.
fn check_entropy_collapse(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    if let Some(latest) = tracker.a2c_updates.last() {
        if latest.policy_entropy < 0.1 {
            flags.push(DiagnosticFlag {
                message: "Entropy below 0.1 — exploration may have collapsed".into(),
                severity: Severity::Warning,
            });
        }
    }
}

/// Three or more consecutive recent updates with clip_fraction > 0.40.
fn check_clip_fraction_high(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    let consecutive = count_consecutive_from_end(&tracker.a2c_updates, |u| u.clip_fraction > 0.40);
    if consecutive >= 3 {
        flags.push(DiagnosticFlag {
            message: format!(
                "Clip fraction above 40% for {consecutive} consecutive updates \
                 — policy changing too aggressively"
            ),
            severity: Severity::Warning,
        });
    }
}

/// Any update with approx_kl > 0.03.
fn check_kl_spike(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    let max_kl = tracker
        .a2c_updates
        .iter()
        .map(|u| u.approx_kl)
        .fold(f32::NEG_INFINITY, f32::max);

    if max_kl > 0.03 {
        flags.push(DiagnosticFlag {
            message: format!(
                "KL divergence spike detected (max {max_kl:.4}) \
                 — policy drifted far from collection policy"
            ),
            severity: Severity::Warning,
        });
    }
}

/// Progress plateau detection via rolling-mean analysis.
fn check_progress_plateau(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    let series = extract_episode_series(tracker);
    if let Some(idx) = detect_plateau(&series.progress, 20, 2.0) {
        let stalled_episodes = series.progress.len() - idx;
        flags.push(DiagnosticFlag {
            message: format!(
                "Progress plateau detected — no improvement in last {stalled_episodes} episodes"
            ),
            severity: Severity::Info,
        });
    }
}

/// Action distribution collapse when steering or throttle std is near zero.
fn check_action_collapse(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    if let Some(latest) = tracker.a2c_updates.last() {
        if latest.steering_std < 0.05 || latest.throttle_std < 0.05 {
            flags.push(DiagnosticFlag {
                message: format!(
                    "Action distribution collapse — steering_std={:.3} throttle_std={:.3}",
                    latest.steering_std, latest.throttle_std,
                ),
                severity: Severity::Warning,
            });
        }
    }
}

/// Crash rate spike: compare last 20 episodes vs previous 20.
fn check_crash_rate_spike(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    if tracker.episodes.len() < 40 {
        return;
    }

    let n = tracker.episodes.len();
    let recent = &tracker.episodes[n - 20..];
    let prior = &tracker.episodes[n - 40..n - 20];

    let crash_rate = |eps: &[crate::analytics::models::EpisodeRecord]| -> f32 {
        let crashes = eps.iter().filter(|e| e.end_reason == "crash").count();
        crashes as f32 / eps.len() as f32 * 100.0
    };

    let recent_rate = crash_rate(recent);
    let prior_rate = crash_rate(prior);

    if recent_rate > prior_rate && (recent_rate - prior_rate) > 30.0 {
        flags.push(DiagnosticFlag {
            message: format!(
                "Crash rate spike — {recent_rate:.0}% recent vs {prior_rate:.0}% prior"
            ),
            severity: Severity::Warning,
        });
    }
}

/// Explained variance negative for 3+ consecutive recent updates.
fn check_value_function_drift(tracker: &EpisodeTracker, flags: &mut Vec<DiagnosticFlag>) {
    let consecutive =
        count_consecutive_from_end(&tracker.a2c_updates, |u| u.explained_variance < 0.0);
    if consecutive >= 3 {
        flags.push(DiagnosticFlag {
            message: format!(
                "Explained variance negative for {consecutive} consecutive updates \
                 — critic may be harmful"
            ),
            severity: Severity::Warning,
        });
    }
}

/// Counts how many elements at the tail of `slice` satisfy `predicate`
/// consecutively, working backwards from the end.
fn count_consecutive_from_end<T>(slice: &[T], predicate: impl Fn(&T) -> bool) -> usize {
    slice.iter().rev().take_while(|item| predicate(item)).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::models::{A2cUpdateRecord, EpisodeTracker};

    /// Helper: builds a minimal A2C update with sensible defaults that should
    /// not trigger any diagnostic flags on its own.
    fn healthy_update() -> A2cUpdateRecord {
        A2cUpdateRecord {
            update_index: 0,
            batch_size: 64,
            policy_loss: -0.01,
            value_loss: 0.5,
            policy_entropy: 0.6,
            explained_variance: 0.8,
            steering_mean: 0.0,
            steering_std: 0.3,
            throttle_mean: 0.5,
            throttle_std: 0.2,
            clamped_action_fraction: 0.05,
            clip_fraction: 0.10,
            approx_kl: 0.005,
            layer_health: Vec::new(),
        }
    }

    #[test]
    fn low_entropy_triggers_warning() {
        let mut update = healthy_update();
        update.policy_entropy = 0.05;

        let tracker = EpisodeTracker {
            episodes: Vec::new(),
            a2c_updates: vec![update],
            episode_traces: Vec::new(),
            last_recorded_update: 0,
        };

        let flags = compute_diagnostic_flags(&tracker);
        assert!(
            flags.iter().any(|f| f.severity == Severity::Warning
                && f.message.contains("Entropy below 0.1")),
            "expected entropy collapse warning, got: {flags:?}",
        );
    }

    #[test]
    fn healthy_tracker_produces_no_warnings() {
        let tracker = EpisodeTracker {
            episodes: Vec::new(),
            a2c_updates: vec![healthy_update()],
            episode_traces: Vec::new(),
            last_recorded_update: 0,
        };

        let flags = compute_diagnostic_flags(&tracker);
        let warnings: Vec<_> = flags
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .collect();
        assert!(
            warnings.is_empty(),
            "healthy tracker should produce no warnings, got: {warnings:?}",
        );
    }
}
