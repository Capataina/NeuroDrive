use crate::analytics::models::{EpisodeTracker, NUM_PROGRESS_SECTORS};

/// Per-sector behavioural profile summarising the agent's control outputs and
/// positioning across recent episode traces.
pub struct SectorBehaviourProfile {
    pub sector: usize,
    pub speed_mean: f32,
    pub speed_var: f32,
    pub steering_mean: f32,
    pub steering_var: f32,
    pub throttle_mean: f32,
    pub throttle_var: f32,
    pub centerline_dist_mean: f32,
    pub centerline_dist_var: f32,
    pub sample_count: usize,
}

/// Accumulator for online mean/variance computation per sector.
#[derive(Clone, Default)]
struct SectorAccumulator {
    count: usize,
    speed_sum: f64,
    speed_sumsq: f64,
    steering_sum: f64,
    steering_sumsq: f64,
    throttle_sum: f64,
    throttle_sumsq: f64,
    centerline_sum: f64,
    centerline_sumsq: f64,
}

impl SectorAccumulator {
    fn push(&mut self, speed: f32, steering: f32, throttle: f32, centerline: f32) {
        self.count += 1;
        let s = speed as f64;
        let st = steering as f64;
        let th = throttle as f64;
        let cl = centerline as f64;
        self.speed_sum += s;
        self.speed_sumsq += s * s;
        self.steering_sum += st;
        self.steering_sumsq += st * st;
        self.throttle_sum += th;
        self.throttle_sumsq += th * th;
        self.centerline_sum += cl;
        self.centerline_sumsq += cl * cl;
    }

    fn mean_var(sum: f64, sumsq: f64, count: usize) -> (f32, f32) {
        if count == 0 {
            return (0.0, 0.0);
        }
        let n = count as f64;
        let mean = sum / n;
        let var = (sumsq / n - mean * mean).max(0.0);
        (mean as f32, var as f32)
    }

    fn into_profile(self, sector: usize) -> SectorBehaviourProfile {
        let (speed_mean, speed_var) = Self::mean_var(self.speed_sum, self.speed_sumsq, self.count);
        let (steering_mean, steering_var) =
            Self::mean_var(self.steering_sum, self.steering_sumsq, self.count);
        let (throttle_mean, throttle_var) =
            Self::mean_var(self.throttle_sum, self.throttle_sumsq, self.count);
        let (centerline_dist_mean, centerline_dist_var) =
            Self::mean_var(self.centerline_sum, self.centerline_sumsq, self.count);

        SectorBehaviourProfile {
            sector,
            speed_mean,
            speed_var,
            steering_mean,
            steering_var,
            throttle_mean,
            throttle_var,
            centerline_dist_mean,
            centerline_dist_var,
            sample_count: self.count,
        }
    }
}

/// Computes per-sector behavioural profiles from the last `window` episode
/// traces. For each of the 20 sectors, gathers all ticks across those traces
/// where `tick.sector_index == sector`, then computes mean and variance of
/// speed, steering, throttle, and centreline distance.
pub fn compute_sector_consistency(
    tracker: &EpisodeTracker,
    window: usize,
) -> Vec<SectorBehaviourProfile> {
    let traces = &tracker.episode_traces;
    let start = traces.len().saturating_sub(window);
    let recent = &traces[start..];

    let mut accumulators = vec![SectorAccumulator::default(); NUM_PROGRESS_SECTORS];

    for trace in recent {
        for tick in &trace.ticks {
            let sector =
                (tick.sector_index as usize).min(NUM_PROGRESS_SECTORS.saturating_sub(1));
            accumulators[sector].push(
                tick.speed,
                tick.steering,
                tick.throttle,
                tick.centerline_distance,
            );
        }
    }

    accumulators
        .into_iter()
        .enumerate()
        .map(|(sector, acc)| acc.into_profile(sector))
        .collect()
}

/// Scalar consistency score: `1.0 / (1.0 + mean_of_variances)`.
///
/// Averages the variance of speed, steering, throttle, and centreline distance
/// across all sectors that have samples. Higher values indicate more consistent
/// behaviour. Returns 1.0 when no sectors have data (vacuously consistent).
pub fn overall_consistency_score(profiles: &[SectorBehaviourProfile]) -> f32 {
    let populated: Vec<&SectorBehaviourProfile> =
        profiles.iter().filter(|p| p.sample_count > 0).collect();

    if populated.is_empty() {
        return 1.0;
    }

    let total_var: f32 = populated
        .iter()
        .map(|p| p.speed_var + p.steering_var + p.throttle_var + p.centerline_dist_var)
        .sum::<f32>();

    let avg_var = total_var / (populated.len() as f32 * 4.0);
    1.0 / (1.0 + avg_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::models::{
        EpisodeTrace, EpisodeTraceMetrics, EpisodeTracker, TickTraceRecord,
    };

    fn make_tick(sector: u32, speed: f32, steering: f32, throttle: f32, cl: f32) -> TickTraceRecord {
        TickTraceRecord {
            env_id: 0,
            tick_index: 0,
            position_x: 0.0,
            position_y: 0.0,
            progress_fraction: 0.0,
            progress_s: 0.0,
            centerline_distance: cl,
            signed_lateral_offset: 0.0,
            speed,
            v_forward: speed,
            v_lateral: 0.0,
            speed_delta: 0.0,
            drift_angle_deg: 0.0,
            heading_error: 0.0,
            min_ray_distance: 100.0,
            velocity_projection: speed,
            centreline_reward: 0.0,
            steering,
            throttle,
            previous_steering: 0.0,
            previous_throttle: 0.0,
            reward: 0.0,
            progress_reward: 0.0,
            time_penalty: 0.0,
            terminal_reward: 0.0,
            done: false,
            done_reason: None,
            sector_index: sector,
            ray_distances: vec![100.0; 11],
            lookahead_heading_deltas: vec![0.0; 4],
            lookahead_curvatures: vec![0.0; 4],
            value_prediction: None,
            policy_steering_mean: None,
            policy_steering_std: None,
            policy_throttle_mean: None,
            policy_throttle_std: None,
        }
    }

    fn make_trace(ticks: Vec<TickTraceRecord>) -> EpisodeTrace {
        EpisodeTrace {
            episode_id: 0,
            end_reason: "timeout".to_string(),
            best_progress: 0.5,
            ticks,
            metrics: EpisodeTraceMetrics::default(),
        }
    }

    #[test]
    fn consistency_score_is_one_for_identical_behaviour() {
        let ticks = vec![
            make_tick(0, 100.0, 0.5, 0.8, 2.0),
            make_tick(0, 100.0, 0.5, 0.8, 2.0),
        ];
        let trace = make_trace(ticks);
        let tracker = EpisodeTracker {
            episodes: Vec::new(),
            a2c_updates: Vec::new(),
            episode_traces: vec![trace],
            last_recorded_update: 0,
        };

        let profiles = compute_sector_consistency(&tracker, 10);
        let score = overall_consistency_score(&profiles);
        assert!(
            (score - 1.0).abs() < 1e-4,
            "identical ticks should yield score ~1.0, got {score}",
        );
    }

    #[test]
    fn consistency_score_decreases_with_variance() {
        let ticks = vec![
            make_tick(0, 50.0, -1.0, 0.0, 0.0),
            make_tick(0, 250.0, 1.0, 1.0, 50.0),
        ];
        let trace = make_trace(ticks);
        let tracker = EpisodeTracker {
            episodes: Vec::new(),
            a2c_updates: Vec::new(),
            episode_traces: vec![trace],
            last_recorded_update: 0,
        };

        let profiles = compute_sector_consistency(&tracker, 10);
        let score = overall_consistency_score(&profiles);
        assert!(
            score < 0.5,
            "high variance ticks should yield low score, got {score}",
        );
    }

    #[test]
    fn empty_tracker_returns_one() {
        let tracker = EpisodeTracker {
            episodes: Vec::new(),
            a2c_updates: Vec::new(),
            episode_traces: Vec::new(),
            last_recorded_update: 0,
        };
        let profiles = compute_sector_consistency(&tracker, 10);
        let score = overall_consistency_score(&profiles);
        assert!((score - 1.0).abs() < 1e-4);
    }
}
