use crate::analytics::metrics::stats::{mean, percentile};
use crate::analytics::models::{EpisodeRecord, EpisodeTraceMetrics, TickTraceRecord};

pub const CURVATURE_DEMAND_THRESHOLD: f32 = 0.015;
const STEERING_ONSET_THRESHOLD: f32 = 0.18;
const THROTTLE_RELEASE_THRESHOLD: f32 = 0.35;
const STEERING_CURVATURE_GAIN: f32 = 0.03;
const STRONG_STEERING_REQUIREMENT: f32 = 0.30;
const UNDERSTEER_RATIO_THRESHOLD: f32 = 0.75;
const LARGE_OFFSET_THRESHOLD: f32 = 30.0;
const LARGE_LINE_GAP_THRESHOLD: f32 = 22.0;
const HIGH_ENTRY_SPEED_THRESHOLD: f32 = 340.0;
const HIGH_CURVATURE_THROTTLE_THRESHOLD: f32 = 0.55;
const LATE_TURN_IN_TICK_THRESHOLD: u32 = 8;

#[derive(Clone, Debug, Default)]
pub struct TurnExecutionSummary {
    pub turn_in_latency_fraction_mean: Option<f32>,
    pub turn_in_latency_fraction_p90: Option<f32>,
    pub turn_in_latency_ticks_mean: Option<f32>,
    pub throttle_release_latency_fraction_mean: Option<f32>,
    pub throttle_release_latency_ticks_mean: Option<f32>,
    pub steering_adequacy_mean: f32,
    pub high_curvature_throttle_mean: f32,
    pub curvature_steering_error_mean: f32,
    pub curvature_steering_bias_mean: f32,
    pub understeer_rate_mean: f32,
    pub turn_entry_speed_mean: Option<f32>,
    pub turn_entry_speed_p90: Option<f32>,
    pub peak_curvature_speed_mean: Option<f32>,
    pub crash_speed_mean: Option<f32>,
    pub entry_lateral_offset_mean: Option<f32>,
    pub peak_lateral_offset_mean: Option<f32>,
    pub peak_centerline_distance_mean: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct FailureModeCount {
    pub label: String,
    pub count: usize,
    pub share: f32,
}

pub fn compute_trace_metrics(ticks: &[TickTraceRecord]) -> EpisodeTraceMetrics {
    if ticks.is_empty() {
        return EpisodeTraceMetrics::default();
    }

    let demand_idx = ticks.iter().enumerate().find_map(|(index, tick)| {
        (curvature_demand(tick) >= CURVATURE_DEMAND_THRESHOLD).then_some(index)
    });
    let peak_idx = ticks
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| curvature_demand(a).total_cmp(&curvature_demand(b)))
        .map(|(index, _)| index);

    let mut turn_in_latency_fraction = None;
    let mut turn_in_latency_ticks = None;
    let mut throttle_release_latency_fraction = None;
    let mut throttle_release_latency_ticks = None;

    if let Some(demand_idx) = demand_idx {
        if let Some(steer_idx) =
            ticks
                .iter()
                .enumerate()
                .skip(demand_idx)
                .find_map(|(index, tick)| {
                    (tick.steering.abs() >= STEERING_ONSET_THRESHOLD).then_some(index)
                })
        {
            turn_in_latency_fraction = Some(wrapped_fraction_delta(
                ticks[demand_idx].progress_fraction,
                ticks[steer_idx].progress_fraction,
            ));
            turn_in_latency_ticks = Some((steer_idx - demand_idx) as u32);
        }

        if let Some(release_idx) =
            ticks
                .iter()
                .enumerate()
                .skip(demand_idx)
                .find_map(|(index, tick)| {
                    (tick.throttle <= THROTTLE_RELEASE_THRESHOLD).then_some(index)
                })
        {
            throttle_release_latency_fraction = Some(wrapped_fraction_delta(
                ticks[demand_idx].progress_fraction,
                ticks[release_idx].progress_fraction,
            ));
            throttle_release_latency_ticks = Some((release_idx - demand_idx) as u32);
        }
    }

    let mut high_curvature_throttle_values = Vec::new();
    let mut curvature_error_values = Vec::new();
    let mut curvature_bias_values = Vec::new();
    let mut understeer_count = 0usize;
    let mut understeer_samples = 0usize;
    let mut all_ray_distances = Vec::new();
    let mut front_ray_distances = Vec::new();
    let mut side_ray_distances = Vec::new();
    let mut centerline_distances = Vec::new();
    let mut abs_lateral_offsets = Vec::new();
    let mut abs_heading_errors_deg = Vec::new();

    for tick in ticks {
        centerline_distances.push(tick.centerline_distance);
        abs_lateral_offsets.push(tick.signed_lateral_offset.abs());
        abs_heading_errors_deg.push(tick.heading_error.abs().to_degrees());
        all_ray_distances.extend(tick.ray_distances.iter().copied());
        front_ray_distances.extend(front_ray_slice(tick).iter().copied());
        side_ray_distances.extend(side_ray_slice(tick).iter().copied());

        if curvature_demand(tick) >= CURVATURE_DEMAND_THRESHOLD {
            high_curvature_throttle_values.push(tick.throttle);
            let required_steering = required_steering(tick);
            let error = required_steering - tick.steering;
            curvature_error_values.push(error.abs());
            curvature_bias_values.push(error);
            if required_steering.abs() >= STRONG_STEERING_REQUIREMENT {
                understeer_samples += 1;
                if tick.steering.abs() < required_steering.abs() * UNDERSTEER_RATIO_THRESHOLD {
                    understeer_count += 1;
                }
            }
        }
    }

    let failure_mode = classify_failure_mode_from_components(
        demand_idx.map(|index| &ticks[index]),
        peak_idx.map(|index| &ticks[index]),
        ticks.last(),
        turn_in_latency_ticks,
        mean(&high_curvature_throttle_values),
        mean(&curvature_error_values),
        if understeer_samples == 0 {
            0.0
        } else {
            understeer_count as f32 / understeer_samples as f32
        },
    );

    EpisodeTraceMetrics {
        turn_in_latency_fraction,
        turn_in_latency_ticks,
        throttle_release_latency_fraction,
        throttle_release_latency_ticks,
        steering_adequacy: if curvature_error_values.is_empty() {
            0.0
        } else {
            (1.0 - mean(&curvature_error_values)).clamp(0.0, 1.0)
        },
        high_curvature_throttle_mean: mean(&high_curvature_throttle_values),
        curvature_steering_error_mean: mean(&curvature_error_values),
        curvature_steering_bias_mean: mean(&curvature_bias_values),
        understeer_rate: if understeer_samples == 0 {
            0.0
        } else {
            understeer_count as f32 / understeer_samples as f32
        },
        turn_entry_speed: demand_idx.map(|index| ticks[index].speed),
        peak_curvature_speed: peak_idx.map(|index| ticks[index].speed),
        crash_speed: ticks
            .last()
            .filter(|tick| tick.done_reason.as_deref() == Some("Crash"))
            .map(|tick| tick.speed),
        entry_lateral_offset: demand_idx.map(|index| ticks[index].signed_lateral_offset),
        peak_lateral_offset: peak_idx.map(|index| ticks[index].signed_lateral_offset),
        peak_centerline_distance: peak_idx.map(|index| ticks[index].centerline_distance),
        mean_centerline_distance: mean(&centerline_distances),
        mean_abs_lateral_offset: mean(&abs_lateral_offsets),
        mean_abs_heading_error_deg: mean(&abs_heading_errors_deg),
        mean_all_ray_distance: mean(&all_ray_distances),
        mean_front_ray_distance: mean(&front_ray_distances),
        mean_side_ray_distance: mean(&side_ray_distances),
        failure_mode,
    }
}

pub fn summarize_turn_execution(episodes: &[EpisodeRecord]) -> TurnExecutionSummary {
    let turn_in_latency_fraction_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.turn_in_latency_fraction)
        .collect();
    let turn_in_latency_ticks_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.turn_in_latency_ticks.map(|value| value as f32))
        .collect();
    let throttle_release_latency_fraction_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.throttle_release_latency_fraction)
        .collect();
    let throttle_release_latency_ticks_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| {
            episode
                .throttle_release_latency_ticks
                .map(|value| value as f32)
        })
        .collect();
    let steering_adequacy_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.steering_adequacy)
        .collect();
    let high_curvature_throttle_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.high_curvature_throttle_mean)
        .collect();
    let curvature_error_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.curvature_steering_error_mean)
        .collect();
    let curvature_bias_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.curvature_steering_bias_mean)
        .collect();
    let understeer_values: Vec<f32> = episodes
        .iter()
        .map(|episode| episode.understeer_rate)
        .collect();
    let turn_entry_speed_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.turn_entry_speed)
        .collect();
    let peak_curvature_speed_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.peak_curvature_speed)
        .collect();
    let crash_speed_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.crash_speed)
        .collect();
    let entry_offset_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.entry_lateral_offset)
        .collect();
    let peak_offset_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.peak_lateral_offset)
        .collect();
    let peak_line_gap_values: Vec<f32> = episodes
        .iter()
        .filter_map(|episode| episode.peak_centerline_distance)
        .collect();

    TurnExecutionSummary {
        turn_in_latency_fraction_mean: (!turn_in_latency_fraction_values.is_empty())
            .then_some(mean(&turn_in_latency_fraction_values)),
        turn_in_latency_fraction_p90: (!turn_in_latency_fraction_values.is_empty())
            .then_some(percentile(turn_in_latency_fraction_values.clone(), 0.90)),
        turn_in_latency_ticks_mean: (!turn_in_latency_ticks_values.is_empty())
            .then_some(mean(&turn_in_latency_ticks_values)),
        throttle_release_latency_fraction_mean: (!throttle_release_latency_fraction_values
            .is_empty())
        .then_some(mean(&throttle_release_latency_fraction_values)),
        throttle_release_latency_ticks_mean: (!throttle_release_latency_ticks_values.is_empty())
            .then_some(mean(&throttle_release_latency_ticks_values)),
        steering_adequacy_mean: mean(&steering_adequacy_values),
        high_curvature_throttle_mean: mean(&high_curvature_throttle_values),
        curvature_steering_error_mean: mean(&curvature_error_values),
        curvature_steering_bias_mean: mean(&curvature_bias_values),
        understeer_rate_mean: mean(&understeer_values),
        turn_entry_speed_mean: (!turn_entry_speed_values.is_empty())
            .then_some(mean(&turn_entry_speed_values)),
        turn_entry_speed_p90: (!turn_entry_speed_values.is_empty())
            .then_some(percentile(turn_entry_speed_values.clone(), 0.90)),
        peak_curvature_speed_mean: (!peak_curvature_speed_values.is_empty())
            .then_some(mean(&peak_curvature_speed_values)),
        crash_speed_mean: (!crash_speed_values.is_empty()).then_some(mean(&crash_speed_values)),
        entry_lateral_offset_mean: (!entry_offset_values.is_empty())
            .then_some(mean(&entry_offset_values)),
        peak_lateral_offset_mean: (!peak_offset_values.is_empty())
            .then_some(mean(&peak_offset_values)),
        peak_centerline_distance_mean: (!peak_line_gap_values.is_empty())
            .then_some(mean(&peak_line_gap_values)),
    }
}

pub fn summarize_failure_modes(episodes: &[EpisodeRecord]) -> Vec<FailureModeCount> {
    let total_classified = episodes
        .iter()
        .filter(|episode| episode.failure_mode.is_some())
        .count()
        .max(1);
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for episode in episodes {
        if let Some(label) = &episode.failure_mode {
            *counts.entry(label.clone()).or_default() += 1;
        }
    }

    let mut rows: Vec<_> = counts
        .into_iter()
        .map(|(label, count)| FailureModeCount {
            label,
            count,
            share: count as f32 / total_classified as f32,
        })
        .collect();
    rows.sort_by(|a, b| b.count.cmp(&a.count));
    rows
}

fn curvature_demand(tick: &TickTraceRecord) -> f32 {
    tick.lookahead_curvatures
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max)
}

fn required_steering(tick: &TickTraceRecord) -> f32 {
    let signed_demand = tick.lookahead_curvatures.first().copied().unwrap_or(0.0);
    (signed_demand / STEERING_CURVATURE_GAIN).clamp(-1.0, 1.0)
}

fn front_ray_slice(tick: &TickTraceRecord) -> &[f32] {
    let len = tick.ray_distances.len();
    let safe_start = len.saturating_sub(3) / 2;
    let safe_end = (safe_start + 3).min(len);
    &tick.ray_distances[safe_start..safe_end]
}

fn side_ray_slice(tick: &TickTraceRecord) -> Vec<f32> {
    let len = tick.ray_distances.len();
    if len < 2 {
        return tick.ray_distances.clone();
    }
    let left = 1.min(len - 1);
    let right = len.saturating_sub(2);
    vec![tick.ray_distances[left], tick.ray_distances[right]]
}

fn classify_failure_mode_from_components(
    entry_tick: Option<&TickTraceRecord>,
    peak_tick: Option<&TickTraceRecord>,
    last_tick: Option<&TickTraceRecord>,
    turn_in_latency_ticks: Option<u32>,
    high_curvature_throttle_mean: f32,
    curvature_error_mean: f32,
    understeer_rate: f32,
) -> Option<String> {
    let last_tick = last_tick?;
    if last_tick.done_reason.as_deref() != Some("Crash") {
        return None;
    }

    let entry_speed = entry_tick.map(|tick| tick.speed).unwrap_or(0.0);
    let entry_offset = entry_tick
        .map(|tick| tick.signed_lateral_offset.abs())
        .unwrap_or(0.0);
    let peak_offset = peak_tick
        .map(|tick| tick.signed_lateral_offset.abs())
        .unwrap_or(0.0);
    let peak_gap = peak_tick
        .map(|tick| tick.centerline_distance)
        .unwrap_or(0.0);

    let label = if turn_in_latency_ticks.unwrap_or(0) >= LATE_TURN_IN_TICK_THRESHOLD {
        "Late turn-in"
    } else if understeer_rate >= 0.55 || curvature_error_mean >= 0.45 {
        "Insufficient steering"
    } else if entry_speed >= HIGH_ENTRY_SPEED_THRESHOLD
        && high_curvature_throttle_mean >= HIGH_CURVATURE_THROTTLE_THRESHOLD
    {
        "Excess entry speed"
    } else if high_curvature_throttle_mean >= HIGH_CURVATURE_THROTTLE_THRESHOLD {
        "Throttle held in curve"
    } else if entry_offset >= LARGE_OFFSET_THRESHOLD {
        "Poor turn setup"
    } else if peak_gap >= LARGE_LINE_GAP_THRESHOLD || peak_offset >= LARGE_OFFSET_THRESHOLD {
        "Line loss through corner"
    } else {
        "Mixed corner failure"
    };

    Some(label.to_string())
}

fn wrapped_fraction_delta(from: f32, to: f32) -> f32 {
    let mut delta = to - from;
    if delta < 0.0 {
        delta += 1.0;
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::compute_trace_metrics;
    use crate::analytics::models::TickTraceRecord;

    fn tick(
        progress_fraction: f32,
        speed: f32,
        steering: f32,
        throttle: f32,
        signed_lateral_offset: f32,
        centerline_distance: f32,
        curvature: f32,
    ) -> TickTraceRecord {
        TickTraceRecord {
            tick_index: 0,
            progress_fraction,
            progress_s: 0.0,
            centerline_distance,
            signed_lateral_offset,
            speed,
            heading_error: 0.0,
            steering,
            throttle,
            reward: 0.0,
            progress_reward: 0.0,
            time_penalty: 0.0,
            terminal_reward: 0.0,
            done: false,
            done_reason: None,
            sector_index: 0,
            ray_distances: vec![100.0; 11],
            lookahead_heading_deltas: vec![0.0; 4],
            lookahead_curvatures: vec![curvature, 0.0, 0.0, 0.0],
            value_prediction: None,
        }
    }

    #[test]
    fn trace_metrics_classify_understeer_when_curvature_demand_is_unmet() {
        let mut ticks = vec![
            tick(0.00, 200.0, 0.0, 0.8, 0.0, 4.0, 0.0),
            tick(0.01, 220.0, 0.0, 0.8, 3.0, 6.0, 0.02),
            tick(0.02, 230.0, 0.0, 0.8, 6.0, 9.0, 0.02),
            tick(0.03, 240.0, 0.25, 0.2, 8.0, 11.0, 0.02),
        ];
        ticks.last_mut().unwrap().done = true;
        ticks.last_mut().unwrap().done_reason = Some("Crash".to_string());

        let metrics = compute_trace_metrics(&ticks);
        assert_eq!(
            metrics.failure_mode.as_deref(),
            Some("Insufficient steering")
        );
        assert_eq!(metrics.turn_in_latency_ticks, Some(2));
    }
}
