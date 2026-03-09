use std::f32::consts::PI;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent::action::ActionState;
use crate::agent::observation::ObservationConfig;
use crate::brain::a2c::{A2cBrain, A2cTrainingStats};
use crate::brain::types::AgentMode;
use crate::game::episode::{EpisodeEndReason, EpisodeState};
use crate::maps::track::Track;

pub const NUM_PROGRESS_SECTORS: usize = 20;

const CURVATURE_DEMAND_THRESHOLD: f32 = 0.015;
const STEERING_ONSET_THRESHOLD: f32 = 0.18;
const THROTTLE_RELEASE_THRESHOLD: f32 = 0.35;
const STEERING_CURVATURE_GAIN: f32 = 0.03;

/// Exported analytics snapshot for a completed episode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub episode_id: u32,
    pub progress: f32,
    pub reward: f32,
    pub pre_terminal_return: f32,
    pub progress_reward_sum: f32,
    pub time_penalty_sum: f32,
    pub terminal_reward_sum: f32,
    pub crash_penalty_sum: f32,
    pub lap_bonus_sum: f32,
    pub ticks: u32,
    pub crashes: u32,
    pub end_reason: String,
    pub lap_completed: bool,
    pub crash_position: Option<[f32; 2]>,
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
    pub turn_in_latency_fraction: Option<f32>,
    pub throttle_release_latency_fraction: Option<f32>,
    pub steering_adequacy: f32,
    pub high_curvature_throttle_mean: f32,
}

/// Tick-level trajectory analytics record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickTraceRecord {
    pub tick_index: u32,
    pub progress_fraction: f32,
    pub progress_s: f32,
    pub speed: f32,
    pub heading_error: f32,
    pub steering: f32,
    pub throttle: f32,
    pub reward: f32,
    pub progress_reward: f32,
    pub time_penalty: f32,
    pub terminal_reward: f32,
    pub done: bool,
    pub done_reason: Option<String>,
    pub sector_index: u32,
    pub lookahead_heading_deltas: Vec<f32>,
    pub lookahead_curvatures: Vec<f32>,
    pub value_prediction: Option<f32>,
}

/// Episode-level trajectory trace with derived control mismatch metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeTrace {
    pub episode_id: u32,
    pub end_reason: String,
    pub lap_completed: bool,
    pub best_progress: f32,
    pub ticks: Vec<TickTraceRecord>,
    pub metrics: EpisodeTraceMetrics,
}

/// Derived mismatch metrics from one episode trace.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EpisodeTraceMetrics {
    pub turn_in_latency_fraction: Option<f32>,
    pub throttle_release_latency_fraction: Option<f32>,
    pub steering_adequacy: f32,
    pub high_curvature_throttle_mean: f32,
}

/// Exported analytics snapshot for one layer after a completed A2C update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct A2cLayerRecord {
    pub layer_name: String,
    pub weight_l2_norm: f32,
    pub gradient_l2_norm: f32,
    pub dead_relu_fraction: Option<f32>,
}

/// Exported analytics snapshot for one completed A2C update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct A2cUpdateRecord {
    pub update_index: u64,
    pub batch_size: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub policy_entropy: f32,
    pub explained_variance: f32,
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
    pub clamped_action_fraction: f32,
    pub layer_health: Vec<A2cLayerRecord>,
}

/// Exported run-level analytics data.
#[derive(Resource, Default, Debug, Serialize, Deserialize)]
pub struct EpisodeTracker {
    pub episodes: Vec<EpisodeRecord>,
    pub a2c_updates: Vec<A2cUpdateRecord>,
    pub episode_traces: Vec<EpisodeTrace>,
    #[serde(skip)]
    pub last_recorded_update: u64,
}

/// Running action statistics for the currently active episode.
#[derive(Clone, Debug, Default)]
pub struct EpisodeActionSummary {
    pub episode_id: u32,
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
}

/// Running action statistics for the currently active episode.
#[derive(Resource, Debug)]
pub struct EpisodeActionAccumulator {
    pub episode_id: u32,
    pub steps: u32,
    pub steering_sum: f32,
    pub steering_sumsq: f32,
    pub throttle_sum: f32,
    pub throttle_sumsq: f32,
    pub last_completed_episode_id: Option<u32>,
    pub last_completed_summary: Option<EpisodeActionSummary>,
}

impl Default for EpisodeActionAccumulator {
    fn default() -> Self {
        Self {
            episode_id: 1,
            steps: 0,
            steering_sum: 0.0,
            steering_sumsq: 0.0,
            throttle_sum: 0.0,
            throttle_sumsq: 0.0,
            last_completed_episode_id: None,
            last_completed_summary: None,
        }
    }
}

impl EpisodeActionAccumulator {
    fn reset_for_episode(&mut self, episode_id: u32) {
        self.episode_id = episode_id;
        self.steps = 0;
        self.steering_sum = 0.0;
        self.steering_sumsq = 0.0;
        self.throttle_sum = 0.0;
        self.throttle_sumsq = 0.0;
    }

    fn record_step(&mut self, action_state: &ActionState) {
        let steering = action_state.applied.steering;
        let throttle = action_state.applied.throttle;
        self.steps = self.steps.saturating_add(1);
        self.steering_sum += steering;
        self.steering_sumsq += steering * steering;
        self.throttle_sum += throttle;
        self.throttle_sumsq += throttle * throttle;
    }

    fn steering_mean(&self) -> f32 {
        mean_from_sum(self.steering_sum, self.steps)
    }

    fn steering_std(&self) -> f32 {
        std_from_sum_and_sumsq(self.steering_sum, self.steering_sumsq, self.steps)
    }

    fn throttle_mean(&self) -> f32 {
        mean_from_sum(self.throttle_sum, self.steps)
    }

    fn throttle_std(&self) -> f32 {
        std_from_sum_and_sumsq(self.throttle_sum, self.throttle_sumsq, self.steps)
    }

    fn snapshot_completed_episode(&mut self, episode_id: u32) {
        self.last_completed_episode_id = Some(episode_id);
        self.last_completed_summary = Some(EpisodeActionSummary {
            episode_id,
            steering_mean: self.steering_mean(),
            steering_std: self.steering_std(),
            throttle_mean: self.throttle_mean(),
            throttle_std: self.throttle_std(),
        });
    }

    fn take_completed_summary(&mut self, episode_id: u32) -> Option<EpisodeActionSummary> {
        if self
            .last_completed_summary
            .as_ref()
            .map(|summary| summary.episode_id)
            == Some(episode_id)
        {
            self.last_completed_summary.take()
        } else {
            None
        }
    }
}

/// Running per-tick trace for the active episode.
#[derive(Resource, Debug)]
pub struct EpisodeTraceAccumulator {
    pub episode_id: u32,
    pub ticks: Vec<TickTraceRecord>,
    pub last_completed_episode_id: Option<u32>,
    pub last_completed_trace: Option<EpisodeTrace>,
}

impl Default for EpisodeTraceAccumulator {
    fn default() -> Self {
        Self {
            episode_id: 1,
            ticks: Vec::new(),
            last_completed_episode_id: None,
            last_completed_trace: None,
        }
    }
}

impl EpisodeTraceAccumulator {
    fn reset_for_episode(&mut self, episode_id: u32) {
        self.episode_id = episode_id;
        self.ticks.clear();
    }

    fn snapshot_completed_episode(
        &mut self,
        episode_id: u32,
        end_reason: EpisodeEndReason,
        best_progress: f32,
    ) {
        let metrics = compute_trace_metrics(&self.ticks);
        self.last_completed_episode_id = Some(episode_id);
        self.last_completed_trace = Some(EpisodeTrace {
            episode_id,
            end_reason: format!("{:?}", end_reason),
            lap_completed: end_reason == EpisodeEndReason::LapComplete,
            best_progress,
            ticks: self.ticks.clone(),
            metrics,
        });
    }

    fn take_completed_trace(&mut self, episode_id: u32) -> Option<EpisodeTrace> {
        if self
            .last_completed_trace
            .as_ref()
            .map(|trace| trace.episode_id)
            == Some(episode_id)
        {
            self.last_completed_trace.take()
        } else {
            None
        }
    }
}

pub fn capture_episode_action_stats_system(
    episode_state: Res<EpisodeState>,
    action_state: Res<ActionState>,
    mut accumulator: ResMut<EpisodeActionAccumulator>,
) {
    if accumulator.episode_id != episode_state.current_episode {
        accumulator.reset_for_episode(episode_state.current_episode);
    }

    accumulator.record_step(&action_state);
}

pub fn capture_episode_tick_trace_system(
    mode: Res<AgentMode>,
    episode_state: Res<EpisodeState>,
    action_state: Res<ActionState>,
    observation_config: Res<ObservationConfig>,
    track_query: Query<&Track>,
    a2c_brain: Option<Res<A2cBrain>>,
    mut accumulator: ResMut<EpisodeTraceAccumulator>,
) {
    let done = episode_state.current_tick_end_reason.is_some();
    let target_episode_id = if done {
        episode_state.current_episode.saturating_sub(1)
    } else {
        episode_state.current_episode
    };

    if accumulator.episode_id != target_episode_id {
        accumulator.reset_for_episode(target_episode_id);
    }

    let (lookahead_heading_deltas, lookahead_curvatures) = if let Ok(track) = track_query.single() {
        compute_lookahead_snapshot(track, &episode_state, &observation_config)
    } else {
        (
            vec![0.0; observation_config.lookahead_distances.len()],
            vec![0.0; observation_config.lookahead_distances.len()],
        )
    };

    let value_prediction = if *mode == AgentMode::Ai {
        a2c_brain.and_then(|brain| {
            let values_len = brain.buffer.values.len();
            let rewards_len = brain.buffer.rewards.len();
            if !brain.buffer.values.is_empty()
                && (values_len == rewards_len || values_len == rewards_len.saturating_add(1))
            {
                brain.buffer.values.last().copied()
            } else {
                None
            }
        })
    } else {
        None
    };

    let tick_index = accumulator.ticks.len() as u32 + 1;
    accumulator.ticks.push(TickTraceRecord {
        tick_index,
        progress_fraction: episode_state.current_tick_progress_fraction,
        progress_s: episode_state.current_tick_progress_s,
        speed: episode_state.current_tick_speed,
        heading_error: episode_state.current_tick_heading_error,
        steering: action_state.applied.steering,
        throttle: action_state.applied.throttle,
        reward: episode_state.current_tick_reward,
        progress_reward: episode_state.current_tick_progress_reward,
        time_penalty: episode_state.current_tick_time_penalty,
        terminal_reward: episode_state.current_tick_terminal_reward,
        done,
        done_reason: episode_state
            .current_tick_end_reason
            .map(|reason| format!("{reason:?}")),
        sector_index: progress_to_sector(episode_state.current_tick_progress_fraction),
        lookahead_heading_deltas,
        lookahead_curvatures,
        value_prediction,
    });
}

pub fn snapshot_completed_episode_action_stats_system(
    episode_state: Res<EpisodeState>,
    mut accumulator: ResMut<EpisodeActionAccumulator>,
) {
    let Some(_reason) = episode_state.current_tick_end_reason else {
        return;
    };

    let finished_episode_id = episode_state.current_episode.saturating_sub(1);
    if accumulator.last_completed_episode_id == Some(finished_episode_id) {
        return;
    }

    accumulator.snapshot_completed_episode(finished_episode_id);
}

pub fn snapshot_completed_episode_trace_system(
    episode_state: Res<EpisodeState>,
    mut accumulator: ResMut<EpisodeTraceAccumulator>,
) {
    let Some(reason) = episode_state.current_tick_end_reason else {
        return;
    };

    let finished_episode_id = episode_state.current_episode.saturating_sub(1);
    if accumulator.last_completed_episode_id == Some(finished_episode_id) {
        return;
    }

    if accumulator.episode_id != finished_episode_id {
        return;
    }

    accumulator.snapshot_completed_episode(
        finished_episode_id,
        reason,
        episode_state.last_episode_best_progress_fraction,
    );
    accumulator.reset_for_episode(episode_state.current_episode);
}

pub fn episode_tracker_system(
    episode_state: Res<EpisodeState>,
    a2c_stats: Option<Res<A2cTrainingStats>>,
    mut action_accumulator: ResMut<EpisodeActionAccumulator>,
    mut trace_accumulator: ResMut<EpisodeTraceAccumulator>,
    mut tracker: ResMut<EpisodeTracker>,
) {
    if let Some(reason) = episode_state.last_end_reason {
        let finished_episode_id = episode_state.current_episode - 1;

        if tracker.episodes.last().map(|r| r.episode_id) != Some(finished_episode_id) {
            let action_summary = action_accumulator
                .take_completed_summary(finished_episode_id)
                .unwrap_or_default();
            let trace = trace_accumulator.take_completed_trace(finished_episode_id);
            let trace_metrics = trace
                .as_ref()
                .map(|episode_trace| episode_trace.metrics.clone())
                .unwrap_or_default();

            if let Some(trace) = trace {
                if tracker
                    .episode_traces
                    .last()
                    .map(|record| record.episode_id)
                    != Some(trace.episode_id)
                {
                    tracker.episode_traces.push(trace);
                }
            }

            tracker.episodes.push(EpisodeRecord {
                episode_id: finished_episode_id,
                progress: episode_state.last_episode_best_progress_fraction,
                reward: episode_state.last_episode_return,
                pre_terminal_return: episode_state.last_episode_pre_terminal_return,
                progress_reward_sum: episode_state.last_episode_progress_reward_sum,
                time_penalty_sum: episode_state.last_episode_time_penalty_sum,
                terminal_reward_sum: episode_state.last_episode_terminal_reward_sum,
                crash_penalty_sum: episode_state.last_episode_crash_penalty_sum,
                lap_bonus_sum: episode_state.last_episode_lap_bonus_sum,
                ticks: episode_state.last_episode_ticks,
                crashes: episode_state.last_episode_crashes,
                end_reason: format!("{:?}", reason),
                lap_completed: reason == EpisodeEndReason::LapComplete,
                crash_position: episode_state
                    .last_episode_crash_position
                    .map(|position| [position.x, position.y]),
                steering_mean: action_summary.steering_mean,
                steering_std: action_summary.steering_std,
                throttle_mean: action_summary.throttle_mean,
                throttle_std: action_summary.throttle_std,
                turn_in_latency_fraction: trace_metrics.turn_in_latency_fraction,
                throttle_release_latency_fraction: trace_metrics.throttle_release_latency_fraction,
                steering_adequacy: trace_metrics.steering_adequacy,
                high_curvature_throttle_mean: trace_metrics.high_curvature_throttle_mean,
            });
        }
    }

    if let Some(a2c_stats) = a2c_stats {
        if a2c_stats.last_completed_update > tracker.last_recorded_update {
            tracker.last_recorded_update = a2c_stats.last_completed_update;
            tracker.a2c_updates.push(A2cUpdateRecord {
                update_index: a2c_stats.last_completed_update,
                batch_size: a2c_stats.batch_size,
                policy_loss: a2c_stats.policy_loss,
                value_loss: a2c_stats.value_loss,
                policy_entropy: a2c_stats.policy_entropy,
                explained_variance: a2c_stats.explained_variance,
                steering_mean: a2c_stats.steering_mean,
                steering_std: a2c_stats.steering_std,
                throttle_mean: a2c_stats.throttle_mean,
                throttle_std: a2c_stats.throttle_std,
                clamped_action_fraction: a2c_stats.clamped_action_fraction,
                layer_health: a2c_stats
                    .layer_health
                    .iter()
                    .map(|layer| A2cLayerRecord {
                        layer_name: layer.layer_name.clone(),
                        weight_l2_norm: layer.weight_l2_norm,
                        gradient_l2_norm: layer.gradient_l2_norm,
                        dead_relu_fraction: layer.dead_relu_fraction,
                    })
                    .collect(),
            });
        }
    }
}

fn compute_lookahead_snapshot(
    track: &Track,
    episode_state: &EpisodeState,
    observation_config: &ObservationConfig,
) -> (Vec<f32>, Vec<f32>) {
    let mut heading_deltas = Vec::with_capacity(observation_config.lookahead_distances.len());
    let mut curvatures = Vec::with_capacity(observation_config.lookahead_distances.len());

    for distance in observation_config.lookahead_distances.iter() {
        let lookahead_s = episode_state.current_tick_progress_s + *distance;
        let lookahead_tangent = track.centerline.tangent_at_s(lookahead_s);
        let heading_delta = signed_angle_between(episode_state.current_tick_forward, lookahead_tangent);
        let turn_delta = signed_angle_between(episode_state.current_tick_tangent, lookahead_tangent);
        let curvature = turn_delta / distance.max(1.0);

        heading_deltas.push(heading_delta);
        curvatures.push(curvature);
    }

    (heading_deltas, curvatures)
}

fn compute_trace_metrics(ticks: &[TickTraceRecord]) -> EpisodeTraceMetrics {
    if ticks.is_empty() {
        return EpisodeTraceMetrics::default();
    }

    let mut demand_idx: Option<usize> = None;
    for (index, tick) in ticks.iter().enumerate() {
        if curvature_demand(tick) >= CURVATURE_DEMAND_THRESHOLD {
            demand_idx = Some(index);
            break;
        }
    }

    let mut turn_in_latency_fraction = None;
    let mut throttle_release_latency_fraction = None;

    if let Some(demand_idx) = demand_idx {
        if let Some(steer_idx) = ticks
            .iter()
            .enumerate()
            .skip(demand_idx)
            .find_map(|(index, tick)| {
                if tick.steering.abs() >= STEERING_ONSET_THRESHOLD {
                    Some(index)
                } else {
                    None
                }
            })
        {
            turn_in_latency_fraction = Some(wrapped_fraction_delta(
                ticks[demand_idx].progress_fraction,
                ticks[steer_idx].progress_fraction,
            ));
        }

        if let Some(release_idx) = ticks
            .iter()
            .enumerate()
            .skip(demand_idx)
            .find_map(|(index, tick)| {
                if tick.throttle <= THROTTLE_RELEASE_THRESHOLD {
                    Some(index)
                } else {
                    None
                }
            })
        {
            throttle_release_latency_fraction = Some(wrapped_fraction_delta(
                ticks[demand_idx].progress_fraction,
                ticks[release_idx].progress_fraction,
            ));
        }
    }

    let mut high_curvature_throttle_sum = 0.0;
    let mut high_curvature_count = 0usize;
    let mut adequacy_error_sum = 0.0;
    let mut adequacy_count = 0usize;
    for tick in ticks {
        let abs_demand = curvature_demand(tick);
        if abs_demand >= CURVATURE_DEMAND_THRESHOLD {
            high_curvature_throttle_sum += tick.throttle;
            high_curvature_count += 1;

            let signed_demand = tick.lookahead_curvatures.first().copied().unwrap_or(0.0);
            let required_steering = (signed_demand / STEERING_CURVATURE_GAIN).clamp(-1.0, 1.0);
            adequacy_error_sum += (required_steering - tick.steering).abs();
            adequacy_count += 1;
        }
    }

    let high_curvature_throttle_mean = if high_curvature_count == 0 {
        0.0
    } else {
        high_curvature_throttle_sum / high_curvature_count as f32
    };

    let steering_adequacy = if adequacy_count == 0 {
        0.0
    } else {
        (1.0 - (adequacy_error_sum / adequacy_count as f32)).clamp(0.0, 1.0)
    };

    EpisodeTraceMetrics {
        turn_in_latency_fraction,
        throttle_release_latency_fraction,
        steering_adequacy,
        high_curvature_throttle_mean,
    }
}

fn progress_to_sector(progress_fraction: f32) -> u32 {
    let idx = (progress_fraction.clamp(0.0, 0.999_999) * NUM_PROGRESS_SECTORS as f32).floor() as usize;
    idx.min(NUM_PROGRESS_SECTORS.saturating_sub(1)) as u32
}

fn curvature_demand(tick: &TickTraceRecord) -> f32 {
    tick.lookahead_curvatures
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max)
}

fn wrapped_fraction_delta(from: f32, to: f32) -> f32 {
    let mut delta = to - from;
    if delta < 0.0 {
        delta += 1.0;
    }
    delta
}

fn signed_angle_between(from: Vec2, to: Vec2) -> f32 {
    let from_n = from.normalize_or_zero();
    let to_n = to.normalize_or_zero();
    if from_n == Vec2::ZERO || to_n == Vec2::ZERO {
        return 0.0;
    }
    wrap_angle(to_n.to_angle() - from_n.to_angle())
}

fn wrap_angle(mut angle: f32) -> f32 {
    while angle > PI {
        angle -= 2.0 * PI;
    }
    while angle < -PI {
        angle += 2.0 * PI;
    }
    angle
}

fn mean_from_sum(sum: f32, count: u32) -> f32 {
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn std_from_sum_and_sumsq(sum: f32, sumsq: f32, count: u32) -> f32 {
    if count == 0 {
        return 0.0;
    }

    let n = count as f32;
    let mean = sum / n;
    ((sumsq / n) - mean * mean).max(0.0).sqrt()
}
