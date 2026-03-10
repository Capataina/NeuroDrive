use std::f32::consts::PI;

use bevy::prelude::*;

use crate::agent::action::ActionState;
use crate::agent::observation::{ObservationConfig, SensorReadings};
use crate::analytics::metrics::turns::compute_trace_metrics;
use crate::analytics::models::{EpisodeTrace, TickTraceRecord};
use crate::brain::a2c::A2cBrain;
use crate::brain::types::AgentMode;
use crate::game::car::Car;
use crate::game::episode::{EpisodeEndReason, EpisodeState};
use crate::maps::track::Track;

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

    pub fn take_completed_trace(&mut self, episode_id: u32) -> Option<EpisodeTrace> {
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

pub fn capture_episode_tick_trace_system(
    mode: Res<AgentMode>,
    episode_state: Res<EpisodeState>,
    action_state: Res<ActionState>,
    observation_config: Res<ObservationConfig>,
    sensor_query: Query<&SensorReadings, With<Car>>,
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

    let Ok(sensors) = sensor_query.single() else {
        return;
    };

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
        centerline_distance: episode_state.current_tick_centerline_distance,
        signed_lateral_offset: sensors.signed_lateral_offset,
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
        ray_distances: sensors.ray_distances.to_vec(),
        lookahead_heading_deltas,
        lookahead_curvatures,
        value_prediction,
    });
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
        let heading_delta =
            signed_angle_between(episode_state.current_tick_forward, lookahead_tangent);
        let turn_delta =
            signed_angle_between(episode_state.current_tick_tangent, lookahead_tangent);
        let curvature = turn_delta / distance.max(1.0);

        heading_deltas.push(heading_delta);
        curvatures.push(curvature);
    }

    (heading_deltas, curvatures)
}

fn progress_to_sector(progress_fraction: f32) -> u32 {
    let idx = (progress_fraction.clamp(0.0, 0.999_999)
        * crate::analytics::models::NUM_PROGRESS_SECTORS as f32)
        .floor() as usize;
    idx.min(crate::analytics::models::NUM_PROGRESS_SECTORS.saturating_sub(1)) as u32
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
