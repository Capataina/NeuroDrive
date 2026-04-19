use std::collections::HashMap;

use bevy::prelude::*;

use crate::agent::action::ActionState;
use crate::agent::observation::{ObservationConfig, SensorReadings};
use crate::analytics::metrics::turns::compute_trace_metrics;
use crate::analytics::models::{EpisodeTrace, TickTraceRecord};
use crate::brain::types::{BrainCar, PolicyOutput, PpoCar};
use crate::game::car::{Car, EnvInstanceId};
use crate::game::episode::{EpisodeEndReason, EpisodeState};
use crate::maps::track::Track;
use crate::sim::signed_angle_between;

/// Running per-tick trace for the active episode.
#[derive(Debug)]
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
            best_progress,
            ticks: self.ticks.clone(),
            metrics,
        });
    }
}

/// Per-car trace accumulators keyed by env_id.
#[derive(Resource, Debug, Default)]
pub struct PerCarTraceAccumulators {
    pub accumulators: HashMap<u32, EpisodeTraceAccumulator>,
}

impl PerCarTraceAccumulators {
    /// Takes the completed trace for a specific car and episode,
    /// returning it if one is ready.
    pub fn take_completed_trace(
        &mut self,
        env_id: u32,
        episode_id: u32,
    ) -> Option<EpisodeTrace> {
        let accumulator = self.accumulators.get_mut(&env_id)?;
        if accumulator
            .last_completed_trace
            .as_ref()
            .map(|t| t.episode_id)
            == Some(episode_id)
        {
            accumulator.last_completed_trace.take()
        } else {
            None
        }
    }
}

/// Captures per-tick trajectory data for every car's active episode.
///
/// `Option<&PpoCar>` and `Option<&BrainCar>` markers discriminate which
/// controller drives this car so we know whether `PolicyOutput` fields
/// represent a Gaussian policy (PPO) or raw output-neuron activations
/// (brain-inspired) — both paths populate the `policy_*` fields, but the
/// semantics differ and analytics downstream needs to distinguish.
pub fn capture_episode_tick_trace_system(
    observation_config: Res<ObservationConfig>,
    car_query: Query<
        (
            &EnvInstanceId,
            &EpisodeState,
            &ActionState,
            &SensorReadings,
            &Transform,
            &PolicyOutput,
            Option<&PpoCar>,
            Option<&BrainCar>,
        ),
        With<Car>,
    >,
    track_query: Query<&Track>,
    mut accumulators: ResMut<PerCarTraceAccumulators>,
) {
    for (env_id, episode_state, action_state, sensors, transform, policy_output, is_ppo, is_brain) in car_query.iter() {
        let done = episode_state.tick.end_reason.is_some();
        let target_episode_id = if done {
            episode_state.current_episode.saturating_sub(1)
        } else {
            episode_state.current_episode
        };

        let accumulator = accumulators
            .accumulators
            .entry(env_id.0)
            .or_default();

        if accumulator.episode_id != target_episode_id {
            accumulator.reset_for_episode(target_episode_id);
        }

        let (lookahead_heading_deltas, lookahead_curvatures) =
            if let Ok(track) = track_query.single() {
                compute_lookahead_snapshot(track, episode_state, &observation_config)
            } else {
                ([0.0f32; 12], [0.0f32; 12])
            };

        // Both PPO (value prediction) and brain-inspired (modulator M, per S2)
        // write `value_prediction` into `PolicyOutput`. Keyboard-driven cars
        // leave it untouched so we suppress it rather than record a stale value.
        let controller_active = is_ppo.is_some() || is_brain.is_some();
        let value_prediction = if controller_active {
            Some(policy_output.value_prediction)
        } else {
            None
        };

        let is_ai = controller_active;

        let tick_index = accumulator.ticks.len() as u32 + 1;
        accumulator.ticks.push(TickTraceRecord {
            env_id: env_id.0,
            tick_index,
            position_x: transform.translation.x,
            position_y: transform.translation.y,
            progress_fraction: episode_state.tick.progress_fraction,
            progress_s: episode_state.tick.progress_s,
            centerline_distance: episode_state.tick.centerline_distance,
            signed_lateral_offset: sensors.signed_lateral_offset,
            speed: episode_state.tick.speed,
            v_forward: sensors.v_forward,
            v_lateral: sensors.v_lateral,
            speed_delta: sensors.speed_delta,
            drift_angle_deg: {
                let spd = sensors.speed;
                if spd > 1.0 {
                    let vf = sensors.v_forward;
                    let vl = sensors.v_lateral;
                    vl.atan2(vf.abs().max(0.001)).abs().to_degrees()
                } else {
                    0.0
                }
            },
            heading_error: episode_state.tick.heading_error,
            min_ray_distance: sensors.ray_distances.iter().copied().fold(f32::MAX, f32::min),
            velocity_projection: episode_state.tick.velocity_projection,
            centreline_reward: episode_state.tick.centreline_reward,
            steering: action_state.applied.steering,
            throttle: action_state.applied.throttle,
            previous_steering: sensors.previous_steering,
            previous_throttle: sensors.previous_throttle,
            reward: episode_state.tick.reward,
            progress_reward: episode_state.tick.progress_reward,
            time_penalty: episode_state.tick.time_penalty,
            terminal_reward: episode_state.tick.terminal_reward,
            done,
            done_reason: episode_state
                .tick.end_reason
                .map(|reason| format!("{reason:?}")),
            sector_index: progress_to_sector(episode_state.tick.progress_fraction),
            ray_distances: sensors.ray_distances,
            lookahead_heading_deltas,
            lookahead_curvatures,
            value_prediction,
            policy_steering_mean: if is_ai { Some(policy_output.steering_mean) } else { None },
            policy_steering_std: if is_ai { Some(policy_output.steering_std) } else { None },
            policy_throttle_mean: if is_ai { Some(policy_output.throttle_mean) } else { None },
            policy_throttle_std: if is_ai { Some(policy_output.throttle_std) } else { None },
        });
    }
}

/// Snapshots completed episode traces for every car that finished an episode this tick.
pub fn snapshot_completed_episode_trace_system(
    car_query: Query<(&EnvInstanceId, &EpisodeState), With<Car>>,
    mut accumulators: ResMut<PerCarTraceAccumulators>,
) {
    for (env_id, episode_state) in car_query.iter() {
        let Some(reason) = episode_state.tick.end_reason else {
            continue;
        };

        let finished_episode_id = episode_state.current_episode.saturating_sub(1);

        let accumulator = accumulators
            .accumulators
            .entry(env_id.0)
            .or_default();

        if accumulator.last_completed_episode_id == Some(finished_episode_id) {
            continue;
        }

        if accumulator.episode_id != finished_episode_id {
            continue;
        }

        accumulator.snapshot_completed_episode(
            finished_episode_id,
            reason,
            episode_state.last.best_progress_fraction,
        );
        accumulator.reset_for_episode(episode_state.current_episode);
    }
}

fn compute_lookahead_snapshot(
    track: &Track,
    episode_state: &EpisodeState,
    observation_config: &ObservationConfig,
) -> ([f32; 12], [f32; 12]) {
    let mut heading_deltas = [0.0f32; 12];
    let mut curvatures = [0.0f32; 12];

    for (index, distance) in observation_config.lookahead_distances.iter().enumerate() {
        let lookahead_s = episode_state.tick.progress_s + *distance;
        let lookahead_tangent = track.centerline.tangent_at_s(lookahead_s);
        let heading_delta =
            signed_angle_between(episode_state.tick.forward, lookahead_tangent);
        let turn_delta =
            signed_angle_between(episode_state.tick.tangent, lookahead_tangent);
        let curvature = turn_delta / distance.max(1.0);

        heading_deltas[index] = heading_delta;
        curvatures[index] = curvature;
    }

    (heading_deltas, curvatures)
}

fn progress_to_sector(progress_fraction: f32) -> u32 {
    let idx = (progress_fraction.clamp(0.0, 0.999_999)
        * crate::analytics::models::NUM_PROGRESS_SECTORS as f32)
        .floor() as usize;
    idx.min(crate::analytics::models::NUM_PROGRESS_SECTORS.saturating_sub(1)) as u32
}

