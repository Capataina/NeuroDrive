use bevy::prelude::*;

use crate::analytics::models::{A2cLayerRecord, A2cUpdateRecord, EpisodeRecord, EpisodeTracker};
use crate::analytics::trackers::action::EpisodeActionAccumulator;
use crate::analytics::trackers::trace::EpisodeTraceAccumulator;
use crate::brain::a2c::A2cTrainingStats;
use crate::game::episode::{EpisodeEndReason, EpisodeState};

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
                turn_in_latency_ticks: trace_metrics.turn_in_latency_ticks,
                throttle_release_latency_fraction: trace_metrics.throttle_release_latency_fraction,
                throttle_release_latency_ticks: trace_metrics.throttle_release_latency_ticks,
                steering_adequacy: trace_metrics.steering_adequacy,
                high_curvature_throttle_mean: trace_metrics.high_curvature_throttle_mean,
                curvature_steering_error_mean: trace_metrics.curvature_steering_error_mean,
                curvature_steering_bias_mean: trace_metrics.curvature_steering_bias_mean,
                understeer_rate: trace_metrics.understeer_rate,
                turn_entry_speed: trace_metrics.turn_entry_speed,
                peak_curvature_speed: trace_metrics.peak_curvature_speed,
                crash_speed: trace_metrics.crash_speed,
                entry_lateral_offset: trace_metrics.entry_lateral_offset,
                peak_lateral_offset: trace_metrics.peak_lateral_offset,
                peak_centerline_distance: trace_metrics.peak_centerline_distance,
                mean_centerline_distance: trace_metrics.mean_centerline_distance,
                mean_abs_lateral_offset: trace_metrics.mean_abs_lateral_offset,
                mean_abs_heading_error_deg: trace_metrics.mean_abs_heading_error_deg,
                mean_all_ray_distance: trace_metrics.mean_all_ray_distance,
                mean_front_ray_distance: trace_metrics.mean_front_ray_distance,
                mean_side_ray_distance: trace_metrics.mean_side_ray_distance,
                failure_mode: trace_metrics.failure_mode,
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
