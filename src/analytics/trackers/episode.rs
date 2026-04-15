use bevy::prelude::*;

use crate::analytics::models::{
    CrashKind, PpoLayerRecord, PpoUpdateRecord, EpisodeRecord, EpisodeTrace, EpisodeTracker,
    TickTraceRecord,
};
use crate::analytics::trackers::action::PerCarActionAccumulators;
use crate::analytics::trackers::trace::PerCarTraceAccumulators;
use crate::brain::ppo::PpoTrainingStats;
use crate::game::car::{Car, EnvInstanceId};
use crate::game::episode::EpisodeState;

/// Classifies a crash based on terminal tick kinematics.
fn classify_crash(tick: &TickTraceRecord) -> CrashKind {
    let speed = tick.speed;
    let vf = tick.v_forward;
    let vl = tick.v_lateral.abs();
    let drift = tick.drift_angle_deg;
    let heading_err = tick.heading_error.abs().to_degrees();

    if speed < 50.0 {
        CrashKind::Stall
    } else if drift > 60.0 {
        CrashKind::Spin
    } else if vl > vf.abs() {
        CrashKind::Slide
    } else if heading_err > 30.0 {
        CrashKind::Overshoot
    } else {
        CrashKind::HeadOn
    }
}

/// Computes episode-level aggregates from the tick trace.
fn compute_trace_aggregates(trace: &EpisodeTrace, is_crash: bool) -> TraceAggregates {
    let ticks = &trace.ticks;
    if ticks.is_empty() {
        return TraceAggregates::default();
    }

    let n = ticks.len() as f32;
    let mut agg = TraceAggregates::default();

    let mut speed_sum = 0.0f32;
    let mut peak_speed = 0.0f32;
    let mut vf_sum = 0.0f32;
    let mut vl_abs_sum = 0.0f32;
    let mut vp_sum = 0.0f32;
    let mut drift_sum = 0.0f32;
    let mut peak_drift = 0.0f32;
    let mut braking_ticks = 0u32;
    let mut accel_ticks = 0u32;
    let mut coasting_ticks = 0u32;
    let mut action_change_sum = 0.0f32;
    let mut wall_prox_ticks = 0u32;
    let mut value_sum = 0.0f32;
    let mut value_count = 0u32;
    let mut policy_steer_std_sum = 0.0f32;
    let mut policy_throttle_std_sum = 0.0f32;
    let mut policy_count = 0u32;
    let mut furthest_sector = 0u32;

    for (i, tick) in ticks.iter().enumerate() {
        speed_sum += tick.speed;
        if tick.speed > peak_speed { peak_speed = tick.speed; }
        vf_sum += tick.v_forward;
        vl_abs_sum += tick.v_lateral.abs();
        vp_sum += tick.velocity_projection;
        drift_sum += tick.drift_angle_deg;
        if tick.drift_angle_deg > peak_drift { peak_drift = tick.drift_angle_deg; }

        if tick.throttle < -0.1 { braking_ticks += 1; }
        else if tick.throttle > 0.1 { accel_ticks += 1; }
        else { coasting_ticks += 1; }

        if i > 0 {
            let prev = &ticks[i - 1];
            action_change_sum += (tick.steering - prev.steering).abs()
                + (tick.throttle - prev.throttle).abs();
        }

        if tick.min_ray_distance < 30.0 { wall_prox_ticks += 1; }
        if tick.sector_index > furthest_sector { furthest_sector = tick.sector_index; }

        if let Some(v) = tick.value_prediction {
            value_sum += v;
            value_count += 1;
        }
        if let Some(s) = tick.policy_steering_std {
            policy_steer_std_sum += s;
            policy_count += 1;
        }
        if let Some(s) = tick.policy_throttle_std {
            policy_throttle_std_sum += s;
        }
    }

    agg.mean_speed = speed_sum / n;
    agg.peak_speed = peak_speed;
    agg.mean_v_forward = vf_sum / n;
    agg.mean_v_lateral_abs = vl_abs_sum / n;
    agg.mean_velocity_projection = vp_sum / n;
    agg.mean_drift_angle_deg = drift_sum / n;
    agg.peak_drift_angle_deg = peak_drift;
    agg.braking_fraction = braking_ticks as f32 / n;
    agg.acceleration_fraction = accel_ticks as f32 / n;
    agg.coasting_fraction = coasting_ticks as f32 / n;
    agg.mean_action_change = if ticks.len() > 1 {
        action_change_sum / (ticks.len() - 1) as f32
    } else {
        0.0
    };
    agg.wall_proximity_fraction = wall_prox_ticks as f32 / n;
    agg.furthest_sector = furthest_sector;
    agg.reward_per_second = if n > 0.0 {
        ticks.last().map(|_| trace.ticks.iter().map(|t| t.reward).sum::<f32>())
            .unwrap_or(0.0) / (n / 60.0)
    } else {
        0.0
    };

    if value_count > 0 {
        agg.mean_value_prediction = Some(value_sum / value_count as f32);
        agg.value_at_start = ticks.first().and_then(|t| t.value_prediction);
    }

    if policy_count > 0 {
        agg.mean_policy_steering_std = Some(policy_steer_std_sum / policy_count as f32);
        agg.mean_policy_throttle_std = Some(policy_throttle_std_sum / policy_count as f32);
    }

    // Crash forensics from the last tick.
    if is_crash {
        if let Some(last) = ticks.last() {
            agg.crash_speed = Some(last.speed);
            agg.crash_v_forward = Some(last.v_forward);
            agg.crash_v_lateral = Some(last.v_lateral);
            agg.crash_drift_angle_deg = Some(last.drift_angle_deg);
            agg.crash_heading_error_deg = Some(last.heading_error.abs().to_degrees());
            agg.crash_min_ray = Some(last.min_ray_distance);
            agg.crash_type = Some(classify_crash(last));
            agg.value_at_crash = last.value_prediction;
        }
    }

    agg
}

#[derive(Default)]
struct TraceAggregates {
    mean_speed: f32,
    peak_speed: f32,
    mean_v_forward: f32,
    mean_v_lateral_abs: f32,
    mean_velocity_projection: f32,
    mean_drift_angle_deg: f32,
    peak_drift_angle_deg: f32,
    braking_fraction: f32,
    acceleration_fraction: f32,
    coasting_fraction: f32,
    mean_action_change: f32,
    wall_proximity_fraction: f32,
    furthest_sector: u32,
    reward_per_second: f32,
    crash_speed: Option<f32>,
    crash_v_forward: Option<f32>,
    crash_v_lateral: Option<f32>,
    crash_drift_angle_deg: Option<f32>,
    crash_heading_error_deg: Option<f32>,
    crash_min_ray: Option<f32>,
    crash_type: Option<CrashKind>,
    mean_value_prediction: Option<f32>,
    value_at_crash: Option<f32>,
    value_at_start: Option<f32>,
    mean_policy_steering_std: Option<f32>,
    mean_policy_throttle_std: Option<f32>,
}

/// Folds completed episodes from all cars into EpisodeTracker, tagged with env_id.
/// Also records PPO update snapshots from the shared training stats resource.
pub fn episode_tracker_system(
    car_query: Query<(&EnvInstanceId, &EpisodeState), With<Car>>,
    ppo_stats: Option<Res<PpoTrainingStats>>,
    mut action_accumulators: ResMut<PerCarActionAccumulators>,
    mut trace_accumulators: ResMut<PerCarTraceAccumulators>,
    mut tracker: ResMut<EpisodeTracker>,
) {
    for (env_id, episode_state) in car_query.iter() {
        let Some(reason) = episode_state.last_end_reason else {
            continue;
        };

        let finished_episode_id = episode_state.current_episode - 1;

        // Avoid duplicate records for the same (env_id, episode_id) pair.
        let already_recorded = tracker.episodes.iter().rev().take(32).any(|r| {
            r.env_id == env_id.0 && r.episode_id == finished_episode_id
        });
        if already_recorded {
            continue;
        }

        let action_summary = action_accumulators
            .take_completed_summary(env_id.0, finished_episode_id)
            .unwrap_or_default();
        let trace = trace_accumulators.take_completed_trace(env_id.0, finished_episode_id);
        let trace_metrics = trace
            .as_ref()
            .map(|episode_trace| episode_trace.metrics.clone())
            .unwrap_or_default();

        let is_crash = reason == crate::game::episode::EpisodeEndReason::Crash;
        let trace_agg = trace.as_ref()
            .map(|t| compute_trace_aggregates(t, is_crash))
            .unwrap_or_default();

        if let Some(trace) = trace {
            let already_traced = tracker
                .episode_traces
                .iter()
                .rev()
                .take(32)
                .any(|t| t.episode_id == trace.episode_id);
            if !already_traced {
                tracker.episode_traces.push(trace);
            }
        }

        let life_seconds = episode_state.last_episode_ticks as f32 / 60.0;

        tracker.episodes.push(EpisodeRecord {
            env_id: env_id.0,
            episode_id: finished_episode_id,
            progress: episode_state.last_episode_best_progress_fraction,
            reward: episode_state.last_episode_return,
            pre_terminal_return: episode_state.last_episode_pre_terminal_return,
            progress_reward_sum: episode_state.last_episode_progress_reward_sum,
            time_penalty_sum: episode_state.last_episode_time_penalty_sum,
            terminal_reward_sum: episode_state.last_episode_terminal_reward_sum,
            crash_penalty_sum: episode_state.last_episode_crash_penalty_sum,
            ticks: episode_state.last_episode_ticks,
            crashes: episode_state.last_episode_crashes,
            end_reason: format!("{:?}", reason),
            distance_driven: episode_state.last_episode_distance_driven,
            crash_position: episode_state
                .last_episode_crash_position
                .map(|position| [position.x, position.y]),

            // Action behaviour
            steering_mean: action_summary.steering_mean,
            steering_std: action_summary.steering_std,
            throttle_mean: action_summary.throttle_mean,
            throttle_std: action_summary.throttle_std,
            braking_fraction: trace_agg.braking_fraction,
            acceleration_fraction: trace_agg.acceleration_fraction,
            coasting_fraction: trace_agg.coasting_fraction,
            mean_action_change: trace_agg.mean_action_change,

            // Speed and momentum
            mean_speed: trace_agg.mean_speed,
            peak_speed: trace_agg.peak_speed,
            mean_v_forward: trace_agg.mean_v_forward,
            mean_v_lateral_abs: trace_agg.mean_v_lateral_abs,
            mean_velocity_projection: trace_agg.mean_velocity_projection,
            mean_drift_angle_deg: trace_agg.mean_drift_angle_deg,
            peak_drift_angle_deg: trace_agg.peak_drift_angle_deg,

            // Crash forensics
            crash_speed: trace_agg.crash_speed,
            crash_v_forward: trace_agg.crash_v_forward,
            crash_v_lateral: trace_agg.crash_v_lateral,
            crash_drift_angle_deg: trace_agg.crash_drift_angle_deg,
            crash_heading_error_deg: trace_agg.crash_heading_error_deg,
            crash_min_ray: trace_agg.crash_min_ray,
            crash_type: trace_agg.crash_type,

            // Value function
            mean_value_prediction: trace_agg.mean_value_prediction,
            value_at_crash: trace_agg.value_at_crash,
            value_at_start: trace_agg.value_at_start,

            // Efficiency and exploration
            reward_per_second: if life_seconds > 0.01 {
                episode_state.last_episode_return / life_seconds
            } else {
                0.0
            },
            furthest_sector: trace_agg.furthest_sector,
            wall_proximity_fraction: trace_agg.wall_proximity_fraction,

            // Policy confidence
            mean_policy_steering_std: trace_agg.mean_policy_steering_std,
            mean_policy_throttle_std: trace_agg.mean_policy_throttle_std,

            // Existing turn metrics
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

    // PPO update recording — reads from a global resource, not per-car.
    if let Some(ppo_stats) = ppo_stats {
        if ppo_stats.last_completed_update > tracker.last_recorded_update {
            tracker.last_recorded_update = ppo_stats.last_completed_update;
            tracker.ppo_updates.push(PpoUpdateRecord {
                update_index: ppo_stats.last_completed_update,
                batch_size: ppo_stats.batch_size,
                policy_loss: ppo_stats.policy_loss,
                value_loss: ppo_stats.value_loss,
                policy_entropy: ppo_stats.policy_entropy,
                explained_variance: ppo_stats.explained_variance,
                steering_mean: ppo_stats.steering_mean,
                steering_std: ppo_stats.steering_std,
                throttle_mean: ppo_stats.throttle_mean,
                throttle_std: ppo_stats.throttle_std,
                clamped_action_fraction: ppo_stats.clamped_action_fraction,
                clip_fraction: ppo_stats.clip_fraction,
                approx_kl: ppo_stats.approx_kl,
                layer_health: ppo_stats
                    .layer_health
                    .iter()
                    .map(|layer| PpoLayerRecord {
                        layer_name: layer.layer_name.clone(),
                        weight_l2_norm: layer.weight_l2_norm,
                        gradient_l2_norm: layer.gradient_l2_norm,
                        saturated_fraction: layer.saturated_fraction,
                    })
                    .collect(),
            });
        }
    }
}
