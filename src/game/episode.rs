use std::collections::VecDeque;
use std::f32::consts::PI;

use bevy::ecs::message::MessageReader;
use bevy::prelude::*;

use crate::game::car::Car;
use crate::game::collision::CollisionEvent;
use crate::game::progress::TrackProgress;
use crate::maps::track::Track;

/// Why an episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeEndReason {
    Crash,
    Timeout,
    LapComplete,
}

/// Core episode loop configuration.
#[derive(Resource, Clone, Copy, Debug)]
pub struct EpisodeConfig {
    /// Timeout duration in seconds.
    pub timeout_s: f32,
    /// Progress threshold that arms lap-wrap detection.
    pub lap_arm_fraction: f32,
    /// Prior-to-wrap threshold used for lap completion.
    pub lap_wrap_from_fraction: f32,
    /// Post-wrap threshold used for lap completion.
    pub lap_wrap_to_fraction: f32,
    /// Reward scaling for positive gain in episode-best progress per tick.
    pub progress_reward_scale: f32,
    /// Small per-tick time penalty to discourage stalling.
    pub time_penalty_per_tick: f32,
    /// Extra per-tick penalty for high-speed heading misalignment.
    pub heading_speed_penalty_scale: f32,
    /// Speed scale used to normalise heading-risk penalty.
    pub speed_norm_max_for_penalty: f32,
    /// Crash penalty applied once on crash episode end.
    pub crash_penalty: f32,
    /// Lap-complete bonus applied once on lap episode end.
    pub lap_bonus: f32,
    /// Number of episodes used for moving averages.
    pub moving_average_window: usize,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            timeout_s: 30.0,
            lap_arm_fraction: 0.25,
            lap_wrap_from_fraction: 0.85,
            lap_wrap_to_fraction: 0.15,
            progress_reward_scale: 140.0,
            time_penalty_per_tick: -0.005,
            heading_speed_penalty_scale: 0.02,
            speed_norm_max_for_penalty: 900.0,
            crash_penalty: -5.0,
            lap_bonus: 100.0,
            moving_average_window: 20,
        }
    }
}

/// Episode state and accumulators.
#[derive(Resource, Debug)]
pub struct EpisodeState {
    pub current_episode: u32,
    pub ticks_in_episode: u32,
    pub previous_progress_fraction: f32,
    pub lap_armed: bool,
    pub current_return: f32,
    pub current_tick_reward: f32,
    pub current_tick_progress_reward: f32,
    pub current_tick_time_penalty: f32,
    pub current_tick_terminal_reward: f32,
    pub current_tick_end_reason: Option<EpisodeEndReason>,
    pub current_tick_progress_fraction: f32,
    pub current_tick_progress_s: f32,
    pub current_tick_speed: f32,
    pub current_tick_heading_error: f32,
    pub current_tick_forward: Vec2,
    pub current_tick_tangent: Vec2,
    pub current_progress_reward_sum: f32,
    pub current_time_penalty_sum: f32,
    pub current_terminal_reward_sum: f32,
    pub current_crash_penalty_sum: f32,
    pub current_lap_bonus_sum: f32,
    pub current_best_progress_fraction: f32,
    pub current_crashes: u32,
    pub last_end_reason: Option<EpisodeEndReason>,
    pub last_episode_return: f32,
    pub last_episode_pre_terminal_return: f32,
    pub last_episode_progress_reward_sum: f32,
    pub last_episode_time_penalty_sum: f32,
    pub last_episode_terminal_reward_sum: f32,
    pub last_episode_crash_penalty_sum: f32,
    pub last_episode_lap_bonus_sum: f32,
    pub last_episode_best_progress_fraction: f32,
    pub last_episode_crashes: u32,
    pub last_episode_ticks: u32,
    pub last_episode_crash_position: Option<Vec2>,
}

impl Default for EpisodeState {
    fn default() -> Self {
        Self {
            current_episode: 1,
            ticks_in_episode: 0,
            previous_progress_fraction: 0.0,
            lap_armed: false,
            current_return: 0.0,
            current_tick_reward: 0.0,
            current_tick_progress_reward: 0.0,
            current_tick_time_penalty: 0.0,
            current_tick_terminal_reward: 0.0,
            current_tick_end_reason: None,
            current_tick_progress_fraction: 0.0,
            current_tick_progress_s: 0.0,
            current_tick_speed: 0.0,
            current_tick_heading_error: 0.0,
            current_tick_forward: Vec2::X,
            current_tick_tangent: Vec2::X,
            current_progress_reward_sum: 0.0,
            current_time_penalty_sum: 0.0,
            current_terminal_reward_sum: 0.0,
            current_crash_penalty_sum: 0.0,
            current_lap_bonus_sum: 0.0,
            current_best_progress_fraction: 0.0,
            current_crashes: 0,
            last_end_reason: None,
            last_episode_return: 0.0,
            last_episode_pre_terminal_return: 0.0,
            last_episode_progress_reward_sum: 0.0,
            last_episode_time_penalty_sum: 0.0,
            last_episode_terminal_reward_sum: 0.0,
            last_episode_crash_penalty_sum: 0.0,
            last_episode_lap_bonus_sum: 0.0,
            last_episode_best_progress_fraction: 0.0,
            last_episode_crashes: 0,
            last_episode_ticks: 0,
            last_episode_crash_position: None,
        }
    }
}

/// Rolling episode-level telemetry for moving averages.
#[derive(Resource, Debug)]
pub struct EpisodeMovingAverages {
    pub returns: VecDeque<f32>,
    pub best_progress_fractions: VecDeque<f32>,
    pub crash_counts: VecDeque<f32>,
    pub return_mean: f32,
    pub best_progress_mean: f32,
    pub crash_mean: f32,
}

impl Default for EpisodeMovingAverages {
    fn default() -> Self {
        Self {
            returns: VecDeque::new(),
            best_progress_fractions: VecDeque::new(),
            crash_counts: VecDeque::new(),
            return_mean: 0.0,
            best_progress_mean: 0.0,
            crash_mean: 0.0,
        }
    }
}

/// Handles per-tick reward accumulation and episode boundaries:
/// crash, timeout, and lap completion.
pub fn episode_loop_system(
    time: Res<Time<bevy::time::Fixed>>,
    config: Res<EpisodeConfig>,
    mut episode_state: ResMut<EpisodeState>,
    mut moving_avg: ResMut<EpisodeMovingAverages>,
    mut collision_events: MessageReader<CollisionEvent>,
    track_query: Query<&Track>,
    mut car_query: Query<(&mut Transform, &mut Car, &mut TrackProgress)>,
) {
    let Ok(track) = track_query.single() else {
        return;
    };
    let Ok((mut transform, mut car, mut progress)) = car_query.single_mut() else {
        return;
    };
    let forward = (transform.rotation * Vec3::X).truncate().normalize_or_zero();

    episode_state.current_tick_reward = 0.0;
    episode_state.current_tick_progress_reward = 0.0;
    episode_state.current_tick_time_penalty = 0.0;
    episode_state.current_tick_terminal_reward = 0.0;
    episode_state.current_tick_end_reason = None;
    episode_state.ticks_in_episode = episode_state.ticks_in_episode.saturating_add(1);
    let previous_best_progress = episode_state.current_best_progress_fraction;
    let progress_gain = (progress.fraction - previous_best_progress).max(0.0);
    episode_state.current_best_progress_fraction = previous_best_progress.max(progress.fraction);
    let progress_reward = progress_gain * config.progress_reward_scale;
    let heading_error = signed_angle_between(forward, progress.tangent);
    let heading_error_norm = (heading_error.abs() / PI).clamp(0.0, 1.0);
    let speed_norm = (car.velocity.length() / config.speed_norm_max_for_penalty).clamp(0.0, 1.0);
    let heading_speed_penalty = -config.heading_speed_penalty_scale * heading_error_norm * speed_norm;
    let time_penalty = config.time_penalty_per_tick + heading_speed_penalty;
    let mut terminal_reward = 0.0;

    if progress.fraction >= config.lap_arm_fraction {
        episode_state.lap_armed = true;
    }

    let crashed = collision_events.read().next().is_some();
    let mut crash_position = None;
    if crashed {
        episode_state.current_crashes = episode_state.current_crashes.saturating_add(1);
        terminal_reward += config.crash_penalty;
        crash_position = Some(transform.translation.truncate());
    }

    let timed_out =
        (episode_state.ticks_in_episode as f32) * time.delta_secs() >= config.timeout_s;
    let lap_complete = episode_state.lap_armed
        && episode_state.previous_progress_fraction >= config.lap_wrap_from_fraction
        && progress.fraction <= config.lap_wrap_to_fraction;

    if lap_complete {
        terminal_reward += config.lap_bonus;
    }
    let tick_reward = progress_reward + time_penalty + terminal_reward;

    episode_state.current_tick_reward = tick_reward;
    episode_state.current_tick_progress_reward = progress_reward;
    episode_state.current_tick_time_penalty = time_penalty;
    episode_state.current_tick_terminal_reward = terminal_reward;
    episode_state.current_tick_progress_fraction = progress.fraction;
    episode_state.current_tick_progress_s = progress.s;
    episode_state.current_tick_speed = car.velocity.length();
    episode_state.current_tick_heading_error = heading_error;
    episode_state.current_tick_forward = forward;
    episode_state.current_tick_tangent = progress.tangent;
    episode_state.current_progress_reward_sum += progress_reward;
    episode_state.current_time_penalty_sum += time_penalty;
    episode_state.current_terminal_reward_sum += terminal_reward;
    if crashed {
        episode_state.current_crash_penalty_sum += config.crash_penalty;
    }
    if lap_complete {
        episode_state.current_lap_bonus_sum += config.lap_bonus;
    }
    episode_state.current_return += tick_reward;

    let end_reason = if crashed {
        Some(EpisodeEndReason::Crash)
    } else if lap_complete {
        Some(EpisodeEndReason::LapComplete)
    } else if timed_out {
        Some(EpisodeEndReason::Timeout)
    } else {
        None
    };

    if let Some(reason) = end_reason {
        episode_state.current_tick_end_reason = Some(reason);
        finalize_episode(
            &config,
            &mut episode_state,
            &mut moving_avg,
            reason,
            crash_position,
        );
        reset_car_to_spawn(&mut transform, &mut car, track);
        sync_progress_to_transform(track, &transform, &mut progress);
    } else {
        episode_state.previous_progress_fraction = progress.fraction;
    }
}

fn reset_car_to_spawn(transform: &mut Transform, car: &mut Car, track: &Track) {
    transform.translation.x = track.spawn_position.x;
    transform.translation.y = track.spawn_position.y;
    transform.rotation = Quat::from_rotation_z(track.spawn_rotation);
    car.velocity = Vec2::ZERO;
}

fn sync_progress_to_transform(track: &Track, transform: &Transform, progress: &mut TrackProgress) {
    let projection = track.centerline.project(transform.translation.truncate());
    progress.s = projection.s;
    progress.fraction = projection.fraction;
    progress.closest_point = projection.closest_point;
    progress.tangent = projection.tangent;
    progress.distance = projection.distance;
}

fn finalize_episode(
    config: &EpisodeConfig,
    episode_state: &mut EpisodeState,
    moving_avg: &mut EpisodeMovingAverages,
    reason: EpisodeEndReason,
    crash_position: Option<Vec2>,
) {
    episode_state.last_end_reason = Some(reason);
    episode_state.last_episode_return = episode_state.current_return;
    episode_state.last_episode_pre_terminal_return =
        episode_state.current_progress_reward_sum + episode_state.current_time_penalty_sum;
    episode_state.last_episode_progress_reward_sum = episode_state.current_progress_reward_sum;
    episode_state.last_episode_time_penalty_sum = episode_state.current_time_penalty_sum;
    episode_state.last_episode_terminal_reward_sum = episode_state.current_terminal_reward_sum;
    episode_state.last_episode_crash_penalty_sum = episode_state.current_crash_penalty_sum;
    episode_state.last_episode_lap_bonus_sum = episode_state.current_lap_bonus_sum;
    episode_state.last_episode_best_progress_fraction = episode_state.current_best_progress_fraction;
    episode_state.last_episode_crashes = episode_state.current_crashes;
    episode_state.last_episode_ticks = episode_state.ticks_in_episode;
    episode_state.last_episode_crash_position = crash_position;

    push_with_limit(
        &mut moving_avg.returns,
        episode_state.last_episode_return,
        config.moving_average_window,
    );
    push_with_limit(
        &mut moving_avg.best_progress_fractions,
        episode_state.last_episode_best_progress_fraction,
        config.moving_average_window,
    );
    push_with_limit(
        &mut moving_avg.crash_counts,
        episode_state.last_episode_crashes as f32,
        config.moving_average_window,
    );
    moving_avg.return_mean = mean(&moving_avg.returns);
    moving_avg.best_progress_mean = mean(&moving_avg.best_progress_fractions);
    moving_avg.crash_mean = mean(&moving_avg.crash_counts);

    episode_state.current_episode = episode_state.current_episode.saturating_add(1);
    episode_state.ticks_in_episode = 0;
    episode_state.previous_progress_fraction = 0.0;
    episode_state.lap_armed = false;
    episode_state.current_return = 0.0;
    episode_state.current_progress_reward_sum = 0.0;
    episode_state.current_time_penalty_sum = 0.0;
    episode_state.current_terminal_reward_sum = 0.0;
    episode_state.current_crash_penalty_sum = 0.0;
    episode_state.current_lap_bonus_sum = 0.0;
    episode_state.current_best_progress_fraction = 0.0;
    episode_state.current_crashes = 0;
}

fn push_with_limit(buffer: &mut VecDeque<f32>, value: f32, limit: usize) {
    buffer.push_back(value);
    while buffer.len() > limit.max(1) {
        let _ = buffer.pop_front();
    }
}

fn mean(values: &VecDeque<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
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

fn signed_angle_between(from: Vec2, to: Vec2) -> f32 {
    let from_n = from.normalize_or_zero();
    let to_n = to.normalize_or_zero();
    if from_n == Vec2::ZERO || to_n == Vec2::ZERO {
        return 0.0;
    }
    wrap_angle(to_n.to_angle() - from_n.to_angle())
}
