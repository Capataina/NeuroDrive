use std::collections::VecDeque;

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
    /// Reward scaling for signed progress delta per tick.
    pub progress_reward_scale: f32,
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
            progress_reward_scale: 100.0,
            crash_penalty: -10.0,
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
    pub current_best_progress_fraction: f32,
    pub current_crashes: u32,
    pub last_end_reason: Option<EpisodeEndReason>,
    pub last_episode_return: f32,
    pub last_episode_best_progress_fraction: f32,
    pub last_episode_crashes: u32,
}

impl Default for EpisodeState {
    fn default() -> Self {
        Self {
            current_episode: 1,
            ticks_in_episode: 0,
            previous_progress_fraction: 0.0,
            lap_armed: false,
            current_return: 0.0,
            current_best_progress_fraction: 0.0,
            current_crashes: 0,
            last_end_reason: None,
            last_episode_return: 0.0,
            last_episode_best_progress_fraction: 0.0,
            last_episode_crashes: 0,
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
    mut car_query: Query<(&mut Transform, &mut Car, &TrackProgress)>,
) {
    let Ok(track) = track_query.single() else {
        return;
    };
    let Ok((mut transform, mut car, progress)) = car_query.single_mut() else {
        return;
    };

    episode_state.ticks_in_episode = episode_state.ticks_in_episode.saturating_add(1);
    episode_state.current_best_progress_fraction = episode_state
        .current_best_progress_fraction
        .max(progress.fraction);

    let mut progress_delta = progress.fraction - episode_state.previous_progress_fraction;
    if progress_delta > 0.5 {
        progress_delta -= 1.0;
    } else if progress_delta < -0.5 {
        progress_delta += 1.0;
    }
    episode_state.current_return += progress_delta * config.progress_reward_scale;

    if progress.fraction >= config.lap_arm_fraction {
        episode_state.lap_armed = true;
    }

    let crashed = collision_events.read().next().is_some();
    if crashed {
        episode_state.current_crashes = episode_state.current_crashes.saturating_add(1);
        episode_state.current_return += config.crash_penalty;
    }

    let timed_out =
        (episode_state.ticks_in_episode as f32) * time.delta_secs() >= config.timeout_s;
    let lap_complete = episode_state.lap_armed
        && episode_state.previous_progress_fraction >= config.lap_wrap_from_fraction
        && progress.fraction <= config.lap_wrap_to_fraction;

    if lap_complete {
        episode_state.current_return += config.lap_bonus;
    }

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
        if reason != EpisodeEndReason::Crash {
            reset_car_to_spawn(&mut transform, &mut car, track);
        }

        finalize_episode(&config, &mut episode_state, &mut moving_avg, reason);
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

fn finalize_episode(
    config: &EpisodeConfig,
    episode_state: &mut EpisodeState,
    moving_avg: &mut EpisodeMovingAverages,
    reason: EpisodeEndReason,
) {
    episode_state.last_end_reason = Some(reason);
    episode_state.last_episode_return = episode_state.current_return;
    episode_state.last_episode_best_progress_fraction = episode_state.current_best_progress_fraction;
    episode_state.last_episode_crashes = episode_state.current_crashes;

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

