use std::collections::VecDeque;

use bevy::prelude::*;
use rand::RngExt;

use crate::game::car::{Car, EnvInstanceId, SpawnConfig, SpawnRng};
use crate::game::collision::Collided;
use crate::game::progress::TrackProgress;
use crate::maps::track::Track;
use crate::sim::signed_angle_between;

/// Why an episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeEndReason {
    Crash,
    Timeout,
}

/// Core episode loop configuration.
#[derive(Resource, Clone, Copy, Debug)]
pub struct EpisodeConfig {
    /// Timeout duration in seconds.
    pub timeout_s: f32,
    /// Velocity projection reward: `dot(velocity, tangent) / speed_reference × scale`.
    pub velocity_reward_scale: f32,
    /// Reference speed for velocity projection normalisation.
    pub speed_reward_reference: f32,
    /// Centreline proximity reward coefficient.
    /// Reward per tick: `coef × (1 - (|lateral_offset| / lateral_max)²)`.
    pub centreline_reward_coef: f32,
    /// Maximum lateral offset for centreline reward (reward reaches 0 at this distance).
    pub centreline_reward_max_distance: f32,
    /// Per-tick time penalty.
    ///
    /// **Policy:** keep at `0.0`. See `context/notes/reward-and-entertainment.md`
    /// — the entertainment-first reward structure is velocity-projection plus
    /// centreline proximity only. A per-tick penalty is the symmetric failure
    /// mode of a survival bonus: it incentivises fast crashing when progress
    /// gain is marginal, which the velocity-projection reward already handles
    /// implicitly (no drive → no reward → no optimum in standing still).
    pub time_penalty_per_tick: f32,
    /// Crash penalty applied once on crash episode end.
    ///
    /// **Policy:** keep at `0.0`. See `context/notes/reward-and-entertainment.md`
    /// — non-zero crash penalties produce "stay still / brake constantly"
    /// policies and are incompatible with the entertainment-first reward
    /// philosophy. Episode termination is the cost; no explicit penalty.
    pub crash_penalty: f32,
    /// Number of episodes used for moving averages.
    pub moving_average_window: usize,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            timeout_s: 30.0,
            velocity_reward_scale: 1.0,
            speed_reward_reference: 200.0,
            centreline_reward_coef: 0.3,
            centreline_reward_max_distance: 50.0,
            time_penalty_per_tick: 0.0,
            crash_penalty: 0.0,
            moving_average_window: 20,
        }
    }
}

/// Per-tick reward and sensor snapshot.
#[derive(Clone, Debug)]
pub struct TickSnapshot {
    pub reward: f32,
    pub progress_reward: f32,
    pub time_penalty: f32,
    pub terminal_reward: f32,
    pub end_reason: Option<EpisodeEndReason>,
    pub progress_fraction: f32,
    pub progress_s: f32,
    pub centerline_distance: f32,
    pub speed: f32,
    pub heading_error: f32,
    pub forward: Vec2,
    pub tangent: Vec2,
    pub velocity_projection: f32,
    pub centreline_reward: f32,
}

impl Default for TickSnapshot {
    fn default() -> Self {
        Self {
            reward: 0.0,
            progress_reward: 0.0,
            time_penalty: 0.0,
            terminal_reward: 0.0,
            end_reason: None,
            progress_fraction: 0.0,
            progress_s: 0.0,
            centerline_distance: 0.0,
            speed: 0.0,
            heading_error: 0.0,
            forward: Vec2::X,
            tangent: Vec2::X,
            velocity_projection: 0.0,
            centreline_reward: 0.0,
        }
    }
}

/// Running accumulators for the current episode.
#[derive(Clone, Debug, Default)]
pub struct EpisodeAccum {
    pub return_sum: f32,
    pub progress_reward_sum: f32,
    pub time_penalty_sum: f32,
    pub terminal_reward_sum: f32,
    pub crash_penalty_sum: f32,
    /// Best distance_driven / track_length achieved this episode.
    pub best_progress_fraction: f32,
    pub crashes: u32,
}

/// Summary of the most recently completed episode.
#[derive(Clone, Debug)]
pub struct LastEpisodeSummary {
    pub end_reason: Option<EpisodeEndReason>,
    pub return_sum: f32,
    pub pre_terminal_return: f32,
    pub progress_reward_sum: f32,
    pub time_penalty_sum: f32,
    pub terminal_reward_sum: f32,
    pub crash_penalty_sum: f32,
    pub best_progress_fraction: f32,
    pub distance_driven: f32,
    pub crashes: u32,
    pub ticks: u32,
    pub crash_position: Option<Vec2>,
}

impl Default for LastEpisodeSummary {
    fn default() -> Self {
        Self {
            end_reason: None,
            return_sum: 0.0,
            pre_terminal_return: 0.0,
            progress_reward_sum: 0.0,
            time_penalty_sum: 0.0,
            terminal_reward_sum: 0.0,
            crash_penalty_sum: 0.0,
            best_progress_fraction: 0.0,
            distance_driven: 0.0,
            crashes: 0,
            ticks: 0,
            crash_position: None,
        }
    }
}

/// Per-car episode state and accumulators.
#[derive(Component, Debug)]
pub struct EpisodeState {
    pub current_episode: u32,
    pub ticks_in_episode: u32,
    /// Arc-length position of the car on the previous tick (for delta computation).
    pub previous_s: f32,
    /// Cumulative forward arc-length driven from spawn this episode.
    pub distance_driven: f32,
    /// Arc-length position where this episode's spawn occurred.
    pub spawn_s: f32,
    pub tick: TickSnapshot,
    pub accum: EpisodeAccum,
    pub last: LastEpisodeSummary,
}

impl Default for EpisodeState {
    fn default() -> Self {
        Self {
            current_episode: 1,
            ticks_in_episode: 0,
            previous_s: 0.0,
            distance_driven: 0.0,
            spawn_s: 0.0,
            tick: TickSnapshot::default(),
            accum: EpisodeAccum::default(),
            last: LastEpisodeSummary::default(),
        }
    }
}

/// Per-car rolling episode-level telemetry for moving averages.
#[derive(Component, Debug)]
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

/// Handles per-tick reward accumulation and episode boundaries for all cars.
/// Each car runs its own independent episode lifecycle.
///
/// Progress is measured as cumulative forward arc-length from spawn.
/// Episode ends on crash or timeout only — no finish line, no lap detection.
pub fn episode_loop_system(
    time: Res<Time<bevy::time::Fixed>>,
    config: Res<EpisodeConfig>,
    track_query: Query<&Track>,
    mut spawn_rng: ResMut<SpawnRng>,
    mut car_query: Query<(
        &mut Transform,
        &mut Car,
        &mut TrackProgress,
        &mut EpisodeState,
        &mut EpisodeMovingAverages,
        &mut SpawnConfig,
        &EnvInstanceId,
        Has<Collided>,
    )>,
) {
    let Ok(track) = track_query.single() else {
        return;
    };
    let total_length = track.centerline.total_length();

    for (mut transform, mut car, mut progress, mut episode_state, mut moving_avg, mut spawn_config, _env_id, crashed) in
        car_query.iter_mut()
    {
        let forward = (transform.rotation * Vec3::X)
            .truncate()
            .normalize_or_zero();

        episode_state.tick.reward = 0.0;
        episode_state.tick.progress_reward = 0.0;
        episode_state.tick.time_penalty = 0.0;
        episode_state.tick.terminal_reward = 0.0;
        episode_state.tick.end_reason = None;
        episode_state.ticks_in_episode = episode_state.ticks_in_episode.saturating_add(1);

        // Compute forward arc-length delta with wrap handling (for distance tracking).
        let mut raw_delta = progress.s - episode_state.previous_s;
        let half_length = total_length * 0.5;
        if raw_delta < -half_length {
            raw_delta += total_length; // wrapped forward past the seam
        }
        if raw_delta > half_length {
            raw_delta -= total_length; // wrapped backward past the seam
        }
        let forward_delta = raw_delta.max(0.0);
        episode_state.distance_driven += forward_delta;

        let progress_fraction = episode_state.distance_driven / total_length;
        episode_state.accum.best_progress_fraction =
            episode_state.accum.best_progress_fraction.max(progress_fraction);

        // Velocity projection reward: how much speed is aligned with the track.
        let v_along_track = car.velocity.dot(progress.tangent);
        let progress_reward = (v_along_track / config.speed_reward_reference)
            * config.velocity_reward_scale;

        // Centreline proximity reward: gentle bonus for staying near the racing line.
        let d_norm = (progress.distance / config.centreline_reward_max_distance).min(1.0);
        let centreline_reward = config.centreline_reward_coef * (1.0 - d_norm * d_norm);

        let heading_error = signed_angle_between(forward, progress.tangent);
        let time_penalty = config.time_penalty_per_tick;
        let mut terminal_reward = 0.0;

        let mut crash_position = None;
        if crashed {
            episode_state.accum.crashes = episode_state.accum.crashes.saturating_add(1);
            terminal_reward += config.crash_penalty;
            crash_position = Some(transform.translation.truncate());
        }

        let timed_out =
            (episode_state.ticks_in_episode as f32) * time.delta_secs() >= config.timeout_s;

        let tick_reward = progress_reward + centreline_reward + time_penalty + terminal_reward;

        episode_state.tick.reward = tick_reward;
        episode_state.tick.progress_reward = progress_reward + centreline_reward;
        episode_state.tick.time_penalty = time_penalty;
        episode_state.tick.terminal_reward = terminal_reward;
        episode_state.tick.progress_fraction = progress_fraction;
        episode_state.tick.progress_s = progress.s;
        episode_state.tick.centerline_distance = progress.distance;
        episode_state.tick.speed = car.velocity.length();
        episode_state.tick.heading_error = heading_error;
        episode_state.tick.forward = forward;
        episode_state.tick.tangent = progress.tangent;
        episode_state.tick.velocity_projection = v_along_track;
        episode_state.tick.centreline_reward = centreline_reward;
        episode_state.accum.progress_reward_sum += progress_reward;
        episode_state.accum.time_penalty_sum += time_penalty;
        episode_state.accum.terminal_reward_sum += terminal_reward;
        if crashed {
            episode_state.accum.crash_penalty_sum += config.crash_penalty;
        }
        episode_state.accum.return_sum += tick_reward;

        let end_reason = if crashed {
            Some(EpisodeEndReason::Crash)
        } else if timed_out {
            Some(EpisodeEndReason::Timeout)
        } else {
            None
        };

        if let Some(reason) = end_reason {
            episode_state.tick.end_reason = Some(reason);
            finalize_episode(
                &config,
                &mut episode_state,
                &mut moving_avg,
                reason,
                crash_position,
            );
            // All cars get a new random spawn position each episode.
            let s = spawn_rng.0.random::<f32>() * total_length;
            let position = track.centerline.point_at_s(s);
            let tangent = track.centerline.tangent_at_s(s);
            let rotation = tangent.y.atan2(tangent.x);
            spawn_config.position = position;
            spawn_config.rotation = rotation;

            reset_car_to_spawn(&mut transform, &mut car, &spawn_config);
            sync_progress_to_transform(track, &transform, &mut progress);
            // Seed distance tracking for the new episode.
            episode_state.previous_s = progress.s;
            episode_state.spawn_s = progress.s;
        } else {
            episode_state.previous_s = progress.s;
        }
    }
}

fn reset_car_to_spawn(transform: &mut Transform, car: &mut Car, spawn: &SpawnConfig) {
    transform.translation.x = spawn.position.x;
    transform.translation.y = spawn.position.y;
    transform.rotation = Quat::from_rotation_z(spawn.rotation);
    car.velocity = Vec2::ZERO;
}

fn sync_progress_to_transform(track: &Track, transform: &Transform, progress: &mut TrackProgress) {
    let projection = track.centerline.project(transform.translation.truncate(), None);
    progress.s = projection.s;
    progress.fraction = projection.fraction;
    progress.closest_point = projection.closest_point;
    progress.tangent = projection.tangent;
    progress.distance = projection.distance;
    progress.last_segment = projection.segment_index;
}

fn finalize_episode(
    config: &EpisodeConfig,
    episode_state: &mut EpisodeState,
    moving_avg: &mut EpisodeMovingAverages,
    reason: EpisodeEndReason,
    crash_position: Option<Vec2>,
) {
    episode_state.last.end_reason = Some(reason);
    episode_state.last.return_sum = episode_state.accum.return_sum;
    episode_state.last.pre_terminal_return =
        episode_state.accum.progress_reward_sum + episode_state.accum.time_penalty_sum;
    episode_state.last.progress_reward_sum = episode_state.accum.progress_reward_sum;
    episode_state.last.time_penalty_sum = episode_state.accum.time_penalty_sum;
    episode_state.last.terminal_reward_sum = episode_state.accum.terminal_reward_sum;
    episode_state.last.crash_penalty_sum = episode_state.accum.crash_penalty_sum;
    episode_state.last.best_progress_fraction =
        episode_state.accum.best_progress_fraction;
    episode_state.last.distance_driven = episode_state.distance_driven;
    episode_state.last.crashes = episode_state.accum.crashes;
    episode_state.last.ticks = episode_state.ticks_in_episode;
    episode_state.last.crash_position = crash_position;

    push_with_limit(
        &mut moving_avg.returns,
        episode_state.last.return_sum,
        config.moving_average_window,
    );
    push_with_limit(
        &mut moving_avg.best_progress_fractions,
        episode_state.last.best_progress_fraction,
        config.moving_average_window,
    );
    push_with_limit(
        &mut moving_avg.crash_counts,
        episode_state.last.crashes as f32,
        config.moving_average_window,
    );
    moving_avg.return_mean = mean(&moving_avg.returns);
    moving_avg.best_progress_mean = mean(&moving_avg.best_progress_fractions);
    moving_avg.crash_mean = mean(&moving_avg.crash_counts);

    episode_state.current_episode = episode_state.current_episode.saturating_add(1);
    episode_state.ticks_in_episode = 0;
    episode_state.distance_driven = 0.0;
    episode_state.accum = EpisodeAccum::default();
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


#[cfg(test)]
mod tests {
    use super::*;

    // ── EpisodeConfig default regression guards (2026-04-18 policy locks) ─

    #[test]
    fn default_time_penalty_is_zero() {
        // The entertainment-first reward philosophy (see notes/reward-and-entertainment.md)
        // requires NO per-tick survival penalty. A non-zero default was a drift
        // bug caught and fixed during the 2026-04-18 CHA audit.
        let cfg = EpisodeConfig::default();
        assert_eq!(cfg.time_penalty_per_tick, 0.0,
            "time_penalty_per_tick must stay 0.0 — see notes/reward-and-entertainment.md");
    }

    #[test]
    fn default_crash_penalty_is_zero() {
        // Same entertainment-first philosophy: crashes cost future discounted
        // reward, not an explicit penalty. Non-zero values produce
        // "stay still / brake constantly" policies.
        let cfg = EpisodeConfig::default();
        assert_eq!(cfg.crash_penalty, 0.0,
            "crash_penalty must stay 0.0 — see notes/reward-and-entertainment.md");
    }

    #[test]
    fn default_reward_weights_match_documented_values() {
        // Regression guard: if any of these defaults drift, the README's
        // §Reward Structure table will go stale. Flagging here forces the
        // documentation update alongside the code change.
        let cfg = EpisodeConfig::default();
        assert_eq!(cfg.velocity_reward_scale, 1.0);
        assert_eq!(cfg.speed_reward_reference, 200.0);
        assert_eq!(cfg.centreline_reward_coef, 0.3);
        assert_eq!(cfg.centreline_reward_max_distance, 50.0);
    }

    #[test]
    fn default_timeout_is_thirty_seconds() {
        let cfg = EpisodeConfig::default();
        assert_eq!(cfg.timeout_s, 30.0);
    }

    #[test]
    fn default_moving_average_window_is_twenty() {
        let cfg = EpisodeConfig::default();
        assert_eq!(cfg.moving_average_window, 20);
    }

    #[test]
    fn push_with_limit_drops_oldest_when_full() {
        let mut buf = VecDeque::new();
        for i in 0..5 {
            push_with_limit(&mut buf, i as f32, 3);
        }
        // Only last 3 kept
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 2.0);
        assert_eq!(buf[1], 3.0);
        assert_eq!(buf[2], 4.0);
    }

    #[test]
    fn push_with_limit_zero_limit_keeps_one() {
        // A zero limit is treated as 1 (degenerate but well-defined).
        let mut buf = VecDeque::new();
        push_with_limit(&mut buf, 1.0, 0);
        push_with_limit(&mut buf, 2.0, 0);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], 2.0);
    }

    #[test]
    fn mean_of_empty_deque_is_zero() {
        let buf: VecDeque<f32> = VecDeque::new();
        assert_eq!(mean(&buf), 0.0);
    }

    #[test]
    fn mean_computes_arithmetic_mean() {
        let mut buf = VecDeque::new();
        buf.push_back(1.0);
        buf.push_back(2.0);
        buf.push_back(3.0);
        assert!((mean(&buf) - 2.0).abs() < 1e-6);
    }
}
