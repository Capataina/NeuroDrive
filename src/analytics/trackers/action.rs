use crate::agent::action::ActionState;
use crate::game::episode::EpisodeState;
use bevy::prelude::*;

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

    pub fn take_completed_summary(&mut self, episode_id: u32) -> Option<EpisodeActionSummary> {
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

fn mean_from_sum(sum: f32, count: u32) -> f32 {
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn std_from_sum_and_sumsq(sum: f32, sumsq: f32, count: u32) -> f32 {
    if count == 0 {
        return 0.0;
    }

    let n = count as f32;
    let mean = sum / n;
    ((sumsq / n) - mean * mean).max(0.0).sqrt()
}
