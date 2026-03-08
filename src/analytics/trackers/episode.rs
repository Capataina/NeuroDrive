use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::agent::action::ActionState;
use crate::brain::a2c::A2cTrainingStats;
use crate::game::episode::{EpisodeEndReason, EpisodeState};

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

pub fn episode_tracker_system(
    episode_state: Res<EpisodeState>,
    a2c_stats: Option<Res<A2cTrainingStats>>,
    mut accumulator: ResMut<EpisodeActionAccumulator>,
    mut tracker: ResMut<EpisodeTracker>,
) {
    if let Some(reason) = episode_state.last_end_reason {
        let finished_episode_id = episode_state.current_episode - 1;

        if tracker.episodes.last().map(|r| r.episode_id) != Some(finished_episode_id) {
            let action_summary = accumulator
                .take_completed_summary(finished_episode_id)
                .unwrap_or_default();
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
