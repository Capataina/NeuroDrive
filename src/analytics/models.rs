use bevy::prelude::*;
use serde::{Deserialize, Serialize};

pub const NUM_PROGRESS_SECTORS: usize = 20;

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
    pub turn_in_latency_ticks: Option<u32>,
    pub throttle_release_latency_fraction: Option<f32>,
    pub throttle_release_latency_ticks: Option<u32>,
    pub steering_adequacy: f32,
    pub high_curvature_throttle_mean: f32,
    pub curvature_steering_error_mean: f32,
    pub curvature_steering_bias_mean: f32,
    pub understeer_rate: f32,
    pub turn_entry_speed: Option<f32>,
    pub peak_curvature_speed: Option<f32>,
    pub crash_speed: Option<f32>,
    pub entry_lateral_offset: Option<f32>,
    pub peak_lateral_offset: Option<f32>,
    pub peak_centerline_distance: Option<f32>,
    pub mean_centerline_distance: f32,
    pub mean_abs_lateral_offset: f32,
    pub mean_abs_heading_error_deg: f32,
    pub mean_all_ray_distance: f32,
    pub mean_front_ray_distance: f32,
    pub mean_side_ray_distance: f32,
    pub failure_mode: Option<String>,
}

/// Tick-level trajectory analytics record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickTraceRecord {
    pub tick_index: u32,
    pub progress_fraction: f32,
    pub progress_s: f32,
    pub centerline_distance: f32,
    pub signed_lateral_offset: f32,
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
    pub ray_distances: Vec<f32>,
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

/// Derived mismatch and turn-execution metrics from one episode trace.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EpisodeTraceMetrics {
    pub turn_in_latency_fraction: Option<f32>,
    pub turn_in_latency_ticks: Option<u32>,
    pub throttle_release_latency_fraction: Option<f32>,
    pub throttle_release_latency_ticks: Option<u32>,
    pub steering_adequacy: f32,
    pub high_curvature_throttle_mean: f32,
    pub curvature_steering_error_mean: f32,
    pub curvature_steering_bias_mean: f32,
    pub understeer_rate: f32,
    pub turn_entry_speed: Option<f32>,
    pub peak_curvature_speed: Option<f32>,
    pub crash_speed: Option<f32>,
    pub entry_lateral_offset: Option<f32>,
    pub peak_lateral_offset: Option<f32>,
    pub peak_centerline_distance: Option<f32>,
    pub mean_centerline_distance: f32,
    pub mean_abs_lateral_offset: f32,
    pub mean_abs_heading_error_deg: f32,
    pub mean_all_ray_distance: f32,
    pub mean_front_ray_distance: f32,
    pub mean_side_ray_distance: f32,
    pub failure_mode: Option<String>,
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
