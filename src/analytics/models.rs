use bevy::prelude::*;
use serde::{Deserialize, Serialize};

pub const NUM_PROGRESS_SECTORS: usize = 20;

/// Exported analytics snapshot for a completed episode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub env_id: u32,
    pub episode_id: u32,
    pub progress: f32,
    pub reward: f32,
    pub pre_terminal_return: f32,
    pub progress_reward_sum: f32,
    pub time_penalty_sum: f32,
    pub terminal_reward_sum: f32,
    pub crash_penalty_sum: f32,
    pub ticks: u32,
    pub crashes: u32,
    pub end_reason: String,
    pub distance_driven: f32,
    pub crash_position: Option<[f32; 2]>,

    // Action behaviour
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
    pub braking_fraction: f32,
    pub acceleration_fraction: f32,
    pub coasting_fraction: f32,
    pub mean_action_change: f32,

    // Speed and momentum
    pub mean_speed: f32,
    pub peak_speed: f32,
    pub mean_v_forward: f32,
    pub mean_v_lateral_abs: f32,
    pub mean_velocity_projection: f32,
    pub mean_drift_angle_deg: f32,
    pub peak_drift_angle_deg: f32,

    // Crash forensics
    pub crash_speed: Option<f32>,
    pub crash_v_forward: Option<f32>,
    pub crash_v_lateral: Option<f32>,
    pub crash_drift_angle_deg: Option<f32>,
    pub crash_heading_error_deg: Option<f32>,
    pub crash_min_ray: Option<f32>,
    pub crash_type: Option<String>,

    // Value function
    pub mean_value_prediction: Option<f32>,
    pub value_at_crash: Option<f32>,
    pub value_at_start: Option<f32>,

    // Efficiency and exploration
    pub reward_per_second: f32,
    pub furthest_sector: u32,
    pub wall_proximity_fraction: f32,

    // Policy confidence
    pub mean_policy_steering_std: Option<f32>,
    pub mean_policy_throttle_std: Option<f32>,

    // Existing turn metrics
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
    pub env_id: u32,
    pub tick_index: u32,
    pub position_x: f32,
    pub position_y: f32,
    pub progress_fraction: f32,
    pub progress_s: f32,
    pub centerline_distance: f32,
    pub signed_lateral_offset: f32,
    pub speed: f32,
    pub v_forward: f32,
    pub v_lateral: f32,
    pub speed_delta: f32,
    pub drift_angle_deg: f32,
    pub heading_error: f32,
    pub min_ray_distance: f32,
    pub velocity_projection: f32,
    pub centreline_reward: f32,
    pub steering: f32,
    pub throttle: f32,
    pub previous_steering: f32,
    pub previous_throttle: f32,
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
    pub policy_steering_mean: Option<f32>,
    pub policy_steering_std: Option<f32>,
    pub policy_throttle_mean: Option<f32>,
    pub policy_throttle_std: Option<f32>,
}

/// Episode-level trajectory trace with derived control mismatch metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeTrace {
    pub episode_id: u32,
    pub end_reason: String,
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

/// Exported analytics snapshot for one layer after a completed PPO update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PpoLayerRecord {
    pub layer_name: String,
    pub weight_l2_norm: f32,
    pub gradient_l2_norm: f32,
    pub saturated_fraction: Option<f32>,
}

/// Exported analytics snapshot for one completed PPO update.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PpoUpdateRecord {
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
    pub clip_fraction: f32,
    pub approx_kl: f32,
    pub layer_health: Vec<PpoLayerRecord>,
}

/// Exported run-level analytics data.
#[derive(Resource, Default, Debug, Serialize, Deserialize)]
pub struct EpisodeTracker {
    pub episodes: Vec<EpisodeRecord>,
    pub ppo_updates: Vec<PpoUpdateRecord>,
    pub episode_traces: Vec<EpisodeTrace>,
    #[serde(skip)]
    pub last_recorded_update: u64,
}

/// Controls which analytics artefacts are exported on exit.
#[derive(Resource, Debug)]
pub struct AnalyticsConfig {
    /// When true, a full trace JSON (with per-tick data) is written alongside
    /// the compact export. Default: false.
    pub full_trace_export: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            full_trace_export: false,
        }
    }
}

/// Metadata describing the training run — hyperparameters, environment shape,
/// and session identity. Serialised into the compact JSON export header.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub car_count: usize,
    pub track_name: String,
    pub session_timestamp: u64,
    pub ppo_epochs: usize,
    pub clip_epsilon: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub samples_per_tick: usize,
}

/// Compact export schema — episodes and PPO updates with run metadata,
/// but no per-tick trace data. This is the default export format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactRunExport {
    pub metadata: RunMetadata,
    pub episodes: Vec<EpisodeRecord>,
    pub ppo_updates: Vec<PpoUpdateRecord>,
}
