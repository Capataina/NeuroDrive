use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::brain::inspired::BrainUpdateRecord;

pub const NUM_PROGRESS_SECTORS: usize = 20;

/// Classification of how a crash occurred, based on terminal tick kinematics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrashKind {
    Stall,
    Spin,
    Slide,
    Overshoot,
    HeadOn,
}

impl std::fmt::Display for CrashKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrashKind::Stall => write!(f, "Stall"),
            CrashKind::Spin => write!(f, "Spin"),
            CrashKind::Slide => write!(f, "Slide"),
            CrashKind::Overshoot => write!(f, "Overshoot"),
            CrashKind::HeadOn => write!(f, "HeadOn"),
        }
    }
}

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
    pub crash_type: Option<CrashKind>,

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

    /// Which controller drove this episode's car. Serialised as a short
    /// string ("Ppo", "Brain", "Keyboard") for report readability. Defaults
    /// to "Unknown" via `default_controller` so pre-S6 exports deserialise.
    #[serde(default = "default_controller")]
    pub controller: String,

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
    pub ray_distances: [f32; 11],
    pub lookahead_heading_deltas: [f32; 12],
    pub lookahead_curvatures: [f32; 12],
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
///
/// The round-2 fields (`return_*`, `value_norm_*`, `epochs_completed`,
/// `early_stopped`) are defaulted on deserialisation so older JSON exports
/// can still be read after the schema was extended in 2026-04-19.
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

    // ── Round-2 diagnostics (2026-04-19) ──
    /// Minimum of the GAE returns seen in this update's training chunk.
    #[serde(default)]
    pub return_min: f32,
    /// Mean of the GAE returns seen in this update's training chunk.
    #[serde(default)]
    pub return_mean: f32,
    /// Maximum of the GAE returns seen in this update's training chunk.
    #[serde(default)]
    pub return_max: f32,
    /// Standard deviation of the GAE returns in this update's training chunk.
    #[serde(default)]
    pub return_std: f32,
    /// PopArt running-mean of returns after this update. `0.0` when PopArt is
    /// disabled; tracks the critic's value-target distribution otherwise.
    #[serde(default)]
    pub value_norm_mu: f32,
    /// PopArt running-std of returns after this update. `1.0` when PopArt is
    /// disabled; with PopArt this should track the growth of return magnitude.
    #[serde(default = "one_f32")]
    pub value_norm_sigma: f32,
    /// Number of PPO epochs that actually ran for this update. Equals
    /// `ppo_epochs` when target-KL early-stop is disabled or never triggered;
    /// smaller when KL exceeded the configured threshold.
    #[serde(default)]
    pub epochs_completed: u32,
    /// True if the target-KL early-stop fired on this update.
    #[serde(default)]
    pub early_stopped: bool,
}

fn one_f32() -> f32 {
    1.0
}

fn default_controller() -> String {
    "Unknown".to_string()
}

/// Exported run-level analytics data.
#[derive(Resource, Default, Debug, Serialize, Deserialize)]
pub struct EpisodeTracker {
    pub episodes: Vec<EpisodeRecord>,
    pub ppo_updates: Vec<PpoUpdateRecord>,
    pub episode_traces: Vec<EpisodeTrace>,
    /// Brain-inspired learner diagnostics, one record per structural cadence.
    /// Empty when no brain cars ran in this session (`#[serde(default)]` so
    /// historical JSON exports without this field still deserialise).
    #[serde(default)]
    pub brain_records: Vec<BrainUpdateRecord>,
    #[serde(skip)]
    pub last_recorded_update: u64,
    /// Count of brain records already copied from `BrainTrainingStats.history`
    /// into `brain_records`. Kept outside the serialised form — rebuilds from
    /// `brain_records.len()` on load if needed.
    #[serde(skip)]
    pub last_recorded_brain_records: usize,
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
    /// `TrainerLayout::label()` — one of "Keyboard", "AllPpo", "AllBrain",
    /// "SideBySide". Present so the markdown exporter can decide whether to
    /// emit the Fleet Comparison section (only meaningful in SideBySide runs).
    #[serde(default = "default_layout_label")]
    pub layout: String,
    /// Number of PPO cars in this run.
    #[serde(default)]
    pub ppo_cars: usize,
    /// Number of brain-inspired cars in this run.
    #[serde(default)]
    pub brain_cars: usize,
}

fn default_layout_label() -> String {
    "AllPpo".to_string()
}

/// Compact export schema — episodes and PPO updates with run metadata,
/// but no per-tick trace data. This is the default export format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactRunExport {
    pub metadata: RunMetadata,
    pub episodes: Vec<EpisodeRecord>,
    pub ppo_updates: Vec<PpoUpdateRecord>,
    /// Brain-inspired learner diagnostics. `#[serde(default)]` so older
    /// exports (before M6) deserialise cleanly.
    #[serde(default)]
    pub brain_records: Vec<BrainUpdateRecord>,
}
