//! Brain-inspired learner (Milestone 6).
//!
//! A sparse directed graph of rate-coded tanh neurons trained by local
//! three-factor plasticity with eligibility traces, raw per-tick reward as
//! modulator, homeostasis, and continual-backprop-style structural plasticity.
//!
//! See `context/notes/brain-v1-design.md` for the design rationale and
//! `context/plans/` for the implementation plan.
//!
//! ## Stages
//!
//! - S1 (this file): plumbing + forward pass, no learning.
//! - S2: three-factor plasticity + eligibility traces.
//! - S3: synaptic scaling + intrinsic excitability.
//! - S4: structural plasticity (replacement, neurogenesis, prune/sprout).
//! - S5: analytics integration.
//! - S6: side-by-side vs PPO mode.

pub mod config;
pub mod forward;
pub mod graph;
pub mod homeostasis;
pub mod plasticity;
pub mod structural;

use std::collections::VecDeque;

use bevy::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::agent::action::{
    ActionState, CarAction, action_smoothing_system, keyboard_action_input_system,
};
use crate::agent::observation::ObservationVector;
use crate::brain::types::{BrainCar, PolicyOutput};
use crate::game::car::{Car, EnvInstanceId, TrainerConfig};
use crate::game::episode::EpisodeState;

use self::config::BrainInspiredConfig;
pub use self::forward::{NeuronActivations, forward_tick};
use self::graph::BrainGraph;

/// Running statistics populated every tick and drained into
/// `BrainTrainingStats` (S5) on the structural cadence.
#[derive(Clone, Debug, Default)]
pub struct BrainRunningStats {
    /// Number of plasticity weight updates applied since last flush.
    pub plasticity_updates: u64,
    /// Number of neurons replaced via continual backprop.
    pub replacement_events: u64,
    /// Number of neurogenesis events (plateau-triggered).
    pub neurogenesis_events: u64,
    /// Number of synapse prune operations.
    pub prune_events: u64,
    /// Number of synapse sprout operations.
    pub sprout_events: u64,
    /// Most recent sampled mean |w|.
    pub mean_abs_weight: f32,
    /// Most recent sampled mean |eligibility|.
    pub mean_abs_eligibility: f32,
    /// Most recent sampled dead-neuron fraction (`mean_rate < 0.01`).
    pub dead_neuron_fraction: f32,
    /// Most recent sampled saturation fraction (`|curr| > 0.95` averaged over cars).
    pub saturation_fraction: f32,
    /// Most recent observed modulator M (mean over cars this tick).
    pub last_mean_m: f32,
}

/// One flushed snapshot of brain-inspired diagnostics. Populated every
/// `structural_cadence` ticks and pushed onto `BrainTrainingStats.history`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BrainUpdateRecord {
    pub tick_start: u64,
    pub tick_end: u64,
    pub neuron_count: u32,
    pub hidden_count: u32,
    pub synapse_count: u32,
    pub mean_abs_weight: f32,
    pub weight_sigma: f32,
    pub mean_abs_eligibility: f32,
    pub mean_utility: f32,
    pub utility_p10: f32,
    pub utility_p90: f32,
    pub replacement_count: u32,
    pub neurogenesis_count: u32,
    pub prune_count: u32,
    pub sprout_count: u32,
    pub dead_neuron_fraction: f32,
    pub saturation_fraction: f32,
    pub mean_m: f32,
}

/// Aggregated brain-inspired training health. Mirrors the role of
/// `PpoTrainingStats` — exposed to analytics, HUD, and leaderboard.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct BrainTrainingStats {
    /// The most recent fully-flushed record (may be `BrainUpdateRecord::default()`
    /// before the first flush).
    pub latest: BrainUpdateRecord,
    /// All records produced so far, in chronological order. Used by analytics
    /// to build sparklines in the markdown report (S5).
    pub history: Vec<BrainUpdateRecord>,
    /// Total ticks the brain has run (cumulative across F4 resets in a session).
    pub total_ticks: u64,
}

/// Resource owning the shared brain graph and its RNG.
///
/// One graph, many embodiments: every car marked `BrainCar` reads
/// observations and writes actions through this single graph.
#[derive(Resource)]
pub struct BrainBrain {
    pub graph: BrainGraph,
    pub config: BrainInspiredConfig,
    pub rng: StdRng,
    pub tick_counter: u64,
    pub stats: BrainRunningStats,
    /// Rolling episode-mean reward window used for plateau detection in S4.
    pub reward_window: VecDeque<f32>,
}

impl BrainBrain {
    pub fn new(config: BrainInspiredConfig, num_cars: usize) -> Self {
        let mut seed_rng = match config.rng_seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        let graph = BrainGraph::seed(&config, num_cars.max(1), &mut seed_rng);
        Self {
            graph,
            config,
            rng: seed_rng,
            tick_counter: 0,
            stats: BrainRunningStats::default(),
            reward_window: VecDeque::with_capacity(128),
        }
    }

    /// Wipes state back to a fresh seed graph, keeping config. Called by the
    /// F4 handler when layouts change so brain-inspired runs start clean.
    pub fn reset_to_seed(&mut self, num_cars: usize) {
        let mut seed_rng = match self.config.rng_seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        self.graph = BrainGraph::seed(&self.config, num_cars.max(1), &mut seed_rng);
        self.rng = seed_rng;
        self.tick_counter = 0;
        self.stats = BrainRunningStats::default();
        self.reward_window.clear();
    }
}

impl Default for BrainBrain {
    fn default() -> Self {
        // Size for the worst-case default (side-by-side uses 8 brain cars).
        // The per-car eligibility vectors are short `Vec<f32>`s, so this is
        // cheap even at 16.
        Self::new(BrainInspiredConfig::default(), 16)
    }
}

pub struct BrainInspiredPlugin;

impl Plugin for BrainInspiredPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrainBrain>()
            .init_resource::<BrainTrainingStats>()
            .add_systems(
                FixedUpdate,
                brain_act_all_cars_system
                    .after(keyboard_action_input_system)
                    .before(action_smoothing_system)
                    .in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                brain_learn_all_cars_system
                    .after(crate::game::episode::episode_loop_system)
                    .after(crate::agent::observation::build_observation_vector_system)
                    .in_set(crate::sim::sets::SimSet::Measurement),
            );
    }
}

/// Runs the shared brain graph's forward pass for every `BrainCar`, writes
/// per-car actions, and publishes `PolicyOutput` diagnostics.
///
/// ## Field repurposing in `PolicyOutput`
/// - `steering_mean` / `throttle_mean` ← raw output-neuron activations.
/// - `steering_std` / `throttle_std` ← 0.0 (brain is deterministic given
///   observations; no sampling distribution).
/// - `value_prediction` ← 0.0 in S1. Populated with the modulator M (per-car
///   reward) by the learn system starting in S2.
pub fn brain_act_all_cars_system(
    mut car_query: Query<
        (
            &EnvInstanceId,
            &ObservationVector,
            &mut ActionState,
            &mut PolicyOutput,
            &mut NeuronActivations,
        ),
        (With<Car>, With<BrainCar>),
    >,
    mut brain: ResMut<BrainBrain>,
) {
    let mut any_car = false;
    for (_env_id, obs, mut action_state, mut policy_output, mut activations) in car_query.iter_mut() {
        any_car = true;
        let (steering, throttle) = forward_tick(&brain.graph, &mut activations, &obs.values);
        let raw_action = CarAction { steering, throttle };
        let applied = raw_action.clamped();
        action_state.desired = applied;

        // Diagnostic surface — keep PolicyOutput populated so downstream
        // analytics/HUD keep working without mode-specific branching.
        policy_output.steering_mean = steering;
        policy_output.steering_std = 0.0;
        policy_output.throttle_mean = throttle;
        policy_output.throttle_std = 0.0;
        // `value_prediction` stays at whatever the last system wrote to it;
        // S2 will populate it with the modulator M per car.
    }

    if any_car {
        brain.tick_counter += 1;
    }
}

/// Per-tick plasticity system. Implemented in S2.
///
/// S1 stub: no-op. Registered in the plugin so S2 is a one-function swap with
/// no plugin wiring changes.
pub fn brain_learn_all_cars_system(
    _car_query: Query<
        (&EnvInstanceId, &EpisodeState, &NeuronActivations),
        (With<Car>, With<BrainCar>),
    >,
    _brain: ResMut<BrainBrain>,
    _stats: ResMut<BrainTrainingStats>,
    _trainer_config: Res<TrainerConfig>,
) {
    // S2 will implement eligibility trace updates + weight updates.
    // S3 will hook in homeostasis on the structural cadence.
    // S4 will hook in structural plasticity on the structural cadence.
    // S5 will drain BrainRunningStats into BrainTrainingStats.
}
