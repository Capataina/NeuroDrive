pub mod buffer;
pub mod model;
pub mod update;

use std::collections::HashMap;

use bevy::app::AppExit;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::agent::action::{
    ActionState, CarAction, action_smoothing_system, keyboard_action_input_system,
};
use crate::agent::observation::{OBSERVATION_DIM, ObservationVector};
use crate::brain::types::{AgentMode, PolicyOutput};
use crate::game::car::{Car, EnvInstanceId};
use crate::game::episode::EpisodeState;

use self::buffer::TrainerRolloutBuffer;
use self::model::ActorCritic;
use self::update::{
    PreparedUpdate, ppo_finish_epoch, ppo_prepare_update, ppo_process_chunk,
    ppo_update_blocking, squashed_gaussian_log_prob,
};

/// All PPO hyperparameters in a single canonical location.
#[derive(Clone, Debug)]
pub struct PpoConfig {
    // Rollout
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub min_update_steps: usize,
    pub ppo_epochs: usize,
    pub clip_epsilon: f32,
    pub samples_per_tick: usize,
    // Network
    pub actor_hidden_dim: usize,
    pub critic_hidden_dim: usize,
    // Optimiser
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub critic_weight_decay: f32,
    pub entropy_coef: f32,
    pub actor_grad_clip: f32,
    pub critic_grad_clip: f32,
    pub value_huber_delta: f32,
    // Exploration
    pub log_std_floor: f32,
    pub log_std_ceil: f32,
    pub log_std_lr: f32,
    /// Target KL divergence for PPO early-stop guardrail. When set to
    /// `Some(target)`, the PPO epoch loop halts as soon as the observed
    /// approximate KL at the end of an epoch exceeds `1.5 * target`. This is
    /// the SB3 default pattern; see `context/references/ppo-tuning-knobs-racing.md`
    /// Section B.2 for derivation. `None` disables the guardrail.
    pub target_kl: Option<f32>,
    /// PopArt enabled flag. When true, critic targets are normalised to
    /// `~N(0, 1)` inside the training loop and `c_value` output is
    /// denormalised at inference time. The POP rescale step preserves
    /// existing predictions across µ/σ updates. See
    /// `context/references/value-target-normalisation.md`.
    pub popart_enabled: bool,
    /// EMA decay for PopArt running mean/std per PPO update. 1e-4 is the
    /// torchbeastpopart convention adapted to our per-update (not
    /// per-step) cadence.
    pub popart_beta: f32,
    /// Numerical floor on PopArt's σ — prevents division by zero when
    /// the return distribution temporarily collapses.
    pub popart_sigma_floor: f32,
}

/// PopArt state — running mean/std of the critic's value targets.
///
/// `mu` and `sigma` are updated by an EMA of per-batch statistics. On each
/// update, the `c_value` layer's weights and bias are rescaled so that
/// externally-observed value predictions (`σ·z + µ`) are preserved across
/// the statistics change. This keeps training on a stationary target
/// distribution (`~N(0, 1)`) while the network itself continues to track
/// the real return scale.
#[derive(Clone, Debug)]
pub struct ValueNorm {
    pub mu: f32,
    pub sigma: f32,
}

impl Default for ValueNorm {
    fn default() -> Self {
        // Identity transform at init — no rescale applied until the first
        // update supplies real statistics.
        Self { mu: 0.0, sigma: 1.0 }
    }
}

impl ValueNorm {
    /// Denormalises a raw critic output `z` back to reward units.
    #[inline]
    pub fn denormalise(&self, raw: f32) -> f32 {
        self.sigma * raw + self.mu
    }

    /// Normalises a reward-unit return into the target space the critic
    /// actually regresses to. Used by tests and reserved for a future
    /// consumer in the exit-flush path; the hot update-loop version is
    /// inlined in `update.rs` to avoid a function-call in the inner loop.
    #[allow(dead_code)]
    #[inline]
    pub fn normalise(&self, value: f32) -> f32 {
        (value - self.mu) / self.sigma
    }
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            // Round-2 (2026-04-19): γ raised 0.99 → 0.995 per
            // `context/references/ppo-tuning-knobs-racing.md` section C.
            // Effective credit horizon 1.67s → 3.33s to match the observation
            // lookahead window (~2.6s at current speeds).
            gamma: 0.995,
            gae_lambda: 0.95,
            max_steps: 512,
            min_update_steps: 128,
            ppo_epochs: 4,
            clip_epsilon: 0.2,
            samples_per_tick: 32,
            actor_hidden_dim: 64,
            critic_hidden_dim: 128,
            actor_lr: 3e-4,
            critic_lr: 5e-4,
            critic_weight_decay: 3e-4,
            entropy_coef: 0.01,
            actor_grad_clip: 0.5,
            critic_grad_clip: 0.5,
            value_huber_delta: 1.0,
            log_std_floor: -1.0,
            log_std_ceil: 0.5,
            log_std_lr: 3e-4,
            // Round-2: target-KL early stop — halts PPO epochs when the
            // approximate KL divergence exceeds 1.5 × target_kl (SB3
            // convention). `None` disables the guardrail entirely.
            target_kl: Some(0.03),
            popart_enabled: true,
            // Raised from 1e-4 → 3e-2 after the first post-round-2 training
            // run (`reports/analytics/run_1776556719.md`). At 1e-4 per
            // update, PopArt retained ~85% of its zero-initial state after
            // 1,582 updates — barely tracking the actual return
            // distribution (batch_mean 262 vs PopArt µ 30). The critic
            // became biased LOW (residual mean -339, section 14) because
            // σ·z + µ ≈ 80·z + 30 could not reach the ~500+ reward-unit
            // returns some episodes delivered. At 3e-2 the EMA tracks the
            // batch mean within ~30 updates.
            popart_beta: 3e-2,
            popart_sigma_floor: 1e-4,
        }
    }
}

/// Shared PPO brain resource. Owns the policy/value network and hyperparameters.
/// The rollout buffer is now a separate `TrainerRolloutBuffer` resource.
#[derive(Resource)]
pub struct PpoBrain {
    pub model: ActorCritic,
    pub rng: StdRng,
    pub config: PpoConfig,
    pub step_counter: usize,
    /// PopArt running statistics on the critic's value target.
    pub value_norm: ValueNorm,
    /// Reusable scratch for `ppo_act_all_cars_system` — carries `(entity, env_id)`
    /// pairs from the observation-stacking pass to the sampling pass. Cleared
    /// and refilled each tick; capacity preserved so no per-frame allocation.
    act_entity_buffer: Vec<(bevy::ecs::entity::Entity, u32)>,
}

impl Default for PpoBrain {
    fn default() -> Self {
        let config = PpoConfig::default();
        let mut init_rng = rand::rng();
        Self {
            model: ActorCritic::new(OBSERVATION_DIM, config.actor_hidden_dim, config.critic_hidden_dim, 2, config.actor_lr, config.critic_lr, config.critic_weight_decay, &mut init_rng),
            rng: StdRng::from_rng(&mut init_rng),
            config,
            step_counter: 0,
            value_norm: ValueNorm::default(),
            act_entity_buffer: Vec::with_capacity(16),
        }
    }
}

/// Snapshot of one layer's parameter and activation health after a PPO update.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PpoLayerHealth {
    pub layer_name: String,
    pub weight_l2_norm: f32,
    pub gradient_l2_norm: f32,
    pub saturated_fraction: Option<f32>,
}

/// Aggregated learning-health metrics for the most recent completed PPO update.
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct PpoTrainingStats {
    pub last_completed_update: u64,
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
    pub layer_health: Vec<PpoLayerHealth>,

    // ── Round-2 diagnostics (2026-04-19) ──
    /// Minimum / mean / max / std of the returns seen by this update.
    pub return_min: f32,
    pub return_mean: f32,
    pub return_max: f32,
    pub return_std: f32,
    /// PopArt running mean of returns (identity when disabled).
    pub value_norm_mu: f32,
    /// PopArt running std of returns (1.0 when disabled).
    pub value_norm_sigma: f32,
    /// PPO epochs that actually ran for this update (may be less than
    /// `ppo_epochs` when target-KL early-stop triggers).
    pub epochs_completed: u32,
    /// True if target-KL early-stop fired on this update.
    pub early_stopped: bool,
}

impl Default for PpoTrainingStats {
    fn default() -> Self {
        Self {
            last_completed_update: 0,
            batch_size: 0,
            policy_loss: 0.0,
            value_loss: 0.0,
            policy_entropy: 0.0,
            explained_variance: 0.0,
            steering_mean: 0.0,
            steering_std: 0.0,
            throttle_mean: 0.0,
            throttle_std: 0.0,
            clamped_action_fraction: 0.0,
            clip_fraction: 0.0,
            approx_kl: 0.0,
            layer_health: Vec::new(),
            return_min: 0.0,
            return_mean: 0.0,
            return_max: 0.0,
            return_std: 0.0,
            value_norm_mu: 0.0,
            value_norm_sigma: 1.0,
            epochs_completed: 0,
            early_stopped: false,
        }
    }
}

/// Holds an in-progress PPO update that is amortised across frames.
/// One epoch runs per `FixedUpdate` tick to keep the simulation smooth.
#[derive(Resource, Default)]
pub struct PpoUpdateState {
    prepared: Option<PreparedUpdate>,
}

pub struct PpoPlugin;

impl Plugin for PpoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PpoBrain>()
            .init_resource::<PpoTrainingStats>()
            .init_resource::<TrainerRolloutBuffer>()
            .init_resource::<PpoUpdateState>()
            .add_systems(
                FixedUpdate,
                ppo_act_all_cars_system
                    .after(keyboard_action_input_system)
                    .before(action_smoothing_system)
                    .in_set(crate::sim::sets::SimSet::Input),
            )
            .add_systems(
                FixedUpdate,
                (
                    ppo_collect_rewards_all_cars_system,
                    ppo_epoch_system.after(ppo_collect_rewards_all_cars_system),
                )
                    .after(crate::game::episode::episode_loop_system)
                    .after(crate::agent::observation::build_observation_vector_system)
                    .in_set(crate::sim::sets::SimSet::Measurement),
            )
            .add_systems(Last, ppo_flush_on_exit_system);
    }
}

/// Runs the shared policy for all cars, writes per-car actions, and pushes
/// all transitions to the `TrainerRolloutBuffer` with `env_id` tagging.
///
/// Structured as three passes to batch **both** forwards:
/// 1. Stack observations into `batch_io.obs_batch`; record `(entity, env_id)` per car.
/// 2. Single batched actor forward + single batched critic forward — one mat-mat
///    each instead of 2N mat-vecs.
/// 3. Per-car: read mean from scratch, sample action, write `PolicyOutput`,
///    push transition to the rollout buffer.
///
/// The Gaussian sampling in pass 3 must stay sequential to preserve the
/// shared-RNG determinism contract (same seed → same action stream).
pub fn ppo_act_all_cars_system(
    mode: Res<AgentMode>,
    mut car_query: Query<(bevy::ecs::entity::Entity, &EnvInstanceId, &ObservationVector, &mut ActionState, &mut PolicyOutput), With<Car>>,
    mut brain: ResMut<PpoBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    let obs_dim = brain.model.scratch.obs_dim;
    let act_dim = brain.model.scratch.act_dim;

    // ── Pass 1: stack observations into batch_io.obs_batch ──────────
    // Reuses the pre-allocated input scratch; no per-frame heap alloc.
    // Destructure `brain` into disjoint field borrows so `act_entity_buffer`
    // and `model.batch_io.obs_batch` can both be mutated in the same loop.
    {
        let PpoBrain { act_entity_buffer, model, .. } = &mut *brain;
        act_entity_buffer.clear();
        let obs_batch = &mut model.batch_io.obs_batch;
        for (entity, env_id, obs, _action_state, _policy_output) in car_query.iter() {
            let idx = act_entity_buffer.len();
            obs_batch[idx * obs_dim..(idx + 1) * obs_dim].copy_from_slice(&obs.values);
            act_entity_buffer.push((entity, env_id.0));
        }
    }
    let car_count = brain.act_entity_buffer.len();
    if car_count == 0 {
        return;
    }

    // ── Pass 2: batched actor + critic forward (one mat-mat each) ───
    brain.model.forward_actor_batch(car_count);
    brain.model.forward_critic_batch(car_count);

    // ── Pass 3: per-car sampling, PolicyOutput writes, buffer pushes ─
    // std is shared across all cars (single a_log_std vector on the model),
    // so we compute it once outside the loop.
    let std_vals: [f32; 2] = [
        brain.model.a_log_std[0].exp(),
        brain.model.a_log_std[1].exp(),
    ];

    // PopArt denormalisation constants snapshot — read once outside the
    // loop since `value_norm` is not mutated during action selection.
    // When PopArt is disabled the state is `(µ=0, σ=1)` and denormalisation
    // is an identity transform.
    let value_norm_mu = brain.value_norm.mu;
    let value_norm_sigma = brain.value_norm.sigma;

    for i in 0..car_count {
        let (entity, env_id) = brain.act_entity_buffer[i];
        let mean: [f32; 2] = [
            brain.model.scratch.a_out[i * act_dim],
            brain.model.scratch.a_out[i * act_dim + 1],
        ];
        // Critic raw output represents the normalised value. Denormalise
        // before exposing to consumers (PolicyOutput, rollout buffer, GAE).
        let value = value_norm_sigma * brain.model.scratch.c_out[i] + value_norm_mu;

        // Sample latent actions from the squashed Gaussian, then squash/remap.
        let mut actions = [0.0f32; 2];
        let mut latent_actions = [0.0f32; 2];
        for j in 0..2 {
            let latent = crate::brain::common::math::sample_normal(
                mean[j],
                std_vals[j],
                &mut brain.rng,
            );
            latent_actions[j] = latent;
            let squashed = latent.tanh();
            actions[j] = if j == 1 { 0.5 * (squashed + 1.0) } else { squashed };
        }

        let raw_action = CarAction { steering: actions[0], throttle: actions[1] };
        let applied_action = raw_action.clamped();
        let safety_clamp_hits = [
            (applied_action.steering - raw_action.steering).abs() > 1e-6,
            (applied_action.throttle - raw_action.throttle).abs() > 1e-6,
        ];
        actions[0] = applied_action.steering;
        actions[1] = applied_action.throttle;

        let mut old_log_prob = 0.0;
        for j in 0..2 {
            let squashed = if j == 0 { actions[j] } else { 2.0 * actions[j] - 1.0 };
            old_log_prob += squashed_gaussian_log_prob(
                latent_actions[j],
                squashed,
                mean[j],
                std_vals[j],
                j,
            );
        }

        if let Ok((_entity, _env_id, obs, mut action_state, mut policy_output)) = car_query.get_mut(entity) {
            action_state.desired = applied_action;
            policy_output.value_prediction = value;
            policy_output.steering_mean = mean[0];
            policy_output.steering_std = std_vals[0];
            policy_output.throttle_mean = mean[1];
            policy_output.throttle_std = std_vals[1];

            buffer.push_pre_step(
                env_id,
                &obs.values,
                &actions,
                &latent_actions,
                safety_clamp_hits,
                value,
                old_log_prob,
            );
        }
    }

    brain.step_counter += car_count;
}

/// Collects per-car rewards and done flags. When the buffer reaches the
/// horizon, prepares a PPO update (GAE + frozen buffer) for the epoch system
/// to process one epoch per tick.
pub fn ppo_collect_rewards_all_cars_system(
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<PpoBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut update_state: ResMut<PpoUpdateState>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    if buffer.pending_rewards() == 0 {
        return;
    }

    let mut any_done = false;
    for (_, _, episode_state) in car_query.iter() {
        let done = episode_state.tick.end_reason.is_some();
        buffer.push_reward(episode_state.tick.reward, done);
        if done {
            any_done = true;
        }
    }

    debug_assert!(
        buffer.is_aligned(),
        "Trainer rollout buffer misaligned: states={}, rewards={}, env_ids={}",
        buffer.pre_step_count(),
        buffer.len(),
        buffer.env_ids.len(),
    );

    let reached_horizon = buffer.len() >= brain.config.max_steps;
    let reached_terminal_batch = any_done && buffer.len() >= brain.config.min_update_steps;

    // Only start a new update if there is no epoch-spread update in progress
    if (reached_horizon || reached_terminal_batch) && update_state.prepared.is_none() {
        let mut bootstrap_values: HashMap<u32, f32> = HashMap::new();
        for (env_id, obs, episode_state) in car_query.iter() {
            let done = episode_state.tick.end_reason.is_some();
            if done {
                bootstrap_values.insert(env_id.0, 0.0);
            } else {
                // forward_critic returns the raw (normalised) critic output;
                // GAE expects reward-unit values, so denormalise via PopArt
                // state (identity when PopArt is disabled).
                let raw = brain.model.forward_critic(&obs.values);
                bootstrap_values.insert(env_id.0, brain.value_norm.denormalise(raw));
            }
        }

        if let Some(prepared) = ppo_prepare_update(&mut brain, &mut buffer, &bootstrap_values) {
            update_state.prepared = Some(prepared);
            // buffer is already empty — ppo_prepare_update took its contents.
        } else {
            buffer.clear();
        }
    }
}

/// Processes a chunk of samples per tick from the in-progress PPO update.
/// A full epoch is split across multiple ticks (samples_per_tick samples each)
/// so the simulation stays smooth.
///
/// Round-2 (2026-04-19): after each completed epoch, checks the
/// approximate KL against `config.target_kl`. If KL exceeds the SB3-style
/// `1.5 * target_kl` guardrail, the current epoch is treated as the final
/// one — remaining scheduled epochs are skipped and `stats.early_stopped`
/// is set. See `context/references/ppo-tuning-knobs-racing.md` Section B.
pub fn ppo_epoch_system(
    mode: Res<AgentMode>,
    mut brain: ResMut<PpoBrain>,
    mut update_state: ResMut<PpoUpdateState>,
    mut stats: ResMut<PpoTrainingStats>,
) {
    if *mode != AgentMode::Ai {
        return;
    }

    let Some(prepared) = update_state.prepared.as_mut() else {
        return;
    };

    if !prepared.is_active() {
        update_state.prepared = None;
        return;
    }

    let chunk_size = brain.config.samples_per_tick;
    let epoch_complete = ppo_process_chunk(&mut brain, prepared, chunk_size);

    if epoch_complete {
        // Compute this epoch's KL from the accumulator (still intact before
        // the next chunk resets it). Used to decide target-KL early stop.
        let batch_size_f32 = (prepared.frozen_buffer.len() as f32).max(1.0);
        let epoch_kl = prepared.accum.approx_kl_sum / batch_size_f32;
        let kl_early_stop = brain
            .config
            .target_kl
            .map(|target| epoch_kl > 1.5 * target)
            .unwrap_or(false);

        let is_final = prepared.epochs_remaining == 1 || kl_early_stop;
        ppo_finish_epoch(&mut brain, prepared, &mut stats, is_final);
        prepared.epochs_remaining -= 1;
        prepared.sample_offset = 0;

        if is_final {
            // Overwrite caller-owned fields on stats. `ppo_finish_epoch` wrote
            // `epochs_completed = config.ppo_epochs` as a default for the
            // no-early-stop case; correct it if we stopped early.
            stats.early_stopped = kl_early_stop;
            if kl_early_stop {
                let epochs_done = brain
                    .config
                    .ppo_epochs
                    .saturating_sub(prepared.epochs_remaining);
                stats.epochs_completed = epochs_done as u32;
            }
            prepared.epochs_remaining = 0;
        }

        if prepared.epochs_remaining == 0 {
            update_state.prepared = None;
        }
    }
}

/// Flushes any in-progress PPO epochs and remaining rollout data on exit.
/// Runs synchronously since frame budget does not matter at shutdown.
pub fn ppo_flush_on_exit_system(
    mut exit_events: MessageReader<AppExit>,
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState), With<Car>>,
    mut brain: ResMut<PpoBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut update_state: ResMut<PpoUpdateState>,
    mut stats: ResMut<PpoTrainingStats>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    if *mode != AgentMode::Ai {
        buffer.clear();
        update_state.prepared = None;
        return;
    }

    // Finish any in-progress staged update
    if let Some(prepared) = update_state.prepared.as_mut() {
        let batch_size = prepared.frozen_buffer.len();
        // Finish current epoch if partially processed
        if prepared.sample_offset > 0 {
            ppo_process_chunk(&mut brain, prepared, batch_size);
            let is_final = prepared.epochs_remaining == 1;
            ppo_finish_epoch(&mut brain, prepared, &mut stats, is_final);
            prepared.epochs_remaining -= 1;
            prepared.sample_offset = 0;
        }
        // Run any remaining full epochs
        while prepared.is_active() {
            let is_final = prepared.epochs_remaining == 1;
            ppo_process_chunk(&mut brain, prepared, batch_size);
            ppo_finish_epoch(&mut brain, prepared, &mut stats, is_final);
            prepared.epochs_remaining -= 1;
            prepared.sample_offset = 0;
        }
        update_state.prepared = None;
    }

    // Flush any remaining buffer data
    if buffer.len() == 0 {
        return;
    }

    let mut bootstrap_values: HashMap<u32, f32> = HashMap::new();
    for (env_id, obs, episode_state) in car_query.iter() {
        let done = episode_state.tick.end_reason.is_some();
        if done {
            bootstrap_values.insert(env_id.0, 0.0);
        } else {
            let raw = brain.model.forward_critic(&obs.values);
            bootstrap_values.insert(env_id.0, brain.value_norm.denormalise(raw));
        }
    }

    ppo_update_blocking(&mut brain, &mut buffer, &mut stats, &bootstrap_values);
    // buffer is already empty — ppo_update_blocking took its contents.
}
