# System — Brain PPO Baseline

## Scope / Purpose

- Provide the current autonomous-controller baseline used to validate that the environment and observation contract are learnable.
- Keep the baseline self-contained in Rust without external ML frameworks.
- PPO (upgraded from A2C) exists as a **diagnostic tool**, not the intended final learning architecture. The project's long-term direction is brain-inspired local plasticity.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/brain/plugin.rs` | `BrainPlugin`, `AgentMode` toggle (F4), registers `TrainerLiveRanking`, wires ranking and visual-role systems, rollout-buffer reset on mode switch | Policy implementation details |
| `src/brain/types.rs` | `AgentMode` enum (`Keyboard` / `Ai`), `PolicyOutput` component | Observation or reward production |
| `src/brain/ppo/` | `PpoBrain` (model + hyperparams + seeded RNG + `act_entity_buffer` reusable scratch, **no buffer**), `TrainerRolloutBuffer` (separate resource with env_id tagging + old log-probs + reusable `EnvGrouping` scratch), `PpoUpdateState` (staged update resource), `PpoPlugin`, vectorised act/collect/epoch/flush systems, per-env GAE, `ActorCritic` model (asymmetric: actor 2×64, critic 2×128, with `BatchIo`+`BatchScratch`+`SampleScratch` pre-allocation), PPO clipped update logic, `PpoTrainingStats`, `PpoLayerHealth` | Environment truth, observation construction |
| `src/brain/ranking.rs` | `TrainerLiveRanking` resource, ranking computation, `update_car_visual_roles_system` (best-car highlighting via alpha + z-order), `CarColour`-aware sprite updates | Policy, observation, reward |
| `src/brain/common/` | Reusable handwritten ML primitives: `Linear` (flat `Vec<f32>` weight storage, batched forward/backward routed through `gemm_backend`), `Tanh` (with saturation tracking), `AdamOptimizer` (AdamW-style with decoupled weight decay, precomputed bias correction, ε=1e-5), Gaussian math, orthogonal init, `gemm_backend` (three-way GEMM dispatch: scalar / matrixmultiply / accelerate) | Algorithm-specific logic |

## Current Implemented Reality

### Mode Switching

- `AgentMode` defaults to `Ai`.
- `F4` toggles between `Ai` and `Keyboard` in the `Update` schedule.
- Toggling clears `TrainerRolloutBuffer` (separate resource, not brain-owned) and resets `step_counter` to avoid mixed-control trajectories.

### Model Architecture

```text
Actor:                                Critic:
obs (43) → Linear(43,64)  → Tanh     obs (43) → Linear(43,128) → Tanh
         → Linear(64,64)  → Tanh              → Linear(128,128) → Tanh
         → Linear(64,2)   → mean              → Linear(128,1)   → value
         + learnable log_std (2)
```

- **Asymmetric actor-critic** — the critic uses 2×128 hidden layers (double the actor's 2×64) to provide sufficient capacity for value prediction. The actor is kept smaller because it converges faster and doesn't need the extra capacity.
- Separate actor and critic stacks (no shared backbone).
- Orthogonal initialisation: √2 scale for hidden layers, 0.01× for actor mean output (near-zero initial policy), 1.0× for critic value output.
- `log_std` initialised to `[0.0, 0.0]` (initial σ = 1.0). **Floor clamped at -1.0** (minimum σ ≈ 0.37) to prevent throttle exploration collapse.
- Actor LR: 3e-4 (standard Adam, weight decay 0.0). Critic LR: 5e-4 (**AdamW with weight decay λ=3e-4**) to prevent unbounded weight growth that drives tanh saturation.
- Activation: Tanh throughout (switched from ReLU — eliminates dead-neuron capacity loss that was starving the actor at 34–57% dead neurons).
- The model exposes three forward paths, all **batched**:
  - `forward_actor_batch(batch_size)` — actor-only batched forward for multi-car action selection. Reads observations from `self.batch_io.obs_batch`, writes means into `self.scratch.a_out`. Does not cache intermediates (inference only).
  - `forward_critic_batch(batch_size)` — critic-only batched forward for multi-car bootstrap values. Reads from `batch_io.obs_batch`, writes values into `scratch.c_out`. Does not cache intermediates.
  - `forward_critic(obs)` — single-sample critic forward using `SampleScratch`. Used only on the reward-collection bootstrap path (per non-terminal car at update prepare time) and the exit-flush bootstrap path.
- `forward_batch(batch_size)` (full actor + critic) and `backward_batch_actor(batch_size)` / `backward_batch_critic(batch_size)` are the training-path entry points. All read from `batch_io` and write into `scratch`; the mat-mat work routes through the active GEMM backend.

### Performance Optimisations

- **Flat `Vec<f32>` weight storage** — `Linear.weights` is a single contiguous vector in row-major order (`weights[i * in_dim + j]`), enabling cache-friendly traversal and LLVM auto-vectorisation. Previously used `Vec<Vec<f32>>`.
- **Three-backend GEMM dispatch** (2026-04-18) — every mat-mat in `Linear::forward_batch` / `Linear::backward_batch` routes through `src/brain/common/gemm_backend.rs`, which selects one of three implementations at compile time: scalar naive loops, `matrixmultiply` (pure-Rust BLIS-style NEON microkernel), or Apple Accelerate (`cblas_sgemm`, macOS-only, dispatches to AMX). Default auto-selects Accelerate on macOS, matrixmultiply elsewhere. Opt-in overrides via `force-scalar` / `force-matrixmultiply` / `force-accelerate` Cargo features. Delivered a 30× speedup on PPO Epoch (13.5 ms → 0.45 ms).
- **Batched multi-car action selection** (2026-04-18) — `ppo_act_all_cars_system` stacks all N cars' observations into `batch_io.obs_batch` in one pass, then runs a single batched actor forward + single batched critic forward (one mat-mat each) rather than N sequential per-car mat-vec pairs. Delivered a 16× speedup on action selection (1.98 ms → 0.13 ms). Per-car Gaussian sampling remains sequential to preserve the shared-RNG determinism contract.
- **Pre-allocated scratch buffers** — `BatchIo` (obs_batch + grad_seed_values + grad_seed_means — all the input sides), `BatchScratch` (forward/backward intermediates for the training path), `SampleScratch` (critic-only intermediates for single-sample forward). Kept as sibling fields on `ActorCritic` so Rust's disjoint-field borrow inference lets `forward_batch` mutably borrow `scratch` while reading `batch_io` — replaces three previous `unsafe { slice::from_raw_parts }` blocks.
- **Reusable `act_entity_buffer`** on `PpoBrain` carries `(Entity, env_id)` pairs from Pass 1 (stack observations) to Pass 3 (sample + write components + push buffer) of `ppo_act_all_cars_system` without per-frame heap allocation.
- **Reusable `EnvGrouping`** on `TrainerRolloutBuffer` — replaces a per-call `HashMap<u32, Vec<usize>>` allocation in `compute_gae_per_env` with a `Vec<Vec<usize>>` indexed by `env_id as usize`. Faster (no hashing, no probing) and deterministic iteration order (was HashMap-order-dependent).
- **Iterator-based inner loops** — dot products and gradient accumulation use iterator chains for LLVM optimisation in the scalar backend.
- **Swap instead of clone for frozen buffer** — `PreparedUpdate` takes ownership of buffer data via swap rather than cloning.
- **Adam precomputed bias correction** — `1/(1-β^t)` computed once per step rather than per-parameter.
- **Accelerate thread pin** — `main.rs` sets `VECLIB_MAXIMUM_THREADS=1` on macOS at startup to prevent Accelerate's default thread pool from fighting Bevy's render pipeline at our small matrix sizes.

### PolicyOutput Component

- `PolicyOutput` is a **per-car Component** written by `ppo_act_all_cars_system` each tick.
- Contains: `value_prediction`, `steering_mean`, `steering_std`, `throttle_mean`, `throttle_std`.
- Exposes brain internals for analytics capture without requiring analytics to call model forward passes.

### Action Selection

- The policy samples Gaussian latent actions from `N(mean, exp(log_std))`.
- Applies `tanh` squashing: steering uses tanh directly to `[-1, 1]`; throttle uses `0.5*(tanh+1)` to map to `[0, 1]`.
- Safety clamping applied after squashing; clamp-hit flags tracked per step.
- RNG is a **seeded `StdRng`** stored in `PpoBrain` — deterministic within a session. Initialised from `rand::rng()` at brain construction, then reused for all sampling.
- All cars receive actions from the shared policy each tick (not just one car).

### Rollout Collection

- `ppo_act_all_cars_system` runs in `SimSet::Input` after keyboard input and before action smoothing. Structured as three passes:
  1. **Stack observations:** iterate all cars, copy each `ObservationVector` into `brain.model.batch_io.obs_batch`, record `(Entity, env_id)` in `brain.act_entity_buffer`. No heap allocation on the hot path.
  2. **Batched forward:** one `forward_actor_batch(car_count)` call (writes means into `scratch.a_out`), then one `forward_critic_batch(car_count)` call (writes values into `scratch.c_out`). Two mat-mats total, regardless of car count.
  3. **Per-car sample + write:** for each car, read mean from `scratch.a_out[i*act_dim..i*act_dim+2]`, sample Gaussian latent via the seeded RNG, tanh-squash, remap throttle, compute old_log_prob, write `ActionState.desired` + `PolicyOutput`, push to `TrainerRolloutBuffer` with `env_id` + `old_log_prob`. Sequential by design to preserve shared-RNG determinism.
- `ppo_collect_rewards_all_cars_system` runs in `SimSet::Measurement` after episode truth and observation rebuild. Pushes per-car reward and done flag for all cars. When the buffer reaches the horizon and no update is in progress, calls `ppo_prepare_update` to compute per-env GAE, freeze the buffer into a `PreparedUpdate` (via swap, not clone), and clear the live buffer.
- `ppo_epoch_system` runs in `SimSet::Measurement` after the collect system. Processes a chunk of `samples_per_tick` (default 32 as of 2026-04-18) samples from the `PreparedUpdate` per tick using batched forward/backward passes through the active GEMM backend. When an epoch's samples are exhausted, calls `ppo_finish_epoch` (clip gradients, step optimiser). Advances to the next epoch or clears the update state when all epochs are done.
- `TrainerRolloutBuffer` stores: `states`, `actions`, `latent_actions`, `safety_clamp_hits`, `old_log_probs`, `rewards`, `values`, `dones`, `env_ids`, plus a reusable `env_grouping: EnvGrouping` scratch for per-env GAE without per-call allocation.
- GAE is computed **per-env** to prevent cross-env value leakage. Transitions are grouped by `env_id` via `EnvGrouping` (a `Vec<Vec<usize>>` indexed by `env_id as usize` — dense small integers make this a better fit than the previous `HashMap<u32, Vec<usize>>`). GAE runs within each group independently. Iteration order is deterministic (ascending env_id). Advantages are normalised **per-minibatch** (per-chunk) rather than globally. Sample indices are shuffled at the start of each epoch via Fisher-Yates shuffle to ensure diverse gradient updates.
- Bootstrap values are computed per-env at prepare time: non-terminal envs get a fresh `forward_critic()` single-sample pass; terminal envs get 0.

### Update Triggering

| Condition | Trigger |
|-----------|---------|
| Rollout horizon | `buffer.len() >= max_steps` (512) — total transitions across **all** cars |
| Terminal batch | **Any car** terminal AND `buffer.len() >= min_update_steps` (128) |
| App exit | Residual rollout data or in-progress update exists |

Update is **amortised across ticks** to avoid frame stutter:

1. `ppo_prepare_update` computes per-env GAE, freezes buffer into `PreparedUpdate`, clears live buffer.
2. `ppo_epoch_system` processes `samples_per_tick` (64) samples per tick via batched forward/backward passes, accumulating gradients into the model.
3. When all samples in an epoch are done, `ppo_finish_epoch` clips gradients and steps the optimiser.
4. After all `ppo_epochs` (4) complete, the `PreparedUpdate` is dropped and the update state cleared.
5. On app exit, `ppo_flush_on_exit_system` finishes any in-progress epochs synchronously, then processes any remaining buffer data via `ppo_update_blocking`.

Each epoch:
  1. Forwards each sample through the current policy to get new log-probs.
  2. Computes ratio `π_new / π_old` from stored old log-probs.
  3. Applies clipped surrogate objective: `min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)`.
  4. Gradient flows through ratio when unclipped; zero when clipped.
  5. Value loss is Huber on returns vs values (unchanged from A2C).
  6. Entropy bonus applied to log-std gradients regardless of clipping.
  7. After all samples: clip gradients (actor: 0.5, critic: 0.5), step Adam (actor) / AdamW (critic with weight decay), update log-std (clamped to floor -1.0).

### Training Stats

`PpoTrainingStats` records the most recent completed update (final epoch values):
- `policy_loss`, `value_loss`, `policy_entropy`, `explained_variance`
- `steering_mean/std`, `throttle_mean/std`, `clamped_action_fraction`
- `clip_fraction` — fraction of samples where the PPO ratio was clipped (healthy: 10–30%)
- `approx_kl` — approximate KL divergence between old and new policy (healthy: < 0.02)
- Per-layer `PpoLayerHealth`: weight L2 norm, gradient L2 norm, tanh saturation fraction
- **Round-2 diagnostics (2026-04-19):**
  - `return_min / return_mean / return_max / return_std` — distribution of returns seen by this update (for PopArt tracking audit)
  - `value_norm_mu / value_norm_sigma` — PopArt running stats after the update
  - `epochs_completed` — how many PPO epochs actually ran (≤ `ppo_epochs`; less when target-KL early stop fires)
  - `early_stopped` — true when target-KL early-stop fired on this update

### Round-2 Interventions (2026-04-19)

The critic target-scaling round landed four orthogonal changes against the
anticipatory-value failure diagnosed from `reports/analytics/run_1776543971.md`.
The round-2 plan file (`context/plans/critic-target-scaling.md`) has been pruned after its completion criteria were fully met; its durable content lives in commit messages `c80d2ca`, `3bed996`, `a0b2cb6`, `e86e737` and in this file. The three
research references under `context/references/` for derivation.

**γ = 0.995.** Credit horizon 1.67s → 3.33s. Previously, a wall-at-2s signal
discounted to 0.30× of its present value before the critic's target update
— muting anticipatory-braking pressure. The new horizon matches the
observation lookahead's ~2.6s reach.

**PopArt on `c_value`.** New state `ValueNorm { mu, sigma }` on `PpoBrain`.
Once per update (in `ppo_prepare_update`, before any training chunk),
`popart_absorb_batch` computes batch mean/variance of GAE returns,
EMA-updates `mu`/`sigma`, and applies the POP rescale to `c_value` weights
and bias so externally-observable predictions `σ·z + µ` are preserved
across the statistics change. The training loss in `ppo_process_chunk`
treats `c_out[s]` as a normalised prediction and regresses against
`(ret − µ) / σ`; this keeps the critic targeting a stationary ~N(0, 1)
distribution regardless of return scale. Denormalisation happens at
inference call sites (bootstrap, action-selection PolicyOutput write,
exit flush) via `brain.value_norm.denormalise(raw)`. When
`popart_enabled=false`, `ValueNorm` stays at `(0, 1)` — all normalisation
becomes identity and the pipeline is numerically equivalent to the
pre-PopArt path.

**Target-KL early stop.** After each completed PPO epoch,
`ppo_epoch_system` checks `approx_kl > 1.5 × target_kl`. When it fires,
the current epoch is treated as the final one — `finish_epoch` writes
detailed stats, `early_stopped = true` is recorded, and the remaining
scheduled epochs are skipped. Guardrail against policy overshoot while
the critic adapts; also a passive diagnostic.

**Observation normaliser** — see `agent-interface.md`.

### Trainer Ranking

- `TrainerLiveRanking` resource tracks best/worst `env_id` with hysteresis (5% margin to prevent flicker).
- Score formula: `0.7 * best_progress_mean + 0.3 * normalised_return_mean`.
- Recomputed once per second (60-tick cadence, gated by `update_cadence_ticks`).
- `update_car_visual_roles_system` sets the best car to full opacity + z=11; all others are dimmed + z=10.
- Each car has a unique colour from a 25-colour palette via its `CarColour` component; ranking only adjusts alpha and z-order, never the base colour.
- Owned by `src/brain/ranking.rs`; registered by `BrainPlugin` in `src/brain/plugin.rs`.

### Hyperparameters (defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `gamma` | **0.995** | Round-2 (2026-04-19): raised from 0.99 to extend credit horizon 1.67s → 3.33s |
| `gae_lambda` | 0.95 | |
| `max_steps` (rollout horizon) | 512 | |
| `min_update_steps` | 128 | |
| `ppo_epochs` | 4 | May early-stop — see `target_kl` below |
| `clip_epsilon` | 0.2 | |
| `samples_per_tick` | 32 | |
| Actor hidden dim | 64 | |
| Critic hidden dim | 128 | |
| Actor LR | 3e-4 (Adam, weight decay 0.0) | |
| Critic LR | 5e-4 (AdamW, weight decay 3e-4) | |
| Actor grad clip | 0.5 | |
| Critic grad clip | 0.5 | |
| Entropy coefficient | 0.01 | |
| `log_std` floor | -1.0 (min σ ≈ 0.37) | |
| `log_std` ceil | 0.5 | |
| **`target_kl`** | **Some(0.03)** | Round-2: PPO early-stops when approx-KL > 1.5 × target |
| **`popart_enabled`** | **true** | Round-2: PopArt critic target normalisation |
| **`popart_beta`** | **1e-4** | EMA decay per PPO update |
| **`popart_sigma_floor`** | **1e-4** | Minimum σ for numerical stability |

## Key Interfaces / Data Flow

| Interface | Producer | Consumer | Notes |
|-----------|----------|----------|-------|
| `ObservationVector` | agent | PPO act path | Fixed-size model input (dim 43) |
| `ActionState.desired` | PPO act system | smoothing → physics | Same control boundary as keyboard |
| `PolicyOutput` | PPO act system | analytics capture | Per-car value prediction, policy means/stds |
| `EpisodeState.current_tick_reward` | game | PPO reward collector | Authoritative per-step reward |
| `EpisodeState.current_tick_end_reason` | game | PPO reward collector | Terminal-step truth |
| `PpoTrainingStats` | PPO update path | debug HUD, analytics tracker | Snapshot of latest completed update |

```text
Tick lifecycle (vectorised):
  observation_t (all cars) → ppo_act_all_cars_system → forward_actor per car → desired action + PolicyOutput + buffer push
  → smoothing → physics → environment step (all cars)
  → episode_loop_system computes reward_t and done_t (per car)
  → observation_t+1 rebuilt (post-reset if terminal, per car)
  → ppo_collect_rewards_all_cars_system appends per-car reward_t, done_t
  → optionally triggers ppo_prepare_update() with per-env bootstrap values (forward_critic)
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `AgentMode`, `PpoBrain` (no buffer), `PpoTrainingStats`, `TrainerRolloutBuffer`, `TrainerLiveRanking`
- **Runtime components (per car):** `PolicyOutput` (value_prediction, steering_mean/std, throttle_mean/std)
- **Handwritten ML primitives:** `Linear` (flat weight storage, forward/backward + batched variants, orthogonal init), `Tanh` (with saturation tracking), `AdamOptimizer` (AdamW-style with decoupled weight decay, per-layer, ε=1e-5), `sample_normal`, `log_prob_normal`, `tanh_correction`, `orthogonal_init`
- **Unit tests in `buffer.rs`:** single-env GAE regression test (verifies per-env GAE matches flat GAE for one env), multi-env GAE isolation test (verifies no cross-env value leakage in interleaved buffer).

## Known Issues / Active Risks

- **No save/load path**, no evaluation mode, no headless training loop.
- **No dedicated PPO integration tests**, no explicit behavioural success threshold, limited protection against silent training regressions beyond runtime stats (unit tests cover GAE only).
- All rollout buffer alignment is checked by `debug_assert!` only — not active in release builds.

## Partial / In Progress

- The baseline is integrated and live. Cars are confirmed to learn meaningful behaviour (drifting corners observed with PPO).
- Several earlier timing and contract bugs have been corrected:
  - reward/order alignment,
  - terminal-step handling,
  - bounded action semantics (tanh squashing),
  - next-state bootstrap handling,
  - flush-on-exit,
  - rollout reset on mode switches.

## Planned / Missing / Likely Changes

- **Headless training, persistence, and evaluation mode** are likely to matter before longer experiments become credible.
- The final project direction points toward biological/local-plasticity systems; PPO should stay modular enough to be **retired later** without distorting the rest of the runtime.

## Durable Notes / Discarded Approaches

- The **bounded tanh-squashed action contract** is a deliberate improvement over sampling unconstrained values and relying on later clamping. It ensures actions are naturally within physical bounds.
- PPO exists as a **baseline for learnability validation**, not the intended final learning architecture. Engineering investment should stay proportionate.
- **ReLU was replaced by tanh** after observing 34–57% dead neuron rates across all hidden layers. The dead neurons starved the actor of capacity, preventing corner-learning. Tanh eliminates this problem entirely (0% saturation observed). This is consistent with the Andrychowicz et al. finding that tanh outperforms ReLU in on-policy continuous control.
- External A2C/PPO research and a NeuroDrive-specific implementation ladder live in `context/references/a2c-for-neurodrive.md` — deep algorithm research belongs there rather than in this system file.
- **PPO upgrade resolved the A2C policy oscillation problem.** The clipped surrogate objective prevents any single update from destabilising the policy. The amortised epoch processing (64 samples/tick) resolved frame stutter that appeared when all 4 epochs ran in a single tick.
- **Critic saturation problem:** with the symmetric 2×64 architecture, the critic's fc2 layer reached 40.6% tanh saturation and weight norms of 19.3, preventing accurate crash-value prediction. The fix was twofold: (1) asymmetric sizing — critic widened to 2×128 for more capacity, (2) AdamW with weight decay λ=3e-4 on the critic to bound weight growth.
- **Log-std floor raised from -2.0 to -1.0** to prevent throttle exploration collapse (std was reaching 0.07, making it impossible to discover throttle modulation for cornering).
- **Braking was tried and reverted.** The `[-1, 1]` throttle range with negative-as-brake caused the policy to converge to "mostly brake" (throttle mean -0.60) as a safe local optimum. Throttle was reverted to `[0, 1]` with `0.5*(tanh+1)` remapping restored. The log-prob code was never actually updated for braking (it still had the `[0,1]` affine correction), so the revert also fixed an inconsistency.

## Obsolete / No Longer Relevant

- Any document treating A2C as a future-only milestone is obsolete — PPO is present and participates in the live runtime path.
- Any reference to `a2c/` directory, `A2cBrain`, `A2cPlugin`, `A2cTrainingStats`, or `A2cLayerHealth` is obsolete — all renamed to `ppo/`, `PpoBrain`, `PpoPlugin`, `PpoTrainingStats`, `PpoLayerHealth`.
- Any reference to the `Brain` trait, `Relu` struct, `glorot_uniform`, or `Linear::new` is obsolete — these were dead code and have been removed.
- Any reference to `DrivingHudEpisodeAccumulator` is obsolete — removed as dead code.
- Any reference to `biological/` or `sessions/` placeholder directories is obsolete — these empty directories have been removed.
- Any reference to `samples_per_tick = 128` or symmetric 2×64 critic is obsolete — now 64 samples/tick and asymmetric 2×128 critic.
- Any reference to observation dimension 23 or 27 is obsolete — the model now takes 43-dimensional input (12 lookahead samples × 2 features replaced the old 4 × 2).
- Any reference to throttle using raw tanh directly to [-1,1] is obsolete — throttle uses `0.5*(tanh+1)` to [0,1].
- Any reference to analytics/HUD using temporary first-car shims is obsolete — the full analytics overhaul is complete.
