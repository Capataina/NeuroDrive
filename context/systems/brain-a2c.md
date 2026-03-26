# System — Brain A2C Baseline

## Scope / Purpose

- Provide the current autonomous-controller baseline used to validate that the environment and observation contract are learnable.
- Keep the baseline self-contained in Rust without external ML frameworks.
- A2C exists as a **diagnostic tool**, not the intended final learning architecture. The project's long-term direction is brain-inspired local plasticity.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/brain/plugin.rs` | `BrainPlugin`, `AgentMode` toggle (F4), registers `TrainerLiveRanking`, wires ranking and visual-role systems, rollout-buffer reset on mode switch | Policy implementation details |
| `src/brain/types.rs` | `AgentMode` enum (`Keyboard` / `Ai`), `Brain` trait (**dead code** — unused, tagged for Stage 5 removal) | Observation or reward production |
| `src/brain/a2c/` | `A2cBrain` (model + hyperparams + seeded RNG, **no buffer**), `TrainerRolloutBuffer` (separate resource with env_id tagging), `A2cPlugin`, vectorised act/collect/flush systems, per-env GAE, `ActorCritic` model, update logic, training stats | Environment truth, observation construction |
| `src/brain/ranking.rs` | `TrainerLiveRanking` resource, ranking computation, `update_car_visual_roles_system` (best-car highlighting via alpha + z-order), `CarColour`-aware sprite updates | Policy, observation, reward |
| `src/brain/common/` | Reusable handwritten ML primitives: `Linear`, `Relu`, `AdamOptimizer`, Gaussian math | Algorithm-specific logic |
| `src/brain/biological/` | **Empty placeholder** for future local-plasticity brain | Nothing yet |

## Current Implemented Reality

### Mode Switching

- `AgentMode` defaults to `Ai`.
- `F4` toggles between `Ai` and `Keyboard` in the `Update` schedule.
- Toggling clears `TrainerRolloutBuffer` (separate resource, not brain-owned) and resets `step_counter` to avoid mixed-control trajectories.

### Model Architecture

```text
Actor:                              Critic:
obs (23) → Linear(23,64) → ReLU    obs (23) → Linear(23,64) → ReLU
         → Linear(64,64) → ReLU             → Linear(64,64) → ReLU
         → Linear(64,2)  → mean             → Linear(64,1)  → value
         + learnable log_std (2)
```

- Separate actor and critic stacks (no shared backbone).
- Glorot initialisation for all weights.
- `log_std` initialised to `[0.0, 0.0]` (initial σ = 1.0).
- Actor LR: 3e-4, Critic LR: 5e-4 (both Adam).
- Activation: ReLU throughout.

### Action Selection

- The policy samples Gaussian latent actions from `N(mean, exp(log_std))`.
- Applies `tanh` squashing, then maps:
  - steering: tanh output directly → `[-1, 1]`
  - throttle: `0.5 * (tanh + 1.0)` → `[0, 1]`
- Safety clamping applied after squashing; clamp-hit flags tracked per step.
- RNG is a **seeded `StdRng`** stored in `A2cBrain` — deterministic within a session. Initialised from `rand::rng()` at brain construction, then reused for all sampling.
- All cars receive actions from the shared policy each tick (not just one car).

### Rollout Collection

- `a2c_act_all_cars_system` (replaces old `a2c_act_system`) runs in `SimSet::Input` after keyboard input and before action smoothing. Iterates **all** cars, calls `model.forward()` for each, samples stochastic actions via the seeded RNG, writes per-car `ActionState`, and pushes transitions tagged with `env_id` to `TrainerRolloutBuffer`.
- `a2c_collect_rewards_all_cars_system` (replaces old `a2c_collect_reward_system`) runs in `SimSet::Measurement` after episode truth and observation rebuild. Pushes per-car reward and done flag for all cars.
- `TrainerRolloutBuffer` stores: `states`, `actions`, `latent_actions`, `safety_clamp_hits`, `rewards`, `values`, `dones`, `env_ids`.
- GAE is computed **per-env** to prevent cross-env value leakage. Transitions are grouped by `env_id`; GAE runs within each group independently. Advantages are normalised globally across all envs in the batch.
- Bootstrap values are computed per-env at update time: non-terminal envs get a fresh `model.forward()` value; terminal envs get 0.

### Update Triggering

| Condition | Trigger |
|-----------|---------|
| Rollout horizon | `buffer.len() >= max_steps` (512) — total transitions across **all** cars |
| Terminal batch | **Any car** terminal AND `buffer.len() >= min_update_steps` (128) |
| App exit | Residual rollout data exists |

- Per-env bootstrap values are computed at update time (non-terminal envs forward-pass, terminal envs 0).
- Update calls `a2c_update(brain, buffer, stats, bootstrap_values: &HashMap<u32, f32>)`, which:
  1. Computes GAE advantages and returns.
  2. Standardises advantages (zero mean, unit variance).
  3. Computes policy loss (negative log-prob × advantage + entropy bonus).
  4. Computes value loss (Huber loss on returns vs values).
  5. Backpropagates through actor and critic separately.
  6. Clips gradients (actor: 0.5, critic: 0.5).
  7. Steps Adam optimisers.
  8. Snapshots `A2cTrainingStats` with losses, entropy, explained variance, action spread, clamp fraction, and per-layer health.

### Training Stats

`A2cTrainingStats` records the most recent completed update:
- `policy_loss`, `value_loss`, `policy_entropy`, `explained_variance`
- `steering_mean/std`, `throttle_mean/std`, `clamped_action_fraction`
- Per-layer `A2cLayerHealth`: weight L2 norm, gradient L2 norm, dead-ReLU fraction

### Trainer Ranking

- `TrainerLiveRanking` resource tracks best/worst `env_id` with hysteresis (5% margin to prevent flicker).
- Score formula: `0.7 * best_progress_mean + 0.3 * normalised_return_mean`.
- Recomputed once per second (60-tick cadence, gated by `update_cadence_ticks`).
- `update_car_visual_roles_system` sets the best car to full opacity + z=11; all others are dimmed + z=10.
- Each car has a unique colour from a 25-colour palette via its `CarColour` component; ranking only adjusts alpha and z-order, never the base colour.
- Owned by `src/brain/ranking.rs`; registered by `BrainPlugin` in `src/brain/plugin.rs`.

### Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `max_steps` (rollout horizon) | 512 |
| `min_update_steps` | 128 |
| Actor hidden dim | 64 |
| Critic hidden dim | 64 |
| Actor LR | 3e-4 |
| Critic LR | 5e-4 |
| Actor grad clip | 0.5 |
| Critic grad clip | 0.5 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |

## Key Interfaces / Data Flow

| Interface | Producer | Consumer | Notes |
|-----------|----------|----------|-------|
| `ObservationVector` | agent | A2C act path | Fixed-size model input (dim 23) |
| `ActionState.desired` | A2C act system | smoothing → physics | Same control boundary as keyboard |
| `EpisodeState.current_tick_reward` | game | A2C reward collector | Authoritative per-step reward |
| `EpisodeState.current_tick_end_reason` | game | A2C reward collector | Terminal-step truth |
| `A2cTrainingStats` | A2C update path | debug HUD, analytics tracker | Snapshot of latest completed update |

```text
Tick lifecycle (vectorised):
  observation_t (all cars) → a2c_act_all_cars_system → per-car desired action + buffer push
  → smoothing → physics → environment step (all cars)
  → episode_loop_system computes reward_t and done_t (per car)
  → observation_t+1 rebuilt (post-reset if terminal, per car)
  → a2c_collect_rewards_all_cars_system appends per-car reward_t, done_t
  → optionally triggers a2c_update() with per-env bootstrap values
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `AgentMode`, `A2cBrain` (no buffer), `A2cTrainingStats`, `TrainerRolloutBuffer`, `TrainerLiveRanking`
- **Handwritten ML primitives:** `Linear` (forward + backward + Glorot init), `Relu` (with dead-neuron tracking), `AdamOptimizer` (per-layer), `sample_normal`, `log_prob_normal`, `tanh_correction`
- **Unit tests in `buffer.rs`:** single-env GAE regression test (verifies per-env GAE matches flat GAE for one env), multi-env GAE isolation test (verifies no cross-env value leakage in interleaved buffer).

## Known Issues / Active Risks

- **No save/load path**, no evaluation mode, no headless training loop.
- **No dedicated A2C integration tests**, no explicit behavioural success threshold, limited protection against silent training regressions beyond runtime stats (unit tests cover GAE only).
- The `Brain` trait is **dead code** — unused by the vectorised path, tagged for Stage 5 removal.
- **Analytics and HUD systems use temporary shims** (target first car only) pending a full overhaul.
- **Observed policy oscillation** (learn-then-forget cycle) — PPO upgrade planned to address via clipped policy ratio.
- All rollout buffer alignment is checked by `debug_assert!` only — not active in release builds.

## Partial / In Progress

- The baseline is integrated and live, but better described as a **validation harness** than a trusted learning subsystem.
- Several earlier timing and contract bugs have been corrected:
  - reward/order alignment,
  - terminal-step handling,
  - bounded action semantics (tanh squashing),
  - next-state bootstrap handling,
  - flush-on-exit,
  - rollout reset on mode switches.

## Planned / Missing / Likely Changes

- **PPO upgrade** planned to clip the policy ratio and address observed oscillation — see `context/plans/ppo-upgrade-brief.md`.
- **Full analytics overhaul** planned (visual outputs: heat maps, graphs, charts) — see `context/plans/analytics-overhaul-brief.md`.
- **Headless training, persistence, and evaluation mode** are likely to matter before longer experiments become credible.
- The final project direction points toward biological/local-plasticity systems; A2C should stay modular enough to be **retired later** without distorting the rest of the runtime.
- Activation switch from ReLU to tanh is a credible upgrade candidate based on on-policy literature.

## Durable Notes / Discarded Approaches

- The **bounded tanh-squashed action contract** is a deliberate improvement over sampling unconstrained values and relying on later clamping. It ensures actions are naturally within physical bounds.
- A2C exists as a **baseline for learnability validation**, not the intended final learning architecture. Engineering investment should stay proportionate.
- External A2C research and a NeuroDrive-specific implementation ladder live in `context/references/a2c-for-neurodrive.md` — deep algorithm research belongs there rather than in this system file.

## Obsolete / No Longer Relevant

- Any document treating A2C as a future-only milestone is obsolete — the code is present and participates in the live runtime path.
