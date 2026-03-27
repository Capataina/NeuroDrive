# System — Brain PPO Baseline

## Scope / Purpose

- Provide the current autonomous-controller baseline used to validate that the environment and observation contract are learnable.
- Keep the baseline self-contained in Rust without external ML frameworks.
- PPO (upgraded from A2C) exists as a **diagnostic tool**, not the intended final learning architecture. The project's long-term direction is brain-inspired local plasticity.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/brain/plugin.rs` | `BrainPlugin`, `AgentMode` toggle (F4), registers `TrainerLiveRanking`, wires ranking and visual-role systems, rollout-buffer reset on mode switch | Policy implementation details |
| `src/brain/types.rs` | `AgentMode` enum (`Keyboard` / `Ai`), `Brain` trait (**dead code** — unused, tagged for Stage 5 removal) | Observation or reward production |
| `src/brain/a2c/` | `A2cBrain` (model + hyperparams + seeded RNG, **no buffer**), `TrainerRolloutBuffer` (separate resource with env_id tagging + old log-probs), `PpoUpdateState` (staged update resource), `A2cPlugin`, vectorised act/collect/epoch/flush systems, per-env GAE, `ActorCritic` model, PPO clipped update logic, training stats | Environment truth, observation construction |
| `src/brain/ranking.rs` | `TrainerLiveRanking` resource, ranking computation, `update_car_visual_roles_system` (best-car highlighting via alpha + z-order), `CarColour`-aware sprite updates | Policy, observation, reward |
| `src/brain/common/` | Reusable handwritten ML primitives: `Linear`, `Tanh`, `Relu` (legacy), `AdamOptimizer`, Gaussian math, orthogonal init | Algorithm-specific logic |
| `src/brain/biological/` | **Empty placeholder** for future local-plasticity brain | Nothing yet |

## Current Implemented Reality

### Mode Switching

- `AgentMode` defaults to `Ai`.
- `F4` toggles between `Ai` and `Keyboard` in the `Update` schedule.
- Toggling clears `TrainerRolloutBuffer` (separate resource, not brain-owned) and resets `step_counter` to avoid mixed-control trajectories.

### Model Architecture

```text
Actor:                              Critic:
obs (23) → Linear(23,64) → Tanh    obs (23) → Linear(23,64) → Tanh
         → Linear(64,64) → Tanh             → Linear(64,64) → Tanh
         → Linear(64,2)  → mean             → Linear(64,1)  → value
         + learnable log_std (2)
```

- Separate actor and critic stacks (no shared backbone).
- Orthogonal initialisation: √2 scale for hidden layers, 0.01× for actor mean output (near-zero initial policy), 1.0× for critic value output.
- `log_std` initialised to `[0.0, 0.0]` (initial σ = 1.0).
- Actor LR: 3e-4, Critic LR: 5e-4 (both Adam).
- Activation: Tanh throughout (switched from ReLU — eliminates dead-neuron capacity loss that was starving the actor at 34–57% dead neurons).

### Action Selection

- The policy samples Gaussian latent actions from `N(mean, exp(log_std))`.
- Applies `tanh` squashing, then maps:
  - steering: tanh output directly → `[-1, 1]`
  - throttle: `0.5 * (tanh + 1.0)` → `[0, 1]`
- Safety clamping applied after squashing; clamp-hit flags tracked per step.
- RNG is a **seeded `StdRng`** stored in `A2cBrain` — deterministic within a session. Initialised from `rand::rng()` at brain construction, then reused for all sampling.
- All cars receive actions from the shared policy each tick (not just one car).

### Rollout Collection

- `a2c_act_all_cars_system` runs in `SimSet::Input` after keyboard input and before action smoothing. Iterates **all** cars, calls `model.forward()` for each, samples stochastic actions via the seeded RNG, writes per-car `ActionState`, computes old log-prob (sum of squashed-Gaussian log-probs across action dimensions), and pushes transitions tagged with `env_id` and `old_log_prob` to `TrainerRolloutBuffer`.
- `a2c_collect_rewards_all_cars_system` runs in `SimSet::Measurement` after episode truth and observation rebuild. Pushes per-car reward and done flag for all cars. When the buffer reaches the horizon and no update is in progress, calls `ppo_prepare_update` to compute per-env GAE, freeze the buffer into a `PreparedUpdate`, and clear the live buffer.
- `ppo_epoch_system` runs in `SimSet::Measurement` after the collect system. Processes a chunk of `samples_per_tick` (default 128) samples from the `PreparedUpdate` per tick. When an epoch's samples are exhausted, calls `ppo_finish_epoch` (clip gradients, step optimiser). Advances to the next epoch or clears the update state when all epochs are done.
- `TrainerRolloutBuffer` stores: `states`, `actions`, `latent_actions`, `safety_clamp_hits`, `old_log_probs`, `rewards`, `values`, `dones`, `env_ids`.
- GAE is computed **per-env** to prevent cross-env value leakage. Transitions are grouped by `env_id`; GAE runs within each group independently. Advantages are normalised **per-minibatch** (per-chunk) rather than globally. Sample indices are shuffled at the start of each epoch via Fisher-Yates shuffle to ensure diverse gradient updates.
- Bootstrap values are computed per-env at prepare time: non-terminal envs get a fresh `model.forward()` value; terminal envs get 0.

### Update Triggering

| Condition | Trigger |
|-----------|---------|
| Rollout horizon | `buffer.len() >= max_steps` (512) — total transitions across **all** cars |
| Terminal batch | **Any car** terminal AND `buffer.len() >= min_update_steps` (128) |
| App exit | Residual rollout data or in-progress update exists |

Update is **amortised across ticks** to avoid frame stutter:

1. `ppo_prepare_update` computes per-env GAE, freezes buffer into `PreparedUpdate`, clears live buffer.
2. `ppo_epoch_system` processes `samples_per_tick` (128) samples per tick, accumulating gradients into the model.
3. When all samples in an epoch are done, `ppo_finish_epoch` clips gradients and steps the optimiser.
4. After all `ppo_epochs` (4) complete, the `PreparedUpdate` is dropped and the update state cleared.
5. On app exit, `a2c_flush_on_exit_system` finishes any in-progress epochs synchronously, then processes any remaining buffer data via `ppo_update_blocking`.

Each epoch:
  1. Forwards each sample through the current policy to get new log-probs.
  2. Computes ratio `π_new / π_old` from stored old log-probs.
  3. Applies clipped surrogate objective: `min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)`.
  4. Gradient flows through ratio when unclipped; zero when clipped.
  5. Value loss is Huber on returns vs values (unchanged from A2C).
  6. Entropy bonus applied to log-std gradients regardless of clipping.
  7. After all samples: clip gradients (actor: 0.5, critic: 0.5), step Adam, update log-std.

### Training Stats

`A2cTrainingStats` records the most recent completed update (final epoch values):
- `policy_loss`, `value_loss`, `policy_entropy`, `explained_variance`
- `steering_mean/std`, `throttle_mean/std`, `clamped_action_fraction`
- `clip_fraction` — fraction of samples where the PPO ratio was clipped (healthy: 10–30%)
- `approx_kl` — approximate KL divergence between old and new policy (healthy: < 0.02)
- Per-layer `A2cLayerHealth`: weight L2 norm, gradient L2 norm, tanh saturation fraction

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
| `ppo_epochs` | 4 |
| `clip_epsilon` | 0.2 |
| `samples_per_tick` | 128 |
| Actor hidden dim | 64 |
| Critic hidden dim | 64 |
| Actor LR | 3e-4 |
| Critic LR | 5e-4 |
| Actor grad clip | 0.5 |
| Critic grad clip | 0.5 |
| Entropy coefficient | 0.01 |

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
- **Handwritten ML primitives:** `Linear` (forward + backward + Glorot/orthogonal init), `Tanh` (with saturation tracking), `Relu` (legacy, unused), `AdamOptimizer` (per-layer, ε=1e-5), `sample_normal`, `log_prob_normal`, `tanh_correction`, `orthogonal_init`
- **Unit tests in `buffer.rs`:** single-env GAE regression test (verifies per-env GAE matches flat GAE for one env), multi-env GAE isolation test (verifies no cross-env value leakage in interleaved buffer).

## Known Issues / Active Risks

- **No save/load path**, no evaluation mode, no headless training loop.
- **No dedicated PPO integration tests**, no explicit behavioural success threshold, limited protection against silent training regressions beyond runtime stats (unit tests cover GAE only).
- The `Brain` trait is **dead code** — unused by the vectorised path.
- **Analytics and HUD systems use temporary shims** (target first car only) pending a full overhaul.
- All rollout buffer alignment is checked by `debug_assert!` only — not active in release builds.
- The module is still named `a2c/` and structs still use `A2c` prefixes (e.g., `A2cBrain`, `A2cPlugin`) despite now implementing PPO. A rename would be cosmetic churn with no functional benefit at this stage.

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

- **Full analytics overhaul** planned (multi-car capture, visual outputs, diagnostic automation) — see `context/plans/analytics-overhaul-brief.md`.
- **Headless training, persistence, and evaluation mode** are likely to matter before longer experiments become credible.
- The final project direction points toward biological/local-plasticity systems; PPO should stay modular enough to be **retired later** without distorting the rest of the runtime.


## Durable Notes / Discarded Approaches

- The **bounded tanh-squashed action contract** is a deliberate improvement over sampling unconstrained values and relying on later clamping. It ensures actions are naturally within physical bounds.
- PPO exists as a **baseline for learnability validation**, not the intended final learning architecture. Engineering investment should stay proportionate.
- **ReLU was replaced by tanh** after observing 34–57% dead neuron rates across all hidden layers. The dead neurons starved the actor of capacity, preventing corner-learning. Tanh eliminates this problem entirely (0% saturation observed). This is consistent with the Andrychowicz et al. finding that tanh outperforms ReLU in on-policy continuous control.
- External A2C/PPO research and a NeuroDrive-specific implementation ladder live in `context/references/a2c-for-neurodrive.md` — deep algorithm research belongs there rather than in this system file.
- **PPO upgrade resolved the A2C policy oscillation problem.** The clipped surrogate objective prevents any single update from destabilising the policy. The amortised epoch processing (128 samples/tick) resolved frame stutter that appeared when all 4 epochs ran in a single tick.

## Obsolete / No Longer Relevant

- Any document treating A2C as a future-only milestone is obsolete — the code is present and participates in the live runtime path.
