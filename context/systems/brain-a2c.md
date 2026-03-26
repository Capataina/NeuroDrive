# System — Brain A2C Baseline

## Scope / Purpose

- Provide the current autonomous-controller baseline used to validate that the environment and observation contract are learnable.
- Keep the baseline self-contained in Rust without external ML frameworks.
- A2C exists as a **diagnostic tool**, not the intended final learning architecture. The project's long-term direction is brain-inspired local plasticity.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/brain/plugin.rs` | `BrainPlugin`, `AgentMode` toggle (F4), rollout-buffer reset on mode switch | Policy implementation details |
| `src/brain/types.rs` | `AgentMode` enum (`Keyboard` / `Ai`), minimal `Brain` trait | Observation or reward production |
| `src/brain/a2c/` | `A2cBrain`, `A2cPlugin`, act/reward-collect/flush systems, `ActorCritic` model, `RolloutBuffer`, update logic, training stats | Environment truth, observation construction |
| `src/brain/common/` | Reusable handwritten ML primitives: `Linear`, `Relu`, `AdamOptimizer`, Gaussian math | Algorithm-specific logic |
| `src/brain/biological/` | **Empty placeholder** for future local-plasticity brain | Nothing yet |

## Current Implemented Reality

### Mode Switching

- `AgentMode` defaults to `Ai`.
- `F4` toggles between `Ai` and `Keyboard` in the `Update` schedule.
- Toggling clears the A2C rollout buffer and resets `step_counter` to avoid mixed-control trajectories.

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
- RNG is created via `rand::rng()` **per act call** — no centralised seed ownership.

### Rollout Collection

- `a2c_act_system` runs in `SimSet::Input` after keyboard input and before action smoothing.
- Each act call appends to `RolloutBuffer`: state, action, latent action, clamp-hit flags, critic value.
- `a2c_collect_reward_system` runs in `SimSet::Measurement` after episode truth and observation rebuild.
- Reward collection appends one reward and one done flag per step.

### Update Triggering

| Condition | Trigger |
|-----------|---------|
| Rollout horizon | `buffer.states.len() >= max_steps` (512) |
| Terminal batch | Episode ended AND `buffer.states.len() >= min_update_steps` (128) |
| App exit | Residual rollout data exists |

- Non-terminal rollouts bootstrap from the current observation.
- Update calls `a2c_update()`, which:
  1. Computes GAE advantages and returns.
  2. Standardises advantages (zero mean, unit variance).
  3. Computes policy loss (negative log-prob × advantage + entropy bonus).
  4. Computes value loss (Huber loss on returns vs values).
  5. Backpropagates through actor and critic separately.
  6. Clips gradients (actor: 0.5, critic: 1.0).
  7. Steps Adam optimisers.
  8. Snapshots `A2cTrainingStats` with losses, entropy, explained variance, action spread, clamp fraction, and per-layer health.

### Training Stats

`A2cTrainingStats` records the most recent completed update:
- `policy_loss`, `value_loss`, `policy_entropy`, `explained_variance`
- `steering_mean/std`, `throttle_mean/std`, `clamped_action_fraction`
- Per-layer `A2cLayerHealth`: weight L2 norm, gradient L2 norm, dead-ReLU fraction

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
| Critic grad clip | 1.0 |
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
Tick lifecycle:
  observation_t → a2c_act_system → desired action
  → smoothing → physics → environment step
  → episode_loop_system computes reward_t and done_t
  → observation_t+1 rebuilt (post-reset if terminal)
  → a2c_collect_reward_system appends reward_t, done_t
  → optionally triggers a2c_update()
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `AgentMode`, `A2cBrain`, `A2cTrainingStats`
- **Handwritten ML primitives:** `Linear` (forward + backward + Glorot init), `Relu` (with dead-neuron tracking), `AdamOptimizer` (per-layer), `sample_normal`, `log_prob_normal`, `tanh_correction`
- **No tests** specific to the A2C path itself.

## Known Issues / Active Risks

- **RNG ownership is ad hoc** — `rand::rng()` created per act call. Deterministic replay does not extend into the A2C path.
- **No save/load path**, no evaluation mode, no headless training loop.
- **No dedicated A2C integration tests**, no explicit behavioural success threshold, limited protection against silent training regressions beyond runtime stats.
- The `Brain` trait is minimal and does not yet drive a broader pluggable-brain architecture.
- `a2c_act_system` uses `single()` for observation query — singleton-car assumption.
- All rollout buffer alignment is checked by `debug_assert_eq!` only — not active in release builds.

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

- **Controlled seeding** is the most obvious missing prerequisite for reproducible learning experiments.
- **Headless training, persistence, and evaluation mode** are likely to matter before longer experiments become credible.
- **Vectorised multi-car training** has a concrete implementation plan in `context/plans/vectorised-a2c-visual-trainer.md`.
- The final project direction points toward biological/local-plasticity systems; A2C should stay modular enough to be **retired later** without distorting the rest of the runtime.
- Activation switch from ReLU to tanh is a credible upgrade candidate based on on-policy literature.

## Durable Notes / Discarded Approaches

- The **bounded tanh-squashed action contract** is a deliberate improvement over sampling unconstrained values and relying on later clamping. It ensures actions are naturally within physical bounds.
- A2C exists as a **baseline for learnability validation**, not the intended final learning architecture. Engineering investment should stay proportionate.
- External A2C research and a NeuroDrive-specific implementation ladder live in `context/references/a2c-for-neurodrive.md` — deep algorithm research belongs there rather than in this system file.

## Obsolete / No Longer Relevant

- Any document treating A2C as a future-only milestone is obsolete — the code is present and participates in the live runtime path.
