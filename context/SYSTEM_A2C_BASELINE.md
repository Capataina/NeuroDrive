# System — A2C Baseline Brain

## Scope / Purpose

- Provide a minimal autonomous learning baseline to validate that the environment, observation contract, and reward signal are sufficient for online learning.
- Keep the baseline self-contained in Rust without external ML frameworks, while preserving a stable controller boundary through `CarAction`.

## Current Implemented System

- A `BrainPlugin` is wired into the app, initialises `AgentMode`, and toggles keyboard versus AI control on `F4` (`src/brain/plugin.rs`, `src/brain/types.rs`).
- `AgentMode` defaults to `Ai`, so the A2C path is the default controller mode in the current source (`src/brain/types.rs`).
- An `A2cPlugin` is registered from the brain layer and initialises an `A2cBrain` resource (`src/brain/plugin.rs`, `src/brain/a2c/mod.rs`).
- `A2cBrain` currently owns a handwritten `ActorCritic`, a `RolloutBuffer`, discount parameters, rollout length, and a step counter (`src/brain/a2c/mod.rs`).
- `ActorCritic` is a handwritten dual-head MLP with separate actor and critic stacks, Gaussian action parameters, and custom Adam optimisers (`src/brain/a2c/model.rs`, `src/brain/common/mlp.rs`, `src/brain/common/optim.rs`).
- `a2c_act_system` reads the current `ObservationVector`, samples a bounded 2D continuous action, writes it into `ActionState.desired`, and appends state/action/latent/value data to the rollout buffer (`src/brain/a2c/mod.rs`).
- `a2c_collect_reward_system` appends `EpisodeState.current_tick_reward` and done flags into the rollout buffer and triggers `a2c_update()` on either rollout horizon or terminal-step minimum batch size (`src/brain/a2c/mod.rs`).
- `a2c_update()` computes GAE, applies actor/critic gradients with gradient clipping, uses Huber loss for critic targets, steps custom Adam optimisers, updates log-std parameters, and clears the rollout buffer (`src/brain/a2c/update.rs`).
- A dedicated `A2cTrainingStats` resource now captures the most recent update’s policy entropy, policy loss, value loss, explained variance, steering/throttle distribution, clamped-action fraction, and per-layer weight/gradient/dead-ReLU health for analytics export (`src/brain/a2c/mod.rs`, `src/brain/a2c/update.rs`).
- The rollout pipeline now uses a per-tick terminal marker (`current_tick_end_reason`) rather than the sticky episode-summary end reason, so only the actual terminal step is marked done for A2C (`src/game/episode.rs`, `src/brain/a2c/mod.rs`).
- Terminal-step rewards are now preserved through the measurement phase so the A2C collector can record crash penalties and other terminal rewards correctly (`src/game/episode.rs`, `src/brain/a2c/mod.rs`).
- Input-stage ordering now explicitly runs A2C action selection before action smoothing so `ActionState.applied` is derived from the same-tick AI action (`src/brain/a2c/mod.rs`, `src/agent/action.rs`).
- The policy output now uses a tanh-squashed Gaussian contract aligned to bounded control ranges (steering `[-1, 1]`, throttle `[0, 1]`) instead of unconstrained sampling plus post-hoc clamping (`src/brain/a2c/mod.rs`, `src/brain/a2c/update.rs`).
- Non-terminal rollout tails now bootstrap value targets from the current next observation supplied at reward-collection time rather than reusing the last state in the buffer (`src/brain/a2c/mod.rs`, `src/brain/a2c/update.rs`).
- Measurement ordering now forces reward finalisation before sensor rebuild, with reset-time progress resync, so post-terminal observations consumed by A2C align to the spawn reset state (`src/agent/plugin.rs`, `src/game/episode.rs`).
- The A2C plugin now flushes partial rollouts on app exit so residual transitions are not silently discarded at shutdown (`src/brain/a2c/mod.rs`).
- Brain mode switches now clear rollout state to prevent mixed control-mode trajectories from contaminating updates (`src/brain/plugin.rs`, `src/brain/a2c/mod.rs`).
- Clamp-health telemetry now records real safety-clamp hits captured at action execution time rather than inferring clamps from already bounded stored actions (`src/brain/a2c/buffer.rs`, `src/brain/a2c/update.rs`).

## Implemented Outputs / Artifacts (if applicable)

- Runtime resources: `AgentMode` and `A2cBrain` (`src/brain/types.rs`, `src/brain/a2c/mod.rs`).
- Runtime controller path that can write AI actions into the same stable `CarAction` boundary used by keyboard control (`src/brain/a2c/mod.rs`, `src/agent/action.rs`).
- Handwritten neural-network primitives in `src/brain/common/`.
- Runtime `A2cTrainingStats` snapshots consumed by the analytics subsystem (`src/brain/a2c/mod.rs`, `src/analytics/trackers/episode.rs`).

## In Progress / Partially Implemented

- The subsystem is integrated into the runtime but not yet validated as a trustworthy baseline.
- The rollout path still assumes `EpisodeState.current_tick_reward` is the correct per-step reward source, and while terminal-step timing has been corrected and reward shaping has been revised, the quality of that signal still needs behavioural validation.
- The new training-health metrics are snapshot-based rather than historical inside the brain layer; long-horizon analysis depends on the analytics subsystem persisting them.
- Deterministic reproducibility is still weak because rollout/action sampling RNG ownership is not yet centralised and seed-controlled.
- The current cadence still depends on static `max_steps`/`min_update_steps` defaults and has not yet been tuned against observed episode lengths.

## Planned / Missing / To Be Changed

- Build and behavioural validation are still missing: there are no A2C-specific tests, no compile-verified training loop outcome, and no baseline success criteria encoded in code or docs.
- There is no headless training mode, snapshot/save-load path, deterministic seeding strategy, evaluation mode, or rollout/report integration.
- There is no abstraction yet for future non-A2C brain implementations beyond the minimal `Brain` trait.
- More learning-health depth is still missing, especially TD-error distributions, update-to-update policy drift, and richer exploration diagnostics.

## Notes / Design Considerations (optional)

- This subsystem should remain a baseline, not the final architecture; it exists to validate learnability and instrument the controller boundary.
- The current codebase mixes algorithm scaffolding and runtime integration before the verification layer is complete, so documentation should continue to mark it as partial rather than “implemented”.
- Because the project goal is eventually brain-inspired local plasticity, A2C should be kept modular and removable once it has served the learnability-validation purpose.

## Discarded / Obsolete / No Longer Relevant

- The older context that described A2C purely as a future Milestone 1 item is obsolete; the repository now contains a live but incomplete implementation.
