# Immediate Implementation Plan — A2C

## Header

- [ ] Status: Proposed, not yet executed.
- [ ] Scope: Stabilise the current A2C baseline so it is compile-safe, schedule-correct, observable, and testable as a real learnability validation layer.
- [ ] Exit rule: stop when A2C control, reward collection, rollout updates, and verification paths are coherent, documented, and this file is archived or removed.

## Execution TODO (2026-03-09)

- [x] Fix reward-ordering bug so progress gain is computed before best-progress state is advanced.
- [x] Remove crash-reset side effects from collision handling so terminal reward/progress are computed from true crash state.
- [x] Enforce fixed-tick input ordering so AI action selection runs before action smoothing applies controls.
- [x] Correct GAE bootstrap source so non-terminal rollouts bootstrap from the current next observation rather than the last state.
- [x] Replace clamp-after-Gaussian training contract with bounded tanh-squashed action semantics.
- [x] Repair clamped-action telemetry by recording real safety clamp hits rather than checking already-clamped stored actions.
- [x] Resolve post-terminal observation ambiguity by ordering sensor rebuild after episode finalisation and resyncing track progress on reset.
- [x] Add dual-trigger A2C updates (rollout horizon and terminal-batch trigger) plus exit-time flush for partial rollouts.
- [x] Reset A2C rollout state on mode switches to prevent mixed keyboard/AI trajectories in one buffer.
- [x] Stabilise critic updates with lower learning rate, Huber value loss, and gradient clipping.
- [x] Rebalance reward defaults to reduce “sprint then crash” incentives.

## Future Vision Roadmap (Ranked)

- [x] 1. Add ego-relative centreline lookahead scalars (heading delta and curvature at multiple distances ahead).
- [ ] 2. Add an ego-relative waypoint bundle (next K centreline points in car frame) to improve compound-turn anticipation.
- [ ] 3. Add curvature-aware speed assist (target-speed governor) while keeping steering policy-driven.
- [ ] 4. Extend forward sensor coverage (longer range + denser front rays) as a low-complexity anticipatory boost.
- [ ] 5. Add a small egocentric occupancy patch around the car only if lookahead features saturate.
- [ ] 6. Add recurrent memory (GRU/LSTM) only after geometric lookahead is validated and stable.
- [ ] 7. Consider minimap/image-style encoders only as a late-stage option after lower-complexity options fail.

## Implementation Structure

- [ ] Modules / files affected (expected):
  - `src/brain/plugin.rs`
  - `src/brain/types.rs`
  - `src/brain/a2c/mod.rs`
  - `src/brain/a2c/model.rs`
  - `src/brain/a2c/buffer.rs`
  - `src/brain/a2c/update.rs`
  - `src/brain/common/mlp.rs`
  - `src/brain/common/optim.rs`
  - `src/agent/action.rs`
  - `src/agent/observation.rs`
  - `src/game/episode.rs`
  - `context/SYSTEM_A2C_BASELINE.md`
  - `context/SYSTEM_SENSORS_OBSERVATIONS.md`
  - `context/SYSTEM_TELEMETRY.md`
- [ ] Responsibility boundaries:
  - observation production stays in `agent`.
  - environment truth and reward truth stay in `game`.
  - A2C owns policy/value inference, rollout storage, and parameter updates.
  - debug/analytics expose A2C state but must not mutate training state.
- [ ] Function inventory:
  - `a2c_act_system` in `src/brain/a2c/mod.rs`
    - Inputs/outputs: reads the current observation and active mode, writes desired action, appends rollout step state.
    - Kind: orchestrator.
    - Called by: `A2cPlugin` during the fixed tick.
  - `a2c_collect_reward_system` in `src/brain/a2c/mod.rs`
    - Inputs/outputs: reads episode reward/end state, appends reward and done markers, may trigger update.
    - Kind: orchestrator.
    - Called by: `A2cPlugin` after measurement truth is available.
  - `ActorCritic::forward` in `src/brain/a2c/model.rs`
    - Inputs/outputs: maps one observation vector to action distribution parameters and a scalar value estimate.
    - Kind: helper with internal caches for backprop.
    - Called by: action selection and update.
  - `RolloutBuffer::compute_gae` in `src/brain/a2c/buffer.rs`
    - Inputs/outputs: maps stored rewards/values/dones to normalised advantages and returns.
    - Kind: helper, pure with respect to external state.
    - Called by: `a2c_update`.
  - `a2c_update` in `src/brain/a2c/update.rs`
    - Inputs/outputs: consumes the rollout buffer, mutates model parameters, clears rollout storage.
    - Kind: orchestrator.
    - Called by: reward collection path after sufficient rollout length.
- [ ] Wiring summary:
  - observation is built in measurement
  - next fixed tick A2C reads that observation and chooses an action
  - physics applies the action
  - episode logic computes per-tick reward and done state
  - A2C records reward and terminal flag
  - once rollout length is reached, update the actor-critic and continue

## Algorithm / System Sections

### Schedule and dataflow correctness

The first job is to make the training loop temporally coherent. Correct behaviour means the action for tick `t` is chosen from a known observation, the reward for tick `t` is recorded exactly once after the environment step completes, and update triggering happens from complete rollout entries.

The recommended default is to keep action selection in `SimSet::Input` and reward collection in `SimSet::Measurement`, with each system registered once. The alternative is to run the full training loop in `Update`; that is worse because it breaks the fixed-tick alignment that the environment relies on.

- [ ] Discovery (bounded):
  - [ ] Read `src/brain/a2c/mod.rs`.
  - [ ] Read `src/agent/plugin.rs` and `src/game/plugin.rs`.
  - [ ] Trace the exact order of observation build, action write, physics step, episode reward update, and rollout append.
- [ ] Implementation playbook:
  - [ ] Remove duplicate or contradictory system registration.
  - [ ] Ensure reward collection runs only after `episode_loop_system`.
  - [ ] Make the state/action/reward/done tuple lengths stay aligned across all normal and terminal paths.
  - [ ] Record clearly which observation belongs to which action step and keep that contract stable.
- [ ] Stop & verify checkpoints:
  - [ ] Add temporary instrumentation or assertions for buffer lengths after a few steps.
  - [ ] Confirm no reward is appended without a corresponding state/action entry.
- [ ] Invariants / sanity checks:
  - [ ] `states.len() == actions.len() == values.len()`.
  - [ ] `rewards.len() == dones.len()`.
  - [ ] Reward entries must never exceed state/action entries.
  - [ ] Each fixed-tick step contributes at most one rollout transition.
- [ ] Minimal explicit test requirements:
  - [ ] At least one focused unit/integration test or runtime assertion for buffer alignment.

### Policy output and action semantics

The policy produces continuous actions, but the environment applies clamped car controls. Correct behaviour means the training distribution and the applied-control semantics are deliberately aligned rather than accidentally diverging through later clamping.

The recommended default is to keep Gaussian sampling but explicitly define where clamping happens and whether the sampled or clamped action is what training treats as executed. The alternative is to continue sampling unconstrained values and rely on later clamping; that is worse because policy gradients then optimise against an action distribution the environment never actually executes.

- [ ] Discovery (bounded):
  - [ ] Read `src/brain/a2c/mod.rs`.
  - [ ] Read `src/agent/action.rs`.
  - [ ] Confirm how `CarAction::clamped()` is applied before physics.
- [ ] Implementation playbook:
  - [ ] Decide whether the policy action should be clamped immediately at sampling time or transformed through a bounded output parameterisation.
  - [ ] Keep steering and throttle semantics explicit and consistent with the environment contract.
  - [ ] Preserve the stable `CarAction` boundary so keyboard and AI stay interchangeable.
- [ ] Stop & verify checkpoints:
  - [ ] Log a short sample of raw versus applied AI actions during one run.
  - [ ] Confirm throttle stays meaningful and steering remains bounded.
- [ ] Invariants / sanity checks:
  - [ ] Applied steering must remain in `[-1, 1]`.
  - [ ] Applied throttle must remain in `[0, 1]`.
  - [ ] Stored rollout actions must reflect the chosen training contract.
- [ ] Minimal explicit test requirements:
  - [ ] Add a small deterministic test around action post-processing if the contract changes.

### Update stability and baseline verification

The handwritten actor-critic update path is the highest-risk part of the baseline. Correct behaviour means gradients flow through the intended layers, advantage/return calculation is numerically sane, and updates improve or at least visibly respond to data rather than silently diverging.

The recommended default is to add light verification around the current handwritten implementation before expanding features. The alternative is to add more A2C features first, such as snapshots or headless mode; that is worse because it compounds uncertainty before the core update is trusted.

- [ ] Discovery (bounded):
  - [ ] Read `src/brain/a2c/update.rs`.
  - [ ] Read `src/brain/a2c/buffer.rs`.
  - [ ] Read `src/brain/a2c/model.rs` and the optimiser code in `src/brain/common/optim.rs`.
- [ ] Implementation playbook:
  - [ ] Check that value bootstrap, done masking, and advantage normalisation are correct for terminal and non-terminal endings.
  - [ ] Add lightweight assertions or diagnostics for NaN, Inf, and exploding variance in actor outputs.
  - [ ] Surface a minimal set of learning diagnostics to HUD and/or analytics once core correctness is established.
  - [ ] Keep model primitives modular so later brain replacements do not depend on A2C-specific assumptions.
- [ ] Stop & verify checkpoints:
  - [ ] Verify that one update pass changes parameters from their initial values.
  - [ ] Verify that no NaN/Inf values appear in log-std, value predictions, or returns during a short run.
  - [ ] Verify that rollout buffers clear only after a completed update path.
- [ ] Invariants / sanity checks:
  - [ ] `compute_gae` must handle terminal steps correctly.
  - [ ] Optimiser state must stay shape-aligned with layer parameters.
  - [ ] Update must not read past buffer bounds or rely on mismatched vector lengths.
- [ ] Minimal explicit test requirements:
  - [ ] Unit test for `compute_gae` on a small hand-checked sequence.
  - [ ] At least one smoke-level training run inspected for non-finite values.

## Integration Points

- [ ] Where it plugs into the existing pipeline:
  - reads `ObservationVector` from `agent`
  - writes `ActionState.desired` into the action boundary
  - reads reward/end state from `game::episode`
  - should eventually expose summary metrics to `analytics` and `debug`
- [ ] Order of execution and lifecycle placement:
  - observation available
  - select action
  - apply physics
  - compute reward and done
  - append rollout reward
  - update model when rollout is full
- [ ] Pre-conditions:
  - observation size and model input size must match.
  - reward source must be defined once per step.
  - active mode must be explicit.
- [ ] Post-conditions:
  - desired action is updated when AI mode is active.
  - rollout buffers remain aligned.
  - parameter updates occur only on valid rollout data.

## Current Reality Note

- [x] Reward-ordering bug is fixed: progress gain is now computed before best-progress state is advanced.
- [x] Fresh reports now show non-zero progress reward variance; remaining bottleneck is first-turn anticipation and critic quality.

## Debugging / Verification

- [ ] Required logs, assertions, or inspection steps:
  - log mode switches between keyboard and AI.
  - inspect buffer lengths and update count during a short run.
  - inspect for non-finite values in model outputs and optimiser state.
  - inspect one short analytics/HUD view for learning-specific telemetry once added.
- [ ] Common failure patterns:
  - duplicated system registration causes reward collection to drift from action collection.
  - unclamped or mismatched action semantics make training optimise the wrong behaviour.
  - GAE/value bootstrap mistakes produce unstable or sign-inverted learning.
  - ad hoc RNG creation prevents reproducible A2C behaviour.

## Completion Criteria

- [ ] Functional correctness: A2C control and update loop are coherent and free of obvious numerical pathologies.
- [ ] Integration correctness: schedule placement, reward collection, and action semantics are aligned with the fixed-tick environment.
- [ ] Tests passing: at minimum compile checks plus targeted tests for the highest-risk helper logic.
- [ ] Context documents updated to reflect reality.
- [ ] File archived or removed.
