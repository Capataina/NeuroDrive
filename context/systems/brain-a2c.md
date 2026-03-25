# System — Brain A2C Baseline

## Scope / Purpose

- Provide the current autonomous-controller baseline used to test whether the environment and observation contract are learnable.
- Keep the baseline self-contained in Rust without external ML frameworks.

## Boundaries / Ownership

- `src/brain/plugin.rs` owns mode-switch setup and rollout reset on mode toggle.
- `src/brain/types.rs` owns `AgentMode` and the minimal `Brain` trait.
- `src/brain/a2c/` owns the current algorithm implementation:
  - action selection,
  - rollout storage,
  - GAE computation,
  - actor/critic updates,
  - training-health snapshots.
- `src/brain/common/` owns reusable handwritten ML primitives used by A2C.
- The brain reads observations and reward truth; it does not own reward definition or raw environment state.

## Current Implemented Reality

- `AgentMode` defaults to `Ai`; `F4` toggles between AI and keyboard.
- Toggling mode clears the A2C rollout buffer and resets the brain step counter to avoid mixed-control trajectories.
- `A2cBrain` currently contains:
  - a handwritten `ActorCritic`,
  - `RolloutBuffer`,
  - discount and GAE parameters,
  - rollout/update thresholds,
  - a simple step counter.
- The model is a handwritten two-hidden-layer actor-critic MLP with separate actor and critic stacks and learnable log-standard deviations.
- Action semantics are bounded by design:
  - the policy samples Gaussian latent actions,
  - applies `tanh`,
  - maps steering to `[-1, 1]`,
  - maps throttle to `[0, 1]`.
- `a2c_act_system` runs in `SimSet::Input` after keyboard input and before action smoothing.
- The act path appends state, action, latent action, clamp-hit flags, and critic value to the rollout buffer.
- `a2c_collect_reward_system` runs in `SimSet::Measurement` after episode truth and observation rebuild.
- Reward collection appends one reward and one done flag per step and triggers updates on:
  - rollout horizon,
  - or a terminal step once the minimum batch size is met.
- Partial non-terminal rollouts can bootstrap from the current observation.
- `a2c_flush_on_exit_system` updates on app exit if residual rollout data remains.
- `A2cTrainingStats` records the most recent completed update’s:
  - losses,
  - entropy,
  - explained variance,
  - action spread,
  - clamp fraction,
  - layer weight/gradient/dead-ReLU health.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer | Notes |
|---|---|---|---|
| `ObservationVector` | agent | A2C act path | fixed-size model input |
| `ActionState.desired` | A2C act system | smoothing/physics | same control boundary as keyboard |
| `EpisodeState.current_tick_reward` | game | A2C reward collector | authoritative per-step reward |
| `EpisodeState.current_tick_end_reason` | game | A2C reward collector | terminal-step truth |
| `A2cTrainingStats` | A2C update path | debug HUD, analytics tracker | snapshot of latest completed update |

```text
observation_t
  -> a2c_act_system
  -> desired action
  -> physics / environment step
  -> episode_loop_system computes reward_t and done_t
  -> a2c_collect_reward_system appends reward_t, done_t
  -> optional a2c_update()
```

## Implemented Outputs / Artifacts

- Runtime resources:
  - `AgentMode`
  - `A2cBrain`
  - `A2cTrainingStats`
- Runtime update code:
  - `RolloutBuffer::compute_gae()`
  - `a2c_update()`
- Reusable handwritten ML primitives in `src/brain/common/`.

## Known Issues / Active Risks

- RNG ownership is still ad hoc via `rand::rng()`, so deterministic replay does not extend meaningfully into the A2C path yet.
- There is no snapshot/save-load path, no evaluation mode, and no headless training loop.
- Verification is still light for the highest-risk algorithm logic:
  - no dedicated A2C integration test,
  - no explicit behavioural success threshold,
  - limited protection against silent training regressions beyond runtime stats.
- The `Brain` trait is minimal and currently does not drive a broader pluggable-brain architecture.

## Partial / In Progress

- The baseline is integrated and live, but it is still better described as a validation harness than as a trusted learning subsystem.
- Several earlier timing and contract bugs have already been corrected in code:
  - reward/order alignment,
  - terminal-step handling,
  - bounded action semantics,
  - next-state bootstrap handling,
  - flush-on-exit,
  - rollout reset on mode switches.

## Planned / Missing / Likely Changes

- Controlled seeding is the most obvious missing prerequisite for reproducible learning experiments.
- Headless training, persistence, and explicit evaluation mode are likely to matter before longer experiments become credible.
- The final project direction still points toward biological/local-plasticity systems; A2C should stay modular enough to be retired later without distorting the rest of the runtime.
- A concrete implementation plan now exists for a visible 25-car vectorised trainer in `context/plans/vectorised-a2c-visual-trainer.md`; if this work starts, that plan should be kept current and then removed or archived once the runtime reality lands.

## Durable Notes / Discarded Approaches

- The bounded tanh-squashed action contract is a deliberate improvement over sampling unconstrained values and relying on later clamping.
- A2C exists here as a baseline for learnability validation, not as the intended final learning architecture.
- External A2C research and a NeuroDrive-specific implementation ladder now live in `context/references/a2c-for-neurodrive.md`; keep deep algorithm research there rather than expanding this system file into a mixed implementation/research document.

## Obsolete / No Longer Relevant

- Any document that still treats A2C as a future-only milestone is obsolete; the code is present and participates in the live runtime path.
