# Implementation-First Path

## Who This Path Is For

This path is for experienced engineers who already understand reinforcement learning and want to reach contribution-ready understanding of the NeuroDrive codebase as quickly as possible. It skips foundational theory and goes straight to the code-grounded project files, filling in RL-specific gaps only where needed.

It is the wrong path if you are not already comfortable with GAE, actor-critic architectures, and rollout buffers. Use `paths/reinforcement-learning-path.md` first if you need that foundation.

## What This Path Assumes

- Solid understanding of on-policy RL (actor-critic, GAE, rollout buffers)
- Comfortable reading Rust
- No prior Bevy knowledge required (a brief primer is included)

## What You Will Understand by the End

- The full NeuroDrive 23-dimensional observation vector and how each feature is computed
- The fixed-tick execution pipeline and where each system fits
- The A2C rollout buffer structure, GAE implementation, and update path
- The action contract: tanh squashing, log-probability correction, clamp hit tracking
- The reward structure and episode lifecycle
- The analytics capture pipeline and export format
- The current implementation's known gaps and planned improvements

## Recommended Sequence

- [ ] `references/observation-vector-reference.md`
  - Start here. This reference file lists all 23 input features, their normalisation, and their origin. It is the fastest way to ground yourself in what the policy sees.

- [ ] `project/architecture/fixed-tick-pipeline.md`
  - The SimSet chain. Read this before anything else about the brain, because schedule placement is essential context for understanding why the A2C systems are ordered as they are.

- [ ] `project/systems/agent-interface.md`
  - How observations are constructed and how the action contract works. Focus on the `ActionState.desired` vs `ActionState.applied` separation and the smoothing path.

- [ ] `project/systems/environment-system.md`
  - Reward structure, episode boundaries, and the progress measurement system. Pay attention to the `EpisodeState` fields and how they feed the A2C reward collector.

- [ ] `project/systems/a2c-brain.md`
  - The A2C implementation in full. Rollout buffer, GAE, update trigger semantics, bootstrap handling, training stats, and mode switching.

- [ ] `concepts/core/advantage-estimation.md`
  - Read the GAE section (Section 3 onwards) to fill in theory gaps around the advantage normalisation and return computation. Skip the early sections if you already know them.

- [ ] `exercises/core/implement-gae.md`
  - Implement GAE independently and compare to the NeuroDrive `compute_gae` function. This is the best single verification that you understand the update core.

- [ ] `project/decisions/tanh-squashed-actions.md`
  - Why tanh squashing is used, and why the log-probability computation must include the Jacobian correction. This is easy to miss and easy to implement wrongly.

- [ ] `project/systems/analytics-system.md`
  - What data is captured per tick and per episode, how it flows through trackers and metrics, and what the exported reports contain.

- [ ] `project/evolution/from-baseline-to-brain.md`
  - Where the project is going. Read this to understand A2C's intended scope and eventual retirement.

## After This Path

From here, proceed to:

- `paths/neuroscience-path.md` — understand what comes after A2C
- `paths/project-architecture-path.md` — deeper architectural coverage if you want to extend any subsystem
- `context/plans/vectorised-a2c-visual-trainer.md` — if you are planning to implement the next major A2C upgrade

## Notes

- This path is aggressive about skipping theory. If you hit anything in the project files that you do not understand, check `concepts/core/` for the relevant background.
- The observation vector reference is the single most useful quick-reference file for understanding the model input. Keep it open while reading `project/systems/a2c-brain.md`.
- The `project/decisions/tanh-squashed-actions.md` file is short but important. The log-probability correction is a common source of bugs in actor-critic implementations.
