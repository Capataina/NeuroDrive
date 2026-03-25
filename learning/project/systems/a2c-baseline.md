# A2C Baseline

## What This System Does

This is the current autonomous learner. Its job is not to become the final project architecture. Its job is to answer a narrower question: can the current environment and observation contract support autonomous driving improvement at all?

## Where It Fits

The A2C baseline sits behind the stable agent boundary. It consumes `ObservationVector`, writes `ActionState.desired`, reads episode reward and terminal truth, and emits training-health snapshots for analytics and debug.

## Key Mechanics

- `src/brain/plugin.rs` owns mode switching between keyboard and AI.
- `src/brain/a2c/mod.rs` owns the live act path, reward collection, and flush-on-exit behaviour.
- `src/brain/a2c/model.rs` contains the handwritten actor-critic model.
- `src/brain/a2c/buffer.rs` stores rollouts.
- `src/brain/a2c/update.rs` performs optimisation and updates `A2cTrainingStats`.

Current implemented behaviour:

- actor and critic are separate stacks,
- actions are sampled as Gaussians, squashed with `tanh`, and mapped into steering/throttle bounds,
- rollouts update on horizon or eligible terminal conditions,
- GAE is used,
- learning-health stats flow into the HUD and analytics.

## Important Trade-Offs

- This is a strong baseline for learnability validation, but not yet a disciplined experiment harness.
- Controlled RNG ownership is still missing, which weakens reproducibility.
- There is no headless mode, no save/load path, and no evaluation-only mode.
- The repo already contains a concrete plan for vectorised A2C, but the current runtime remains singleton-car.

## Learning Links

- Related concepts: `learning/concepts/core/actor-critic-and-gae.md`
- Related comparison: `learning/project/comparisons/a2c-baseline-vs-biological-target.md`
- Related exercise: `learning/exercises/project/debug-a2c-reproducibility.md`

## Status

Current for this project, but intentionally temporary in long-term architecture terms.
