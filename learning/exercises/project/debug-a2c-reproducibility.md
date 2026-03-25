# Debug A2C Reproducibility

This exercise asks you to diagnose the current determinism gap in the learner and propose a practical repair path.

## Goal

Explain why NeuroDrive is deterministic in core simulation terms but not yet meaningfully reproducible as an end-to-end A2C experiment.

## Starting Point

Read:

- `learning/concepts/foundations/fixed-timestep-simulation.md`
- `learning/project/systems/a2c-baseline.md`
- `learning/project/evolution/current-state-and-next-gaps.md`

Then inspect:

- `src/brain/a2c/mod.rs`
- `src/game/physics.rs`
- `context/systems/determinism.md`

## Deliverable

Write a proposal with:

- the current sources of strong determinism,
- the current sources of weak determinism,
- one recommended next implementation step,
- one alternative and why it is worse right now.

## Hints

1. Compare the pure replay test with the live A2C act path.
2. Look for ad hoc RNG creation.
3. Separate “deterministic simulation” from “reproducible experiment”.
