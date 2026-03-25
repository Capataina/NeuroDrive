# Reason About Schedule Order

This exercise is about reconstructing why NeuroDrive's fixed-tick ordering works the way it does.

## Goal

Explain, in your own words, why the current `Input -> Physics -> Collision -> Measurement` ordering exists and identify at least three bugs that would appear if specific systems were moved earlier or later.

## Starting Point

Read:

- `learning/concepts/foundations/fixed-timestep-simulation.md`
- `learning/concepts/domain-patterns/ecs-plugin-scheduling.md`
- `learning/project/architecture/runtime-architecture.md`

Then inspect:

- `src/game/plugin.rs`
- `src/agent/plugin.rs`
- `src/analytics/plugin.rs`
- `src/brain/a2c/mod.rs`

## Deliverable

Produce a short note with:

- the broad set ordering,
- the extra `.after()` and `.before()` relationships that matter,
- three concrete failure modes from incorrect ordering.

## Hints

1. Track when `desired` action becomes `applied` action.
2. Track when reward becomes authoritative.
3. Track why analytics trace capture is placed before A2C reward collection.
