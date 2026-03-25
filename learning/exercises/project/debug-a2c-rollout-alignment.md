# Exercise: Debug A2C Rollout Alignment

## Goal

Reason about where a reward/action/state misalignment bug could arise in the current schedule and how you would detect it.

## Starting Point

Read:

- `project/architecture/data-flow-and-schedule.md`
- `project/systems/a2c-baseline.md`
- `project/systems/analytics.md`

Then inspect:

- `src/brain/a2c/mod.rs`
- `src/brain/a2c/update.rs`
- `src/game/episode.rs`

## Tasks

- explain the intended order from `ObservationVector` to action to reward collection,
- identify one place where stale observation or stale reward would silently damage learning,
- propose one test or assertion that would make this safer,
- explain what analytics symptom you might expect if rollout alignment were wrong.

## Hints

- Pay attention to terminal versus non-terminal update triggers.
- Pay attention to reset timing and post-reset observations.
