# Exercise: Extend The Analytics Schema

## Goal

Design a safe extension to the analytics output that improves experiment quality.

## Starting Point

Read:

- `project/systems/analytics.md`
- `project/evolution/project-state-and-next-pressure-points.md`

Then inspect:

- `src/analytics/models.rs`
- `src/analytics/exporters/json.rs`
- `src/analytics/exporters/markdown.rs`

## Tasks

- choose one missing metadata field such as seed, config snapshot, git revision, or track identity,
- explain where it should be captured,
- explain which exported artefacts should include it,
- explain how the change improves run comparison rather than only adding detail.

## Hints

- Prefer changes that improve experiment discipline.
- Think about whether the field belongs at run level, episode level, or update level.
