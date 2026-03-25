# Extend Observation Vector

This exercise is about reasoning safely about interface changes, not about typing code quickly.

## Goal

Design a change that adds one new observation feature without breaking the controller boundary, analytics expectations, or debug assumptions.

## Starting Point

Read:

- `learning/concepts/core/observation-design.md`
- `learning/project/systems/agent-interface.md`
- `learning/project/systems/analytics-and-debugging.md`

Then inspect:

- `src/agent/observation.rs`
- `src/brain/a2c/mod.rs`
- `src/analytics/trackers/trace.rs`

## Deliverable

Write a change plan covering:

- where the new raw signal would come from,
- how it would be normalised,
- what constant or dimension changes would be required,
- which downstream consumers would need review,
- what test or assertion you would add first.

## Hints

1. Start from the shared dimension constant rather than from the model.
2. Ask whether analytics trace capture should store the raw signal, derived signal, or neither.
3. Ask whether the feature adds meaningful information or just leaks an easier proxy for progress.
