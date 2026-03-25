# Exercise: Inspect The Observation Pipeline

## Goal

Trace the current observation contract from geometry truth to policy input.

## Starting Point

Read:

- `project/systems/maps-and-centreline.md`
- `project/systems/agent-interface.md`
- `concepts/core/observations-actions-and-representation.md`

Then inspect:

- `src/agent/observation.rs`
- `src/game/progress.rs`
- `src/maps/centerline.rs`

## Tasks

- identify where each of the 23 observation dimensions comes from,
- explain why `TrackProgress` itself is not directly passed into the observation vector,
- describe one reason the current hybrid design may be better than rays alone,
- describe one reason observation versioning will matter later.

## Expected Outcome

You should finish with a concrete mapping from project geometry to learner-visible features rather than a vague sense that "the sensors come from the environment".
