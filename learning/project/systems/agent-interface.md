# Agent Interface

## Status

Current in the project runtime.

## What This System Does

The agent interface is the stable boundary between:

- what the environment exposes to controllers,
- and what controllers are allowed to send back.

It provides:

- the `CarAction` contract,
- desired versus applied action state,
- optional smoothing,
- sensor readings,
- observation vector construction.

## Why This Boundary Is Important

It decouples several concerns:

- the environment can stay the source of truth,
- controllers can vary,
- analytics can record a stable action surface,
- debug views can inspect both raw sensor readings and controller-facing vectors.

This is one of the cleanest subsystem boundaries in the current repository.

## Current Observation Design

The observation vector dimension is `23`.

The current schema is geometry-rich rather than minimalist:

- ray distances,
- speed,
- signed lateral offset,
- heading error,
- angular velocity,
- lookahead heading deltas,
- lookahead curvatures.

This is a deliberate design choice. The project is not trying to prove that a learner can infer everything from sparse raw inputs.

## Action Design

The action surface stays intentionally small:

- steering,
- throttle.

There is no separate brake channel yet.

That keeps the control problem manageable, but it also means future changes to action space would have a wide blast radius across:

- the policy output layer,
- analytics schemas,
- debug displays,
- exercise expectations.

## Important Design Lesson

`TrackProgress` is not part of the observation vector. This is a strong boundary choice and should not be broken casually. It preserves the difference between environment truth and controller-accessible representation.

## Current Risks

- no observation schema versioning,
- no stronger compatibility guard for snapshots or offline comparison,
- no brake channel,
- no broader input-health instrumentation yet.

## Related Files

- `concepts/core/observations-actions-and-representation.md`
- `project/systems/a2c-baseline.md`
- `project/systems/debug-runtime.md`
