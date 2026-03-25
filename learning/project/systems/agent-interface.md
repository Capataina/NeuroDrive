# Agent Interface

## What This System Does

The agent interface is the stable boundary between the environment and any controller. It defines what the controller can observe and how it can act.

## Where It Fits

This system sits between `game` truth and `brain` logic. It is the most important contract to preserve if the repo later replaces A2C with a biological learner.

## Key Mechanics

- `src/agent/action.rs` defines `CarAction` and the `ActionState` resource.
- `desired` action is controller output.
- `applied` action is what physics actually consumes after optional smoothing.

On the observation side:

- `SensorReadings` stores world-derived measurements,
- `ObservationVector` stores the normalised fixed-size controller input,
- `ObservationConfig` centralises sensor and scaling parameters.

The live observation vector currently has 23 dimensions:

- 11 rays,
- speed,
- signed lateral offset,
- heading error,
- angular velocity,
- 4 lookahead heading-delta features,
- 4 lookahead curvature features.

## Important Trade-Offs

- The interface is richer than the original Milestone 1 sketch because lookahead geometry has already been added.
- `TrackProgress` is intentionally excluded from the observation vector to avoid direct progress leakage.
- The interface is stable across keyboard and AI modes, which is good for analytics and future brain replacements.

## Learning Links

- Related concepts: `learning/concepts/core/observation-design.md`
- Related systems: `learning/project/systems/environment.md`
- Related exercises: `learning/exercises/project/extend-observation-vector.md`

## Status

Current for this project.
