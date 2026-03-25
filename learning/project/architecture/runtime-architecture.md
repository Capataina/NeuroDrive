# Runtime Architecture

## What This System Does

This file teaches the current NeuroDrive runtime as a whole. It is the learner-friendly counterpart to `context/architecture.md`.

## Where It Fits

NeuroDrive currently has five main runtime layers:

- `maps` for track geometry and spatial truth,
- `game` for car state, physics, collisions, progress, and rewards,
- `agent` for the controller-facing action and observation boundary,
- `brain` for the current A2C baseline and mode switching,
- `analytics` and `debug` for observability.

## Key Mechanics

The runtime starts in `src/main.rs`, where Bevy plugins are wired and the fixed 60 Hz simulation step is configured.

The fixed-tick pipeline is intentionally ordered:

1. input systems write desired actions,
2. physics mutates car state,
3. collision checks terminal truth,
4. measurement systems rebuild progress, rewards, observations, analytics, and HUD data.

That order matters because later systems assume earlier state is already authoritative.

## Important Trade-Offs

- The architecture is no longer “environment only”; A2C and analytics are live.
- The project intent is still biological local plasticity, so the current A2C code should stay modular rather than expanding into the permanent centre of the repo.
- Several current systems still assume one car, which is fine for the present runtime but blocks the proposed vectorised trainer work.

## Learning Links

- Related concepts: `learning/concepts/domain-patterns/ecs-plugin-scheduling.md`
- Related systems: `learning/project/systems/environment.md`
- Related evolution: `learning/project/evolution/current-state-and-next-gaps.md`
