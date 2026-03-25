# Environment

## What This System Does

The environment owns the world the controller acts in: the track, the car, physics, collision truth, progress measurement, reward shaping, and episode boundaries.

## Where It Fits

This is the foundation of the runtime. The brain learns against environment truth; it does not define that truth.

## Key Mechanics

- `src/maps/monaco.rs` builds the current hard-coded track.
- `src/game/car.rs` spawns one car with transform, physics parameters, and agent-facing components.
- `src/game/physics.rs` applies the current action on the fixed tick.
- `src/game/progress.rs` projects the car onto the centreline and updates progress state.
- `src/game/collision.rs` checks the rotated car rectangle against the road grid.
- `src/game/episode.rs` computes reward, terminal reasons, resets, and moving averages.

The live reward is more specific than the early README summary:

- progress gain reward,
- per-tick time penalty,
- heading-speed penalty,
- crash penalty,
- lap-complete bonus.

## Important Trade-Offs

- Reward is dense and interpretable, but still hand-shaped.
- Lap completion is currently progress-wrap based rather than explicit finish-line crossing.
- The current environment is singleton-car, which keeps the runtime simple but makes the planned vectorised trainer structurally invasive.

## Learning Links

- Related concepts: `learning/concepts/domain-patterns/deterministic-racing-environment.md`
- Related systems: `learning/project/systems/agent-interface.md`
- Related exercises: `learning/exercises/project/reason-about-schedule-order.md`

## Status

Current for this project.
