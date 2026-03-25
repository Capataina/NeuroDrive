# System — Environment

## Scope / Purpose

- Own the deterministic racing world that the controller acts within.
- Provide track geometry, spawn state, car physics, collision truth, progress measurement, reward shaping, and episode boundaries.

## Boundaries / Ownership

- `src/maps/` owns static track topology, tile semantics, centreline derivation, and track rendering.
- `src/game/` owns runtime car state, physics, collisions, progress measurement, reward accumulation, resets, and moving averages.
- The environment owns reward truth and terminal truth.
- It does not own observations, policy updates, or analytics export.

## Current Implemented Reality

- The application currently ships one hard-coded Sepang-inspired closed loop built from tile parts in `src/maps/monaco.rs`.
- `TrackGrid` is the authoritative driveable-area query used by both collisions and raycasts.
- A closed `TrackCenterline` is derived from tile connectivity and stored on the `Track` component together with spawn pose.
- `GamePlugin` configures a fixed `SimSet` chain: `Input -> Physics -> Collision -> Measurement`.
- A single car entity is spawned with:
  - sprite and transform,
  - kinematic parameters,
  - `TrackProgress`,
  - sensor and observation components.
- Physics is deterministic and centralised in `car_physics_system`, backed by the pure helper `step_car_dynamics()`.
- Collision detection checks rotated car-rectangle corners against road occupancy and emits `CollisionEvent` when any corner leaves the road.
- `episode_loop_system` owns:
  - per-tick reward accumulation,
  - crash/timeout/lap-complete termination,
  - reset-to-spawn,
  - progress resynchronisation after reset,
  - rolling episode means.
- Reward composition currently includes:
  - positive gain in best-so-far episode progress,
  - a per-tick time penalty,
  - a heading-speed penalty,
  - one-off crash penalty,
  - one-off lap bonus.
- Lap completion is still progress-wrap based rather than finish-line-crossing based.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer | Notes |
|---|---|---|---|
| `Track` | `maps` startup | physics-adjacent queries, observations, debug | single authoritative track entity |
| `CollisionEvent` | collision system | episode system, HUD stats | emitted before reward finalisation |
| `TrackProgress` | progress system | episode logic, debug, analytics, observations | environment measurement, not policy input |
| `EpisodeState` | episode system | A2C reward collection, HUD, analytics | carries both current-tick and last-episode summaries |
| `EpisodeMovingAverages` | episode system | HUD | rolling return/progress/crash means |

- The reward and episode lifecycle contract is order-sensitive:
  - physics mutates the car,
  - progress is recomputed,
  - episode logic records reward and terminal state,
  - resets happen only after terminal bookkeeping is complete.

## Implemented Outputs / Artifacts

- Runtime resources:
  - `EpisodeConfig`
  - `EpisodeState`
  - `EpisodeMovingAverages`
- Runtime messages:
  - `CollisionEvent`
- Runtime components:
  - `Track`
  - `Car`
  - `TrackProgress`
- Tests:
  - pure deterministic replay test for car dynamics in `src/game/physics.rs`

## Known Issues / Active Risks

- Environment-level regression coverage is still thin; there are no ECS-level tests for collision timing, lap wrapping, reset correctness, or episode-summary edge cases.
- The track layer currently assumes a single closed loop with no branching.
- Finish-line rendering is cosmetic only; lap completion still relies on progress thresholds rather than explicit crossing logic.
- The environment currently runs in a normal Bevy windowed loop only; there is no headless or accelerated training mode.

## Partial / In Progress

- Reward shaping has already been revised to reduce “sprint then crash” incentives, but that behavioural effect still depends on analytics validation rather than strong formal tests.
- Progress and reward state now feed more downstream systems than earlier versions of the project:
  - A2C,
  - HUD,
  - analytics,
  - turn-diagnosis metrics.

## Planned / Missing / Likely Changes

- A headless training mode is a likely future requirement once longer experiments matter.
- Multi-track support would force sharper contracts around spawn, progress semantics, and run metadata.
- Lap detection may eventually move from wrap heuristics to explicit finish-line crossing.
- ECS-level regression tests are the most obvious missing verification layer in this subsystem.

## Durable Notes / Discarded Approaches

- Reset ownership is now centralised in `episode.rs`. Earlier split reset behaviour between collision and episode handling was a bad fit because it obscured terminal reward truth.
- `TrackProgress` is shared widely but remains environment truth, not observation truth. Preserving that separation is important as the learning stack evolves.

## Obsolete / No Longer Relevant

- Any architecture note that still describes the repository as “just the environment layer” is obsolete; the environment is still central, but no longer the only substantial runtime subsystem.
