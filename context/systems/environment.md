# System — Environment

## Scope / Purpose

- Own the deterministic racing world that the controller acts within.
- Provide track geometry, spawn state, car physics, collision truth, progress measurement, reward shaping, and episode boundaries.
- The environment is the spatial and temporal foundation that every other subsystem reads from.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/maps/` | Static track topology, tile semantics, centreline derivation, spawn pose, visual tile rendering | Runtime car state, observations, policy logic |
| `src/game/` | Car entity lifecycle, physics step, collisions, progress measurement, reward accumulation, episode resets, moving averages | Observation construction, policy updates, analytics export |

- The environment owns **reward truth** and **terminal truth**. All downstream consumers (brain, analytics, debug) must read these rather than computing their own.
- `maps` is the only subsystem that produces spatial truth (`Track`, `TrackGrid`, `TrackCenterline`).

## Current Implemented Reality

### Track

- One hard-coded Sepang-inspired closed loop built from tile parts in `src/maps/monaco.rs` via `MonacoPlugin`.
- `TrackGrid` is the authoritative driveable-area occupancy query used by both collisions and raycasts.
- `TrackCenterline` is a closed polyline derived from tile connectivity, stored on the `Track` component. It supports:
  - arc-length projection (`project(pos) → s, fraction, closest_point, tangent, distance`),
  - tangent queries at arbitrary arc-length (`tangent_at_s(s)`).
- `Track` component stores `grid`, `spawn_position`, `spawn_rotation`, and `centerline`.

### Car

- Multiple car entities spawned by `setup_game` in `GamePlugin` after the track is ready.
- `TrainerConfig` controls the number of cars (default 3), lateral spawn spread, and alpha values.
- Cars are spawned with deterministic lateral offsets perpendicular to the track heading at the spawn point.
- **Random spawn positions:** Car 0 always spawns at the canonical track start position. Cars 1–N (ghost cars) spawn at random positions along the centreline, facing the tangent direction. On each episode reset, ghost cars receive a new random position via the `SpawnRng` resource. Car 0 always resets to its fixed start position.
- Each car entity carries:
  - `Sprite` (unique colour from 25-colour palette, 12×6 world units),
  - `Transform`,
  - `Car` component (velocity, rotation_speed=4.0, thrust=750.0, drag=0.985),
  - `EnvInstanceId(u32)` — stable identity for the car's lifetime,
  - `SpawnConfig { position, rotation }` — per-car reset target,
  - `CarColour { r, g, b }` — unique visual colour,
  - `ActionState` — per-car desired/applied action (Component, not Resource),
  - `EpisodeState` — per-car episode counters, rewards, progress (Component, not Resource),
  - `EpisodeMovingAverages` — per-car rolling statistics (Component, not Resource),
  - `TrackProgress`,
  - `SensorReadings`,
  - `ObservationVector`.

### Physics

- `car_physics_system` runs in `SimSet::Physics`. It is the **only place** where actions mutate car state.
- Backed by the pure helper `step_car_dynamics()`, which takes kinematic state + action + dt + params and produces the next state. This helper is shared with the deterministic replay test.
- Dynamics model:
  - heading change: `heading += -steering * rotation_speed * dt`
  - thrust: `velocity += forward * thrust * throttle * dt` (only when throttle > 0)
  - drag: `velocity *= drag`
  - position: `position += velocity * dt`

### Collision

- `collision_detection_system` runs in `SimSet::Collision`.
- Iterates all car entities and checks the four rotated corners of each car's bounding rectangle against `track.grid.is_road_at()`.
- Adds a `Collided` marker component (SparseSet storage) to cars with an off-road corner; removes it from cars that are on-road.
- Each car's collision state is independent — one car crashing does not affect others.

### Progress

- `update_track_progress_system` runs first in `SimSet::Measurement`.
- Projects the car's position onto the centreline, producing `TrackProgress` fields: `s`, `fraction`, `closest_point`, `tangent`, `distance`.
- Uses `for` loops over car queries — already supports multi-car iteration.

### Episode Loop

- `episode_loop_system` runs after progress update in `SimSet::Measurement`.
- Iterates all car entities, processing each car's episode independently via per-car `EpisodeState` and `EpisodeMovingAverages` components.
- Per-tick processing (for each car):
  1. Computes per-tick forward progress delta: `(fraction - previous_tick_fraction).max(0)`.
  2. Computes speed-weighted progress reward: `progress_delta * (speed / speed_reward_reference) * progress_reward_scale`. Reference speed: 200.0, scale: 100.0. Zero reward when stationary or going backward.
  3. Adds flat time penalty: -0.005 per tick. No heading-speed component.
  4. Checks terminal conditions in priority: crash → timeout (no lap detection — finish line has been removed).
  5. Applies terminal reward: crash penalty (-5.0).
  6. On terminal: finalises episode state, pushes to moving averages, resets car. Ghost cars (env_id ≠ 0) get a new random spawn position along the centreline; car 0 always resets to the canonical start.

### Reward Composition

| Term | Value | When |
|------|-------|------|
| Speed-weighted progress | `+delta × (speed/200) × 100` | Every tick with forward progress |
| Time penalty | `-0.005` | Every tick |
| Crash penalty | `-5.0` | One-off on crash |
| Lap bonus | `+100.0` | One-off on lap completion (legacy — planned for removal) |

### Episode Termination

| Condition | Detection | Reset |
|-----------|-----------|-------|
| **Crash** | `Collided` marker present on car entity | Immediate |
| **Lap complete** | Lap armed (≥25% progress) AND previous fraction ≥0.85 AND current ≤0.15 | Immediate |
| **Timeout** | Ticks × dt ≥ 30 seconds | Immediate |

- Lap detection is **progress-wrap based**, not finish-line-crossing based.
- Reset returns each car to its own `SpawnConfig` position/rotation (not the canonical track spawn) and re-syncs `TrackProgress`.
- **Planned change:** Lap detection is scheduled for removal. The finish-line concept will be replaced by a distance-from-spawn paradigm where progress measures cumulative arc-length driven from the spawn point, with no special finish position.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer(s) | Notes |
|-----------|----------|-------------|-------|
| `Track` | `maps` startup | physics queries, observations, debug | Single authoritative track entity |
| `Collided` marker | collision system | episode system, HUD stats | Per-car component, checked before reward finalisation |
| `TrackProgress` | progress system | episode logic, debug, analytics, observations | Environment measurement, not policy input |
| `EpisodeState` | episode system (per-car Component) | A2C reward collection, HUD, analytics | Current-tick and last-episode summaries per car |
| `EpisodeMovingAverages` | episode system (per-car Component) | HUD, ranking | Rolling return/progress/crash means (window=20) per car |
| `EpisodeConfig` | resource defaults | episode system | All reward and timing parameters |

```text
Physics mutates car → Progress recomputed → Episode records reward and checks terminal
→ If terminal: finalise, push moving avg, reset car, re-sync progress
→ If not: store previous progress fraction for next tick
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `EpisodeConfig`, `TrainerConfig`
- **Runtime components (per car):** `Car`, `EnvInstanceId`, `SpawnConfig`, `CarColour`, `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, `TrackProgress`, `Collided` (marker)
- **Runtime components (track):** `Track`
- **Tests:** Deterministic replay test for `step_car_dynamics()` in `src/game/physics.rs`

## Known Issues / Active Risks

- **Singleton-car assumptions have been removed** from the game layer. Collision, episode, physics, and progress systems all iterate all cars. `EpisodeState`, `EpisodeMovingAverages`, and `ActionState` are per-car Components.
- Analytics and HUD systems still use **temporary shims** that target the first car only, pending a full analytics overhaul.
- **Environment regression coverage is thin:** no ECS-level tests for collision timing, lap wrapping, reset correctness, or episode-summary edge cases.
- The track layer assumes a **single closed loop** with no branching.
- Finish-line rendering is cosmetic only; lap completion relies on **progress thresholds** rather than explicit crossing logic.
- The environment runs in a **normal Bevy windowed loop only** — no headless or accelerated training mode exists.
- `wrap_angle` and `signed_angle_between` are duplicated across `observation.rs` and `episode.rs`.

## Partial / In Progress

- Reward shaping has been revised to reduce "sprint then crash" incentives, but that behavioural effect depends on analytics validation rather than formal tests.
- Progress and reward state now feed more downstream systems than originally intended: A2C, HUD, analytics, and turn-diagnosis metrics all depend on `EpisodeState`.

## Planned / Missing / Likely Changes

- **Headless training mode** is a likely requirement once longer experiments matter.
- **Multi-track support** would force sharper contracts around spawn, progress semantics, and run metadata.
- Lap detection may move from wrap heuristics to explicit finish-line crossing.
- **ECS-level regression tests** are the most obvious missing verification layer.
- A brake action channel may eventually be added as a separate action-space change.
- **Finish-line removal and distance-from-spawn paradigm:** The current lap-complete detection and lap bonus will be removed. Progress will be measured as cumulative forward arc-length from spawn, not absolute track position. This removes the gaming incentive for cars spawning near the wrap point and makes progress metrics honest across random spawn positions.
- **Analytics rework required:** The analytics pipeline currently reports absolute track position as progress. With random spawns, this inflates ghost car metrics. Analytics needs to track distance-driven-from-spawn and report car 0 separately as the honest benchmark.
- **Turning speed may be a physics bottleneck:** At rotation_speed=4.0 rad/s and 60 Hz, maximum heading change is 3.8°/tick. At high speeds (>300 u/s), this may be insufficient for tight corners. Investigation needed before adding more reward complexity.

## Durable Notes / Discarded Approaches

- Reset ownership is centralised in `episode.rs`. Earlier split reset behaviour between collision and episode handling was a bad fit because it obscured terminal reward truth.
- `TrackProgress` is shared widely but remains **environment truth, not observation truth**. Preserving that separation is important as the learning stack evolves.
- The `CAR_WIDTH` × `CAR_HEIGHT` (12×6) collision rectangle is intentionally tight — it detects edge-level road departure rather than centre-only checking.
- **Centreline proximity reward was removed** after the policy learned to farm it by sitting still near the centreline, earning steady reward without driving. The speed-weighted progress reward makes a separate centreline term redundant — you can't earn progress reward without following the track.
- **Episode-end progress bonus was removed** because it added critic instability without meaningfully improving learning. The cumulative speed-weighted progress reward already rewards longer, faster runs.
- **Heading-speed penalty was removed** because it's redundant with speed-weighted progress — a misaligned car makes less forward progress per tick, so the penalty is implicit.
- **Crash penalty was reduced from -10 to -5** because harsh crash penalties discouraged the aggressive driving that produces learning. Dying is how cars learn; the crash penalty is a tiebreaker, not a deterrent.

## Obsolete / No Longer Relevant

- Any architecture note that treats the repository as "just the environment layer" is obsolete; the environment is still central, but no longer the only substantial runtime subsystem.
