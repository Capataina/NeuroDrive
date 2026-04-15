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
- `Track` component stores `grid` and `centerline`.
- Finish line sprite has been removed from `monaco.rs`.

### Car

- Multiple car entities spawned by `setup_game` in `GamePlugin` after the track is ready.
- `TrainerConfig` controls the number of cars (default 8) and alpha values.
- **Random spawn positions:** All cars (including car 0) spawn at random positions along the centreline, facing the tangent direction. On each episode reset, every car receives a new random position via the `SpawnRng` resource. There is no special-case car 0 that always resets to a fixed start.
- Each car entity carries:
  - `Sprite` (unique colour from 25-colour palette, 12×6 world units),
  - `Transform`,
  - `Car` component (velocity, rotation_speed=8.0, thrust=750.0, drag=0.985),
  - `EnvInstanceId(u32)` — stable identity for the car's lifetime,
  - `SpawnConfig { position, rotation }` — per-car reset target,
  - `CarColour { r, g, b }` — unique visual colour,
  - `ActionState` — per-car desired/applied action (Component, not Resource),
  - `EpisodeState` — per-car episode counters, rewards, progress (Component, not Resource). Contains `distance_driven`, `spawn_s`, `previous_s` instead of lap-related fields,
  - `EpisodeMovingAverages` — per-car rolling statistics (Component, not Resource),
  - `TrackProgress`,
  - `SensorReadings`,
  - `ObservationVector`.

### Physics

- `car_physics_system` runs in `SimSet::Physics`. It is the **only place** where actions mutate car state.
- Backed by the pure helper `step_car_dynamics()`, which takes kinematic state + action + dt + params and produces the next state. This helper is shared with the deterministic replay test.
- Dynamics model:
  - heading change: `heading += -steering * rotation_speed * dt` (rotation_speed = 8.0)
  - thrust: `velocity += forward * thrust * throttle * dt` (throttle in [0, 1])
  - drag: `velocity *= drag` (sole deceleration mechanism — no braking)
  - position: `position += velocity * dt`

### Collision

- `collision_detection_system` runs in `SimSet::Collision`.
- Iterates all car entities and checks the four rotated corners of each car's bounding rectangle against `track.grid.is_road_at()`.
- Adds a `Collided` marker component (SparseSet storage) to cars with an off-road corner; removes it from cars that are on-road.
- Each car's collision state is independent — one car crashing does not affect others.

### Progress

- `update_track_progress_system` runs first in `SimSet::Measurement`.
- Projects the car's position onto the centreline, producing `TrackProgress` fields: `s`, `fraction`, `closest_point`, `tangent`, `distance`.
- Progress is measured as **cumulative forward arc-length from spawn** with wrap handling. The `EpisodeState` tracks `spawn_s` and `previous_s` to compute `distance_driven` — the total forward arc-length accumulated since the episode began, irrespective of absolute track position. This makes progress metrics honest across random spawn positions.
- Uses `for` loops over car queries — already supports multi-car iteration.

### Episode Loop

- `episode_loop_system` runs after progress update in `SimSet::Measurement`.
- Iterates all car entities, processing each car's episode independently via per-car `EpisodeState` and `EpisodeMovingAverages` components.
- Per-tick processing (for each car):
  1. Computes velocity projection reward: `dot(velocity, tangent) / 200 × scale` — rewards forward motion along the centreline tangent direction.
  2. Computes centreline proximity reward: `0.3 × (1 - (d/50)²)` — rewards staying close to the centreline, scaled by distance `d`.
  3. Adds flat time penalty: -0.005 per tick.
  4. Checks terminal conditions in priority: crash → timeout (30 seconds). No lap detection — finish line has been removed.
  5. Applies terminal reward: crash penalty (0.0 — crashes end the episode but carry no explicit penalty).
  6. On terminal: finalises episode state, pushes to moving averages, resets car. All cars get a new random spawn position along the centreline on reset.

### Reward Composition

| Term | Value | When |
|------|-------|------|
| Velocity projection | `+dot(velocity, tangent) / 200 × scale` | Every tick |
| Centreline proximity | `+0.3 × (1 - (d/50)²)` | Every tick |
| Time penalty | `-0.005` | Every tick |
| Crash penalty | `0.0` | One-off on crash (episode ends, no explicit penalty) |

### Episode Termination

| Condition | Detection | Reset |
|-----------|-----------|-------|
| **Crash** | `Collided` marker present on car entity | Immediate |
| **Timeout** | Ticks × dt ≥ 30 seconds | Immediate |

- There is no lap detection and no finish line. Episodes end only on crash or timeout.
- Reset returns each car to a new random `SpawnConfig` position/rotation along the centreline and re-syncs `TrackProgress`.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer(s) | Notes |
|-----------|----------|-------------|-------|
| `Track` | `maps` startup | physics queries, observations, debug | Single authoritative track entity |
| `Collided` marker | collision system | episode system, HUD stats | Per-car component, checked before reward finalisation |
| `TrackProgress` | progress system | episode logic, debug, analytics, observations | Environment measurement, not policy input |
| `EpisodeState` | episode system (per-car Component) | PPO reward collection, HUD, analytics | Current-tick and last-episode summaries per car. Contains distance_driven, spawn_s, previous_s |
| `EpisodeMovingAverages` | episode system (per-car Component) | HUD, ranking | Rolling return/progress/crash means (window=20) per car |
| `EpisodeConfig` | resource defaults | episode system | All reward and timing parameters |

```text
Physics mutates car → Progress recomputed → Episode records reward and checks terminal
→ If terminal: finalise, push moving avg, reset car to new random spawn, re-sync progress
→ If not: store previous_s for next tick's distance calculation
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `EpisodeConfig`, `TrainerConfig`
- **Runtime components (per car):** `Car`, `EnvInstanceId`, `SpawnConfig`, `CarColour`, `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, `TrackProgress`, `Collided` (marker)
- **Runtime components (track):** `Track`
- **Tests:** Deterministic replay test for `step_car_dynamics()` in `src/game/physics.rs`

## Known Issues / Active Risks

- **Singleton-car assumptions have been removed** from the game layer. Collision, episode, physics, and progress systems all iterate all cars. `EpisodeState`, `EpisodeMovingAverages`, and `ActionState` are per-car Components.
- **Environment regression coverage is thin:** no ECS-level tests for collision timing, reset correctness, or episode-summary edge cases.
- The track layer assumes a **single closed loop** with no branching.
- The environment runs in a **normal Bevy windowed loop only** — no headless or accelerated training mode exists.
- `wrap_angle` and `signed_angle_between` have been consolidated into `src/sim/mod.rs` as shared geometry utilities.

## Partial / In Progress

- Reward shaping has been revised to a velocity-projection + centreline-proximity model, but that behavioural effect depends on analytics validation rather than formal tests.
- Progress and reward state now feed more downstream systems than originally intended: PPO, HUD, analytics, and turn-diagnosis metrics all depend on `EpisodeState`.

## Planned / Missing / Likely Changes

- **Headless training mode** is a likely requirement once longer experiments matter.
- **Multi-track support** would force sharper contracts around spawn, progress semantics, and run metadata.
- **ECS-level regression tests** are the most obvious missing verification layer.

## Durable Notes / Discarded Approaches

- Reset ownership is centralised in `episode.rs`. Earlier split reset behaviour between collision and episode handling was a bad fit because it obscured terminal reward truth.
- `TrackProgress` is shared widely but remains **environment truth, not observation truth**. Preserving that separation is important as the learning stack evolves.
- The `CAR_WIDTH` × `CAR_HEIGHT` (12×6) collision rectangle is intentionally tight — it detects edge-level road departure rather than centre-only checking.
- **Centreline proximity reward was re-added** after the velocity-projection reward model replaced the old speed-weighted-progress scheme. The proximity term now works alongside velocity projection without the earlier farming problem because the time penalty and velocity projection together ensure stationary cars earn negative reward.
- **Episode-end progress bonus was removed** because it added critic instability without meaningfully improving learning.
- **Crash penalty was reduced to 0.0** — crashes end the episode (which is already a significant implicit penalty through lost future reward) but carry no explicit negative reward. This avoids discouraging the exploratory driving that produces learning.
- **Lap detection and finish line were removed.** Progress is now cumulative forward arc-length from spawn, removing the gaming incentive for cars spawning near the wrap point. The lap bonus (+100.0) no longer exists.
- **rotation_speed was increased from 4.0 to 8.0** to address the turning bottleneck at high speeds (previous maximum heading change of 3.8°/tick at 60 Hz was insufficient for tight corners).

## Obsolete / No Longer Relevant

- Any architecture note that treats the repository as "just the environment layer" is obsolete; the environment is still central, but no longer the only substantial runtime subsystem.
- Any reference to lap detection, lap completion, finish line crossing, or lap bonus is obsolete — these concepts have been fully removed.
- Any reference to car 0 having a fixed/canonical spawn position is obsolete — all cars now spawn randomly.
- Any reference to speed-weighted progress reward (`delta × (speed/200) × 100`) is obsolete — replaced by velocity projection reward.
