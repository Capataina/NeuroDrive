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

- A single car entity spawned by `setup_game` in `GamePlugin` after the track is ready.
- The car entity carries:
  - `Sprite` (red, 12×6 world units),
  - `Transform`,
  - `Car` component (velocity, rotation_speed=4.0, thrust=750.0, drag=0.985),
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
- Checks the four rotated corners of the car's bounding rectangle against `track.grid.is_road_at()`.
- Emits a zero-payload `CollisionEvent` message when any corner leaves the road.
- Currently uses `single()` query — assumes one car.

### Progress

- `update_track_progress_system` runs first in `SimSet::Measurement`.
- Projects the car's position onto the centreline, producing `TrackProgress` fields: `s`, `fraction`, `closest_point`, `tangent`, `distance`.
- Uses `for` loops over car queries — already supports multi-car iteration.

### Episode Loop

- `episode_loop_system` runs after progress update in `SimSet::Measurement`.
- **Currently uses `single_mut()` for the car query** — assumes one car.
- Per-tick processing:
  1. Computes progress-gain reward: `(fraction - previous_best) * progress_reward_scale` (140.0).
  2. Computes heading-speed penalty: `-heading_speed_penalty_scale * |heading_error|/π * speed/speed_norm_max`.
  3. Adds fixed time penalty: -0.005 per tick.
  4. Checks terminal conditions in priority: crash → lap complete → timeout.
  5. Applies terminal rewards: crash penalty (-5.0) or lap bonus (100.0).
  6. On terminal: finalises episode state, pushes to moving averages, resets car to spawn, re-syncs progress.

### Reward Composition

| Term | Value | When |
|------|-------|------|
| Progress gain | `+gain × 140.0` | Every tick where episode-best progress increases |
| Time penalty | `-0.005` | Every tick |
| Heading-speed penalty | `-0.02 × |heading_error|/π × speed/900` | Every tick |
| Crash penalty | `-5.0` | One-off on crash |
| Lap bonus | `+100.0` | One-off on lap completion |

### Episode Termination

| Condition | Detection | Reset |
|-----------|-----------|-------|
| **Crash** | `CollisionEvent` received | Immediate |
| **Lap complete** | Lap armed (≥25% progress) AND previous fraction ≥0.85 AND current ≤0.15 | Immediate |
| **Timeout** | Ticks × dt ≥ 30 seconds | Immediate |

- Lap detection is **progress-wrap based**, not finish-line-crossing based.
- Reset always returns the car to the canonical track spawn pose and re-syncs `TrackProgress`.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer(s) | Notes |
|-----------|----------|-------------|-------|
| `Track` | `maps` startup | physics queries, observations, debug | Single authoritative track entity |
| `CollisionEvent` | collision system | episode system, HUD stats | Emitted before reward finalisation |
| `TrackProgress` | progress system | episode logic, debug, analytics, observations | Environment measurement, not policy input |
| `EpisodeState` | episode system | A2C reward collection, HUD, analytics | Current-tick and last-episode summaries |
| `EpisodeMovingAverages` | episode system | HUD | Rolling return/progress/crash means (window=20) |
| `EpisodeConfig` | resource defaults | episode system | All reward and timing parameters |

```text
Physics mutates car → Progress recomputed → Episode records reward and checks terminal
→ If terminal: finalise, push moving avg, reset car, re-sync progress
→ If not: store previous progress fraction for next tick
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `EpisodeConfig`, `EpisodeState`, `EpisodeMovingAverages`
- **Runtime messages:** `CollisionEvent`
- **Runtime components:** `Track`, `Car`, `TrackProgress`
- **Tests:** Deterministic replay test for `step_car_dynamics()` in `src/game/physics.rs`

## Known Issues / Active Risks

- **Singleton-car assumptions** are spread across collision (`single()`), episode logic (`single_mut()`), and implicitly in `EpisodeState` being a global resource. This is the main structural blocker for multi-car work.
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

## Durable Notes / Discarded Approaches

- Reset ownership is centralised in `episode.rs`. Earlier split reset behaviour between collision and episode handling was a bad fit because it obscured terminal reward truth.
- `TrackProgress` is shared widely but remains **environment truth, not observation truth**. Preserving that separation is important as the learning stack evolves.
- The `CAR_WIDTH` × `CAR_HEIGHT` (12×6) collision rectangle is intentionally tight — it detects edge-level road departure rather than centre-only checking.

## Obsolete / No Longer Relevant

- Any architecture note that treats the repository as "just the environment layer" is obsolete; the environment is still central, but no longer the only substantial runtime subsystem.
