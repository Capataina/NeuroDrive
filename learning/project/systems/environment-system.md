# The Environment System

## What This File Covers

The environment system is the foundation of NeuroDrive. It owns everything that constitutes the world the car inhabits: the track, the physics, the collision detection, the progress measurement, the reward computation, and the episode lifecycle. This file explains each of these components, how they work, why they are designed the way they are, and how they interact with the rest of the runtime.

**Status:** Current implementation.

## Prerequisites

- `concepts/foundations/bevy-ecs-primer.md` — ECS, components, resources, events, fixed timestep
- `concepts/core/reinforcement-learning.md` — episodes, rewards, terminal states
- `project/architecture/fixed-tick-pipeline.md` — SimSet ordering contract
- `project/architecture/module-boundaries.md` — why game/ owns reward truth

---

## The Two Halves: maps/ and game/

The environment is split across two modules with different roles:

| Module | What it owns |
|---|---|
| `maps/` | Static track topology, tile semantics, centreline, spawn pose, grid queries |
| `game/` | Runtime car state, physics, collision, progress, reward, episode lifecycle |

This split is intentional. The track is a static structure created at startup that never changes at runtime. The car state, progress, rewards, and episode boundaries evolve every tick. Separating these two concerns keeps the static spatial world isolated from the dynamic learning loop.

---

## The Track: maps/

### What the Track Contains

The track is built from tile parts in `src/maps/monaco.rs`. The layout is a Sepang-inspired closed circuit — a hard-coded single loop with a variety of turn radii and straight sections.

Three data structures are derived from the tile layout at startup and attached to a `Track` entity:

**TrackGrid**

A discrete 2D grid where each cell is either driveable or not. This is the authoritative source for:
- collision detection (is any car corner off-road?)
- raycasts from the observation system (how far can the car see in each direction?)

The grid resolves per-tile occupancy into a fast lookup table that is cheaper than per-tile polygon intersection at runtime.

**TrackCenterline**

A closed polyline tracing the centre of the driveable path. Derived from the tile connectivity graph. Used for:
- progress measurement (project car position onto centreline, compute arc length)
- lookahead sampling (trace centreline ahead of the car to get curvature and heading features)
- spawn pose (the car starts at the centreline origin, oriented tangentially)
- debug overlays (geometry visualisation of the centreline)

**Spawn Pose**

The initial position and orientation for the car. On every episode reset, the car returns exactly to this pose. This determinism is intentional — starting from the same position every episode means the difficulty is constant and episode-to-episode performance changes reflect learning, not start-position variation.

### Why Hard-Coded?

The current track is fully hard-coded. This is an explicit early-milestone choice:
- a hard-coded track removes variability from early learning experiments,
- it makes environment bugs easier to diagnose because the world is always identical,
- generalisation to multiple tracks is Milestone 6 work — adding it now would be premature.

---

## Car Physics: game/car.rs and game/physics.rs

### The Car Entity

A single `Car` entity is spawned at startup by `GamePlugin`. The car entity has:
- `Transform` (position and rotation)
- `Car` component (velocity, kinematic parameters: steering sensitivity, acceleration, drag, max speed)
- `TrackProgress` (current centreline projection state)
- `SensorReadings` and `ObservationVector` (owned by the agent layer but attached to the car entity)

### The Physics Step

Every fixed tick, `car_physics_system` runs in `SimSet::Physics`. It reads `ActionState.applied` and calls the pure helper:

```rust
pub fn step_car_dynamics(car: &Car, dt: f32, action: &CarAction) -> (Vec2, f32, f32)
```

The return value is `(new_position_offset, new_heading, new_speed)`. The physics model is kinematic (not physically-based rigid-body dynamics). Key properties:

- **Steering:** heading changes proportionally to steering input, scaled by current speed (more speed = more turning effect up to a cap)
- **Throttle:** speed updates with acceleration proportional to throttle and drag
- **Maximum speed:** clamped at `max_speed`
- **Determinism:** `step_car_dynamics` is a pure function — no ECS queries, no global state, no randomness. This means it can be tested in isolation and the physics is exactly reproducible given the same inputs.

A dedicated unit test exercises this: it feeds a known action sequence into `step_car_dynamics()` and verifies the resulting trajectory matches a stored golden output. This is the deepest determinism guarantee in the codebase.

### Why a Kinematic Model?

A full physics engine (friction, inertia, slip angles, suspension) would make the car harder to control and harder for a learning algorithm to master. The kinematic model:
- produces plausible-looking driving behaviour,
- is learnable with a relatively small observation vector,
- avoids numeric instabilities from rigid-body physics,
- eliminates engine and physics library dependencies.

---

## Collision Detection: game/collision.rs

### How Collision Works

`collision_detection_system` runs in `SimSet::Collision`, after physics has updated the car's position. It:

1. Takes the car's current `Transform` (position + rotation)
2. Computes the four corners of the rotated car rectangle
3. For each corner, queries `TrackGrid` to check whether that cell is driveable
4. If any corner is off-road: emits `CollisionEvent`

`CollisionEvent` is a **zero-payload message**. It carries no per-car identity, no crash location, no severity. This is sufficient for the current single-car runtime because the only thing the episode system needs to know is: *did the car crash this tick?*

### Why the Episode System Reads the Event

`CollisionEvent` is consumed by `episode_loop_system` in `SimSet::Measurement`. The episode system is the single point of truth about whether this tick ends the episode. It reads the collision event, combines it with timeout and lap detection, and produces the authoritative terminal flag.

**This separation matters:** collision detection tells you what the physical world observed. Episode logic decides what that means for the learning loop (episode over, apply crash penalty, reset car). These are different responsibilities, separated into different systems.

### Ordering Requirement

Collision must run **after** physics (to check the new car position) and **before** measurement (so the episode system can read the collision truth). This is enforced by `SimSet::Collision` sitting between `SimSet::Physics` and `SimSet::Measurement` in the chain.

---

## Progress Measurement: game/progress.rs

### What TrackProgress Contains

After every physics step, `update_track_progress_system` runs first in `SimSet::Measurement` and updates the `TrackProgress` component:

| Field | What it stores |
|---|---|
| `s` | Arc-length distance along the centreline from origin |
| `fraction` | `s / total_centreline_length` — lap completion fraction in [0, 1) |
| `closest_point` | The nearest centreline point to the car's current position |
| `tangent` | Centreline tangent direction at the closest point |
| `distance` | Signed perpendicular distance from centreline (positive = left) |

### Why This Matters for Learning

`TrackProgress.fraction` is the basis for lap detection and for the progress-gain reward term. Without accurate progress measurement, the agent would have no way to know how far around the track it has driven.

`TrackProgress` is also used by the observation system to compute centreline-relative features (heading error, lateral offset), though `TrackProgress` itself is intentionally **excluded** from the observation vector. The agent sees geometry-derived features but not raw progress state — this prevents the policy from learning a position-lookup strategy rather than a general driving strategy.

### Why Progress Must Run First in Measurement

`episode_loop_system` uses `progress.fraction` to:
- detect lap completion (did the fraction wrap around?)
- compute the progress-gain reward term (how much did the fraction increase this tick?)

If `episode_loop_system` ran first, it would compute rewards and terminals from the stale previous-tick progress. The ordering `progress → episode → observation_rebuild` ensures that rewards always reflect the post-physics world state.

---

## The Episode Lifecycle: game/episode.rs

### EpisodeState

`EpisodeState` is a global singleton resource that carries the complete state of the current and most-recently-completed episode:

| Field group | What it stores |
|---|---|
| Current tick | `current_tick_reward`, `current_tick_end_reason` |
| Current episode | `steps_in_episode`, `episode_total_return`, `episode_best_progress` |
| Last episode | `last_episode_return`, `last_episode_steps`, `last_episode_end_reason`, `last_episode_best_progress` |
| Summary | `total_episodes`, `total_steps` |

### EpisodeConfig

`EpisodeConfig` stores the tunable parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `max_steps` | 3600 | Episode timeout (60 ticks/s × 60s = 60-second max) |
| `progress_scale` | 140.0 | Multiplier applied to progress-gain reward |
| `time_penalty` | -0.005 | Per-tick penalty to discourage stalling |
| `heading_speed_penalty` | 0.02 | Weight for heading-speed penalty term |
| `crash_penalty` | -5.0 | One-off penalty on collision |
| `lap_bonus` | +100.0 | One-off bonus for completing a full lap |
| `moving_average_window` | 50 | Episode count for rolling performance means |

### Per-Tick Reward Computation

Every tick, `episode_loop_system` computes the reward from several components:

**1. Progress gain reward**

```
progress_gain = fraction_this_tick - best_progress_so_far_this_episode
if progress_gain > 0:
    progress_reward = progress_gain * progress_scale
```

Key design choice: rewards only **new** progress. The agent must continue pushing forward rather than repeatedly covering familiar ground. This is a best-progress frontier rather than a simple cumulative distance reward.

**2. Time penalty**

```
time_reward = time_penalty  (always -0.005)
```

Applied every tick regardless of other rewards. This discourages the agent from surviving by staying still. It creates constant pressure to make progress.

**3. Heading-speed penalty**

```
heading_speed_reward = -heading_speed_penalty * |heading_error| * speed
```

Penalises moving fast while pointed incorrectly. This reduces the incentive for the agent to sprint toward the wall at full throttle. At low speed or zero heading error, this term is small; it grows as the agent drives dangerously.

**4. Crash penalty**

```
if CollisionEvent this tick:
    crash_reward = crash_penalty  (-5.0)
```

One-off. Does not repeat for consecutive collision ticks.

**5. Lap bonus**

```
if progress.fraction wraps around (lap detected):
    lap_reward = lap_bonus  (+100.0)
```

Only triggered by fraction wrap-around, not by crossing a physical finish line.

**Total per-tick reward:**

```
r_t = progress_reward + time_reward + heading_speed_reward + crash_reward + lap_reward
```

This reward structure is an example of **shaped reward design** — multiple terms each contributing a learning signal for a different behavioural objective. The `concepts/domain-patterns/reward-shaping.md` file covers the theory behind this design in detail.

### Terminal Conditions

An episode ends when any of the following is true:

| Condition | Trigger |
|---|---|
| **Crash** | `CollisionEvent` was emitted this tick |
| **Timeout** | `steps_in_episode >= max_steps` |
| **Lap complete** | `progress.fraction` wrap-around detected |

All three conditions are checked inside `episode_loop_system`. When a terminal condition fires:
1. The episode summary is finalised into `EpisodeState`
2. The car is reset to spawn pose
3. Progress is resynchronised to the new spawn position
4. `EpisodeMovingAverages` is updated

### Episode Reset

After terminal:

```rust
// Reset car transform
transform.translation = spawn_pose.position;
transform.rotation = spawn_pose.rotation;
car.velocity = 0.0;

// Resync progress
progress.update(spawn_pose.position, &track_centerline);
```

The car reset happens **inside** `episode_loop_system`, before `update_sensor_readings_system` and `build_observation_vector_system` run. This is the critical ordering requirement: observations must be built from the post-reset state, not the crash state. If the observation rebuild ran before the reset, the A2C brain would receive a stale crash observation as the starting state for the new episode.

### Moving Averages

`EpisodeMovingAverages` maintains a sliding-window mean over the last `moving_average_window` (default: 50) episodes for:
- mean episode return
- mean best-progress fraction
- crash rate

These moving averages are used by the debug HUD for the recent-quarter assessment display. They are not used in the learning loop itself.

---

## What the Environment Guarantees

The environment system provides several guarantees to the rest of the runtime:

1. **Physics determinism:** Given the same action sequence from the same spawn state, the car follows an identical trajectory. This is tested.

2. **Single source of reward truth:** `EpisodeState.current_tick_reward` is computed exactly once per tick by `episode_loop_system` and never modified outside that system.

3. **Single source of terminal truth:** `EpisodeState.current_tick_end_reason` is set by `episode_loop_system` and consulted by the A2C reward collector. There is no other place that can declare a tick terminal.

4. **Controller-agnostic behaviour:** The same reward, collision, and progress logic runs whether the current controller is keyboard or A2C. The environment does not import from `brain/`.

---

## Known Limitations

| Limitation | Consequence |
|---|---|
| Singleton car (`single()` queries) | Cannot run multiple cars simultaneously |
| No headless mode | Training requires a window; cannot run faster than real-time |
| Progress via fraction wrap | Lap detection relies on crossing the 0/1 boundary; unusual positions near origin could create false positives |
| No ECS-level regression tests | Collision timing, reset correctness, and lap detection edge cases are untested at the system level |
| Single hard-coded track | Generalisation experiments require multi-track infrastructure (Milestone 6) |

---

## Related Files

- `project/architecture/fixed-tick-pipeline.md` — SimSet ordering with correctness requirements
- `project/architecture/module-boundaries.md` — why game/ owns reward truth
- `concepts/domain-patterns/reward-shaping.md` — theory of the reward decomposition
- `project/systems/agent-interface.md` — how observations are built from environment state
- `project/evolution/milestone-roadmap.md` — multi-track and headless plans
