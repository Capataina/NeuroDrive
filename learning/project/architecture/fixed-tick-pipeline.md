# Fixed-Tick Pipeline

## Why This Matters

The order in which NeuroDrive's systems run within each 60 Hz tick is not arbitrary. Multiple correctness requirements constrain the ordering, and several bugs in earlier development arose from systems running in the wrong sequence. Understanding the pipeline is essential for contributing to any of the simulation, learning, or analytics subsystems.

**Status:** Current implementation.

## Prerequisites

- `concepts/foundations/bevy-ecs-primer.md` — schedules, SimSet, system ordering

---

## The SimSet Contract

NeuroDrive defines four ordered system sets within `FixedUpdate`:

```rust
pub enum SimSet {
    Input,
    Physics,
    Collision,
    Measurement,
}
```

These are chained:

```
Input → Physics → Collision → Measurement
```

Every system that runs in the fixed tick belongs to exactly one of these sets.

---

## What Runs in Each Set

### SimSet::Input

**Purpose:** Determine what control action the car will take this tick.

| System | What it does |
|---|---|
| `keyboard_action_input_system` | Reads keyboard and writes desired steering/throttle (keyboard mode only) |
| `a2c_act_system` | Runs the A2C forward pass, samples action, writes desired action (AI mode only) |
| `action_smoothing_system` | Optionally smooths desired → applied action (currently disabled) |

**Ordering constraints:**
- `a2c_act_system` must run after `keyboard_action_input_system` (both write to `ActionState.desired`, mode gate prevents conflict, but registration order matters)
- `action_smoothing_system` must run last in this set

**Result:** `ActionState.applied` is set for this tick.

---

### SimSet::Physics

**Purpose:** Advance the physical state of the car given the applied action.

| System | What it does |
|---|---|
| `car_physics_system` | Applies steering and throttle to the car's velocity and position via `step_car_dynamics()` |
| `capture_episode_action_stats_system` | Records the applied action in the analytics accumulator |

**Ordering constraints:**
- Physics must run after Input (the car must have an applied action before we step)
- Physics must run before Collision (collision detection checks the new car position)
- Action stats capture runs in Physics set because it reads `ActionState.applied` after smoothing

**Result:** `Transform` (position, rotation) and `Car.velocity` are updated for the new tick state.

---

### SimSet::Collision

**Purpose:** Detect whether the car has left the driveable area.

| System | What it does |
|---|---|
| `collision_detection_system` | Checks car corner positions against `TrackGrid`, emits `CollisionEvent` if off-track |

**Ordering constraints:**
- Must run after Physics (needs the new car position)
- Must run before Measurement (reward computation needs the collision truth)

**Result:** `CollisionEvent` is emitted (or not) for this tick.

---

### SimSet::Measurement

**Purpose:** Compute all derived quantities from the new physical state. This is the most populated set.

The ordering *within* Measurement also matters:

```
update_track_progress_system
    ↓
episode_loop_system
    ↓
update_sensor_readings_system
build_observation_vector_system
    ↓
capture_episode_tick_trace_system
snapshot_completed_episode_trace_system
snapshot_completed_episode_action_stats_system
    ↓
a2c_collect_reward_system
    ↓
update_driving_hud_stats_system
capture_driving_hud_episode_metrics_system
```

#### update_track_progress_system

Updates `TrackProgress` (s, fraction, closest point, tangent, distance) based on the car's new position. Must run before episode logic because the episode system uses `progress.fraction` for lap detection and reward computation.

#### episode_loop_system

The central episode system. It:
- Computes the per-tick reward (progress gain, time penalty, heading-speed penalty)
- Detects terminal conditions (crash via `CollisionEvent`, timeout, lap completion)
- On terminal: finalises episode summary, resets the car to spawn, resyncs progress
- Updates `EpisodeState` with all current-tick and last-episode fields

**Why this must run before observation rebuild:** If the episode ends this tick and the car resets to spawn, the new observation must reflect the post-reset state, not the crash state. If observation rebuild ran before reset, the A2C would see the stale crash observation as the starting observation for the new episode.

#### update_sensor_readings_system / build_observation_vector_system

Rebuild the `SensorReadings` (raycasts, geometry) and `ObservationVector` (normalised 23-dim input) based on the current car position. These run after the episode system so they capture the post-reset state when applicable.

#### capture_episode_tick_trace_system

Appends one `TickTraceRecord` to the analytics accumulator. This runs after observation rebuild so it captures the new observation alongside this tick's reward.

#### a2c_collect_reward_system

Appends `reward_t` and `done_t` to the rollout buffer. Must run:
- **After** `episode_loop_system` — needs `EpisodeState.current_tick_reward` and `current_tick_end_reason`
- **After** `build_observation_vector_system` — needs the new observation for bootstrap when done = true

This placement is the most ordering-sensitive in the pipeline. Earlier versions had subtle bugs here.

#### HUD systems

Read from `EpisodeState`, `EpisodeMovingAverages`, `A2cTrainingStats`. Run last in Measurement so they see the fully updated state.

---

## The Non-Fixed (Update) Systems

Some systems run in Bevy's uncapped `Update` schedule rather than `FixedUpdate`:

| System | Why Update and not FixedUpdate |
|---|---|
| `toggle_agent_mode_system` | Responds to keyboard input (F4); does not need deterministic 60Hz cadence |
| `episode_tracker_system` | Folds completed-episode snapshots into `EpisodeTracker`; runs after snapshot capture |
| Debug overlay rendering | World-space gizmos; frame-rate dependent visual only |
| HUD text update | Bevy UI; frame-rate dependent |

These systems read from resources and components updated in `FixedUpdate`. They are not part of the deterministic learning loop.

---

## The Last Schedule

| System | Purpose |
|---|---|
| `a2c_flush_on_exit_system` | If the app is closing with a partial rollout, run one final A2C update |
| `analytics export` | Write JSON and Markdown reports |

These run in `Last`, after all other systems have completed for the final frame.

---

## Why Getting This Wrong Breaks Things

A table of ordering requirements and what breaks if violated:

| Requirement | If violated |
|---|---|
| Physics before Collision | Collision checks the old position; car can pass through walls |
| Progress before Episode | Lap completion detection uses wrong progress value |
| Episode before Observation rebuild | Post-reset observations reflect crash state |
| Episode before A2C reward collection | `done` flag is wrong; bootstrap logic is inverted |
| Observation rebuild before A2C reward | A2C collects stale observation for bootstrapping |
| A2C act before smoothing | Desired action bypasses smoothing |

---

## Related Files

- `concepts/foundations/bevy-ecs-primer.md` — how Bevy schedules work
- `project/architecture/runtime-overview.md` — full subsystem map
- `project/systems/a2c-brain.md` — the A2C system placements
- `project/systems/environment-system.md` — the episode and physics systems
