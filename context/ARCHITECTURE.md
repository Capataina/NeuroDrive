# NeuroDrive Architecture

## Repository Overview

NeuroDrive is a brain-inspired AI research project built around a 2D top-down racing environment.
The codebase is written in Rust using Bevy for simulation and rendering.

## Directory Structure

```
NeuroDrive/
|-- src/
|   |-- main.rs                  # Application entry point
|   |-- agent/
|   |   |-- mod.rs               # Agent interface module exports
|   |   |-- action.rs            # Stable action interface + keyboard adapter
|   |   |-- observation.rs       # Raycast sensors + normalised observation vector
|   |   `-- plugin.rs            # AgentPlugin wiring (FixedUpdate)
|   |-- debug/
|   |   |-- mod.rs               # Debug module exports
|   |   |-- hud.rs               # F3 driving-state HUD (UI + stats)
|   |   |-- overlays.rs          # Gizmo-based overlay rendering + toggles
|   |   `-- plugin.rs            # DebugPlugin wiring
|   |-- sim/
|   |   |-- mod.rs               # Simulation scheduling exports
|   |   `-- sets.rs              # Fixed-tick pipeline ordering (SimSet)
|   |-- game/
|   |   |-- mod.rs               # Game module exports
|   |   |-- car.rs               # Car component and spawn helper
|   |   |-- physics.rs           # Fixed-tick vehicle dynamics (consumes actions)
|   |   |-- collision.rs         # Off-track detection and reset logic
|   |   |-- episode.rs           # Episode loop + reward + rolling telemetry
|   |   |-- progress.rs          # Centreline projection + progress component
|   |   `-- plugin.rs            # GamePlugin bundling game systems
|   `-- maps/
|       |-- mod.rs               # Maps module exports
|       |-- grid.rs              # TrackGrid definition and tile rendering
|       |-- centerline.rs         # Grid-derived centreline polyline + projection
|       |-- track.rs             # Track component
|       |-- parts/
|       |   `-- mod.rs           # TilePart enum and connectivity rules
|       `-- monaco.rs            # Sepang-inspired circuit definition
|-- context/
|   |-- ARCHITECTURE.md          # This file
|   |-- SYSTEM_GAME_ENVIRONMENT.md
|   |-- SYSTEM_DETERMINISM_REPLAY.md
|   |-- SYSTEM_SENSORS_OBSERVATIONS.md
|   |-- SYSTEM_TRACK_PROGRESS_AND_LAPS.md
|   |-- SYSTEM_DEBUG_OVERLAYS.md
|   `-- SYSTEM_TELEMETRY.md
|-- Cargo.toml
`-- README.md                    # Project vision (immutable)
```

## Current Implementation Status

### Implemented

- Modular Bevy application structure (track plugin then game plugin).
- Fixed-timestep simulation configured at 60 Hz (`src/main.rs`), with explicit fixed-tick pipeline ordering (`src/sim/sets.rs`).
- Stable action interface (`CarAction`) with a fixed-tick keyboard adapter and optional smoothing (`src/agent/action.rs`).
- 2D car entity with deterministic fixed-tick velocity integration and drag (`src/game/physics.rs`).
- Grid-based tile track system (`TrackGrid` + `TilePart`) with per-tile connectivity (`open_edges()`).
- Tile types: straights (H/V), corners (NW/NE/SW/SE), T-junctions, crossroads, spawn point.
- Sepang-inspired circuit layout on a 14x9 tile grid.
- Tile rendering: filled road surfaces, straight wall bars on closed edges, quarter-circle arc walls on corner tiles.
- Corner road surfaces rendered as clipped quarter-circle meshes to prevent surface leakage past curved walls.
- Off-track detection via `TrackGrid::is_road_at()` with wall-thickness insets.
- Automatic reset on off-track collision (in-place transform reset + velocity reset).
- Start/finish line marker rendering.
- Grid-derived closed centreline polyline and closest-point projection for continuous progress (`src/maps/centerline.rs`, `src/game/progress.rs`).
- Episode loop with crash/timeout/lap-complete end conditions, reward accumulation, and rolling episode averages (`src/game/episode.rs`).
- Geometry debug overlay (F1) for centreline + projection visualisation via Bevy gizmos (`src/debug/overlays.rs`).
- Raycast sensor system with fixed ray layout and road-boundary intersection against `TrackGrid::is_road_at()` (`src/agent/observation.rs`).
- Stable normalised observation vector composed from rays, speed, heading error, and angular velocity (`src/agent/observation.rs`).
- Sensor debug overlay (F2) for ray lines and hit points (`src/debug/overlays.rs`).
- Driving-state HUD (F3) with progress, heading error, death count, best progress + best life, reward, episode crash count, and moving averages (`src/debug/hud.rs`).
- Deterministic replay unit test for the pure car dynamics stepper (`src/game/physics.rs` tests).

### Not Yet Implemented

- Replay harness over full ECS runtime action streams (record/playback mode) is still missing.
- Full learning telemetry UI (e.g. policy/brain-specific diagnostics) is still missing.
- Neural brain system and learning mechanisms.
- Multiple tracks.

## Subsystem Responsibilities

### `src/main.rs`

- Bevy app initialisation.
- Window configuration.
- Plugin registration order (track before game).
- Fixed timestep configuration and global plugin wiring (agent + game + debug).

### `src/agent/action.rs`

- `CarAction` stable action interface (`steering`, `throttle`).
- `ActionState` resource for fixed-tick desired/applied actions.
- `keyboard_action_input_system()` maps keyboard state to `ActionState.desired` each fixed tick.
- `action_smoothing_system()` optionally filters desired actions into applied actions.

### `src/agent/observation.rs`

- `SensorReadings` component stores ray hits/distances and derived kinematics.
- `ObservationVector` component stores the fixed-size normalised observation features.
- `update_sensor_readings_system()` updates raycasts and heading/speed/angular-velocity measurements per fixed tick.
- `build_observation_vector_system()` normalises measured features into a stable observation contract.

### `src/sim/sets.rs`

- Defines `SimSet` to keep fixed-tick ordering explicit:
  - Input â†’ Physics â†’ Collision â†’ Measurement.

### `src/game/car.rs`

- `Car` component (velocity, thrust, drag, rotation_speed).
- `spawn_car()` spawns the car sprite and component.

### `src/game/physics.rs`

- `car_physics_system()` consumes `ActionState` and updates car transform/velocity on the fixed tick.
- `step_car_dynamics()` provides a pure deterministic dynamics step used both by runtime physics and replay testing.

### `src/game/collision.rs`

- `CollisionEvent` message signals that the car has left the driveable road surface.
- `collision_detection_system()` checks quaternion-rotated car sprite corners against `TrackGrid::is_road_at()`.
- `handle_collision_system()` resets the car to the track spawn pose on collision (no despawn/respawn).

### `src/game/progress.rs`

- `TrackProgress` component stores centreline projection and progress fraction.
- `update_track_progress_system()` updates `TrackProgress` each fixed tick from the track centreline.

### `src/game/episode.rs`

- `EpisodeConfig` defines timeout, lap detection thresholds, and reward constants.
- `EpisodeState` tracks current episode lifecycle, reward, per-episode crash count, and end reason.
- `EpisodeMovingAverages` stores rolling means for return/progress/crashes.
- `episode_loop_system()` advances episode state and handles crash/timeout/lap boundaries.

### `src/game/plugin.rs`

- `GamePlugin` bundles game systems.
- Registers collision messages and schedules fixed-tick simulation systems using `SimSet` ordering.
- Spawns camera and car after the track is available (`PostStartup`).

### `src/maps/parts/mod.rs`

- `TilePart` enum defining all tile types and the `Empty` sentinel.
- `open_edges()` returns `(north, south, east, west)` connectivity per tile.
- `is_road()` and `is_corner()` classification helpers.

### `src/maps/grid.rs`

- `TrackGrid` struct: row-major tile storage, origin, tile_size.
- Spatial queries: `cell_center()`, `world_to_cell()`, `is_road_at()`.
- `find_spawn()` locates the `SpawnPoint` tile.
- `find_spawn_cell()` locates the `SpawnPoint` tile cell coordinates.
- `render_tile_grid()` spawns all visual geometry:
  - Road surfaces (flat squares for non-corners, quarter-circle meshes for corners).
  - Straight wall bars on closed edges of non-corner tiles.
  - Quarter-circle arc walls on corner tiles.

### `src/maps/track.rs`

- `Track` component carrying the grid, spawn pose, and the computed centreline.
- Queried by game and collision systems.

### `src/maps/monaco.rs`

- `MonacoPlugin` spawns the Sepang-inspired track on startup.
- `build_tiles()` defines the 14x9 tile grid layout.
- `render_finish_line()` places the start/finish marker.
- Derives spawn position (tile centre) and heading (east) from the `SpawnPoint` tile.

## Execution Flow

1. `MonacoPlugin` runs first (Startup): builds tile grid, renders visuals, spawns the `Track` entity.
2. `GamePlugin` runs next (PostStartup): spawns the 2D camera and the car at the track spawn pose.
3. Each fixed tick (FixedUpdate):
   - `SimSet::Input` (AgentPlugin): latch keyboard actions and optionally smooth them.
   - `SimSet::Physics` (GamePlugin): integrate car dynamics from the stable action interface.
   - `SimSet::Collision` (GamePlugin): detect off-track collisions and reset the car pose.
   - `SimSet::Measurement` (GamePlugin + AgentPlugin): update centreline projection, advance episode/reward logic, then sensors, then the normalised observation vector.
4. Each frame (Update):
   - `DebugPlugin`: handle F1/F2/F3 toggles, render geometry/sensor gizmos, and update driving-state HUD visibility/text.

## Dependencies

- `bevy 0.18.0` — Game engine for ECS, rendering, and input handling.
