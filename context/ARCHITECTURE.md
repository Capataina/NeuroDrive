# NeuroDrive Architecture

## Repository Overview

NeuroDrive is a Rust/Bevy project for a deterministic top-down racing environment that is being extended toward autonomous learning.
The repository now contains four main runtime layers: environment simulation, agent I/O, brain implementations, and analytics/debug observability.

## Directory Structure

```text
NeuroDrive/
|-- src/
|   |-- main.rs
|   |-- agent/
|   |   |-- mod.rs
|   |   |-- action.rs
|   |   |-- observation.rs
|   |   `-- plugin.rs
|   |-- analytics/
|   |   |-- mod.rs
|   |   |-- plugin.rs
|   |   |-- exporters/
|   |   |   |-- mod.rs
|   |   |   |-- json.rs
|   |   |   `-- markdown.rs
|   |   |-- metrics/
|   |   |   |-- mod.rs
|   |   |   `-- chunking.rs
|   |   `-- trackers/
|   |       |-- mod.rs
|   |       `-- episode.rs
|   |-- brain/
|   |   |-- mod.rs
|   |   |-- plugin.rs
|   |   |-- types.rs
|   |   |-- a2c/
|   |   |   |-- mod.rs
|   |   |   |-- model.rs
|   |   |   |-- buffer.rs
|   |   |   `-- update.rs
|   |   `-- common/
|   |       |-- mod.rs
|   |       |-- math.rs
|   |       |-- mlp.rs
|   |       `-- optim.rs
|   |-- debug/
|   |   |-- mod.rs
|   |   |-- hud.rs
|   |   |-- overlays.rs
|   |   `-- plugin.rs
|   |-- game/
|   |   |-- mod.rs
|   |   |-- car.rs
|   |   |-- collision.rs
|   |   |-- episode.rs
|   |   |-- physics.rs
|   |   |-- progress.rs
|   |   `-- plugin.rs
|   |-- maps/
|   |   |-- mod.rs
|   |   |-- centerline.rs
|   |   |-- grid.rs
|   |   |-- monaco.rs
|   |   |-- track.rs
|   |   `-- parts/
|   |       `-- mod.rs
|   `-- sim/
|       |-- mod.rs
|       `-- sets.rs
|-- context/
|   |-- ARCHITECTURE.md
|   |-- SYSTEM_A2C_BASELINE.md
|   |-- SYSTEM_ANALYTICS.md
|   |-- SYSTEM_DEBUG_OVERLAYS.md
|   |-- SYSTEM_DETERMINISM_REPLAY.md
|   |-- SYSTEM_GAME_ENVIRONMENT.md
|   |-- SYSTEM_SENSORS_OBSERVATIONS.md
|   |-- SYSTEM_TELEMETRY.md
|   |-- SYSTEM_TRACK_PROGRESS_AND_LAPS.md
|   |-- IMPLEMENT_NOW_ANALYTICS.md
|   `-- IMPLEMENT_NOW_A2C.md
|-- Cargo.toml
|-- Cargo.lock
|-- README.md
`-- agents.md
```

## Current Implementation Status

### Implemented and Compiling Before the Current New Subsystems

- The deterministic Bevy environment stack is implemented: fixed 60 Hz simulation, explicit system ordering, a tile-grid track, deterministic car physics, centreline projection, collision/reset handling, episode lifecycle, sensor raycasts, a normalised observation vector, and debug overlays.
- The repository includes a single Sepang-inspired closed-loop track with grid-derived rendering and a centreline used for progress and lap logic.
- The debug layer includes geometry overlays, sensor overlays, and an on-screen driving HUD.
- The simulation still supports manual validation through a keyboard controller and a stable `CarAction` interface.
- A deterministic replay unit test exists for the pure car dynamics stepper only.

### Newly Added but Still Partial

- A `brain/` subsystem now exists with an `AgentMode` switch, an A2C plugin, a handwritten actor-critic model, a rollout buffer, and an update path.
- An `analytics/` subsystem now exists with an episode tracker, chunked metrics, and JSON/Markdown exporters intended to write run reports under `reports/`.
- The A2C code is integrated into the fixed-tick schedule and is no longer just a roadmap item; it is partial runtime code.

### Current Repository Reality

- The project currently passes `cargo check` and `cargo test`.
- The analytics exit path is now aligned with the Bevy 0.18 message API and exports reports on shutdown from the `Last` schedule.
- The A2C implementation is present but still incomplete as a validated baseline: there is no persistence, no policy evaluation mode, no headless training mode, and no deterministic seeding strategy.

## Subsystem Responsibilities

### `src/main.rs`

- Creates the Bevy app, configures the window, sets the fixed timestep, and wires plugins in runtime order.
- Runtime plugin order is: maps, agent, brain, analytics, game, debug.

### `src/maps/`

- Owns track topology, tile semantics, rendering geometry, spawn lookup, and centreline construction.
- `grid.rs` defines `TrackGrid` spatial queries and track rendering.
- `centerline.rs` derives a closed-loop polyline and projection model from tile connectivity.
- `monaco.rs` builds and spawns the current Sepang-inspired track.
- `track.rs` defines the `Track` component consumed by gameplay and measurements.

### `src/game/`

- Owns the runtime environment state and episode lifecycle.
- `car.rs` defines the car entity and spawns its measurement components.
- `physics.rs` is the only runtime location where actions mutate motion state.
- `collision.rs` detects off-track occupancy and emits/resolves collision messages.
- `progress.rs` projects the car onto the centreline every fixed tick.
- `episode.rs` converts progress/collisions/timeouts into reward accumulation, lap detection, and episode boundaries.

### `src/agent/`

- Owns the stable controller-facing boundary.
- `action.rs` defines `CarAction`, desired/applied action state, optional smoothing, and keyboard input.
- `observation.rs` owns raw sensor readings plus the fixed-size normalised observation vector.
- `plugin.rs` schedules action input and observation building into the fixed simulation pipeline.

### `src/brain/`

- Owns controller selection and learning implementations.
- `types.rs` defines `AgentMode` and the generic `Brain` trait.
- `plugin.rs` initialises the active mode and toggles keyboard vs AI control on `F4`.
- `a2c/` contains the current baseline learning implementation attempt.
- `common/` contains handwritten neural-network and optimiser primitives used by A2C.

### `src/analytics/`

- Owns run-level aggregation and export of episode results.
- `trackers/episode.rs` stores one record per completed episode.
- `metrics/chunking.rs` aggregates episode records into chunked summary metrics.
- `exporters/` serialises the tracked data to JSON and Markdown on app exit.
- `plugin.rs` initialises the tracker and attempts to trigger export when the app exits.

### `src/debug/`

- Owns developer-facing visual inspection tools.
- `overlays.rs` toggles and draws world-space geometry and sensor overlays.
- `hud.rs` maintains HUD-specific derived stats and renders the driving state panel.

### `src/sim/`

- Owns the fixed pipeline ordering contract used across agent, brain, and game systems.

## Dependency Direction

- `main` depends on all runtime plugins and defines startup order.
- `maps` is a base subsystem and does not depend on `game`, `agent`, `brain`, `analytics`, or `debug`.
- `game` depends on `maps` and `sim`.
- `agent` depends on `game`, `maps`, `brain` types, and `sim`.
- `brain` depends on `agent`, `game`, and `sim`.
- `analytics` currently depends on `game`.
- `debug` depends on `agent`, `game`, `maps`, and `sim`.

## Core Execution Flow

1. `main.rs` configures Bevy, the window, and fixed-time simulation.
2. `MonacoPlugin` runs at startup and spawns the track plus its visuals.
3. `GamePlugin` runs at `PostStartup` and spawns the camera and the car entity.
4. Each fixed tick runs the ordered simulation pipeline:
   - `SimSet::Input`: keyboard control and A2C action selection write desired actions.
   - `SimSet::Physics`: car dynamics consume the applied action.
   - `SimSet::Collision`: collision detection and reset handling run.
   - `SimSet::Measurement`: progress, episode accounting, and observation rebuilding run.
5. Each frame in `Update`:
   - brain mode toggling runs,
   - analytics tracking records finished episodes,
   - debug overlays/HUD update,
   - analytics export is intended to run on app exit.

## Notes on Structural Drift

- The architecture has moved beyond a pure Milestone 0 environment: the repository now includes live A2C and analytics modules.
- The context folder must now treat analytics and A2C as real subsystems, not roadmap-only items.
- The current highest-priority structural issue is not layout but integration correctness: the new analytics subsystem breaks the build and the A2C subsystem is integrated before its verification layer exists.
