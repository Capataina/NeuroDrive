# NeuroDrive Architecture

## Scope / Purpose

- Provide the top-down structural map for the repository as it exists now.
- Orient a new engineer to the runtime layers, ownership boundaries, dependency direction, and the main fixed-tick execution flow.
- Keep detailed subsystem behaviour in `context/systems/` rather than duplicating it here.

## Repository Overview

- NeuroDrive is a Rust application built on Bevy 0.18.
- The current runtime is a deterministic 2D top-down racing environment with a fixed controller boundary, a live A2C baseline brain, analytics export, and a debug HUD/overlay layer.
- The repository intent in `README.md` remains brain-inspired online learning from first principles; the current implementation reality is still at the baseline-validation stage rather than the final biological-learning architecture.
- `cargo check` and `cargo test` both pass in the current workspace state.

## Repository Structure

```text
NeuroDrive/
|-- src/
|   |-- main.rs                     # app entrypoint, plugin order, window config, fixed timestep
|   |-- agent/
|   |   |-- action.rs              # stable car-control boundary and optional smoothing
|   |   |-- observation.rs         # raycasts, centreline-relative features, observation vector
|   |   `-- plugin.rs              # fixed-tick scheduling for action and observation systems
|   |-- analytics/
|   |   |-- models.rs              # canonical run/episode/tick/update analytics schemas
|   |   |-- plugin.rs              # tracker setup and on-exit export orchestration
|   |   |-- trackers/              # fixed-tick accumulation for actions, traces, episodes
|   |   |-- metrics/               # derived diagnostics and trend computation
|   |   |-- exporters/             # JSON and Markdown report writers
|   |   `-- sessions/              # currently empty placeholder directory
|   |-- brain/
|   |   |-- plugin.rs              # agent-mode resource and mode toggle wiring
|   |   |-- types.rs               # Brain trait and keyboard/AI mode switch
|   |   |-- a2c/                   # current handwritten A2C baseline implementation
|   |   |-- common/                # handwritten MLP, math, and optimiser primitives
|   |   `-- biological/            # currently empty placeholder directory
|   |-- debug/
|   |   |-- overlays.rs            # F1/F2/F3 toggles and world-space gizmo rendering
|   |   |-- hud.rs                 # runtime diagnostics panel and recent-quarter summaries
|   |   `-- plugin.rs              # debug resource setup and scheduling
|   |-- game/
|   |   |-- car.rs                 # car entity spawn and core kinematic parameters
|   |   |-- physics.rs             # deterministic car dynamics and pure replay stepper
|   |   |-- collision.rs           # off-track detection and collision message emission
|   |   |-- progress.rs            # centreline projection and progress measurement
|   |   |-- episode.rs             # reward logic, lap logic, resets, moving averages
|   |   `-- plugin.rs              # fixed-tick environment schedule and startup spawn
|   |-- maps/
|   |   |-- monaco.rs              # hard-coded Sepang-inspired track construction
|   |   |-- grid.rs                # driveable-area queries and tile-grid rendering helpers
|   |   |-- centerline.rs          # closed-loop centreline derivation and projection
|   |   |-- track.rs               # Track component shared across runtime systems
|   |   `-- parts/                 # tile semantics for track assembly
|   `-- sim/
|       `-- sets.rs                # fixed pipeline ordering contract
|-- context/
|   |-- architecture.md            # this file
|   `-- systems/                   # subsystem-level implementation memory
|-- learning/                      # user-facing teaching material, not startup truth
|-- reports/                       # exported analytics run reports
|-- Cargo.toml
`-- README.md
```

## Subsystem Responsibilities

| Subsystem | Owns | Main neighbours |
|---|---|---|
| `maps` | Track topology, tile semantics, centreline derivation, spawn pose, visual track geometry | `game`, `agent`, `debug` |
| `game` | Car entity lifecycle, physics, collision truth, progress measurement, rewards, episode boundaries | `maps`, `agent`, `analytics`, `debug`, `brain` |
| `agent` | Stable action boundary and policy observation contract | `game`, `maps`, `brain`, `debug` |
| `brain` | Controller mode switching and the current A2C baseline | `agent`, `game`, `analytics` |
| `analytics` | Episode/update capture, derived diagnostics, report export | `game`, `agent`, `brain` |
| `debug` | Live overlays, HUD, and recent-run assessment | `game`, `agent`, `brain`, `maps` |
| `sim` | Named fixed-tick ordering sets shared across runtime plugins | all fixed-update subsystems |

## Dependency Direction

- `main` wires plugin order and global Bevy configuration.
- `maps` is foundational and does not depend on runtime control or analytics layers.
- `game` depends on `maps` for spatial truth and on `sim` for fixed-update ordering.
- `agent` depends on `game` and `maps` for measurable world state, and references `brain::types` for mode-aware keyboard suppression.
- `brain` depends on `agent` for observations/actions and `game` for reward/terminal state.
- `analytics` depends on `game`, `agent`, and `brain` data but should not mutate environment or training truth.
- `debug` depends on runtime state from `maps`, `game`, `agent`, and optionally `brain`.

## Core Execution / Data Flow

```text
Startup:
main -> MonacoPlugin spawns Track
     -> GamePlugin spawns camera + Car
     -> other plugins initialise resources and schedules

FixedUpdate:
SimSet::Input
  keyboard_action_input_system
  a2c_act_system
  action_smoothing_system

SimSet::Physics
  car_physics_system
  capture_episode_action_stats_system

SimSet::Collision
  collision_detection_system

SimSet::Measurement
  update_track_progress_system
  episode_loop_system
  update_sensor_readings_system
  build_observation_vector_system
  capture_episode_tick_trace_system
  snapshot_completed_episode_trace_system
  snapshot_completed_episode_action_stats_system
  a2c_collect_reward_system
  update_driving_hud_stats_system
  capture_driving_hud_episode_metrics_system

Update:
  toggle_agent_mode_system
  episode_tracker_system
  debug overlay + HUD rendering

Last:
  a2c_flush_on_exit_system
  analytics on-exit export
```

- The environment contract is fixed-tick and order-sensitive. A2C action selection must happen before smoothing, reward collection must happen after episode truth is computed, and analytics trace capture intentionally sits between observation rebuild and A2C reward collection.
- The `agent` boundary is stable: observations go from environment to controller, and actions go from controller to physics through `ActionState`.
- The current analytics path is append-only during runtime and flushes only on exit.

## Structural Notes / Current Reality

- The codebase is no longer environment-only. Any documentation that still treats A2C or analytics as roadmap-only is stale.
- The repository now also contains a substantial `learning/` tree. It is useful teaching material, but it should stay distinct from `context/`, which remains the implementation-facing memory layer.
- `src/analytics/sessions/` and `src/brain/biological/` currently exist as empty placeholders. They should not be treated as implemented subsystems until files land there.
- The project is still in a transitional architecture state:
  - the repository intent targets brain-inspired local plasticity,
  - the implemented learning path is currently a handwritten A2C baseline used to validate the environment and observation contract.
- `README.md` is directionally accurate about project intent and milestone ordering, but its Milestone 1 checklist now understates implementation reality:
  - the A2C baseline is live,
  - the debug learning HUD exists,
  - analytics export is implemented.
- The highest current documentation pressure is around validation and interface truth:
  - A2C is live but still lacks persistence, evaluation mode, headless training, and controlled RNG ownership.
  - Analytics reports are useful but still miss run metadata such as config snapshots and seeds.
  - Environment regression coverage remains narrow beyond a few targeted unit tests.
