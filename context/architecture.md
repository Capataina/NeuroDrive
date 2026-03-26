# NeuroDrive Architecture

## Scope / Purpose

- Provide the top-down structural map for the repository as it exists now.
- Orient a new engineer to the runtime layers, ownership boundaries, dependency direction, and the main fixed-tick execution flow.
- Keep detailed subsystem behaviour in `context/systems/` rather than duplicating it here.

## Repository Overview

- NeuroDrive is a Rust application built on **Bevy 0.18**, targeting brain-inspired online learning from first principles.
- The current runtime is a **deterministic 2D top-down racing environment** with:
  - a fixed 60 Hz simulation timestep,
  - a single hard-coded track,
  - a **multi-car vectorised trainer** (configurable car count via `TrainerConfig`, default 3) with per-car components and one shared rollout buffer,
  - a live handwritten A2C baseline brain,
  - a per-tick analytics capture pipeline with JSON and Markdown export,
  - a debug HUD, world-space overlay layer, and live leaderboard panel.
- The project intent in `README.md` is biologically inspired local plasticity. The current implementation reality is still at **baseline validation** — proving the environment and observation contract are learnable before transitioning to biological learning rules.
- `cargo check` and `cargo test` both pass in the current workspace state.

## Repository Structure

```text
NeuroDrive/
├── src/
│   ├── main.rs                          # App entrypoint: plugin registration, window config, 60 Hz fixed timestep
│   ├── sim/
│   │   ├── mod.rs
│   │   └── sets.rs                      # SimSet enum: Input → Physics → Collision → Measurement ordering contract
│   ├── maps/
│   │   ├── mod.rs
│   │   ├── monaco.rs                    # Hard-coded Sepang-inspired track tile assembly → MonacoPlugin
│   │   ├── track.rs                     # Track component: grid, spawn pose, centreline
│   │   ├── grid.rs                      # TrackGrid: driveable-area occupancy queries, tile rendering
│   │   ├── centerline.rs               # TrackCenterline: closed-loop polyline, arc-length projection, tangent queries
│   │   └── parts/
│   │       └── mod.rs                   # Tile semantics for track construction (straights, curves, spawn)
│   ├── game/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # GamePlugin: SimSet chain config, camera + multi-car spawn (TrainerConfig.num_envs cars with lateral offsets)
│   │   ├── car.rs                       # Car component (velocity, dynamics params), spawn_car(); EnvInstanceId, SpawnConfig, CarColour components
│   │   ├── physics.rs                   # car_physics_system + pure step_car_dynamics() helper
│   │   ├── collision.rs                 # Corner-based off-road detection → Collided marker component (all cars)
│   │   ├── progress.rs                  # TrackProgress component, centreline projection each tick
│   │   └── episode.rs                   # EpisodeState (Component), rewards, terminal conditions, resets, EpisodeMovingAverages (Component)
│   ├── agent/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # AgentPlugin: action + observation scheduling in SimSet
│   │   ├── action.rs                    # CarAction, ActionState (Component, desired/applied), smoothing, keyboard input
│   │   └── observation.rs               # SensorReadings, ObservationVector (dim 23), ray + centreline features
│   ├── brain/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # BrainPlugin: AgentMode toggle (F4), A2C buffer reset on switch
│   │   ├── types.rs                     # AgentMode enum, Brain trait
│   │   ├── a2c/
│   │   │   ├── mod.rs                   # A2cBrain, A2cPlugin, act/reward-collect/flush systems
│   │   │   ├── model.rs                 # ActorCritic: 2×64 MLP with separate actor + critic stacks
│   │   │   ├── buffer.rs               # TrainerRolloutBuffer: env_id-tagged transitions from all cars, per-env GAE computation
│   │   │   └── update.rs               # a2c_update(): GAE, policy/value losses, gradient clipping, health stats
│   │   ├── common/
│   │   │   ├── mod.rs
│   │   │   ├── mlp.rs                   # Handwritten Linear layer, ReLU, Glorot init, forward/backward
│   │   │   ├── math.rs                  # Gaussian sampling, log-prob, tanh correction utilities
│   │   │   └── optim.rs                # Adam optimiser with per-layer state
│   │   ├── ranking.rs                   # TrainerLiveRanking resource, car colour-based visual roles, best-car highlighting
│   │   └── biological/                  # Empty placeholder for future local-plasticity brain
│   ├── analytics/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # AnalyticsPlugin: tracker init, capture scheduling, on-exit export
│   │   ├── models.rs                    # EpisodeTracker, EpisodeRecord, TickTraceRecord, A2cUpdateRecord
│   │   ├── trackers/
│   │   │   ├── mod.rs
│   │   │   ├── action.rs               # Per-episode steering/throttle accumulation and snapshot
│   │   │   ├── trace.rs                # Per-tick trajectory capture and episode snapshot
│   │   │   └── episode.rs              # Folds completed episode + trace + A2C snapshots into EpisodeTracker
│   │   ├── metrics/
│   │   │   ├── mod.rs
│   │   │   ├── stats.rs                # Basic episode statistics computation
│   │   │   ├── chunking.rs             # Temporal chunked trend analysis
│   │   │   ├── inputs.rs               # Input-learning summaries (ray, offset, heading distributions)
│   │   │   ├── turns.rs                # Turn-execution diagnostics (latency, adequacy, understeer)
│   │   │   ├── critic.rs               # Critic health diagnostics (value drift, explained variance)
│   │   │   ├── sectors.rs              # Progress-sector breakdown summaries
│   │   │   ├── trajectory.rs           # Trajectory-level derived summaries
│   │   │   └── insights.rs             # Narrative insight bullet generation
│   │   ├── exporters/
│   │   │   ├── mod.rs
│   │   │   ├── json.rs                 # Timestamped JSON run export
│   │   │   └── markdown.rs             # Timestamped Markdown report with tables and narrative
│   │   └── sessions/                    # Empty placeholder for future session management
│   └── debug/
│       ├── mod.rs
│       ├── plugin.rs                    # DebugPlugin: overlay + HUD resource init and scheduling
│       ├── overlays.rs                  # F1/F2 world-space gizmos, F3 toggles, geometry + sensor drawing
│       ├── hud.rs                       # Bevy UI diagnostics panel, quarter summaries, run assessment
│       └── leaderboard.rs              # Live leaderboard HUD panel (top-right, F3-toggled) with per-car colour swatches
├── context/                             # Repository memory layer (this folder)
├── learning/                            # User-facing educational archive (not startup context)
├── reports/                             # Exported analytics run reports (JSON + Markdown)
├── Cargo.toml                           # Workspace root, bevy 0.18 + rand + serde dependencies
└── README.md                            # Project intent, brain-inspired learning vision, milestone roadmap
```

## Subsystem Responsibilities

| Subsystem | Owns | Main neighbours | Key source root |
|-----------|------|-----------------|-----------------|
| **sim** | Named fixed-tick ordering sets shared across all runtime plugins | all fixed-update subsystems | `src/sim/sets.rs` |
| **maps** | Track topology, tile semantics, centreline derivation, spawn pose, visual track geometry | game, agent, debug | `src/maps/` |
| **game** | Car entity lifecycle, physics, collision truth, progress measurement, reward shaping, episode boundaries | maps, agent, analytics, debug, brain | `src/game/` |
| **agent** | Stable action boundary (CarAction ↔ ActionState) and policy observation contract (ObservationVector) | game, maps, brain, debug | `src/agent/` |
| **brain** | Controller mode switching (F4), the current A2C baseline implementation, trainer live ranking, and car visual roles | agent, game, analytics | `src/brain/` |
| **analytics** | Episode/update capture, derived diagnostics, JSON + Markdown report export | game, agent, brain | `src/analytics/` |
| **debug** | Live world-space overlays, runtime HUD panel, live leaderboard panel, and recent-run assessment | game, agent, brain, maps | `src/debug/` |

## Dependency Direction

```text
                    ┌──────────┐
                    │   main   │  wires plugin order + global Bevy config
                    └────┬─────┘
                         │
          ┌──────────────┼──────────────────────┐
          │              │                      │
     ┌────▼────┐    ┌────▼────┐           ┌─────▼─────┐
     │  maps   │    │   sim   │           │  debug    │
     │(spatial │    │(ordering│           │(read-only │
     │ truth)  │    │contract)│           │ inspector)│
     └────┬────┘    └────┬────┘           └───────────┘
          │              │                      ▲
     ┌────▼────┐         │                      │
     │  game   │◄────────┘              reads from all
     │(env     │                        runtime state
     │ truth)  │
     └────┬────┘
          │
     ┌────▼────┐
     │  agent  │  reads game + maps for observations
     │(control │  references brain::types for mode gating
     │boundary)│
     └────┬────┘
          │
     ┌────▼────┐
     │  brain  │  reads agent for obs, game for reward/done
     └────┬────┘
          │
     ┌────▼──────┐
     │ analytics │  reads game, agent, brain data
     │(consumer  │  must not mutate environment or training truth
     │ only)     │
     └───────────┘
```

**Key rules:**
- `maps` is foundational and does not depend on runtime control or analytics layers.
- `game` depends on `maps` for spatial truth and on `sim` for fixed-update ordering.
- `agent` depends on `game` and `maps` for measurable world state, and references `brain::types` for mode-aware keyboard suppression.
- `brain` depends on `agent` for observations/actions and `game` for reward/terminal state.
- `analytics` depends on `game`, `agent`, and `brain` data but must not mutate environment or training truth.
- `debug` depends on runtime state from all layers but must not become a source of simulation truth.

## Core Execution / Data Flow

### Plugin Registration Order (main.rs)

```text
DefaultPlugins → MonacoPlugin → AgentPlugin → BrainPlugin → AnalyticsPlugin → GamePlugin → DebugPlugin
```

`MonacoPlugin` must run before `GamePlugin` because car spawn queries the `Track` entity. `GamePlugin` configures the `SimSet` chain that all other plugins place their fixed-update systems into.

### Fixed-Tick Pipeline (FixedUpdate at 60 Hz)

```text
SimSet::Input
├── keyboard_action_input_system          (agent — mode-gated, writes ActionState.desired)
├── a2c_act_all_cars_system               (brain — mode-gated, writes ActionState.desired for all cars, appends to rollout buffer with env_id)
└── action_smoothing_system               (agent — copies or smooths desired → applied)

SimSet::Physics
├── car_physics_system                    (game — mutates car transform and velocity from ActionState.applied)
└── capture_episode_action_stats_system   (analytics — records applied steering/throttle stats)

SimSet::Collision
└── collision_detection_system            (game — checks all car corners against road grid → Collided marker component)

SimSet::Measurement
├── update_track_progress_system          (game — centreline projection for progress)
├── episode_loop_system                   (game — reward, terminal check, reset, moving averages — iterates all cars)
├── update_sensor_readings_system         (agent — raycasts, kinematics, lookahead — after episode for post-reset state)
├── build_observation_vector_system       (agent — normalises sensors into ObservationVector)
├── capture_episode_tick_trace_system     (analytics — per-tick trace record)
├── snapshot_completed_episode_trace_system     (analytics)
├── snapshot_completed_episode_action_stats_system  (analytics)
├── a2c_collect_rewards_all_cars_system   (brain — appends reward/done for all cars, triggers update at horizon with per-env GAE)
├── update_driving_hud_stats_system       (debug — updates live HUD values)
└── capture_driving_hud_episode_metrics_system  (debug — captures episode-end data for quarter summaries)
```

### Update Schedule (every frame)

```text
├── toggle_agent_mode_system              (brain — F4 toggles AI ↔ Keyboard, clears rollout buffer)
├── episode_tracker_system                (analytics — folds completed snapshots into EpisodeTracker)
├── debug_overlay_toggle_system           (debug — F1/F2/F3 toggle handling)
├── draw_geometry_overlay_system          (debug — centreline, tangent, forward, velocity gizmos)
├── draw_sensor_overlay_system            (debug — ray segments and hit points)
├── update_driving_hud_visibility_system  (debug — shows/hides HUD based on F3)
├── update_driving_hud_text_system        (debug — refreshes HUD text sections)
├── update_trainer_ranking_system         (brain — updates best/worst car ranking with hysteresis)
├── update_car_visual_roles_system        (brain — assigns visual highlight roles based on ranking)
└── update_leaderboard_system             (debug — refreshes live leaderboard panel with per-car colour swatches)
```

### Last Schedule (before exit)

```text
├── a2c_flush_on_exit_system              (brain — updates from any residual rollout data)
└── on_exit_system                        (analytics — exports JSON + Markdown to reports/)
```

### Key Data Flow Contracts

- The environment contract is **fixed-tick and order-sensitive**. A2C action selection must happen before smoothing, reward collection must happen after episode truth is computed, and analytics trace capture sits between observation rebuild and A2C reward collection.
- The `agent` boundary is stable: observations go from environment to controller, and actions go from controller to physics through `ActionState`.
- The analytics path is **append-only** during runtime and flushes only on app exit.

## Structural Notes / Current Reality

- The codebase is **not** environment-only. A2C, analytics, and the debug HUD are live and substantial subsystems. Documentation treating them as roadmap-only is obsolete.
- **Singleton-car assumptions have been removed** from `game`, `agent`, and `brain`. `ActionState`, `EpisodeState`, and `EpisodeMovingAverages` are now per-car **Components** (not Resources). `CollisionEvent` has been replaced by a `Collided` marker component. All fixed-tick systems iterate over multiple cars.
- The runtime is a **multi-car vectorised trainer**:
  - `TrainerConfig` controls car count (default 3). Cars are spawned with lateral offsets, each assigned a unique colour from a 25-colour palette.
  - Per-car components: `EnvInstanceId`, `SpawnConfig`, `CarColour`, `ActionState`, `EpisodeState`, `EpisodeMovingAverages`.
  - One shared `TrainerRolloutBuffer` collects transitions from all cars with `env_id` tagging; GAE is computed per-env (no cross-env value leakage).
  - A `TrainerLiveRanking` resource tracks best/worst car with hysteresis; `ranking.rs` assigns visual highlight roles.
  - A live leaderboard panel (top-right, F3-toggled) shows per-car performance with colour swatches.
- **Analytics and HUD systems use temporary shims** (target first car) pending a full analytics overhaul to support multi-car reporting.
- The brain layer now owns **ranking logic** (`src/brain/ranking.rs`) in addition to A2C. A seeded `StdRng` lives in `A2cBrain` for deterministic policy sampling.
- The project is in a **transitional architecture state**:
  - The repository intent targets brain-inspired local plasticity (Milestones 2–9).
  - The implemented learning path is a handwritten A2C baseline used to validate the environment and observation contract (Milestone 1).
- **`src/brain/biological/`** and **`src/analytics/sessions/`** are empty placeholder directories. They should not be treated as implemented subsystems.
- `README.md` is directionally accurate but its Milestone 1 checklist understates implementation reality — A2C, debug HUD, and analytics export are all live.
- The highest current documentation pressure is around **validation and experiment discipline**: A2C lacks persistence, evaluation mode, headless training, and multi-car analytics integration.
