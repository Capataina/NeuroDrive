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
  - a **multi-car vectorised trainer** (configurable car count via `TrainerConfig`, default 8) with per-car components and one shared rollout buffer,
  - all cars spawning at **random centreline positions** (re-randomised each reset, no privileged car),
  - throttle axis `[0, 1]` (coast to full thrust, no braking — drag is the sole deceleration mechanism),
  - a **velocity-projection reward** (dot of velocity onto centreline tangent) plus centreline proximity reward,
  - **43-dimensional observations** (rays, kinematics with v_forward/v_lateral split, speed_delta, 12-point lookahead with heading deltas + curvatures spanning 30–650 units, previous actions),
  - a live handwritten **PPO** brain (upgraded from A2C — clipped surrogate objective, multi-epoch updates amortised across ticks, **asymmetric architecture**: actor 2×64, critic 2×128; round-2 extensions 2026-04-19: **PopArt value-target normalisation** on `c_value`, **γ raised to 0.995**, **target-KL early stop**, **running observation mean/var normaliser** per Andrychowicz 2021),
  - a **PolicyOutput** component per car exposing value prediction and policy distribution parameters,
  - a comprehensive analytics pipeline: 16 tick-level fields, 25 episode-level aggregates, crash classification, 10-section Markdown report with auto-generated takeaways (exports to `reports/json/analytics/` and `reports/analytics/`),
  - a debug HUD, world-space overlay layer, and live leaderboard panel.
- Episodes end on **crash or 30-second timeout only** — there is no finish line or lap concept.
- The project intent in `README.md` is biologically inspired local plasticity. The current implementation reality is still at **baseline validation** — proving the environment and observation contract are learnable before transitioning to biological learning rules.
- `cargo check` and `cargo test` both pass in the current workspace state.

## Repository Structure

```text
NeuroDrive/
├── src/
│   ├── main.rs                          # App entrypoint: plugin registration, window config, 60 Hz fixed timestep
│   ├── sim/
│   │   ├── mod.rs                       # Shared geometry utilities: wrap_angle(), signed_angle_between()
│   │   └── sets.rs                      # SimSet enum: Input → Physics → Collision → Measurement ordering contract
│   ├── maps/
│   │   ├── mod.rs
│   │   ├── monaco.rs                    # Hard-coded Sepang-inspired track tile assembly → MonacoPlugin (no finish line sprite)
│   │   ├── track.rs                     # Track component: grid, centreline
│   │   ├── grid.rs                      # TrackGrid: driveable-area occupancy queries, tile rendering
│   │   ├── centerline.rs               # TrackCenterline: closed-loop polyline, arc-length projection, tangent queries
│   │   └── parts/
│   │       └── mod.rs                   # Tile semantics for track construction (straights, curves, spawn)
│   ├── game/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # GamePlugin: SimSet chain config, camera + multi-car spawn (all cars at random centreline positions)
│   │   ├── car.rs                       # Car component (velocity, dynamics params), spawn_car(); EnvInstanceId, SpawnRng, CarColour components
│   │   ├── physics.rs                   # car_physics_system + pure step_car_dynamics() helper (rotation_speed=8.0, throttle [0,1])
│   │   ├── collision.rs                 # Corner-based off-road detection → Collided marker component (all cars)
│   │   ├── progress.rs                  # TrackProgress component, cumulative forward arc-length from spawn with wrap handling
│   │   └── episode.rs                   # EpisodeState (Component), velocity-projection + centreline rewards, crash/timeout terminal, resets, EpisodeMovingAverages (Component)
│   ├── agent/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # AgentPlugin: action + observation scheduling in SimSet
│   │   ├── action.rs                    # CarAction, ActionState (Component, desired/applied), smoothing, keyboard input
│   │   └── observation.rs               # SensorReadings (v_forward, v_lateral, speed_delta, previous_steering, previous_throttle), ObservationVector (dim 43), ray + centreline features
│   ├── brain/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # BrainPlugin: AgentMode toggle (F4), PPO buffer reset on switch
│   │   ├── types.rs                     # AgentMode enum, PolicyOutput component
│   │   ├── ppo/
│   │   │   ├── mod.rs                   # PpoBrain (with act_entity_buffer scratch), PpoPlugin, PpoUpdateState, ppo_act_all_cars_system (3-pass batched), collect/epoch/flush systems
│   │   │   ├── model.rs                 # ActorCritic: asymmetric MLP (actor 2×64, critic 2×128), BatchIo + BatchScratch + SampleScratch (critic-only), forward_actor_batch/forward_critic_batch/forward_critic
│   │   │   ├── buffer.rs               # TrainerRolloutBuffer: env_id-tagged transitions + old_log_probs, per-env GAE via reusable EnvGrouping (Vec-indexed by env_id, deterministic iteration)
│   │   │   └── update.rs               # PreparedUpdate, ppo_process_chunk/ppo_finish_epoch, PPO clipped surrogate
│   │   ├── common/
│   │   │   ├── mod.rs
│   │   │   ├── gemm_backend.rs          # GEMM dispatch module — selects scalar/matrixmultiply/accelerate at compile time; backend_name() for profiling reports
│   │   │   ├── gemm_scalar.rs           # Naive nested-loop reference backend (force-scalar)
│   │   │   ├── gemm_matrixmultiply.rs   # Pure-Rust BLIS-style NEON microkernel via matrixmultiply crate
│   │   │   ├── gemm_accelerate.rs       # Apple Accelerate via cblas_sgemm (macOS; dispatches to AMX)
│   │   │   ├── mlp.rs                   # Handwritten Linear (flat weight storage) + Tanh; forward_batch/backward_batch route through gemm_backend; forward_into single-sample scalar path
│   │   │   ├── math.rs                  # Gaussian sampling, log-prob, tanh correction, orthogonal init utilities
│   │   │   └── optim.rs                # AdamW optimiser with per-layer state and decoupled weight decay
│   │   └── ranking.rs                   # TrainerLiveRanking resource, car colour-based visual roles, best-car highlighting
│   ├── analytics/
│   │   ├── mod.rs
│   │   ├── plugin.rs                    # AnalyticsPlugin: tracker + config init, capture scheduling, two-tier on-exit export
│   │   ├── models.rs                    # EpisodeTracker, EpisodeRecord (env_id-tagged, 25 episode-level aggregates), TickTraceRecord (16 tick-level fields), CrashKind (Slide/HeadOn/Overshoot/Spin/Stall), PpoUpdateRecord, AnalyticsConfig, RunMetadata, CompactRunExport
│   │   ├── trackers/
│   │   │   ├── mod.rs
│   │   │   ├── action.rs               # PerCarActionAccumulators: per-car steering/throttle accumulation and snapshot (all cars)
│   │   │   ├── trace.rs                # PerCarTraceAccumulators: per-car per-tick trajectory capture (position, velocity decomposition, drift angle, min ray, velocity projection, centreline reward, policy confidence) and episode snapshot (all cars)
│   │   │   └── episode.rs              # Folds all cars' completed episodes + traces + PPO snapshots into EpisodeTracker
│   │   ├── metrics/
│   │   │   ├── mod.rs
│   │   │   ├── stats.rs                # Basic episode statistics computation
│   │   │   ├── chunking.rs             # Temporal chunked trend analysis
│   │   │   ├── timeseries.rs           # Episode/update time-series extraction, rolling mean, plateau detection
│   │   │   ├── diagnostics.rs          # Automated diagnostic flags (entropy collapse, clip spikes, plateaus, etc.)
│   │   │   ├── consistency.rs          # Per-sector behavioural consistency (speed/steering/throttle variance)
│   │   │   ├── phases.rs               # Learning phase detection (Exploration → Discovery → Refinement → Plateau → Regression)
│   │   │   ├── sparkline.rs            # ASCII visual helpers (sparklines, bar charts, heatmap rows)
│   │   │   ├── turns.rs                # Turn-execution trace metrics (compute_trace_metrics)
│   │   │   ├── sectors.rs              # Progress-sector breakdown summaries
│   │   │   └── trajectory.rs           # Trajectory-level derived summaries
│   │   ├── exporters/
│   │   │   ├── mod.rs
│   │   │   ├── json.rs                 # Two-tier JSON export to reports/json/analytics/: compact (always) + full trace (opt-in)
│   │   │   ├── markdown.rs             # Diagnostic Markdown report to reports/analytics/ with 10 sections, sparklines, heatmaps, crash classification, auto-generated takeaways
│   │   │   ├── context.rs              # RunContext: captures config snapshot for both analytics and profiling exports
│   │   │   └── cleanup.rs              # Retention-limited directory cleanup (auto-deletes oldest reports)
│   ├── profiling/                       # Performance profiling (feature-gated: --features profiling)
│   │   ├── mod.rs                       # ProfilingPlugin: timing capture, auto-exit, on-exit export
│   │   ├── config.rs                    # ProfilingConfig: duration, ring buffer size, category toggles
│   │   ├── timers.rs                    # FrameTimings ring buffer, FrameRecord struct
│   │   ├── capture.rs                   # Frame start/end timing, per-SimSet boundary markers, auto-exit
│   │   └── exporters/
│   │       ├── mod.rs
│   │       ├── json.rs                  # On-exit JSON performance report export
│   │       └── markdown.rs              # Rich Markdown performance report with interpretation and recommendations
│   └── debug/
│       ├── mod.rs
│       ├── plugin.rs                    # DebugPlugin: overlay + HUD resource init and scheduling
│       ├── overlays.rs                  # F1/F2 world-space gizmos, F3 toggles, geometry + sensor drawing
│       ├── hud.rs                       # Bevy UI diagnostics panel, quarter summaries, run assessment
│       └── leaderboard.rs              # Live leaderboard HUD panel (top-right, F3-toggled) with per-car colour swatches
├── context/                             # Repository memory layer (this folder)
├── learning/                            # User-facing educational archive (not startup context)
├── reports/                             # Exported run reports
│   ├── json/analytics/                  # Compact + full-trace JSON analytics exports
│   ├── json/performance/                # JSON performance profiling exports
│   ├── analytics/                       # Markdown analytics reports
│   └── performance/                     # Markdown performance reports
├── tests/                               # Integration tests (unblocked by the 2026-04-18 [lib] target)
│   ├── gemm_correctness.rs              # Cross-validates active GEMM backend against inline scalar reference on every shape PPO uses
│   └── ppo_pipeline.rs                  # End-to-end forward+backward+optimiser pipeline; finite gradients, Adam step, varying batch sizes
├── Cargo.toml                           # bevy 0.18 + rand + serde + matrixmultiply; Accelerate (blas-src + cblas) gated on target_os = "macos"; feature flags: profiling / force-scalar / force-matrixmultiply / force-accelerate
├── src/lib.rs                           # Library target mirroring the binary module tree — enables integration tests in tests/*.rs
└── README.md                            # Project intent, brain-inspired learning vision, milestone roadmap, Building and Running command reference
```

## Subsystem Responsibilities

| Subsystem | Owns | Main neighbours | Key source root |
|-----------|------|-----------------|-----------------|
| **sim** | Named fixed-tick ordering sets shared across all runtime plugins, canonical shared geometry utilities (`wrap_angle`, `signed_angle_between` — consolidated from earlier triplication) | all fixed-update subsystems | `src/sim/` |
| **maps** | Track topology, tile semantics, centreline derivation, visual track geometry (no finish line, no fixed spawn pose) | game, agent, debug | `src/maps/` |
| **game** | Car entity lifecycle (random spawn, drag-only deceleration), collision truth, cumulative progress measurement, velocity-projection + centreline reward shaping, crash/timeout episode boundaries | maps, agent, analytics, debug, brain | `src/game/` |
| **agent** | Stable action boundary (CarAction ↔ ActionState, throttle [0,1]) and policy observation contract (ObservationVector, 43 dims) | game, maps, brain, debug | `src/agent/` |
| **brain** | Controller mode switching (F4), the PPO baseline implementation (clipped surrogate, amortised epochs), PolicyOutput per-car component, trainer live ranking, and car visual roles | agent, game, analytics | `src/brain/` |
| **analytics** | Multi-car episode/update capture (env_id-tagged), 10 derived metric modules, crash classification (Slide/HeadOn/Overshoot/Spin/Stall), two-tier JSON export to `reports/json/analytics/`, diagnostic Markdown to `reports/analytics/`, RunContext snapshot, retention-limited cleanup, auto-generated takeaways | game, agent, brain | `src/analytics/` |
| **profiling** | Feature-gated (`--features profiling`) per-frame and per-system timing capture (17 systems), per-SimSet breakdown, ring buffer storage, auto-exit, JSON + Markdown report export with RunContext snapshot, retention-limited cleanup | sim, brain, analytics | `src/profiling/` |
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
DefaultPlugins → MonacoPlugin → AgentPlugin → BrainPlugin → AnalyticsPlugin → GamePlugin → DebugPlugin + ProfilingPlugin (feature-gated)
```

`MonacoPlugin` must run before `GamePlugin` because car spawn queries the `Track` entity. `GamePlugin` configures the `SimSet` chain that all other plugins place their fixed-update systems into.

### Fixed-Tick Pipeline (FixedUpdate at 60 Hz)

```text
frame_start_system                        (profiling, feature-gated — captures frame start timestamp)

SimSet::Input
├── keyboard_action_input_system          (agent — mode-gated, writes ActionState.desired)
├── ppo_act_all_cars_system               (brain — mode-gated; 3-pass batched: stacks obs into batch_io → single mat-mat actor + critic batched forwards → per-car sampling + PolicyOutput writes + rollout buffer push)
└── action_smoothing_system               (agent — copies or smooths desired → applied)

input_end_system                          (profiling, feature-gated — marks Input→Physics boundary)

SimSet::Physics
├── car_physics_system                    (game — mutates car transform and velocity from ActionState.applied)
└── capture_episode_action_stats_system   (analytics — records applied steering/throttle stats)

physics_end_system                        (profiling, feature-gated — marks Physics→Collision boundary)

SimSet::Collision
└── collision_detection_system            (game — checks all car corners against road grid → Collided marker component)

collision_end_system                      (profiling, feature-gated — marks Collision→Measurement boundary)

SimSet::Measurement
├── update_track_progress_system          (game — centreline projection for progress)
├── episode_loop_system                   (game — velocity-projection + centreline reward, crash/timeout terminal check, random-spawn reset, moving averages — iterates all cars)
├── update_sensor_readings_system         (agent — raycasts, kinematics, lookahead — after episode for post-reset state)
├── build_observation_vector_system       (agent — normalises sensors into ObservationVector)
├── capture_episode_tick_trace_system     (analytics — per-tick trace record)
├── snapshot_completed_episode_trace_system     (analytics)
├── snapshot_completed_episode_action_stats_system  (analytics)
├── ppo_collect_rewards_all_cars_system   (brain — appends reward/done for all cars, prepares PPO update at horizon)
├── ppo_epoch_system                     (brain — processes one samples_per_tick chunk (default 32) from prepared update via active GEMM backend, advances epoch state)
├── update_driving_hud_stats_system       (debug — updates live HUD values)
└── capture_driving_hud_episode_metrics_system  (debug — captures episode-end data for quarter summaries)

frame_end_system                          (profiling, feature-gated — captures frame end timestamp, records per-set durations)
auto_exit_system                          (profiling, feature-gated — exits app after configured duration)
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
├── ppo_flush_on_exit_system              (brain — finishes in-progress PPO epochs + flushes residual rollout data)
├── on_exit_system                        (analytics — exports JSON + Markdown to reports/)
└── on_exit_export_system                 (profiling, feature-gated — exports JSON + Markdown performance reports to reports/)
```

### Key Data Flow Contracts

- The environment contract is **fixed-tick and order-sensitive**. PPO action selection (which also writes `PolicyOutput`) must happen before smoothing, reward collection must happen after episode truth is computed, and analytics trace capture (which reads `PolicyOutput` for policy confidence metrics) sits between observation rebuild and PPO reward collection.
- The PPO update is **amortised across ticks** via `PreparedUpdate` and `ppo_epoch_system`. GAE is computed once at prepare time; samples are processed in `samples_per_tick`-sized chunks (default 32 as of 2026-04-18) across subsequent ticks. New transitions collect into a fresh buffer during the update.
- All mat-mat work in `Linear::forward_batch` / `Linear::backward_batch` and in `forward_actor_batch` / `forward_critic_batch` routes through the active **GEMM backend** selected at compile time by the `force-*` Cargo features (default: Accelerate on macOS via cblas_sgemm → AMX, matrixmultiply elsewhere via pure-Rust BLIS-style NEON microkernel). The backend name is recorded in every profiling report under the Run Context's `### Build` section.
- The `agent` boundary is stable: observations go from environment to controller, and actions go from controller to physics through `ActionState`.
- The analytics path is **append-only** during runtime and flushes only on app exit.

## Reading Guide

| System file | Covers | Source roots |
|-------------|--------|--------------|
| `systems/environment.md` | Track topology (`src/maps/`), car lifecycle, physics, collisions, progress, reward, episodes (`src/game/`) | `src/maps/`, `src/game/` |
| `systems/agent-interface.md` | Action contract, observation contract, scheduling | `src/agent/` |
| `systems/brain-ppo.md` | PPO algorithm, model architecture, rollout buffer, ranking, ML primitives | `src/brain/` |
| `systems/analytics.md` | Capture pipeline, derived metrics, two-tier export, crash classification | `src/analytics/` |
| `systems/profiling.md` | Feature-gated per-system timing, auto-exit, performance reports | `src/profiling/` |
| `systems/debug.md` | Live overlays, HUD panel, leaderboard | `src/debug/` |
| `systems/determinism.md` | Cross-cutting: ordering contract, reproducibility surfaces, RNG state | `src/sim/`, cross-cutting |

## Inter-System Relationships

Individual system files cover their own boundaries. This section is the canonical home for the relationships **between** systems — the connections that matter when reasoning about blast radius or change impact. Each entry names the two sides, the mechanism, the data that flows, and what breaks if the connection is violated.

| A | B | Mechanism | Data | What breaks if broken |
|---|---|-----------|------|-----------------------|
| `game` | `agent` | Read-only: `Car`, `ActionState.applied`, `TrackProgress`, centreline projection | Physics state + projection truth | Observations become stale or wrong-tick; post-reset observations leak crash state |
| `agent` | `game` | Write path: `ActionState.applied` consumed by `car_physics_system` | Steering + throttle | Physics stops executing policy decisions |
| `agent` | `brain` | Read-only: `ObservationVector` (43-dim) consumed by `ppo_act_all_cars_system` | Normalised observation tensor | Any change in `OBSERVATION_DIM` constant or feature ordering desynchronises the model (dim mismatch panics on `forward_actor`); no runtime dimension assertion beyond shared `const OBSERVATION_DIM` |
| `brain` | `agent` | Write path: writes per-car `ActionState.desired` and `PolicyOutput` component | Steering + throttle means/stds + value prediction | Smoothing + physics receive stale or default actions |
| `game` | `brain` | `EpisodeState.current_tick_reward` + `current_tick_end_reason` consumed by `ppo_collect_rewards_all_cars_system` | Per-tick reward + terminal flag | PPO mis-aligns reward-to-observation; GAE becomes invalid |
| `brain` | `analytics` | `PolicyOutput` per-car component read by `capture_episode_tick_trace_system`; `PpoTrainingStats` read by `episode_tracker_system` | Value prediction, policy confidences, update diagnostics | Trace records miss policy stats; Markdown report's "What Does the Car Think" section becomes meaningless |
| `game` | `analytics` | `EpisodeState` (current and finalised), `Collided` marker, `TrackProgress` consumed by per-car capture systems | Episode summary + reward decomposition | Episode records desynchronise with env truth; crash classification becomes unreliable |
| `maps` | `game` | `Track { grid, centerline }` singleton consumed by physics (via progress), collision, and episode reset (spawn RNG draws centreline fractions) | Spatial truth (grid occupancy + arc-length parametrisation) | No collisions, no spawn, no progress — runtime fails before first tick |
| `maps` | `agent` | `TrackGrid` consumed by raycast marching; `TrackCenterline` consumed by lookahead features | Grid occupancy + `tangent_at_s` | Sensor readings collapse; lookahead curvature vanishes |
| `sim` | all fixed-update subsystems | `SimSet` ordering contract: `Input → Physics → Collision → Measurement`, configured by `GamePlugin` | Schedule sets, not data | Any plugin placing systems outside `SimSet` creates silent ordering bugs (e.g., observations built from pre-reset state); only the four-stage chain guarantees the reward/observation alignment that PPO depends on |
| `analytics::exporters::{cleanup, context}` | `profiling::exporters::json` | Direct `use crate::analytics::exporters::{cleanup::enforce_retention, context::RunContext}` | `RunContext` struct (full run config snapshot) + retention-limited directory pruning | Profiling report lose their run-context header and unbounded report directories accumulate; see Shared Infrastructure below |
| `brain::ranking` | `debug::leaderboard` | `TrainerLiveRanking` resource + per-car `CarColour` component | Ranked car order + colour swatches | Leaderboard panel goes blank or shows stale ordering |
| `brain::ranking` | `debug::hud` | Same `TrainerLiveRanking` read by `update_driving_hud_text_system` to pick the "best car" view (falls back to first car if unavailable) | Best car index | HUD silently shows first car instead of best — not fatal but misleading |

### Shared Infrastructure: RunContext and Retention Cleanup

The profiling subsystem and the analytics subsystem both export reports, and both:

- capture an identical `RunContext` snapshot (car count, PPO hyperparameters, reward coefficients, observation layout) via `analytics::exporters::context::RunContext::capture()`,
- enforce a retention limit of 3 reports per directory via `analytics::exporters::cleanup::enforce_retention()`.

These helpers live in `analytics::exporters` and are imported by `profiling::exporters::json`. This is deliberate shared infrastructure, not parallel evolution — but it does mean the profiling feature has a **compile-time dependency on the analytics module** even though nothing in `systems/profiling.md`'s boundaries section makes that obvious from a quick read. If analytics were ever ripped out or relocated, the feature-gated profiling pipeline would fail to compile even with `--features profiling` enabled.

### Dependency Chain Trace — One PPO Training Tick (end-to-end)

The single operation that crosses the most system boundaries is one fixed tick during PPO training (mode = `Ai`, rollout near horizon). Tracing it names the full blast radius that any change to the tick pipeline must respect.

```text
Step  Owner / System                                  Reads                       Writes
────  ──────────────────────────────────────────────  ──────────────────────────  ──────────────────────────────
 1    profiling::frame_start_system (feature-gated)   —                           FrameRecord.frame_start
 2    agent::keyboard_action_input_system             AgentMode, KeyCode          — (exits early in Ai mode)
 3    brain::ppo_act_all_cars_system                  ObservationVector per car,  batch_io.obs_batch stacked,
                                                      model (3-pass batched:      scratch.a_out (batched means),
                                                      forward_actor_batch then    scratch.c_out (batched values),
                                                      forward_critic_batch),      ActionState.desired (per car),
                                                      PpoBrain.rng                PolicyOutput (per car),
                                                                                  TrainerRolloutBuffer push (state +
                                                                                  latent_action + action + old_log_prob
                                                                                  + env_id)
 4    agent::action_smoothing_system                  ActionState.desired         ActionState.applied (per car)
 5    profiling::input_end_system                     —                           FrameRecord.input_end
 6    game::car_physics_system                        ActionState.applied, Car    Car.velocity, Transform
 7    analytics::capture_episode_action_stats_system  ActionState.applied         PerCarActionAccumulators entry
 8    profiling::physics_end_system                   —                           FrameRecord.physics_end
 9    game::collision_detection_system                Transform, TrackGrid        Collided marker (add/remove)
10    profiling::collision_end_system                 —                           FrameRecord.collision_end
11    game::update_track_progress_system              Transform, TrackCenterline  TrackProgress (per car)
12    game::episode_loop_system                       TrackProgress, Collided,    EpisodeState.current_tick_reward,
                                                      Car.velocity, EpisodeConfig current_tick_end_reason,
                                                                                  SpawnRng (on reset), Transform
                                                                                  reset, Car velocity reset
13    agent::update_sensor_readings_system            Transform, TrackGrid,       SensorReadings (per car, now
                                                      TrackCenterline             reflects post-reset state)
14    agent::build_observation_vector_system          SensorReadings              ObservationVector (per car)
15    analytics::capture_episode_tick_trace_system    EpisodeState, PolicyOutput, PerCarTraceAccumulators push
                                                      SensorReadings, Transform
16    analytics::snapshot_completed_episode_*_system  PerCar*Accumulators,        EpisodeTracker.pending_episodes /
                                                      EpisodeState (terminal)     pending_traces on terminal
17    brain::ppo_collect_rewards_all_cars_system      EpisodeState.current_tick_* TrainerRolloutBuffer
                                                      (per car)                    reward + done push;
                                                                                  may call ppo_prepare_update()
                                                                                  at horizon → PreparedUpdate
18    brain::ppo_epoch_system                         PreparedUpdate,             model weights + Adam state
                                                      model (forward_batch +      PpoTrainingStats
                                                      backward_batch)             (on epoch end)
19    debug::update_driving_hud_stats_system          TrackProgress, Collided     DrivingHudStats
20    debug::capture_driving_hud_episode_metrics      EpisodeState (terminal)     DrivingHudHistory
21    profiling::frame_end_system + auto_exit         SystemTimers.durations_us   FrameRecord completed
```

**Failure semantics along the chain:**

- **Step 3 → Step 17 alignment.** The PPO buffer push in step 3 records (state, action, old_log_prob). Step 17 pushes the matching (reward, done). Any step between them that mutates `EpisodeState` out of order or runs in the wrong `SimSet` desynchronises reward from its generating action — silent training corruption. This is why `sim::SimSet` is structural rather than decorative.
- **Step 12 reset ordering.** Episode reset happens in `episode_loop_system` (step 12), **before** sensor readings rebuild (step 13). This is deliberate: if the order were reversed, the first observation of a new episode would be built from pre-reset crash state and the PPO rollout would bootstrap from a lie.
- **Step 11 before Step 12.** Progress projection must run before episode logic because crash classification and velocity-projection reward both read `TrackProgress`. Swapping them gives a zero reward on the terminal tick.
- **Step 18 amortisation.** `ppo_epoch_system` processes only `samples_per_tick` (default 64) samples of the prepared update per tick. A full 4-epoch update over a 512-sample rollout takes `4 × 512 / 64 = 32` ticks. During these 32 ticks, a new rollout buffer is collecting — PPO is **not on-policy in the strict sense** during amortised updates, but the `old_log_prob` captured at step 3 protects ratio calculation regardless.
- **Step 9 Collided as a marker not an event.** The environment never fires Bevy events for collisions; `Collided` is added as a component marker and read by the episode system two steps later. Any system that needs to react to collisions must be placed **in `SimSet::Measurement` after `episode_loop_system`** to see it before it is cleared on reset.

## Coverage (Knowledge Gaps from this 2026-04-18 Pass)

This section is explicit about where this upkeep pass relied on direct code inspection versus inference from existing documentation and the scan-tool output, so the next session knows what still needs verification.

**Directly inspected this session:**

- `src/main.rs` (plugin registration order — verified it matches architecture.md).
- `src/brain/ppo/update.rs` lines 140–300 (the two `unsafe` aliasing blocks — verified and captured as rationale).
- `src/brain/ppo/mod.rs` lines 25–100 (`PpoConfig` struct — verified Phase 4 consolidation).
- `src/game/car.rs` imports + `SpawnRng`, `EnvInstanceId`, `TrainerConfig` struct signatures.
- `src/agent/observation.rs` header through `OBSERVATION_DIM` constant — verified 43-dim layout matches the docs.
- `src/analytics/exporters/context.rs` full `RunContext` struct signature.
- `src/analytics/exporters/cleanup.rs` in full (tiny file).
- Grep for `unsafe`, `WHY/NOTE/HACK/IMPORTANT/TODO/FIXME/SAFETY`, `EventWriter|EventReader`, `StdRng|seed_from|rand::rng|ThreadRng`, `debug_assert`, `#[cfg(feature`, `EpisodeConfig|TrainerConfig|PpoConfig|AnalyticsConfig|ProfilingConfig|RunContext|EnvInstanceId` across all of `src/`.

**Inspected through the scan tool output only** (file existence + import regex, not code bodies read):

- All 67 Rust files were listed by `scripts/scan_repo.py`. Import regex output was consumed for roughly the first 20 files to establish dependency-direction consistency with architecture.md, then trusted by sampling.
- `src/analytics/metrics/` 10 modules — only `chunking.rs`, `consistency.rs`, `diagnostics.rs`, `phases.rs` headers were glanced at via the scan; internal logic was trusted against `systems/analytics.md`.
- `src/profiling/*` — the timer instrumentation structure was trusted against `systems/profiling.md` without re-reading every file body.

**Described from existing context only, not re-verified this pass:**

- Exact HUD layout (`src/debug/hud.rs` — trusted against `systems/debug.md`).
- Centreline binary-search projection implementation (`src/maps/centerline.rs` — trusted against `systems/environment.md` and the Phase 2 audit note).
- The 25 episode-level aggregates enumerated in `systems/analytics.md` — counted in the scan but field-by-field accuracy is from the previous session.
- Analytics markdown exporter structure — architecture and systems both agree; not re-read this pass.

**Known gap items re-surfaced by this pass:**

- `unsafe` in `src/brain/ppo/update.rs` — now inspected and verified. SAFETY comments describe read-only aliasing of scratch buffers (`obs_batch` and `grad_seed_{values,means}`) while the model takes `&mut self`. The aliased slices are distinct from all `&mut` scratch fields used by `forward_batch`/`backward_batch`. This is a borrow-checker workaround, not an unsound pattern. The `notes/session-2026-04-15.md` checkbox for this item can be ticked.
- User-controllable PPO seed — still not implemented; `systems/determinism.md` flags this correctly.
- Headless training mode — still missing; flagged in `systems/environment.md` and `systems/brain-ppo.md`.
- Analytics/profiling parallel evolution hazard — now documented under Shared Infrastructure above.

## Structural Notes / Current Reality

- The codebase is **not** environment-only. PPO, analytics, and the debug HUD are live and substantial subsystems. Documentation treating them as roadmap-only is obsolete.
- **Singleton-car assumptions have been removed** from `game`, `agent`, and `brain`. `ActionState`, `EpisodeState`, and `EpisodeMovingAverages` are now per-car **Components** (not Resources). `CollisionEvent` has been replaced by a `Collided` marker component. All fixed-tick systems iterate over multiple cars.
- The runtime is a **multi-car vectorised trainer**:
  - `TrainerConfig` controls car count (default 8). All cars spawn at **random centreline positions** (re-randomised on each episode reset), each assigned a unique colour from a 25-colour palette. There is no privileged car 0 or fixed spawn position.
  - Per-car components: `EnvInstanceId`, `CarColour`, `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, `PolicyOutput`.
  - One shared `TrainerRolloutBuffer` collects transitions from all cars with `env_id` tagging and old log-probs for PPO ratio computation; GAE is computed per-env (no cross-env value leakage).
  - A `TrainerLiveRanking` resource tracks best/worst car with hysteresis; `ranking.rs` assigns visual highlight roles.
  - A live leaderboard panel (top-right, F3-toggled) shows per-car performance with colour swatches.
- **Episode semantics**: there is no finish line or lap concept. Progress is **cumulative forward arc-length from spawn** with wrap handling. Episodes end on **crash or 30-second timeout** only. `EpisodeEndReason` has `Crash` and `Timeout` variants (no `LapComplete`).
- **Physics**: `rotation_speed` is 8.0 rad/s. The throttle axis is `[0, 1]` — 0 coasts (drag decelerates naturally), 1 is full thrust. Braking was tried and reverted because the policy converged to "mostly brake" as a safe local optimum.
- **Reward**: per-tick velocity projection reward — `dot(velocity, tangent) / speed_reference × velocity_reward_scale` — plus a centreline proximity reward (`centreline_reward_coef`, `centreline_reward_max_distance`). Crash penalty is 0.0. `EpisodeConfig` carries `velocity_reward_scale`, `centreline_reward_coef`, `centreline_reward_max_distance` (not `progress_reward_scale`).
- **Observations** (43 dimensions): rays (11), v_forward + v_lateral, speed_delta, centreline offset/heading/curvature, 12-point lookahead (heading deltas + curvatures, 30–650 units, dense near / sparse far), previous_steering, previous_throttle.
- The brain uses **PPO** (upgraded from A2C): clipped surrogate objective (ε=0.2), 4 epochs per update (with **target-KL early stop** — halts epochs when `approx_kl > 1.5 × target_kl`, default 0.03). The network is an **asymmetric actor-critic** — actor 2×64, critic 2×128 — with **tanh activations**, **orthogonal weight initialisation** (√2 hidden, 0.01× policy head, 1.0× value head), and **per-minibatch advantage normalisation** with sample shuffling. The actor uses standard Adam (LR 3e-4); the critic uses **AdamW with weight decay λ=3e-4** (LR 5e-4). `log_std` is floored at -1.0 (minimum σ ≈ 0.37). **γ = 0.995** (round-2 2026-04-19; was 0.99), giving a credit horizon of ~3.3 s to match the observation lookahead's ~2.6 s reach. Steering uses full `[-1, 1]` tanh output; throttle uses `0.5×(tanh+1)` remapping to `[0, 1]`. The model exposes two batched inference paths (`forward_actor_batch`, `forward_critic_batch`) both reading from the shared `BatchIo::obs_batch`, plus a `forward_critic` single-sample path for bootstrap values on the reward-collection and exit-flush paths. Action selection is **fully batched across all cars** — one mat-mat through the actor followed by one mat-mat through the critic, no per-car sequential forward. Training updates are **amortised across ticks** via `PreparedUpdate` and `ppo_epoch_system` — GAE is computed once, then `samples_per_tick` (default 32) samples are processed per tick. All mat-mat work routes through the active **GEMM backend** (scalar / matrixmultiply / accelerate, selected at compile time) via `src/brain/common/gemm_backend.rs`. Training uses **pre-allocated scratch buffers** (`BatchIo` for inputs + gradient seeds, `BatchScratch` for forward/backward intermediates, `SampleScratch` for critic-only single-sample forward) and **flat `Vec<f32>` weight storage** for cache-friendly traversal. Blocking flush on exit handles residual data.
- **PopArt value-target normalisation** (round-2, 2026-04-19). `PpoBrain` holds a `ValueNorm { mu, sigma }` state updated once per PPO update before any training chunk: batch mean/variance of GAE returns is blended into `(mu, sigma)` via EMA (β = 1e-4), then the POP rescale is applied to `c_value` weights and bias so externally-observed predictions `σ·z + µ` are preserved across the statistics change. The training loss regresses `c_value`'s raw output against the normalised target `(ret − µ) / σ`, so the critic targets a stationary ~N(0, 1) distribution regardless of return scale. Denormalisation `σ·raw + µ` happens at inference call sites (bootstrap `forward_critic`, action-selection `forward_critic_batch` value reads, on-exit flush). When `config.popart_enabled = false`, `ValueNorm` stays at `(0, 1)` and the pipeline is numerically equivalent to the pre-PopArt path. See `context/references/value-target-normalisation.md` for derivation.
- **Observation running mean/var normaliser** (round-2). `ObservationNormalizer` Resource applies Welford online per-dim mean/variance tracking in `build_observation_vector_system` after raw feature assembly. Pass-through during `warmup_samples = 1000`; afterwards centres and scales each of the 43 dims and clips to `[-10, 10]` (SB3 `VecNormalize.clip_obs` convention). Stats persist across episodes. When `enabled = false`, identity pass-through.
- **PolicyOutput** component: written by `ppo_act_all_cars_system` each tick, exposes `value_prediction`, `steering_mean`/`steering_std`, `throttle_mean`/`throttle_std`. Read by the analytics trace capture system for policy confidence metrics.
- **Analytics** has been comprehensively expanded: 16 tick-level trace fields (position, velocity decomposition, drift angle, min ray, velocity projection, centreline reward, policy confidence), 25 episode-level aggregates, a **crash classification system** (`CrashKind`: Slide, HeadOn, Overshoot, Spin, Stall), and a **15-section Markdown report** (sections 11–15 added 2026-04-19 for the round-2 diagnostic pass — pre-crash forensics, layer-health timeseries, PopArt µ/σ tracker, critic prediction-quality histogram, fleet variance). `PpoUpdateRecord` gained `return_min/mean/max/std`, `value_norm_mu/sigma`, `epochs_completed`, `early_stopped` (backwards-compatible via `#[serde(default)]`).
- The brain layer now owns **ranking logic** (`src/brain/ranking.rs`) in addition to PPO. A seeded `StdRng` lives in `PpoBrain` for deterministic policy sampling.
- **Debug overlays default to off** — geometry (F1), sensors (F2), and telemetry HUD (F3) all start disabled. The HUD is a compact 440px panel with blue accent palette, 72% opacity, condensed text lines (no wrapping), PPO-specific metrics (clip %, KL divergence), six-column quarter table, no legend. Leaderboard panel matches the updated colour scheme.
- **Profiling** is feature-gated behind `--features profiling`. When the feature is off, all profiling code is compiled out entirely — zero runtime cost. When enabled, the app auto-exits after a configurable duration (default 30s) and exports JSON + Markdown performance reports to `reports/json/performance/` and `reports/performance/` respectively. Both include a `RunContext` snapshot which records the active **GEMM backend** under a `### Build` section (so every perf artefact is self-documenting about what produced it). Per-system timing covers all 17 FixedUpdate systems via an `instrument!()` macro. Reports have retention limits (3 reports per directory; oldest auto-deleted).
- **GEMM backend** selection is compile-time via Cargo features. Default: Accelerate on macOS (via `cblas_sgemm`, dispatches to AMX coprocessor), `matrixmultiply` (pure-Rust BLIS NEON microkernel) elsewhere. `--features force-scalar` forces the naive reference kernel; `--features force-matrixmultiply` forces the portable Rust kernel on any platform; `--features force-accelerate` forces Apple Accelerate (macOS-only, compile-error elsewhere). Mutually exclusive — compile-error if two are set. macOS `main.rs` pins `VECLIB_MAXIMUM_THREADS=1` at startup to prevent Accelerate's internal thread pool from fighting Bevy for cores at our small matrix sizes.
- **Performance is no longer the constraint**. Post-2026-04-18 dual-backend + batched-actor work, mean frame time dropped from 15.7 ms to 0.735 ms (21× overall), PPO Epoch from 13.5 ms to 0.446 ms (30×), action selection from 1.98 ms to 0.126 ms (16×). Budget utilisation went from 94% to 4.4%. Zero stutters, 0% frames over budget. See `notes/performance-tuning-lessons.md` for the contributing factors.
- The project is in a **transitional architecture state**:
  - The repository intent targets brain-inspired local plasticity (Milestones 2–9).
  - The implemented learning path is a handwritten PPO baseline used to validate the environment and observation contract (Milestone 1). Cars are confirmed to learn — drifting corners observed.
- `README.md` is directionally accurate but its Milestone 1 checklist understates implementation reality — PPO, debug HUD, analytics export, velocity-projection reward, expanded observations, and crash classification are all live.
- The finish-line removal, random-spawn paradigm, velocity-projection reward, expanded observations, and analytics overhaul have all been **implemented**. The system is now in a coherent post-paradigm-shift state.
