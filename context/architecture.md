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
  - a live handwritten **PPO** brain (upgraded from A2C — clipped surrogate objective, multi-epoch updates amortised across ticks, **asymmetric architecture**: actor 2×64, critic 2×128),
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
│   │   │   ├── mod.rs                   # PpoBrain, PpoPlugin, PpoUpdateState, PolicyOutput component, act/collect/epoch/flush systems
│   │   │   ├── model.rs                 # ActorCritic: asymmetric MLP (actor 2×64, critic 2×128), BatchScratch pre-allocation, forward_actor/forward_critic/forward
│   │   │   ├── buffer.rs               # TrainerRolloutBuffer: env_id-tagged transitions + old_log_probs, per-env GAE
│   │   │   └── update.rs               # PreparedUpdate, ppo_process_chunk/ppo_finish_epoch, PPO clipped surrogate
│   │   ├── common/
│   │   │   ├── mod.rs
│   │   │   ├── mlp.rs                   # Handwritten Linear (flat weight storage), Tanh, orthogonal init, forward/backward + batched variants
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
├── Cargo.toml                           # Workspace root, bevy 0.18 + rand + serde dependencies
└── README.md                            # Project intent, brain-inspired learning vision, milestone roadmap
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
├── ppo_act_all_cars_system               (brain — mode-gated, writes ActionState.desired for all cars, writes PolicyOutput component, appends to rollout buffer with env_id + old log-prob)
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
├── ppo_epoch_system                     (brain — processes 64-sample chunk from prepared update, advances epoch state)
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
- The PPO update is **amortised across ticks** via `PreparedUpdate` and `ppo_epoch_system`. GAE is computed once at prepare time; samples are processed in 64-sample chunks across subsequent ticks. New transitions collect into a fresh buffer during the update.
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
- The brain uses **PPO** (upgraded from A2C): clipped surrogate objective (ε=0.2), 4 epochs per update. The network is an **asymmetric actor-critic** — actor 2×64, critic 2×128 — with **tanh activations**, **orthogonal weight initialisation** (√2 hidden, 0.01× policy head, 1.0× value head), and **per-minibatch advantage normalisation** with sample shuffling. The actor uses standard Adam (LR 3e-4); the critic uses **AdamW with weight decay λ=3e-4** (LR 5e-4) to prevent unbounded weight growth. `log_std` is floored at -1.0 (minimum σ ≈ 0.37). Steering uses full `[-1, 1]` tanh output; throttle uses `0.5×(tanh+1)` remapping to `[0, 1]`. The model exposes split forward paths: `forward_actor` (action selection), `forward_critic` (bootstrap values), and `forward` (full pass). Updates are **amortised across ticks** via `PreparedUpdate` and `ppo_epoch_system` — GAE is computed once, then 64 samples are processed per tick to avoid frame stutter. Training uses **batched forward/backward passes** with **pre-allocated scratch buffers** and **flat `Vec<f32>` weight storage** for cache-friendly traversal. Blocking flush on exit handles residual data.
- **PolicyOutput** component: written by `ppo_act_all_cars_system` each tick, exposes `value_prediction`, `steering_mean`/`steering_std`, `throttle_mean`/`throttle_std`. Read by the analytics trace capture system for policy confidence metrics.
- **Analytics** has been comprehensively expanded: 16 tick-level trace fields (position, velocity decomposition, drift angle, min ray, velocity projection, centreline reward, policy confidence), 25 episode-level aggregates, a **crash classification system** (`CrashKind`: Slide, HeadOn, Overshoot, Spin, Stall), and a 10-section Markdown report with auto-generated takeaways.
- The brain layer now owns **ranking logic** (`src/brain/ranking.rs`) in addition to PPO. A seeded `StdRng` lives in `PpoBrain` for deterministic policy sampling.
- **Debug overlays default to off** — geometry (F1), sensors (F2), and telemetry HUD (F3) all start disabled. The HUD is a compact 440px panel with blue accent palette, 72% opacity, condensed text lines (no wrapping), PPO-specific metrics (clip %, KL divergence), six-column quarter table, no legend. Leaderboard panel matches the updated colour scheme.
- **Profiling** is feature-gated behind `--features profiling`. When the feature is off, all profiling code is compiled out entirely — zero runtime cost. When enabled, the app auto-exits after a configurable duration (default 30s) and exports JSON + Markdown performance reports to `reports/json/performance/` and `reports/performance/` respectively. Both include a `RunContext` snapshot (same as analytics reports). Per-system timing covers all 17 FixedUpdate systems via an `instrument!()` macro. Reports have retention limits (3 reports per directory; oldest auto-deleted).
- The project is in a **transitional architecture state**:
  - The repository intent targets brain-inspired local plasticity (Milestones 2–9).
  - The implemented learning path is a handwritten PPO baseline used to validate the environment and observation contract (Milestone 1). Cars are confirmed to learn — drifting corners observed.
- `README.md` is directionally accurate but its Milestone 1 checklist understates implementation reality — PPO, debug HUD, analytics export, velocity-projection reward, expanded observations, and crash classification are all live.
- The finish-line removal, random-spawn paradigm, velocity-projection reward, expanded observations, and analytics overhaul have all been **implemented**. The system is now in a coherent post-paradigm-shift state.
