# NeuroDrive Architecture

## Scope / Purpose

- Provide the top-down structural map for the repository as it exists now.
- Orient a new engineer to the runtime layers, ownership boundaries, dependency direction, and the main fixed-tick execution flow.
- Keep detailed subsystem behaviour in `context/systems/` rather than duplicating it here.

## Repository Overview

- NeuroDrive is a Rust application built on **Bevy 0.18**, implementing brain-inspired online learning from first principles.
- The current runtime is a **deterministic 2D top-down racing environment** with:
  - a fixed 60 Hz simulation timestep,
  - a single hard-coded track,
  - a **multi-controller, multi-car vectorised trainer** — fleet composition is governed by `TrainerConfig.layout: TrainerLayout` with variants `Keyboard` / `AllPpo{count}` / `AllBrain{count}` / `SideBySide{ppo,brain}`. Default is `AllBrain{8}`.
  - **F4 cycles layouts** `AllBrain → SideBySide → AllPpo → AllBrain` (Keyboard is not in the cycle — it is a manual-intervention escape hatch reachable programmatically only).
  - per-car `Controller` enum Component + ZST markers (`PpoCar` / `BrainCar` / `KeyboardCar`) to enforce compile-time partitioning between learners,
  - all cars spawning at **random centreline positions** (re-randomised each reset, no privileged car),
  - throttle axis `[0, 1]` (coast to full thrust, no braking — drag is the sole deceleration mechanism),
  - a **velocity-projection reward** (dot of velocity onto centreline tangent) plus centreline proximity reward,
  - **43-dimensional observations** (rays, kinematics with v_forward/v_lateral split, speed_delta, 12-point lookahead with heading deltas + curvatures spanning 30–650 units, previous actions),
  - **two live learners running in parallel when the layout allows:**
    - a handwritten **PPO** actor-critic (M1–M5 diagnostic baseline — clipped surrogate objective, asymmetric 2×64 / 2×128, PopArt, γ=0.995, target-KL early stop, observation normaliser);
    - a **brain-inspired v1 learner** (M6 — sparse directed graph of rate-coded tanh neurons, three-factor plasticity with eligibility traces, raw-reward modulator, synaptic scaling + intrinsic excitability homeostasis, continual-backprop structural plasticity with plateau-triggered neurogenesis);
  - a **PolicyOutput** component per car whose field semantics depend on which learner drives the car (critic estimate vs modulator M),
  - a comprehensive analytics pipeline with **layout-aware Markdown reports** (up to 19 sections, content-gated per what actually ran — brain-only runs omit PPO-specific sections; side-by-side adds a Fleet Comparison section) exported to `reports/{json/,}analytics/run_<ts>_<slug>.md/json` where slug is `brain` / `side` / `ppo` / `keyboard`,
  - a debug HUD, world-space overlay layer, and live leaderboard panel.
- Episodes end on **crash or 30-second timeout only** — there is no finish line or lap concept.
- The project intent in `README.md` is biologically inspired local plasticity. As of M6 that substrate is live alongside the PPO diagnostic baseline; the empirical acceptance bar (visible brain learning relative to PPO in a ~2000-episode side-by-side run) is pending a real training run.
- `cargo check` and `cargo test` both pass — 133 tests green across default, `force-scalar`, and release builds.

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
│   │   ├── mod.rs                       # Re-exports: common / inspired / plugin / ppo / ranking / types
│   │   ├── plugin.rs                    # BrainPlugin: registers PpoPlugin + BrainInspiredPlugin + ranking/visual-role systems; F4 handler cycle_trainer_layout_system despawns+respawns cars on layout change, resets both learners' state
│   │   ├── types.rs                     # Controller enum Component (Keyboard/Ppo/Brain); PpoCar/BrainCar/KeyboardCar ZST marker components; PolicyOutput (controller-dependent field semantics)
│   │   ├── ppo/                         # PPO diagnostic baseline (M1–M5)
│   │   │   ├── mod.rs                   # PpoBrain (with act_entity_buffer scratch), PpoPlugin, PpoUpdateState, ppo_act_all_cars_system (3-pass batched, With<PpoCar>), collect/epoch/flush systems (all With<PpoCar>)
│   │   │   ├── model.rs                 # ActorCritic: asymmetric MLP (actor 2×64, critic 2×128), BatchIo + BatchScratch + SampleScratch (critic-only), forward_actor_batch/forward_critic_batch/forward_critic
│   │   │   ├── buffer.rs                # TrainerRolloutBuffer: env_id-tagged transitions + old_log_probs, per-env GAE via reusable EnvGrouping (Vec-indexed by env_id, deterministic iteration)
│   │   │   └── update.rs                # PreparedUpdate, ppo_process_chunk/ppo_finish_epoch, PPO clipped surrogate
│   │   ├── inspired/                    # Brain-inspired v1 learner (M6) — see systems/brain-inspired.md
│   │   │   ├── mod.rs                   # BrainInspiredPlugin, BrainBrain resource, BrainRunningStats, BrainUpdateRecord, BrainTrainingStats, brain_act_all_cars_system (With<BrainCar>), brain_learn_all_cars_system (With<BrainCar>), build_brain_update_record cadence flush
│   │   │   ├── config.rs                # BrainInspiredConfig — all dials tagged RESEARCH-ANCHORED or TUNE, plus enable_plasticity/homeostasis/structural ablation flags
│   │   │   ├── graph.rs                 # NeuronId/SynapseId, NeuronRole enum (Input/Hidden/Output), Neuron, Synapse (per-car eligibility Vec), BrainGraph (slot-stable storage + free-lists); seed-graph constructor
│   │   │   ├── forward.rs               # NeuronActivations per-car Component (prev/curr buffers); pure forward_tick (one-step propagation, prev-tick reads, tanh on hidden/output)
│   │   │   ├── plasticity.rs            # CarLearnSample<'a>, apply_plasticity_tick (three-factor rule), sample_plasticity_health diagnostic scan
│   │   │   ├── homeostasis.rs           # apply_synaptic_scaling (cadence, clamped to [0.5, 2.0]), update_intrinsic_homeostat (per tick, mean-rate EMA + bias nudge)
│   │   │   └── structural.rs            # update_utility_tick (CBP η_u=0.99), replace_low_utility, detect_plateau, grow_hidden_neuron, prune_synapses, sprout_synapses
│   │   ├── common/
│   │   │   ├── mod.rs
│   │   │   ├── gemm_backend.rs          # GEMM dispatch module — selects scalar/matrixmultiply/accelerate at compile time; backend_name() for profiling reports
│   │   │   ├── gemm_scalar.rs           # Naive nested-loop reference backend (force-scalar)
│   │   │   ├── gemm_matrixmultiply.rs   # Pure-Rust BLIS-style NEON microkernel via matrixmultiply crate
│   │   │   ├── gemm_accelerate.rs       # Apple Accelerate via cblas_sgemm (macOS; dispatches to AMX)
│   │   │   ├── mlp.rs                   # Handwritten Linear (flat weight storage) + Tanh; forward_batch/backward_batch route through gemm_backend; forward_into single-sample scalar path
│   │   │   ├── math.rs                  # Gaussian sampling, log-prob, tanh correction, orthogonal init utilities (reused by brain-inspired replacement resample)
│   │   │   └── optim.rs                 # AdamW optimiser with per-layer state and decoupled weight decay
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
│   ├── brain_inspired_pipeline.rs       # 21 tests across M6 stages S1–S6 — graph, forward, plasticity, homeostasis, structural, analytics, side-by-side
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
| **agent** | Stable action boundary (CarAction ↔ ActionState, throttle [0,1]), keyboard system (`With<KeyboardCar>`), policy observation contract (ObservationVector, 43 dims), Welford observation normaliser | game, maps, brain, debug | `src/agent/` |
| **brain** | Controller partitioning (per-car `Controller` enum + `PpoCar` / `BrainCar` / `KeyboardCar` ZST markers), F4 layout cycle (`cycle_trainer_layout_system`), the PPO baseline (clipped surrogate, amortised epochs; `With<PpoCar>`-filtered), the brain-inspired v1 learner (sparse graph + three-factor plasticity + homeostasis + structural plasticity; `With<BrainCar>`-filtered — see `systems/brain-inspired.md`), PolicyOutput per-car component, trainer live ranking, and car visual roles | agent, game, analytics | `src/brain/` |
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
├── keyboard_action_input_system          (agent — With<KeyboardCar> filter, writes ActionState.desired)
├── ppo_act_all_cars_system               (brain::ppo — With<PpoCar> filter; 3-pass batched: stacks obs into batch_io → single mat-mat actor + critic batched forwards → per-car sampling + PolicyOutput writes + rollout buffer push)
├── brain_act_all_cars_system             (brain::inspired — With<BrainCar> filter; per-car forward_tick over the shared BrainGraph; writes ActionState.desired + PolicyOutput.*_mean; increments BrainBrain.tick_counter)
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
├── ppo_collect_rewards_all_cars_system   (brain::ppo — With<PpoCar> filter; appends reward/done for PPO cars, prepares PPO update at horizon)
├── ppo_epoch_system                     (brain::ppo — processes one samples_per_tick chunk (default 32) from prepared update via active GEMM backend, advances epoch state)
├── brain_learn_all_cars_system           (brain::inspired — With<BrainCar> filter; pushes terminal-episode returns to reward_window; applies three-factor plasticity across all brain cars; runs intrinsic homeostat every tick; on structural_cadence (default 128): synaptic scaling, utility-based replacement, plateau-triggered neurogenesis, synapse prune/sprout; cadence flush of BrainUpdateRecord into BrainTrainingStats.history)
├── update_driving_hud_stats_system       (debug — updates live HUD values)
└── capture_driving_hud_episode_metrics_system  (debug — captures episode-end data for quarter summaries)

frame_end_system                          (profiling, feature-gated — captures frame end timestamp, records per-set durations)
auto_exit_system                          (profiling, feature-gated — exits app after configured duration)
```

### Update Schedule (every frame)

```text
├── cycle_trainer_layout_system           (brain — F4 cycles TrainerLayout (AllBrain → SideBySide → AllPpo → AllBrain); despawns all cars, resets PPO + brain state, respawns via spawn_cars_for_layout)
├── episode_tracker_system                (analytics — folds completed snapshots into EpisodeTracker; copies new BrainTrainingStats.history entries into brain_records; tags EpisodeRecord.controller from Option<&PpoCar>/&BrainCar/&KeyboardCar)
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
| `systems/brain-ppo.md` | PPO algorithm, model architecture, rollout buffer, ranking, ML primitives, controller partitioning (M6: `Controller` enum + ZST markers) | `src/brain/` |
| `systems/brain-inspired.md` | Brain-inspired v1 learner (M6) — graph topology, three-factor plasticity, homeostasis, structural plasticity, side-by-side coexistence with PPO | `src/brain/inspired/` |
| `systems/analytics.md` | Capture pipeline, derived metrics, two-tier export, crash classification, brain records + layout-aware filenames + section gating (M6) | `src/analytics/` |
| `systems/profiling.md` | Feature-gated per-system timing, auto-exit, performance reports | `src/profiling/` |
| `systems/debug.md` | Live overlays, HUD panel, leaderboard | `src/debug/` |
| `systems/determinism.md` | Cross-cutting: ordering contract, reproducibility surfaces, RNG state | `src/sim/`, cross-cutting |

## Inter-System Relationships

Individual system files cover their own boundaries. This section is the canonical home for the relationships **between** systems — the connections that matter when reasoning about blast radius or change impact. Each entry names the two sides, the mechanism, the data that flows, and what breaks if the connection is violated.

| A | B | Mechanism | Data | What breaks if broken |
|---|---|-----------|------|-----------------------|
| `game` | `agent` | Read-only: `Car`, `ActionState.applied`, `TrackProgress`, centreline projection | Physics state + projection truth | Observations become stale or wrong-tick; post-reset observations leak crash state |
| `agent` | `game` | Write path: `ActionState.applied` consumed by `car_physics_system` | Steering + throttle | Physics stops executing policy decisions |
| `agent` | `brain::ppo` | Read-only: `ObservationVector` (43-dim) consumed by `ppo_act_all_cars_system` via `(With<Car>, With<PpoCar>)` filter | Normalised observation tensor | Any change in `OBSERVATION_DIM` constant or feature ordering desynchronises the model (dim mismatch panics on `forward_actor`); no runtime dimension assertion beyond shared `const OBSERVATION_DIM` |
| `agent` | `brain::inspired` | Read-only: `ObservationVector` (43-dim) consumed by `brain_act_all_cars_system` via `(With<Car>, With<BrainCar>)` filter; written into input-neuron activations by `forward_tick` | Same observation tensor as PPO — contract is stable across learners | Any change to `OBSERVATION_DIM` simultaneously breaks both learners (graph's `input_neurons` Vec length is `config.obs_dim`) |
| `brain::ppo` | `agent` | Write path: writes per-car `ActionState.desired` and `PolicyOutput` component (With<PpoCar> cars only) | Steering + throttle means/stds + critic value prediction | Smoothing + physics receive stale or default actions on PPO cars |
| `brain::inspired` | `agent` | Write path: writes per-car `ActionState.desired` and `PolicyOutput` (With<BrainCar> cars only); `PolicyOutput.value_prediction` carries per-car modulator M (not a critic estimate) | Steering + throttle activations; raw tick reward M | Brain cars stop moving or PolicyOutput semantics leak into PPO analytics paths |
| `game` | `brain::ppo` | `EpisodeState.tick.reward` + `tick.end_reason` consumed by `ppo_collect_rewards_all_cars_system` | Per-tick reward + terminal flag | PPO mis-aligns reward-to-observation; GAE becomes invalid |
| `game` | `brain::inspired` | `EpisodeState.tick.reward` = modulator M; `EpisodeState.tick.end_reason` zeros per-car eligibility; `EpisodeState.last.return_sum` pushed to reward_window on terminal for plateau detection | Per-tick reward, terminal flag, per-episode return sum | Brain learns wrong correlations (bad modulator); eligibility leaks across resets (bad terminal signal); plateau detector never fires (no reward_window entries) |
| `brain` (both) | `analytics` | `PolicyOutput` per-car component read by `capture_episode_tick_trace_system`; `PpoTrainingStats` read by `episode_tracker_system`; `BrainTrainingStats.history` also read by `episode_tracker_system` and copied to `EpisodeTracker.brain_records`; `PpoCar` / `BrainCar` / `KeyboardCar` markers read by `episode_tracker_system` to set `EpisodeRecord.controller` | Value/modulator, policy confidences, PPO update diagnostics, brain cadence snapshots, controller identity | Trace records miss policy stats; Markdown report's "What Does the Car Think" section becomes meaningless; brain sections (16–18) miss data; Fleet Comparison (19) cannot segment by controller |
| `game` | `analytics` | `EpisodeState` (current and finalised), `Collided` marker, `TrackProgress` consumed by per-car capture systems | Episode summary + reward decomposition | Episode records desynchronise with env truth; crash classification becomes unreliable |
| `maps` | `game` | `Track { grid, centerline }` singleton consumed by physics (via progress), collision, and episode reset (spawn RNG draws centreline fractions) | Spatial truth (grid occupancy + arc-length parametrisation) | No collisions, no spawn, no progress — runtime fails before first tick |
| `maps` | `agent` | `TrackGrid` consumed by raycast marching; `TrackCenterline` consumed by lookahead features | Grid occupancy + `tangent_at_s` | Sensor readings collapse; lookahead curvature vanishes |
| `sim` | all fixed-update subsystems | `SimSet` ordering contract: `Input → Physics → Collision → Measurement`, configured by `GamePlugin` | Schedule sets, not data | Any plugin placing systems outside `SimSet` creates silent ordering bugs (e.g., observations built from pre-reset state); only the four-stage chain guarantees the reward/observation alignment that PPO depends on |
| `analytics::exporters::{cleanup, context}` | `profiling::exporters::json` | Direct `use crate::analytics::exporters::{cleanup::enforce_retention, context::RunContext}` | `RunContext` struct (full run config snapshot) + retention-limited directory pruning | Profiling report lose their run-context header and unbounded report directories accumulate; see Shared Infrastructure below |
| `brain::ranking` | `debug::leaderboard` | `TrainerLiveRanking` resource + per-car `CarColour` component | Ranked car order + colour swatches | Leaderboard panel goes blank or shows stale ordering |
| `brain::ranking` | `debug::hud` | Same `TrainerLiveRanking` read by `update_driving_hud_text_system` to pick the "best car" view (falls back to first car if unavailable) | Best car index | HUD silently shows first car instead of best — not fatal but misleading |
| `brain::plugin` | `game::plugin` | `cycle_trainer_layout_system` (in `brain::plugin`) calls `spawn_cars_for_layout` (in `game::plugin`) on every F4 press; also mutates `TrainerConfig.layout` via `set_layout` | Fleet-composition Resource + spawn helper | F4 toggle stops working, or spawn loses layout-awareness — side-by-side mode becomes unreachable |
| `brain::inspired` (BrainBrain) | `brain::plugin` | `cycle_trainer_layout_system` calls `brain_brain.reset_to_seed(num_cars)` to rebuild the graph cleanly when layout changes | Graph, RNG, reward_window reset | Stale eligibility / weights bleed across F4 transitions |
| `brain` (types) | `game` + `brain::ppo` + `brain::inspired` + `analytics` + `agent` | `Controller` enum Component + `PpoCar` / `BrainCar` / `KeyboardCar` ZST markers are **the** controller partitioning contract. Attached once at spawn; read via `With<>` query filters everywhere | Compile-time controller identity | Any system that forgets to filter would accidentally drive or observe the wrong car-set — the compiler prevents this by construction |

### Shared Infrastructure: RunContext and Retention Cleanup

The profiling subsystem and the analytics subsystem both export reports, and both:

- capture an identical `RunContext` snapshot (car count, PPO hyperparameters, reward coefficients, observation layout) via `analytics::exporters::context::RunContext::capture()`,
- enforce a retention limit of 3 reports per directory via `analytics::exporters::cleanup::enforce_retention()`.

These helpers live in `analytics::exporters` and are imported by `profiling::exporters::json`. This is deliberate shared infrastructure, not parallel evolution — but it does mean the profiling feature has a **compile-time dependency on the analytics module** even though nothing in `systems/profiling.md`'s boundaries section makes that obvious from a quick read. If analytics were ever ripped out or relocated, the feature-gated profiling pipeline would fail to compile even with `--features profiling` enabled.

### Dependency Chain Trace — One PPO Training Tick (end-to-end)

The single operation that crosses the most system boundaries is one fixed tick during PPO training (`TrainerLayout::AllPpo` or the PPO half of `SideBySide`, rollout near horizon). Tracing it names the full blast radius that any change to the tick pipeline must respect.

```text
Step  Owner / System                                  Reads                       Writes
────  ──────────────────────────────────────────────  ──────────────────────────  ──────────────────────────────
 1    profiling::frame_start_system (feature-gated)   —                           FrameRecord.frame_start
 2    agent::keyboard_action_input_system             KeyCode                     — (With<KeyboardCar> filter empty in AllPpo — iterates zero cars)
 3    brain::ppo_act_all_cars_system                  ObservationVector per car,  batch_io.obs_batch stacked,
                                                      model (3-pass batched:      scratch.a_out (batched means),
                                                      forward_actor_batch then    scratch.c_out (RAW normalised values),
                                                      forward_critic_batch),      PolicyOutput.value_prediction
                                                      PpoBrain.rng,               (DENORMALISED via value_norm.σ·z+µ),
                                                      PpoBrain.value_norm         ActionState.desired (per car),
                                                                                  TrainerRolloutBuffer push (state +
                                                                                  latent_action + action + old_log_prob
                                                                                  + denormalised value + env_id)
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
14    agent::build_observation_vector_system          SensorReadings,             ObservationVector (per car —
                                                      ObservationNormalizer         centred/scaled/clipped after warmup,
                                                                                    identity pass-through during first
                                                                                    1000 samples; stats updated in-place)
15    analytics::capture_episode_tick_trace_system    EpisodeState, PolicyOutput, PerCarTraceAccumulators push
                                                      SensorReadings, Transform
16    analytics::snapshot_completed_episode_*_system  PerCar*Accumulators,        EpisodeTracker.pending_episodes /
                                                      EpisodeState (terminal)     pending_traces on terminal
17    brain::ppo_collect_rewards_all_cars_system      EpisodeState.current_tick_* TrainerRolloutBuffer
                                                      (per car),                   reward + done push;
                                                      PpoBrain.value_norm         may call ppo_prepare_update()
                                                      (for bootstrap              at horizon → PreparedUpdate;
                                                      denormalisation)             popart_absorb_batch runs here, before
                                                                                  first training chunk, and applies the
                                                                                  POP rescale to c_value weights
18    brain::ppo_epoch_system                         PreparedUpdate,             model weights + Adam state
                                                      model (forward_batch +      PpoTrainingStats (on epoch end),
                                                      backward_batch),            early_stopped flag (when
                                                      PpoConfig.target_kl          approx_kl > 1.5 × target_kl halts
                                                                                  remaining epochs)
19    debug::update_driving_hud_stats_system          TrackProgress, Collided     DrivingHudStats
20    debug::capture_driving_hud_episode_metrics      EpisodeState (terminal)     DrivingHudHistory
21    profiling::frame_end_system + auto_exit         SystemTimers.durations_us   FrameRecord completed
```

**Failure semantics along the chain:**

- **Step 3 → Step 17 alignment.** The PPO buffer push in step 3 records (state, action, old_log_prob). Step 17 pushes the matching (reward, done). Any step between them that mutates `EpisodeState` out of order or runs in the wrong `SimSet` desynchronises reward from its generating action — silent training corruption. This is why `sim::SimSet` is structural rather than decorative.
- **Step 12 reset ordering.** Episode reset happens in `episode_loop_system` (step 12), **before** sensor readings rebuild (step 13). This is deliberate: if the order were reversed, the first observation of a new episode would be built from pre-reset crash state and the PPO rollout would bootstrap from a lie.
- **Step 11 before Step 12.** Progress projection must run before episode logic because crash classification and velocity-projection reward both read `TrackProgress`. Swapping them gives a zero reward on the terminal tick.
- **Step 18 amortisation.** `ppo_epoch_system` processes only `samples_per_tick` (default 32 as of 2026-04-18) samples of the prepared update per tick. A full 4-epoch update over a 512-sample rollout takes `4 × 512 / 32 = 64` ticks. During these 64 ticks, a new rollout buffer is collecting — PPO is **not on-policy in the strict sense** during amortised updates, but the `old_log_prob` captured at step 3 protects ratio calculation regardless.
- **Step 17 PopArt placement.** `popart_absorb_batch` runs inside `ppo_prepare_update`, **after** GAE computes `returns` but **before** any training chunk touches `c_value`. This ordering matters: the POP rescale must complete before the first forward pass sees the updated `c_value` weights, otherwise early chunks train against the old scale and the subsequent ones against the new scale — corrupting the update.
- **Step 3 denormalisation asymmetry.** Inside `brain::ppo_act_all_cars_system`, the critic's `c_out[i]` is raw (normalised) but the `value` written to `PolicyOutput` and pushed to the buffer is denormalised via `value_norm.σ·z + µ`. This asymmetry is load-bearing: GAE in step 17 consumes buffer values in reward units, and analytics in steps 15–16 read `PolicyOutput.value_prediction` in reward units. If either callsite forgot to denormalise, GAE would compute advantages against normalised-scale bootstrap values (silent training corruption) or the analytics report would show values bounded to ~(-3, +3) (meaningless).
- **Step 9 Collided as a marker not an event.** The environment never fires Bevy events for collisions; `Collided` is added as a component marker and read by the episode system two steps later. Any system that needs to react to collisions must be placed **in `SimSet::Measurement` after `episode_loop_system`** to see it before it is cleared on reset.

### Dependency Chain Trace — One Brain-Inspired Training Tick (end-to-end)

The parallel trace for `TrainerLayout::AllBrain` (or the brain half of `SideBySide`). Shares the environment path with PPO; diverges in steps 3 and 17. Steps marked `(— With<PpoCar> empty)` iterate zero cars when no PPO cars exist in the layout; in `SideBySide` they run on the PPO half only.

```text
Step  Owner / System                                  Reads                       Writes
────  ──────────────────────────────────────────────  ──────────────────────────  ──────────────────────────────
 1    profiling::frame_start_system (feature-gated)   —                           FrameRecord.frame_start
 2    agent::keyboard_action_input_system             KeyCode                     — (With<KeyboardCar> empty in AllBrain)
 3a   brain::ppo_act_all_cars_system                  —                           — (With<PpoCar> empty in AllBrain; runs on PPO half in SideBySide)
 3b   brain::brain_act_all_cars_system                ObservationVector per car,  NeuronActivations.prev ← curr,
                                                      NeuronActivations (prev),   input-neuron curr ← observation,
                                                      BrainGraph (shared),        hidden/output curr = tanh(bias + Σ prev·w),
                                                      BrainBrain.tick_counter     ActionState.desired (per car),
                                                                                  PolicyOutput.*_mean (raw output activations);
                                                                                  BrainBrain.tick_counter += 1
 4    agent::action_smoothing_system                  ActionState.desired         ActionState.applied (per car)
 5    profiling::input_end_system                     —                           FrameRecord.input_end
 6    game::car_physics_system                        ActionState.applied, Car    Car.velocity, Transform
 7    analytics::capture_episode_action_stats_system  ActionState.applied         PerCarActionAccumulators entry
 8    profiling::physics_end_system                   —                           FrameRecord.physics_end
 9    game::collision_detection_system                Transform, TrackGrid        Collided marker (add/remove)
10    profiling::collision_end_system                 —                           FrameRecord.collision_end
11    game::update_track_progress_system              Transform, TrackCenterline  TrackProgress (per car)
12    game::episode_loop_system                       TrackProgress, Collided,    EpisodeState.tick.*, SpawnRng on reset,
                                                      Car.velocity, EpisodeConfig Transform + velocity reset on terminal
13    agent::update_sensor_readings_system            Transform, TrackGrid,       SensorReadings (post-reset state)
                                                      TrackCenterline
14    agent::build_observation_vector_system          SensorReadings,             ObservationVector (per car)
                                                      ObservationNormalizer
15    analytics::capture_episode_tick_trace_system    EpisodeState, PolicyOutput, PerCarTraceAccumulators push
                                                      SensorReadings, Transform
16    analytics::snapshot_completed_episode_*_system  PerCar*Accumulators,        EpisodeTracker.pending_episodes /
                                                      EpisodeState (terminal)     pending_traces on terminal
17a   brain::ppo_collect_rewards_all_cars_system      —                           — (With<PpoCar> empty in AllBrain)
17b   brain::brain_learn_all_cars_system              EpisodeState.tick.reward,   Synapse.eligibility[car] updates (all cars),
                                                      EpisodeState.tick.end_reason,Δw accumulated and applied to shared weights,
                                                      EpisodeState.last.return_sum,reward_window push on terminal,
                                                      NeuronActivations (prev/curr),PolicyOutput.value_prediction ← M per car,
                                                      BrainGraph, BrainBrain.rng, Neuron.mean_rate EMA + bias nudge (every tick),
                                                      BrainBrain.tick_counter,    on cadence: Σ|w_in| scaling,
                                                      BrainBrain.config,          utility EMA update,
                                                      BrainBrain.reward_window    replacement + plateau neurogenesis + prune + sprout,
                                                                                  BrainTrainingStats.history push + counter reset
18    brain::ppo_epoch_system                         PreparedUpdate (only when   model weights + Adam state (when applicable)
                                                      PPO cars contributed)       — idle in AllBrain
19    debug::update_driving_hud_stats_system          TrackProgress, Collided     DrivingHudStats
20    debug::capture_driving_hud_episode_metrics      EpisodeState (terminal)     DrivingHudHistory
21    profiling::frame_end_system + auto_exit         SystemTimers.durations_us   FrameRecord completed
```

**Brain-mode failure semantics:**

- **Step 3b buffer rotation timing.** `forward_tick` rotates `prev ← curr` at the *start* of the forward pass. This means when step 17b reads `NeuronActivations.prev`, it sees the previous tick's source activations — exactly the "pre-before-post" semantics the three-factor rule requires. Reversing the rotation (e.g. rotating at end instead of start) would break the eligibility update and silently produce wrong learning.
- **Step 17b cadence ordering.** Per-tick order inside the learn system is: plasticity → homeostasis → structural → stats flush. If structural runs before plasticity, the replacement would happen against a graph that has not absorbed this tick's reward. If homeostasis runs before plasticity, bias updates would fight the just-applied weight step. The current order is load-bearing.
- **Step 17b field-destructure pattern.** Structural operations need simultaneous `&mut` access to `graph`, `rng`, `stats`, and `reward_window`. The learn system uses a `let BrainBrain { graph, rng, stats: brain_stats, reward_window, .. } = &mut *brain;` destructure to get disjoint borrows. Any change that refactors this has to preserve the same borrow-split or compilation fails.
- **Step 17b `PolicyOutput.value_prediction` semantics.** In brain mode this field carries the **per-car modulator M (raw tick reward)**, not a critic estimate. Analytics uses `Option<&PpoCar>` / `Option<&BrainCar>` markers to discriminate — `capture_episode_tick_trace_system` reads the same PolicyOutput field but interprets it differently downstream. Any code that reads `value_prediction` without respecting this dual semantics becomes a silent interpretation bug.
- **Step 17a/17b layout gating.** In `SideBySide` both 17a and 17b fire — PPO runs on its 8 cars, brain-inspired on its 8. The PPO rollout buffer's `env_ids` column contains only PPO env_ids by construction (query filter partitions at compile time). Brain-inspired's `reward_window` contains only brain-car returns.
- **Step 3b / 17b same-tick pairing.** Unlike PPO where `old_log_prob` protects the ratio even across amortised updates, brain-inspired plasticity applies weight changes *within the same tick* as it observes `pre`/`post`. There is no amortisation. This is cheaper but also means any reordering between steps 3b and 17b that decorrelates the activation snapshot from the reward corrupts learning.

## Coverage (Knowledge Gaps from the 2026-04-19 Upkeep Passes)

This section is explicit about where upkeep passes relied on direct code inspection versus inference from existing documentation and the scan-tool output, so the next session knows what still needs verification.

### Second pass — 2026-04-19 (post-M6)

The second upkeep pass this date ran after M6 (brain-inspired v1) shipped — six staged commits `6237aa7..c64ce9b` + wrap `4c5c7c5` + default/analytics fix `3a737d9`. Updates focused on capturing the controller-partitioning migration (AgentMode → per-car markers + TrainerLayout), the new `brain::inspired` module, analytics integration (BrainUpdateRecord, layout slug, section gating), and the Fleet Comparison side-by-side diagnostic.

**Directly inspected this pass:**

- `src/brain/inspired/*` in full — config.rs (every dial tagged), graph.rs (slot-stable Vec + free-lists), forward.rs (NeuronActivations + forward_tick), mod.rs (BrainBrain, BrainInspiredPlugin, act/learn systems, build_brain_update_record flush), plasticity.rs (apply_plasticity_tick + sample_plasticity_health), homeostasis.rs (apply_synaptic_scaling + update_intrinsic_homeostat), structural.rs (utility/replace/plateau/grow/prune/sprout).
- `src/brain/types.rs` — Controller enum + ZST markers (PpoCar/BrainCar/KeyboardCar); verified field semantics on PolicyOutput.
- `src/brain/plugin.rs` — cycle_trainer_layout_system full body; F4 cycle order, despawn+respawn, reset paths.
- `src/brain/ppo/mod.rs` relevant deltas — four systems now filter via (With<Car>, With<PpoCar>); Res<AgentMode> parameter removed; early-exit gate removed.
- `src/agent/action.rs` — keyboard_action_input_system now filters With<KeyboardCar>; env_id==0 special case removed.
- `src/game/car.rs` in full — TrainerLayout enum + variants + default + next + total_cars + ppo_count + brain_count + slug; TrainerConfig::set_layout sync; car_colour_warm / car_colour_cool; spawn_car now attaches Controller + marker.
- `src/game/plugin.rs` — spawn_cars_for_layout (shared between PostStartup setup_game and F4 respawn).
- `src/analytics/models.rs` — EpisodeTracker.brain_records + last_recorded_brain_records; CompactRunExport.brain_records; RunMetadata.layout + ppo_cars + brain_cars; EpisodeRecord.controller (all #[serde(default)]).
- `src/analytics/trackers/episode.rs` — episode_tracker_system gains Option<Res<BrainTrainingStats>> + per-car marker reads; idempotent brain_records sync.
- `src/analytics/exporters/{json.rs, markdown.rs}` — filename slug, section gating on PPO-only sections 9/12/13/14, brain sections 16–18, Fleet Comparison section 19.
- `src/analytics/plugin.rs` — RunMetadata construction populates layout/ppo_cars/brain_cars; filename includes slug.
- `src/analytics/metrics/{consistency, diagnostics, phases}.rs` — test fixtures for new EpisodeTracker / EpisodeRecord fields.
- `tests/brain_inspired_pipeline.rs` in full — 21 tests across S1–S6.
- Full M6 git commit bodies — 613 lines across `9bd2407..3a737d9` inspected for rationale not already captured.
- Grep over `src/brain/inspired/` for `WHY|HACK|IMPORTANT|SAFETY|NOTE:|FIXME|TODO` — zero matches (rationale lives in rustdoc `///` comments + `notes/brain-v1-decisions.md`).
- `context/notes/brain-v1-decisions.md` — 20+3 = 23 decisions, all grounded in the code and commit history inspected above.

**Trusted from M6 commits without re-inspection:**

- `src/brain/common/*` — unchanged by M6; trusted from the 2026-04-18 performance pass.
- `src/profiling/*` — unchanged.
- `src/maps/*` — unchanged.
- `src/debug/*` — unchanged by M6; HUD/leaderboard work in side-by-side by reading PolicyOutput unchanged, but live HUD column split is explicitly deferred (decisions note, outstanding follow-ups).
- `src/game/{episode, physics, collision, progress}.rs` — unchanged by M6.
- `src/sim/*` — unchanged.

**Not re-inspected (first-pass coverage still stands):**

- `src/analytics/metrics/{chunking, sectors, sparkline, stats, timeseries, trajectory, turns, pre_crash}.rs` — unchanged by M6; trusted from the first 2026-04-19 upkeep pass.

**Known gaps still outstanding:**

- User-controllable init seed — still weak (both `SpawnRng` and `PpoBrain.rng` seed from `rand::rng()` at startup; `BrainInspiredConfig.rng_seed` accepts a `Some(u64)` but is not yet wired through a user-facing config source).
- Headless training mode — still missing; becomes more valuable now that brain-mode training runs are the primary workflow.
- ECS-level replay harness — still missing.
- HUD column split in side-by-side — analytics side is done; live HUD still shows single-column PPO stats (decisions note outstanding follow-up).
- Tuning sweep over `TUNE`-flagged brain dials (η, ρ, structural_cadence, plateau_window, prune_threshold, sprout_probability, …) — pending empirical data from a real training run.
- **First real brain-mode training run** to hit the M6 acceptance bar (visible learning relative to PPO) — has not happened yet. Infrastructure is ready; the user is running training while this upkeep pass runs.

### First pass — 2026-04-19 (round-2 PPO critic target-scaling)

The first upkeep pass this date covered the PPO round-2 changes (PopArt, γ=0.995, target-KL early stop, observation normaliser).

**Directly inspected this session:**

- `src/brain/ppo/model.rs` in full — `ActorCritic`, `BatchIo`, `BatchScratch`, `SampleScratch`, the asymmetric actor/critic layout, and the denormalisation call sites.
- `src/brain/ppo/update.rs` in full — PopArt `popart_absorb_batch` implementation, the loss computation with normalised target, target-KL early stop integration path, `return_distribution` helper.
- `src/brain/ppo/mod.rs` in full — `PpoConfig` (with new `target_kl`, `popart_*` fields), `ValueNorm` struct + methods, `ppo_epoch_system` with KL early-stop branch, `ppo_act_all_cars_system` denormalisation, bootstrap denormalisation paths.
- `src/brain/ppo/buffer.rs` in full — no round-2 changes here; GAE computation unchanged.
- `src/agent/observation.rs` in full — `ObservationNormalizer` Welford state, integration into `build_observation_vector_system`, warmup + clip + disable flag.
- `src/agent/plugin.rs` in full — verified `ObservationNormalizer` resource registration.
- `src/analytics/models.rs` in full — `PpoUpdateRecord` with round-2 fields and `#[serde(default)]` back-compat.
- `src/analytics/trackers/episode.rs` in full — stats → record mapping wiring for new fields.
- `src/analytics/exporters/markdown.rs` — the five new Markdown sections (11–15) for round-2 diagnostics.
- `src/analytics/metrics/pre_crash.rs` in full (new module).
- `src/analytics/metrics/diagnostics.rs` (test-fixture construction — relevant to the back-compat story).
- `src/brain/common/mlp.rs` (first 80 lines — the `Linear` layer definition relevant to POP rescale).
- Grep passes: `WHY|HACK|IMPORTANT|SAFETY|FIXME`, `TODO|NOTE:`, `unsafe`, `debug_assert|StdRng|seed_from|EventWriter|EventReader|#[cfg(feature` across all of `src/`.
- Full context/ inventory: 30 context files scanned; 7 system files spot-checked; 10 references catalogued.

**Trusted from recent commits without re-inspection (commits `a0b2cb6`, `e86e737`):**

- `systems/brain-ppo.md` — updated in `a0b2cb6` to reflect PopArt + target-KL; hyperparameters table now shows γ=0.995, `target_kl=Some(0.03)`, `popart_*` fields. Verified freshness via grep; content stands.
- `systems/analytics.md` — updated in `a0b2cb6` with sections 11–15 and the `PpoUpdateRecord` field expansion subsection. Verified; content stands.
- `systems/agent-interface.md` — updated in `a0b2cb6` with the `ObservationNormalizer` paragraph. Verified.
- `systems/determinism.md` — updated in this pass to add the three round-2 session-deterministic surfaces (observation normaliser, PopArt, target-KL early stop).
- `systems/environment.md`, `systems/debug.md`, `systems/profiling.md` — no round-2 impact on these subsystems; trusted from the 2026-04-18 pass.
- `reports/analytics/run_1776556719.md` — the 2,271-episode validation run — trusted from conversation-level inspection without re-reading the file during upkeep; all headline claims are already in the Structural Notes above.

**Not inspected this pass:**

- `src/analytics/metrics/chunking.rs`, `consistency.rs`, `sectors.rs`, `trajectory.rs`, `turns.rs`, `phases.rs`, `sparkline.rs`, `stats.rs`, `timeseries.rs` — trusted from prior passes; no round-2 changes to these modules.
- `src/profiling/*` — unchanged since the 2026-04-18 performance work.
- `src/maps/*` — no round-2 changes; trusted from prior passes.
- `src/debug/*` — no round-2 changes; trusted.
- `src/game/car.rs`, `physics.rs`, `episode.rs`, `collision.rs`, `progress.rs` — no round-2 changes; trusted.
- `src/sim/*` — unchanged.

**Known gap items still outstanding:**

- User-controllable PPO seed — still not implemented; `systems/determinism.md` flags this correctly.
- Headless training mode — still missing; flagged in `systems/environment.md` and `systems/brain-ppo.md`. Becomes more valuable as the brain-inspired phase begins (longer experiments, model comparison).
- Analytics/profiling parallel evolution hazard — documented under Shared Infrastructure; this remains a live coupling to watch.
- ECS-level replay harness — still missing; listed as "Future Work" in determinism.md.

## Structural Notes / Current Reality

- The codebase is **not** environment-only. PPO, brain-inspired v1, analytics, and the debug HUD are live and substantial subsystems. Documentation treating any of them as roadmap-only is obsolete.
- **Singleton-car and single-controller assumptions are both removed.** `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, `PolicyOutput`, `NeuronActivations`, `Controller`, and the three ZST marker components (`PpoCar` / `BrainCar` / `KeyboardCar`) are all per-car Components. `CollisionEvent` is a `Collided` marker. All fixed-tick systems iterate over multiple cars with appropriate query filters.
- The runtime is a **multi-controller, multi-car vectorised trainer**:
  - `TrainerConfig.layout: TrainerLayout` controls fleet composition — variants `Keyboard` (1 car) / `AllPpo{count}` / `AllBrain{count}` / `SideBySide{ppo, brain}`. Default `AllBrain{8}`. `TrainerConfig.num_envs` is kept in sync with `layout.total_cars()` for back-compat.
  - F4 cycles `AllBrain → SideBySide{8,8} → AllPpo → AllBrain` (Keyboard excluded from the cycle — manual-intervention escape hatch only).
  - All cars spawn at **random centreline positions** (re-randomised on each episode reset). Single-controller layouts use the 25-colour palette; side-by-side uses an 8-entry warm palette (reds/oranges/yellows) for PPO and an 8-entry cool palette (blues/teals/cyans) for brain cars.
  - Per-car components: `EnvInstanceId`, `CarColour`, `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, `PolicyOutput`, `NeuronActivations` (lazy-sized), `Controller` + exactly one of the three marker ZSTs.
  - One shared `TrainerRolloutBuffer` collects transitions from **PPO cars only** (enforced by `With<PpoCar>` query filter) with `env_id` tagging and old log-probs; GAE is computed per-env.
  - One shared `BrainGraph` (inside `BrainBrain`) — all brain cars are embodiments of this single graph; per-car activations and eligibility traces live on `NeuronActivations` Components and on `Synapse.eligibility[car]` respectively. 8 weight updates per tick are summed into the shared weights.
  - A `TrainerLiveRanking` resource tracks best/worst car with hysteresis; `ranking.rs` assigns visual highlight roles.
  - A live leaderboard panel (top-right, F3-toggled) shows per-car performance with colour swatches.
- **Episode semantics**: there is no finish line or lap concept. Progress is **cumulative forward arc-length from spawn** with wrap handling. Episodes end on **crash or 30-second timeout** only. `EpisodeEndReason` has `Crash` and `Timeout` variants (no `LapComplete`).
- **Physics**: `rotation_speed` is 8.0 rad/s. The throttle axis is `[0, 1]` — 0 coasts (drag decelerates naturally), 1 is full thrust. Braking was tried and reverted because the policy converged to "mostly brake" as a safe local optimum.
- **Reward**: per-tick velocity projection reward — `dot(velocity, tangent) / speed_reference × velocity_reward_scale` — plus a centreline proximity reward (`centreline_reward_coef`, `centreline_reward_max_distance`). Crash penalty is 0.0. `EpisodeConfig` carries `velocity_reward_scale`, `centreline_reward_coef`, `centreline_reward_max_distance` (not `progress_reward_scale`).
- **Observations** (43 dimensions): rays (11), v_forward + v_lateral, speed_delta, centreline offset/heading/curvature, 12-point lookahead (heading deltas + curvatures, 30–650 units, dense near / sparse far), previous_steering, previous_throttle.
- The brain uses **PPO** (upgraded from A2C): clipped surrogate objective (ε=0.2), 4 epochs per update (with **target-KL early stop** — halts epochs when `approx_kl > 1.5 × target_kl`, default 0.03). The network is an **asymmetric actor-critic** — actor 2×64, critic 2×128 — with **tanh activations**, **orthogonal weight initialisation** (√2 hidden, 0.01× policy head, 1.0× value head), and **per-minibatch advantage normalisation** with sample shuffling. The actor uses standard Adam (LR 3e-4); the critic uses **AdamW with weight decay λ=3e-4** (LR 5e-4). `log_std` is floored at -1.0 (minimum σ ≈ 0.37). **γ = 0.995** (round-2 2026-04-19; was 0.99), giving a credit horizon of ~3.3 s to match the observation lookahead's ~2.6 s reach. Steering uses full `[-1, 1]` tanh output; throttle uses `0.5×(tanh+1)` remapping to `[0, 1]`. The model exposes two batched inference paths (`forward_actor_batch`, `forward_critic_batch`) both reading from the shared `BatchIo::obs_batch`, plus a `forward_critic` single-sample path for bootstrap values on the reward-collection and exit-flush paths. Action selection is **fully batched across all cars** — one mat-mat through the actor followed by one mat-mat through the critic, no per-car sequential forward. Training updates are **amortised across ticks** via `PreparedUpdate` and `ppo_epoch_system` — GAE is computed once, then `samples_per_tick` (default 32) samples are processed per tick. All mat-mat work routes through the active **GEMM backend** (scalar / matrixmultiply / accelerate, selected at compile time) via `src/brain/common/gemm_backend.rs`. Training uses **pre-allocated scratch buffers** (`BatchIo` for inputs + gradient seeds, `BatchScratch` for forward/backward intermediates, `SampleScratch` for critic-only single-sample forward) and **flat `Vec<f32>` weight storage** for cache-friendly traversal. Blocking flush on exit handles residual data.
- **PopArt value-target normalisation** (round-2, 2026-04-19). `PpoBrain` holds a `ValueNorm { mu, sigma }` state updated once per PPO update before any training chunk: batch mean/variance of GAE returns is blended into `(mu, sigma)` via EMA (β = 3e-2 after the 2026-04-19 hotfix — initial 1e-4 retained 85% of the zero-initial state after 1,500 updates and left the critic biased low), then the POP rescale is applied to `c_value` weights and bias so externally-observed predictions `σ·z + µ` are preserved across the statistics change. The training loss regresses `c_value`'s raw output against the normalised target `(ret − µ) / σ`, so the critic targets a stationary ~N(0, 1) distribution regardless of return scale. Denormalisation `σ·raw + µ` happens at inference call sites (bootstrap `forward_critic`, action-selection `forward_critic_batch` value reads, on-exit flush). When `config.popart_enabled = false`, `ValueNorm` stays at `(0, 1)` and the pipeline is numerically equivalent to the pre-PopArt path. See `context/references/value-target-normalisation.md` for derivation.
- **Observation running mean/var normaliser** (round-2). `ObservationNormalizer` Resource applies Welford online per-dim mean/variance tracking in `build_observation_vector_system` after raw feature assembly. Pass-through during `warmup_samples = 1000`; afterwards centres and scales each of the 43 dims and clips to `[-10, 10]` (SB3 `VecNormalize.clip_obs` convention). Stats persist across episodes. When `enabled = false`, identity pass-through.
- **PolicyOutput** component: written by `ppo_act_all_cars_system` each tick, exposes `value_prediction`, `steering_mean`/`steering_std`, `throttle_mean`/`throttle_std`. Read by the analytics trace capture system for policy confidence metrics.
- **Analytics** has been comprehensively expanded: 16 tick-level trace fields (position, velocity decomposition, drift angle, min ray, velocity projection, centreline reward, policy confidence), 25 episode-level aggregates, a **crash classification system** (`CrashKind`: Slide, HeadOn, Overshoot, Spin, Stall), and a **layout-aware Markdown report** — up to 19 sections. Sections 11–15 were added 2026-04-19 (first pass) for the round-2 PPO diagnostics (pre-crash forensics, layer-health timeseries, PopArt µ/σ tracker, critic prediction-quality histogram, fleet variance). Sections 16–19 were added 2026-04-19 (second pass, M6): brain structure trajectory, plasticity health, structural events, and a side-by-side Fleet Comparison. PPO-specific sections (9, 12, 13, 14) skip entirely in brain-only runs. Filenames include a layout slug: `run_<ts>_<brain|side|ppo|keyboard>.md`. `PpoUpdateRecord` gained round-2 fields; `EpisodeRecord` gained `controller: String`; `EpisodeTracker` and `CompactRunExport` gained `brain_records: Vec<BrainUpdateRecord>`; `RunMetadata` gained `layout` / `ppo_cars` / `brain_cars` — all backwards-compatible via `#[serde(default)]`.
- **Brain-inspired v1 learner (M6, shipped).** Sparse directed graph of rate-coded tanh neurons (43 inputs + 15 hidden + 2 outputs + growth), trained by three-factor plasticity with per-car eligibility traces (`e ← λ·e + pre·post; Δw = η·M·e`), raw per-tick reward as modulator M (Option C — no critic), synaptic scaling + intrinsic excitability homeostasis, continual-backprop structural plasticity (utility-based replacement, plateau-triggered neurogenesis, synapse prune/sprout). Coexists with PPO in `TrainerLayout::SideBySide`. Full details in `systems/brain-inspired.md`; design rationale in `notes/brain-v1-design.md`; 23-decision implementation log in `notes/brain-v1-decisions.md`. Acceptance bar (visible learning relative to PPO in a ~2000-episode side-by-side run) is **empirically pending** — infrastructure is ready; the first real training run has not been executed yet.
- The brain layer now owns **ranking logic** (`src/brain/ranking.rs`) in addition to PPO. A seeded `StdRng` lives in `PpoBrain` for deterministic policy sampling.
- **Debug overlays default to off** — geometry (F1), sensors (F2), and telemetry HUD (F3) all start disabled. The HUD is a compact 440px panel with blue accent palette, 72% opacity, condensed text lines (no wrapping), PPO-specific metrics (clip %, KL divergence), six-column quarter table, no legend. Leaderboard panel matches the updated colour scheme.
- **Profiling** is feature-gated behind `--features profiling`. When the feature is off, all profiling code is compiled out entirely — zero runtime cost. When enabled, the app auto-exits after a configurable duration (default 30s) and exports JSON + Markdown performance reports to `reports/json/performance/` and `reports/performance/` respectively. Both include a `RunContext` snapshot which records the active **GEMM backend** under a `### Build` section (so every perf artefact is self-documenting about what produced it). Per-system timing covers all 17 FixedUpdate systems via an `instrument!()` macro. Reports have retention limits (3 reports per directory; oldest auto-deleted).
- **GEMM backend** selection is compile-time via Cargo features. Default: Accelerate on macOS (via `cblas_sgemm`, dispatches to AMX coprocessor), `matrixmultiply` (pure-Rust BLIS NEON microkernel) elsewhere. `--features force-scalar` forces the naive reference kernel; `--features force-matrixmultiply` forces the portable Rust kernel on any platform; `--features force-accelerate` forces Apple Accelerate (macOS-only, compile-error elsewhere). Mutually exclusive — compile-error if two are set. macOS `main.rs` pins `VECLIB_MAXIMUM_THREADS=1` at startup to prevent Accelerate's internal thread pool from fighting Bevy for cores at our small matrix sizes.
- **Performance is no longer the constraint**. Post-2026-04-18 dual-backend + batched-actor work, mean frame time dropped from 15.7 ms to 0.735 ms (21× overall), PPO Epoch from 13.5 ms to 0.446 ms (30×), action selection from 1.98 ms to 0.126 ms (16×). Budget utilisation went from 94% to 4.4%. Zero stutters, 0% frames over budget. See `notes/performance-tuning-lessons.md` for the contributing factors.
- The project is in a **post-baseline state with M1–M6 shipped**:
  - Round-2 training run `reports/analytics/run_1776556719.md` (M5 validation) — 2,271 episodes, 1,582 PPO updates — showed **all 8 cars completing the full track loop**, fleet max-progress spread 1.1%, crash rate falling from 100% to 56% in the best chunk, mean speed rising monotonically, and pre-crash analytics confirming the policy anticipates (96% of crashes had throttle released > 0.25 s before impact). The environment, observation contract, and reward shaping are confirmed learnable.
  - M6 (brain-inspired v1) shipped 2026-04-19 as six staged commits + wrap + default/analytics fix (`6237aa7..3a737d9`, 8 commits pushed to origin/master). 133 tests green across default, `force-scalar`, and release builds.
  - PPO is **stable reference machinery** from here forward — the diagnostic baseline against which brain-inspired learning will be measured. See `notes/baseline-to-brain-inspired.md` for the full transition framing and what carries forward.
  - The next active work is **empirical validation of M6** — a real training run (AllBrain or SideBySide) to check whether the brain-inspired substrate actually learns. Pre-empirical. M7 (brain visualisation) is the next explicit milestone after validation.
- `README.md` progress bar reflects M1–M6 at 100%, M7 (brain visualisation) as the next pointer. The README's Known Biological Simplifications section captures scope compromises (slot-recycling apoptosis, unrestricted neurogenesis location, no spatial constraints on synapse formation, etc.) that are intentional at v1 scale.
- The finish-line removal, random-spawn paradigm, velocity-projection reward, expanded observations, analytics overhaul, round-2 critic target-scaling interventions, and the full brain-inspired v1 learner with side-by-side comparison have all been **implemented**. The system is now in a coherent post-M6 state ready for empirical measurement.
