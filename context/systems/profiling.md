# System — Profiling

## Scope / Purpose

- Measure per-frame, per-SimSet, and **per-system** execution time of the FixedUpdate pipeline to identify performance bottlenecks.
- Feature-gated behind `--features profiling` — when the feature is off, all profiling code is compiled out entirely with zero runtime cost.
- Automatically exit the application after a configurable duration and export both JSON and Markdown performance reports on exit.

## Boundaries / Ownership

- Owns all files under `src/profiling/`.
- Reads from `brain::ppo::buffer::TrainerRolloutBuffer` and `analytics::models::EpisodeTracker` for buffer size snapshots.
- Reads from `TrainerConfig`, `EpisodeConfig`, `ObservationConfig`, `PpoBrain`, `EpisodeTracker` for the run context header.
- Uses `analytics::exporters::cleanup::enforce_retention` and `analytics::exporters::context::RunContext` from the analytics subsystem.
- Conditionally compiled — the `profiling` module only exists when `--features profiling` is active.

## Current Implemented Reality

## Launch Command

```bash
cargo run --release --features profiling
```

`--release` is strongly recommended for profiling — debug-mode numbers are ~10× slower than optimised code and don't represent the real hot-path cost. The profiling feature is additive — it does not disable any existing subsystem. The app runs normally with timing instrumentation layered in.

Feature flags compose freely: `--features "profiling,force-scalar"` profiles the scalar GEMM backend, `--features "profiling,force-matrixmultiply"` profiles the matrixmultiply backend, and so on. Every exported report records the active backend in its Run Context `### Build` section, so benchmarks across different feature combinations are directly comparable.

## Architecture

### Ring Buffer Approach

`FrameTimings` holds a fixed-capacity ring buffer of `FrameRecord` structs. Each record captures:

- Frame start and end timestamps.
- Per-SimSet boundary timestamps (Input, Physics, Collision, Measurement).
- Derived per-set durations computed from adjacent boundary markers.
- **Per-system timings**: a `Vec<(String, u64)>` of named system durations in microseconds.

The ring buffer has a configurable capacity (default 1800 frames = 30s at 60 Hz). Once full, the oldest entries are overwritten. This bounds memory usage regardless of run duration.

### Per-SimSet Boundary Timing

Profiling systems are scheduled as boundary markers around each `SimSet` in the FixedUpdate pipeline:

```text
frame_start_system          ← before SimSet::Input (also resets SystemTimers)
  SimSet::Input
input_end_system            ← between Input and Physics
  SimSet::Physics
physics_end_system          ← between Physics and Collision
  SimSet::Collision
collision_end_system        ← between Collision and Measurement
  SimSet::Measurement
frame_end_system            ← after Measurement (drains SystemTimers, records complete frame)
auto_exit_system            ← checks elapsed time, triggers app exit
```

### Per-System Timing

The `SystemTimers` resource provides a shared timing accumulator. For each system to be profiled, the `ProfilingPlugin` registers a `start_timer("Name")` system `.before()` the target and a `stop_timer("Name")` system `.after()` it, both in the same `SimSet`. This uses an `instrument!()` macro for concise registration.

At frame end, `frame_end_system` drains the `SystemTimers.durations_us` map into the `FrameRecord.system_timings` vector (sorted alphabetically for deterministic output).

#### Instrumented Systems

| SimSet | System | Timer Name |
|--------|--------|------------|
| Input | `keyboard_action_input_system` | Keyboard Input |
| Input | `ppo_act_all_cars_system` | PPO Action Selection |
| Input | `action_smoothing_system` | Action Smoothing |
| Physics | `car_physics_system` | Car Physics |
| Physics | `capture_episode_action_stats_system` | Action Stats Capture |
| Collision | `collision_detection_system` | Collision Detection |
| Measurement | `update_track_progress_system` | Track Progress |
| Measurement | `episode_loop_system` | Episode Loop (Reward + Reset) |
| Measurement | `update_sensor_readings_system` | Sensor Raycasting |
| Measurement | `build_observation_vector_system` | Observation Vector |
| Measurement | `capture_episode_tick_trace_system` | Trace Capture |
| Measurement | `snapshot_completed_episode_trace_system` | Trace Snapshot |
| Measurement | `snapshot_completed_episode_action_stats_system` | Action Stats Snapshot |
| Measurement | `ppo_collect_rewards_all_cars_system` | PPO Reward Collection |
| Measurement | `ppo_epoch_system` | PPO Epoch (Training) |
| Measurement | `update_driving_hud_stats_system` | HUD Stats |
| Measurement | `capture_driving_hud_episode_metrics_system` | HUD Episode Capture |

### Auto-Exit

`auto_exit_system` runs after `frame_end_system` and compares the tick counter against the configured duration (in ticks at 60 Hz). When reached, it sends an `AppExit` event, triggering the `Last` schedule where reports are exported.

## Configuration

`ProfilingConfig` is inserted as a Bevy resource by `ProfilingPlugin`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `duration_seconds` | `f32` | `30.0` | Seconds before auto-exit |
| `ring_buffer_frames` | `usize` | `1800` | Maximum number of `FrameRecord` entries retained |
| `track_set_timings` | `bool` | `true` | Whether to compute per-SimSet boundary durations |
| `track_buffer_sizes` | `bool` | `true` | Whether to capture rollout buffer and trace count |

## Export

### Output Locations

On exit, the profiling system exports two reports:

- **JSON**: `reports/json/performance/perf_{timestamp}.json` — raw frame data + summary statistics.
- **Markdown**: `reports/performance/perf_{timestamp}.md` — explanatory human-readable report.

Both directories enforce a retention limit of 3 reports (oldest are deleted).

### Markdown Report Structure

The Markdown report is designed to be readable by someone unfamiliar with the codebase:

1. **How to Read This Report** — explains the pipeline stages and what a stutter means.
2. **Overall Verdict** — one-paragraph plain-English health assessment.
3. **Frame Budget** — total ticks, mean frame time, utilisation, over-budget count, stutter count.
4. **Frame Time Distribution** — sparkline, percentile table, histogram with bucket labels, interpretation.
5. **Pipeline Breakdown** — per-SimSet mean, max, % of frame, sparkline, interpretation.
6. **Per-System Detail** — tables for each pipeline stage showing individual system mean, max, % of stage, with human-readable descriptions of what each system does.
7. **Stutter Analysis** — worst 5 ticks with full per-system timing breakdown, PPO Epoch stutter correlation analysis, interpretation.
8. **Buffer & Memory Pressure** — rollout buffer and trace count sparklines with explanations.
9. **Recommendations** — ranked optimisation opportunities based on heuristics (dominant system, PPO stutter correlation, sensor raycasting cost, analytics overhead).

### JSON Report Contents

| Section | Contents |
|---------|----------|
| **config** | Snapshot of profiling configuration used |
| **summary** | Mean/min/max/p95/p99 frame times, per-set means and maxes, budget analysis, stutter detection |
| **frames** | Full array of `FrameRecord` structs including per-system timings |

## Key Interfaces / Data Flow

- `ProfilingPlugin` registers systems in `FixedUpdate` (boundary markers, frame start/end, auto-exit) and `Last` (export).
- `SystemTimers` resource is written by `start_timer`/`stop_timer` closure systems and drained by `frame_end_system`.
- `FrameTimings` resource is written by `frame_end_system` and read by the export system on exit.
- Export writes to `reports/json/performance/` and `reports/performance/` with retention limits.

## Implemented Outputs / Artifacts

- JSON performance report: `reports/json/performance/perf_{timestamp}.json`
- Markdown performance report: `reports/performance/perf_{timestamp}.md`
- Both include run context headers captured from live Bevy resources.

## Key Source Files

| File | Responsibility |
|------|---------------|
| `src/profiling/mod.rs` | Plugin registration, per-system `instrument!()` macro |
| `src/profiling/timers.rs` | `FrameRecord`, `FrameTimings` ring buffer, `SystemTimers` accumulator |
| `src/profiling/capture.rs` | `frame_start/end`, boundary markers, `start_timer`/`stop_timer` helpers |
| `src/profiling/config.rs` | `ProfilingConfig` |
| `src/profiling/exporters/json.rs` | JSON export + summary computation |
| `src/profiling/exporters/markdown.rs` | Markdown report generation with interpretation and recommendations |

## Known Issues / Active Risks

- Per-system timing uses wrapper systems scheduled `.before()`/`.after()` the target. This includes Bevy's scheduler dispatch overhead between systems, which inflates some system timings (e.g., Action Smoothing shows ~1.7ms but the actual work is nanoseconds).

## Partial / In Progress

- No live TUI viewer yet — only post-run reports. The performance-optimisation plan describes the vision for a real-time TUI.

## Planned / Missing / Likely Changes

- Live TUI performance viewer (separate binary reading a shared stream).
- Per-car breakdown within per-system timings.
- Render/Update schedule profiling (currently FixedUpdate only).
- Memory allocation tracking.

## Durable Notes / Discarded Approaches

- The initial profiler version used only per-SimSet boundary timing (4 boundary markers). This was insufficient — the "Measurement" set contains 11 systems and saying "Measurement is slow" gave no actionable insight. Per-system timing was added to identify PPO Epoch as the specific bottleneck.

## Obsolete / No Longer Relevant

- Nothing yet.

## Known Limitations

- **Measures FixedUpdate only**: the profiling boundary markers are placed around SimSet stages in the FixedUpdate schedule. Render time, Update schedule systems, and Bevy internal overhead are not captured. Frame time statistics reflect simulation cost, not total application frame time.
- **Ring buffer window**: with the default 1800 capacity at 60 Hz, the buffer holds exactly 30 seconds of data, matching the default profiling duration.
- **Timer overhead**: each instrumented system adds two tiny closure-based systems (start/stop) to the schedule. This is negligible but non-zero — approximately 34 extra systems for the current 17 instrumented targets.
