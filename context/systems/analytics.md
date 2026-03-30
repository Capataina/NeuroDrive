# System — Analytics

## Scope / Purpose

- Persist enough run data to inspect learning and driving behaviour after the app exits.
- Separate raw capture, derived metrics, and export rendering so diagnostics can expand without rewriting the whole subsystem each time.
- Analytics is a **consumer and summariser**, not a source of reward, episode, or environment facts.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/analytics/models.rs` | Canonical analytics schemas: `EpisodeRecord` (env_id-tagged), `TickTraceRecord` (env_id-tagged), `EpisodeTrace`, `A2cUpdateRecord`, `EpisodeTracker`, `AnalyticsConfig`, `RunMetadata`, `CompactRunExport` | Environment truth, reward definitions |
| `src/analytics/trackers/` | Per-car fixed-tick accumulation and episode/update record finalisation | When episodes end (owned by game) |
| `src/analytics/metrics/` | Derived diagnostics, trend synthesis, visual rendering helpers | Raw data capture (owned by trackers) |
| `src/analytics/exporters/` | Two-tier JSON and diagnostic Markdown serialisation | Schema definitions (owned by models) |
| `src/analytics/plugin.rs` | Scheduling, resource registration, and on-exit export orchestration | Runtime state mutation |

## Current Implemented Reality

### Plugin Initialisation

`AnalyticsPlugin` initialises these resources:
- `EpisodeTracker` — accumulates completed episode records, traces, and PPO update snapshots.
- `PerCarActionAccumulators` — `HashMap<u32, EpisodeActionAccumulator>` for per-car steering/throttle running statistics.
- `PerCarTraceAccumulators` — `HashMap<u32, EpisodeTraceAccumulator>` for per-car per-tick trajectory data.
- `AnalyticsConfig` — configuration resource controlling export behaviour (`full_trace_export: bool`, default false).

### Multi-Car Capture

All capture systems iterate **all cars** via `Query<(&EnvInstanceId, ...), With<Car>>`. There are no first-car shims remaining. Each record is tagged with `env_id` for per-car and cohort analysis.

### Capture Pipeline

```text
FixedUpdate (SimSet::Physics):
  capture_episode_action_stats_system     ← records all cars' applied steering/throttle stats

FixedUpdate (SimSet::Measurement):
  capture_episode_tick_trace_system       ← per-tick trajectory record for all cars
  snapshot_completed_episode_trace_system ← finalises per-car trace on episode end
  snapshot_completed_episode_action_stats_system ← finalises per-car action stats on episode end

Update:
  episode_tracker_system                  ← folds all cars' completed episodes + traces + PPO snapshots into EpisodeTracker

Last:
  on_exit_system                          ← builds RunMetadata, triggers two-tier export
```

### Per-Tick Trace Data

Each `TickTraceRecord` captures:
- `env_id` — which car produced this tick
- progress (fraction, arc-length), centreline distance, signed lateral offset
- `position_x`, `position_y` — world-space car position
- `v_forward`, `v_lateral` — velocity decomposition along car axes
- `speed_delta` — frame-to-frame speed change
- `drift_angle_deg` — angle between velocity vector and car forward
- speed, heading error
- applied steering and throttle
- `min_ray_distance` — closest ray hit (proximity to walls)
- reward decomposition (total, `velocity_projection`, `centreline_reward`, time penalty, terminal)
- `policy_steering_mean`, `policy_steering_std`, `policy_throttle_mean`, `policy_throttle_std` — from PolicyOutput component
- `previous_steering`, `previous_throttle` — previous tick's applied actions
- done flag and reason
- sector index (track divided into 20 sectors)
- all ray distances, lookahead heading deltas, lookahead curvatures
- critic value prediction (from PolicyOutput component)

### Per-Episode Record

`EpisodeRecord` combines:
- `env_id` — which car completed this episode
- episode identity and summary (id, progress, return, ticks, crashes, end reason)
- reward decomposition sums (progress, time penalty, terminal, crash — `lap_bonus_sum` and `lap_completed` removed)
- action statistics (steering/throttle mean and std)
- **Speed metrics:** speed mean, max, std across the episode
- **Action behaviour:** braking fraction, acceleration fraction, coast fraction, steering jitter, throttle jitter
- **Crash forensics:** crash type classification, velocity at crash
- **Value function stats:** mean value prediction, value at start, value at crash
- **Exploration metrics:** steering std mean, throttle std mean
- turn-execution diagnostics (turn-in latency, throttle release latency, steering adequacy, understeer rate)
- input-level summaries (mean centreline distance, heading error, ray distances)
- heuristic failure mode classification

### Crash Classification

Terminal episodes are classified into crash types based on the final tick's state:
- **Slide** — high drift angle at impact
- **HeadOn** — low drift angle, high forward velocity
- **Overshoot** — missed turn, ran wide
- **Spin** — high angular velocity at impact
- **Stall** — very low velocity at crash (possibly stuck against a wall)

### Metrics Modules

| Module | Derives |
|--------|---------|
| `stats.rs` | Basic statistical utilities (mean, std, percentile) |
| `chunking.rs` | Temporal chunked trend analysis (10 chunks by default). `ChunkMetrics` expanded with ~15 new trend fields covering speed stats, action behaviour fractions, crash type breakdowns, value function means, and exploration metrics |
| `timeseries.rs` | Episode/update time-series extraction, rolling mean, plateau detection |
| `diagnostics.rs` | Automated diagnostic flags (7 checks: entropy collapse, clip fraction, KL spike, plateau, action collapse, crash rate spike, value drift) |
| `consistency.rs` | Per-sector behavioural consistency (speed/steering/throttle/centreline variance), overall consistency score |
| `phases.rs` | Learning phase detection: Exploration → Discovery → Refinement → Plateau → Regression |
| `sparkline.rs` | ASCII visual helpers: sparklines (▁▂▃▄▅▆▇█), horizontal bar charts, heatmap rows |
| `inputs.rs` | Input-learning summaries (ray, offset, heading distributions) |
| `turns.rs` | Turn-execution diagnostics (latency, adequacy, understeer, failure mode classification) |
| `critic.rs` | Critic health diagnostics (value drift, explained variance by context) |
| `sectors.rs` | Progress-sector breakdown summaries (20 sectors) |
| `trajectory.rs` | Trajectory-level derived summaries and episode selection |
| `insights.rs` | Narrative insight bullet generation |

The new modules (`timeseries`, `diagnostics`, `consistency`, `phases`, `sparkline`) are the primary consumers in the overhauled markdown report. The older modules (`inputs`, `insights`, `critic`) remain valid public API but are not currently wired into the report — they can be re-integrated as diagnostic depth increases.

### Two-Tier Export

Export triggers on `AppExit` message from the `Last` schedule:

| Output | Content | When |
|--------|---------|------|
| `reports/run_<ts>.json` | `CompactRunExport`: `RunMetadata` + all `EpisodeRecord`s + all `A2cUpdateRecord`s. No per-tick traces. | Always |
| `reports/run_<ts>_traces.json` | Full `EpisodeTracker` including per-tick trace data | Only when `AnalyticsConfig.full_trace_export == true` |
| `reports/run_<ts>.md` | Diagnostic Markdown report generated from full in-memory data | Always |

`RunMetadata` captures: car count, track name, session timestamp, PPO hyperparameters (epochs, clip_epsilon, gamma, gae_lambda, max_steps, samples_per_tick).

The compact JSON is typically kilobytes; the full trace JSON can be tens of megabytes for long runs.

### Markdown Report Structure

The report is organised around **diagnostic questions** across 10 sections, each with auto-generated takeaway sentences:

| Section | Answers |
|---------|---------|
| 1. Run Summary | Metadata, learning phase, diagnostic flags |
| 2. Learning Progress | Progress/reward/crash sparklines, 10-chunk trend table |
| 3. Action Behaviour | Braking/acceleration/coast fractions, steering and throttle jitter, action distribution evolution |
| 4. Speed & Momentum | Speed statistics, v_forward/v_lateral trends, speed_delta patterns, drift angle analysis |
| 5. Crash Forensics | Crash type breakdown (Slide/HeadOn/Overshoot/Spin/Stall), velocity at crash, crash heatmap by sector |
| 6. What Does the Car Think | Value function evolution, value at start vs crash, explained variance, policy mean/std trends |
| 7. Track Coverage | Consistency score, per-sector speed/steering profiles, highest-variance sectors |
| 8. Driving Quality | Per-car comparison table, best vs worst contrast, turn-execution diagnostics |
| 9. Training Health | PPO sparklines (entropy, clip%, KL, EV), latest update, layer health, reward decomposition |
| 10. Trajectory Snapshots | Best, latest, latest crash episodes |

ASCII visuals include Unicode sparklines (▁▂▃▄▅▆▇█), horizontal bar charts (█░), and single-row heatmaps.

## Key Interfaces / Data Flow

| Interface | Source | Analytics use |
|-----------|--------|--------------|
| `ActionState.applied` | agent | Per-car action summaries and trace capture |
| `EpisodeState` | game | Reward decomposition, terminal reason, episode summaries (distance_driven, spawn_s, previous_s) |
| `SensorReadings` | agent | Trace capture: v_forward, v_lateral, speed_delta, previous actions, ray data |
| `PolicyOutput` | brain | Per-car value prediction, policy means/stds for trace and episode capture |
| `A2cTrainingStats` | brain | PPO update records (including clip_fraction, approx_kl) |
| `ObservationConfig` and `Track` | agent/maps | Lookahead snapshot reconstruction in traces |
| `TrainerConfig` | game | Car count for RunMetadata |
| `A2cBrain` | brain | PPO hyperparameters for RunMetadata |
| `EnvInstanceId` | game | Per-car tagging in all capture systems |

## Implemented Outputs / Artifacts

- **Runtime resources:** `EpisodeTracker`, `PerCarActionAccumulators`, `PerCarTraceAccumulators`, `AnalyticsConfig`
- **Exported schemas:** `EpisodeRecord`, `EpisodeTrace`, `TickTraceRecord`, `A2cUpdateRecord`, `RunMetadata`, `CompactRunExport`
- **Output files:** compact JSON (always), full trace JSON (opt-in), diagnostic Markdown (always)
- **Unit tests:** 25 tests across timeseries, diagnostics, consistency, phases, sparkline, and turns modules

## Known Issues / Active Risks

- **Exit-triggered only** — abrupt termination (kill signal, panic) loses the entire run.
- No dedicated validation that every finished episode is recorded exactly once across all terminal paths.
- The heuristic failure-mode classification is useful for triage but is **not ground truth**.
- Some older metric modules (`inputs`, `insights`, `critic`) are not wired into the current markdown report and produce dead-code warnings. They remain valid API for future re-integration.

## Partial / In Progress

- The older metric modules (`inputs`, `insights`, `critic`) remain as valid API but are not wired into the current markdown report. They can be re-integrated as diagnostic depth increases.

## Planned / Missing / Likely Changes

- **Crash-safe checkpointing or periodic export** would materially improve experiment robustness.
- **Comparison tooling** across multiple exported runs does not exist.
- If new observation features are added, trace and metrics schemas will need coordinated extension.
- Re-integrating the older metric modules (critic diagnostics by region, input learning trends) into the markdown report would deepen the diagnostic capability.

## Durable Notes / Discarded Approaches

- Keeping raw trackers, derived metrics, and exporters **separate** is a good structural choice — it reduces coupling and makes new diagnostics easier to add.
- Analytics should stay **downstream of runtime truth**. It is a consumer and summariser, not the source of reward, episode, or environment facts.
- The two-tier JSON approach (compact always, full trace opt-in) was chosen over auto-deleting JSON because the compact data enables re-analysis of old runs without the size cost of per-tick traces.
- The markdown report was deliberately restructured around diagnostic questions rather than metric modules, because the primary consumers are a human watching training and an agent reading reports remotely.
- **Reward decomposition columns simplified** — centreline reward and progress bonus columns were removed from the earlier reward model. The current model captures velocity_projection and centreline_reward as separate decomposition terms.

## Obsolete / No Longer Relevant

- Any reference to **first-car shims** in analytics capture is obsolete — all systems now iterate all cars with `env_id` tagging.
- The old single-file JSON export (full EpisodeTracker serialised as one blob) has been replaced by the two-tier model.
- The old markdown report structure (organised by metric module with dense tables) has been replaced by the 10-section diagnostic structure.
- Any reference to `lap_bonus_sum` or `lap_completed` in EpisodeRecord or EpisodeTrace is obsolete — these fields have been removed.
- Any reference to 7 sections in the markdown report is obsolete — it now has 10 sections.
- Any reference to progress metrics being "misleading with random spawns" is obsolete — progress is now cumulative forward arc-length from spawn (distance_driven), which is honest across all spawn positions.
- Any reference to value prediction being `None` for all cars is obsolete — PolicyOutput component now provides per-car value predictions directly.
