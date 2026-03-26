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
- speed, heading error
- applied steering and throttle
- reward decomposition (total, progress, time penalty, terminal)
- done flag and reason
- sector index (track divided into 20 sectors)
- all ray distances, lookahead heading deltas, lookahead curvatures
- critic value prediction (currently None — pending per-car value lookup from shared buffer)

### Per-Episode Record

`EpisodeRecord` combines:
- `env_id` — which car completed this episode
- episode identity and summary (id, progress, return, ticks, crashes, end reason)
- reward decomposition sums (progress, time penalty, terminal, crash, lap bonus)
- action statistics (steering/throttle mean and std)
- turn-execution diagnostics (turn-in latency, throttle release latency, steering adequacy, understeer rate)
- input-level summaries (mean centreline distance, heading error, ray distances)
- heuristic failure mode classification

### Metrics Modules

| Module | Derives |
|--------|---------|
| `stats.rs` | Basic statistical utilities (mean, std, percentile) |
| `chunking.rs` | Temporal chunked trend analysis (10 chunks by default) |
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

The report is organised around **diagnostic questions**, not metric modules:

| Section | Answers |
|---------|---------|
| 1. Run Summary | Metadata, learning phase, diagnostic flags |
| 2. Is the Policy Learning? | Progress/reward/crash sparklines, 10-chunk trend table |
| 3. Has It Found a Route? | Consistency score, speed profile bar chart, highest-variance sectors |
| 4. Per-Car Performance | Per-car comparison table, best vs worst contrast |
| 5. Where Does It Fail? | Crash heatmap by sector, failure modes, corner vs straight analysis |
| 6. Training Health | PPO sparklines (entropy, clip%, KL, EV), latest update, layer health, reward decomposition |
| 7. Trajectory Snapshots | Best, latest, latest crash episodes |

ASCII visuals include Unicode sparklines (▁▂▃▄▅▆▇█), horizontal bar charts (█░), and single-row heatmaps.

## Key Interfaces / Data Flow

| Interface | Source | Analytics use |
|-----------|--------|--------------|
| `ActionState.applied` | agent | Per-car action summaries and trace capture |
| `EpisodeState` | game | Reward decomposition, terminal reason, episode summaries |
| `SensorReadings` | agent | Trace capture and input-oriented metrics |
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
- The value prediction in trace capture currently returns `None` for all cars — per-car value lookup from the shared trainer buffer is pending.
- Some older metric modules (`inputs`, `insights`, `critic`) are not wired into the current markdown report and produce dead-code warnings. They remain valid API for future re-integration.

## Partial / In Progress

- The value prediction field in `TickTraceRecord` currently returns `None` for all cars — per-car lookup from the shared trainer buffer is pending.
- The older metric modules (`inputs`, `insights`, `critic`) remain as valid API but are not wired into the current markdown report. They can be re-integrated as diagnostic depth increases.

## Planned / Missing / Likely Changes

- **Crash-safe checkpointing or periodic export** would materially improve experiment robustness.
- **Comparison tooling** across multiple exported runs does not exist.
- **Per-car value predictions** in trace capture need a per-car lookup from the shared buffer rather than reading the last value.
- If a brake channel or new observation features are added, trace and metrics schemas will need coordinated extension.
- Re-integrating the older metric modules (critic diagnostics by region, input learning trends) into the markdown report would deepen the diagnostic capability.

## Durable Notes / Discarded Approaches

- Keeping raw trackers, derived metrics, and exporters **separate** is a good structural choice — it reduces coupling and makes new diagnostics easier to add.
- Analytics should stay **downstream of runtime truth**. It is a consumer and summariser, not the source of reward, episode, or environment facts.
- The two-tier JSON approach (compact always, full trace opt-in) was chosen over auto-deleting JSON because the compact data enables re-analysis of old runs without the size cost of per-tick traces.
- The markdown report was deliberately restructured around diagnostic questions ("is the policy learning?", "has it found a route?") rather than metric modules, because the primary consumers are a human watching training and an agent reading reports remotely.

## Obsolete / No Longer Relevant

- Any reference to **first-car shims** in analytics capture is obsolete — all systems now iterate all cars with `env_id` tagging.
- The old single-file JSON export (full EpisodeTracker serialised as one blob) has been replaced by the two-tier model.
- The old markdown report structure (organised by metric module with dense tables) has been replaced by the 7-section diagnostic structure.
