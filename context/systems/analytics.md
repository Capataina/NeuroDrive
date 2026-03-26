# System — Analytics

## Scope / Purpose

- Persist enough run data to inspect learning and driving behaviour after the app exits.
- Separate raw capture, derived metrics, and export rendering so diagnostics can expand without rewriting the whole subsystem each time.
- Analytics is a **consumer and summariser**, not a source of reward, episode, or environment facts.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/analytics/models.rs` | Canonical analytics schemas: `EpisodeRecord`, `TickTraceRecord`, `EpisodeTrace`, `A2cUpdateRecord`, `EpisodeTracker` | Environment truth, reward definitions |
| `src/analytics/trackers/` | Fixed-tick accumulation and episode/update record finalisation | When episodes end (owned by game) |
| `src/analytics/metrics/` | Derived diagnostics and trend synthesis | Raw data capture (owned by trackers) |
| `src/analytics/exporters/` | JSON and Markdown serialisation | Schema definitions (owned by models) |
| `src/analytics/plugin.rs` | Scheduling and on-exit export orchestration | Runtime state mutation |

## Current Implemented Reality

### Plugin Initialisation

`AnalyticsPlugin` initialises three trackers as resources:
- `EpisodeTracker` — accumulates completed episode records, traces, and A2C update snapshots.
- `EpisodeActionAccumulator` — per-episode steering/throttle running statistics.
- `EpisodeTraceAccumulator` — per-tick trajectory data for the current episode.

### Capture Pipeline

```text
FixedUpdate (SimSet::Physics):
  capture_episode_action_stats_system     ← records applied steering/throttle stats

FixedUpdate (SimSet::Measurement):
  capture_episode_tick_trace_system       ← per-tick trajectory record (after obs rebuild, before A2C reward)
  snapshot_completed_episode_trace_system ← finalises trace on episode end
  snapshot_completed_episode_action_stats_system ← finalises action stats on episode end

Update:
  episode_tracker_system                  ← folds completed episode + trace + A2C snapshots into EpisodeTracker

Last:
  on_exit_system                          ← triggers export on AppExit
```

### Per-Tick Trace Data

Each `TickTraceRecord` captures:
- progress (fraction, arc-length), centreline distance, signed lateral offset
- speed, heading error
- applied steering and throttle
- reward decomposition (total, progress, time penalty, terminal)
- done flag and reason
- sector index (track divided into 20 sectors)
- all ray distances, lookahead heading deltas, lookahead curvatures
- critic value prediction (when AI is active)

### Per-Episode Record

`EpisodeRecord` combines:
- episode identity and summary (id, progress, return, ticks, crashes, end reason)
- reward decomposition sums (progress, time penalty, terminal, crash, lap bonus)
- action statistics (steering/throttle mean and std)
- turn-execution diagnostics (turn-in latency, throttle release latency, steering adequacy, understeer rate)
- input-level summaries (mean centreline distance, heading error, ray distances)
- heuristic failure mode classification

### Metrics Modules

| Module | Derives |
|--------|---------|
| `stats.rs` | Basic episode statistics |
| `chunking.rs` | Temporal chunked trend analysis |
| `inputs.rs` | Input-learning summaries (ray, offset, heading distributions) |
| `turns.rs` | Turn-execution diagnostics (latency, adequacy, understeer, curvature-steering error) |
| `critic.rs` | Critic health diagnostics (value drift, explained variance) |
| `sectors.rs` | Progress-sector breakdown (20 sectors) |
| `trajectory.rs` | Trajectory-level derived summaries |
| `insights.rs` | Narrative insight bullet generation |

### Export

- Export triggers on `AppExit` message from the `Last` schedule.
- Writes two files per run:
  - `reports/run_<unix_timestamp>.json` — full structured data
  - `reports/run_<unix_timestamp>.md` — human-readable report with tables and narrative

## Key Interfaces / Data Flow

| Interface | Source | Analytics use |
|-----------|--------|--------------|
| `ActionState.applied` | agent | Action summaries and trace capture |
| `EpisodeState` | game | Reward decomposition, terminal reason, episode summaries |
| `SensorReadings` | agent | Trace capture and input-oriented metrics |
| `A2cTrainingStats` | brain | Update records and live learning-health export |
| `ObservationConfig` and `Track` | agent/maps | Lookahead snapshot reconstruction in traces |

## Implemented Outputs / Artifacts

- **Runtime resource:** `EpisodeTracker`
- **Exported schemas:** `EpisodeRecord`, `EpisodeTrace`, `TickTraceRecord`, `A2cUpdateRecord`
- **Output files:** `reports/run_<timestamp>.json`, `reports/run_<timestamp>.md`

## Known Issues / Active Risks

- **Exit-triggered only** — abrupt termination (kill signal, panic) loses the entire run.
- **Missing experiment metadata:** no RNG seed, config snapshot, git revision, active mode, or track identity in reports.
- No dedicated validation that every finished episode is recorded exactly once across all terminal paths.
- The heuristic failure-mode classification is useful for triage but is **not ground truth**.
- All analytics capture systems use **temporary shims** that target the first car (`car_query.iter().next()`) only. Per-car analytics with `env_id` tagging is planned as part of the full analytics overhaul.
- The value prediction in trace capture reads from `TrainerRolloutBuffer.values` (the shared trainer buffer), not a per-car buffer.

## Partial / In Progress

- The subsystem is useful for post-run diagnosis, but the data model is closer to "single-run introspection" than "rigorous experiment tracking".
- Long-horizon training-health history exists only because A2C update snapshots are persisted by analytics; the brain itself only exposes the most recent update stats.

## Planned / Missing / Likely Changes

- **Crash-safe checkpointing or periodic export** would materially improve experiment robustness.
- **Run metadata capture** is the clearest missing capability for cross-run comparisons.
- Comparison tooling across multiple exported runs does not exist.
- If a brake channel or new observation features are added, trace and metrics schemas will need coordinated extension.
- **Full analytics visual overhaul** planned: heat maps, time-series graphs, distribution charts, trajectory overlays, ASCII visualisations. See `context/plans/analytics-overhaul-brief.md`. This will also include per-car `env_id` tagging and cohort summaries (best/worst/quartile breakdowns).

## Durable Notes / Discarded Approaches

- Keeping raw trackers, derived metrics, and exporters **separate** is a good structural choice — it reduces coupling and makes new diagnostics easier to add.
- Analytics should stay **downstream of runtime truth**. It is a consumer and summariser, not the source of reward, episode, or environment facts.

## Obsolete / No Longer Relevant

- The old "telemetry is only the HUD" mental model is obsolete. Analytics is now a substantial second observability layer with its own schemas and reports.
