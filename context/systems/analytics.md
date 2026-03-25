# System — Analytics

## Scope / Purpose

- Persist enough run data to inspect learning and driving behaviour after the app exits.
- Separate raw capture, derived metrics, and export rendering so diagnostics can expand without rewriting the whole subsystem each time.

## Boundaries / Ownership

- `src/analytics/models.rs` owns the canonical analytics schema.
- `src/analytics/trackers/` owns fixed-tick accumulation and episode/update record finalisation.
- `src/analytics/metrics/` owns derived diagnostics and trend synthesis.
- `src/analytics/exporters/` owns JSON and Markdown serialisation.
- `src/analytics/plugin.rs` owns scheduling and on-exit export.
- Analytics reads runtime state but should not create environment truth or mutate training state.

## Current Implemented Reality

- `AnalyticsPlugin` initialises:
  - `EpisodeTracker`,
  - `EpisodeActionAccumulator`,
  - `EpisodeTraceAccumulator`.
- Action tracking records per-episode steering/throttle means and standard deviations from the applied control stream.
- Trace tracking records per-tick episode data including:
  - progress,
  - speed,
  - centreline distance,
  - signed lateral offset,
  - heading error,
  - applied controls,
  - reward decomposition,
  - terminal flags,
  - ray distances,
  - lookahead features,
  - current critic value prediction when AI is active.
- Episode tracking combines episode summaries, trace-derived metrics, and A2C training snapshots into persistent run records.
- Metrics modules derive:
  - chunked trends,
  - input-learning summaries,
  - turn-execution diagnostics,
  - critic diagnostics,
  - sector summaries,
  - trajectory summaries,
  - narrative insight bullets.
- Export currently writes timestamped JSON and Markdown reports under `reports/` on app exit.
- Export triggering uses Bevy 0.18 `AppExit` messages from the `Last` schedule.

## Key Interfaces / Data Flow

| Interface | Source | Analytics use |
|---|---|---|
| `ActionState.applied` | agent | action summaries and trace capture |
| `EpisodeState` | game | reward decomposition, terminal reason, episode summaries |
| `SensorReadings` | agent | trace capture and input-oriented metrics |
| `A2cTrainingStats` | brain | update records and live learning-health export |
| `ObservationConfig` and `Track` | agent/maps | lookahead snapshot reconstruction in traces |

```text
FixedUpdate capture:
  action stats
  tick trace
  completed-episode snapshots

Update:
  episode_tracker_system folds snapshots into EpisodeTracker

Last:
  export_to_json()
  export_to_markdown()
```

## Implemented Outputs / Artifacts

- Runtime resource:
  - `EpisodeTracker`
- Exported record types:
  - `EpisodeRecord`
  - `EpisodeTrace`
  - `TickTraceRecord`
  - `A2cUpdateRecord`
- Output files:
  - `reports/run_<timestamp>.json`
  - `reports/run_<timestamp>.md`

## Known Issues / Active Risks

- Export is exit-triggered only, so abrupt termination can lose the run.
- Reports still lack core experiment metadata:
  - RNG seed,
  - config snapshot,
  - git revision,
  - active mode,
  - track identity.
- There is no dedicated validation harness proving that every finished episode is recorded exactly once across all terminal paths.
- The heuristic failure-mode classification is useful for triage but is not ground truth.

## Partial / In Progress

- The subsystem is already useful for post-run diagnosis, but the data model is still closer to “single-run introspection” than “rigorous experiment tracking”.
- Long-horizon training-health history exists only because update snapshots are persisted by analytics; the brain itself only exposes the most recent update stats.

## Planned / Missing / Likely Changes

- Crash-safe checkpointing or periodic export would materially improve experiment robustness.
- Run metadata capture is the clearest missing capability if comparisons between runs are going to matter.
- Comparison tooling across multiple exported runs does not exist yet.
- If a brake channel or new observation features are added, the trace and metrics schema will need coordinated extension.

## Durable Notes / Discarded Approaches

- Keeping raw trackers, derived metrics, and exporters separate is a good structural choice; it reduces coupling and makes new diagnostics easier to add.
- Analytics should stay downstream of runtime truth. It is a consumer and summariser, not the source of reward, episode, or environment facts.

## Obsolete / No Longer Relevant

- The old “telemetry is only the HUD” mental model is obsolete. Analytics is now a substantial second observability layer with its own schemas and reports.
