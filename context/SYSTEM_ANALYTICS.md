# System — Analytics Export and Run Summaries

## Scope / Purpose

- Persist episode-level results so training behaviour can be inspected after a run rather than only through the live HUD.
- Provide a minimal analysis path that converts raw episode completions into machine-readable and human-readable reports.
- Define a durable analytics taxonomy so performance health, learning health, network health, and exploration health do not get conflated.

## Current Implemented System

- An `AnalyticsPlugin` is wired into the application and initialises an `EpisodeTracker` resource (`src/analytics/plugin.rs`).
- The subsystem is now structurally modular rather than centred on one tracker file: raw analytics schemas live in `src/analytics/models.rs`, fixed-tick capture systems are split across `src/analytics/trackers/action.rs`, `src/analytics/trackers/trace.rs`, and `src/analytics/trackers/episode.rs`, derived metrics live in dedicated files under `src/analytics/metrics/`, and exporters remain isolated under `src/analytics/exporters/`.
- Episode action tracking is implemented as a dedicated accumulator that snapshots steering/throttle mean and standard deviation per completed episode (`src/analytics/trackers/action.rs`).
- Per-tick trace capture is implemented as a dedicated trace accumulator that records progress, speed, heading error, centreline distance, signed lateral offset, raw ray distances, applied controls, reward decomposition, terminal reason, lookahead-derived curvature context, and critic value predictions (`src/analytics/trackers/trace.rs`).
- Completed episode storage is now a thin store/orchestration layer that converts raw accumulators into `EpisodeRecord`, `EpisodeTrace`, and `A2cUpdateRecord` artifacts (`src/analytics/trackers/episode.rs`).
- Each stored episode record now captures reward decomposition, action summaries, lane-following summaries, turn-preparation summaries, curvature-demand-versus-steering summaries, speed-at-turn-entry summaries, lane-position-through-turn summaries, and a heuristic failure-mode classification (`src/analytics/models.rs`, `src/analytics/trackers/episode.rs`).
- Chunked summary metrics now use explicit chunk windows that always produce at most 10 non-empty chunks, preventing the old “11 chunks” drift when episode counts were not divisible in a friendly way (`src/analytics/metrics/chunking.rs`).
- The analytics layer now computes dedicated input-learning chunks for centreline distance, absolute lateral offset, absolute heading error, and grouped ray-clearance signals so the report can state whether the agent is actually learning the intended observation semantics (`src/analytics/metrics/inputs.rs`).
- The analytics layer now computes dedicated turn-execution diagnostics covering curvature-demanded versus actual steering, speed at turn entry and peak curvature, turn preparation latencies, lane position through the corner, and heuristic crash-mode classification (`src/analytics/metrics/turns.rs`).
- Sector diagnostics, critic diagnostics, trajectory snapshots, and section-specific narrative insights are now each implemented in separate metric modules instead of being embedded in the Markdown exporter (`src/analytics/metrics/sectors.rs`, `src/analytics/metrics/critic.rs`, `src/analytics/metrics/trajectory.rs`, `src/analytics/metrics/insights.rs`).
- JSON export now serialises the richer raw tracker state, including the extended tick traces and per-episode derived metrics (`src/analytics/exporters/json.rs`).
- Markdown export is now a renderer over modular metric modules and emits clearer sectioned reports with executive summary bullets, 10-chunk trend tables, grouped ASCII charts, explicit turn-execution tables, failure-mode tables, and section-specific insight callouts (`src/analytics/exporters/markdown.rs`).
- Export triggering on shutdown is now compile-correct and runs from `Last` using Bevy 0.18 `AppExit` messages (`src/analytics/plugin.rs`).
- Episode action statistics are now snapshotted on the fixed tick after episode finalisation, avoiding contamination from the next episode before `Update` tracking runs (`src/analytics/plugin.rs`, `src/analytics/trackers/action.rs`).
- Per-tick trace capture now runs after observation rebuilding and before A2C reward-collection/update, so terminal-step diagnostics include fresh sensor state and pre-update critic predictions (`src/analytics/plugin.rs`, `src/analytics/trackers/trace.rs`).

## Implemented Outputs / Artifacts (if applicable)

- `EpisodeTracker` resource containing run-level episode records, per-tick trajectory traces, and A2C update records (`src/analytics/models.rs`).
- Intended output files under `reports/run_<unix_timestamp>.json` and `reports/run_<unix_timestamp>.md` (`src/analytics/plugin.rs`).

## In Progress / Partially Implemented

- Export triggering is implemented as an on-exit path only; there is no periodic checkpointing, manual export command, or crash-safe flush path.
- The tracked schema now includes first-wave learning-health signals, but it still lacks run metadata such as seed, config snapshot, git revision, and track identity.
- Crash location reporting is currently coarse and summary-oriented rather than a full heatmap representation.
- The new turn-execution failure classification is heuristic rather than ground truth; it is designed to accelerate diagnosis, not to replace direct trace inspection when the labels look suspicious.
- Historical reports generated before the latest environment fixes can still show older reward or observation semantics; those reports should be treated as pre-refactor baselines and re-run before comparing trends.

## Planned / Missing / To Be Changed

- There is no experiment/run metadata such as seed, config values, git revision, active mode, or track identity in the exported files.
- There is no directory/index policy for multiple runs beyond timestamped filenames.
- There is no validation ensuring episode records are recorded exactly once across all end-reason paths.
- There is no offline comparison tooling beyond one Markdown summary report.
- The current implementation still lacks richer visitation-diversity metrics, discounted TD-error distributions, explicit policy-drift metrics, and dedicated run-to-run comparison tooling.
- Brake-aware diagnostics are still absent because the runtime has no brake channel yet; the turn-execution modules are written so that brake timing can be added later without restructuring the exporter again.
- Input-learning diagnostics are currently grouped around interpretable signals (centreline, heading, and ray clearances) rather than a fully generic per-feature schema with metadata-driven “better/worse” semantics.

## Notes / Design Considerations (optional)

- Current analytics taxonomy in code:
  - Raw schemas: `models.rs` stores run-level, episode-level, and tick-level artifacts without presentation logic.
  - Capture layer: `trackers/` owns only fixed-tick accumulation and conversion into stable records.
  - Derived metrics: `metrics/` owns chunking, input-learning trends, turn execution, sector summaries, critic diagnostics, trajectory summaries, and narrative insight generation.
  - Presentation: `exporters/` serialises either raw records (`json.rs`) or a curated report (`markdown.rs`).
- These categories should stay separate in both naming and storage. A high reward with collapsed entropy, negative explained variance, or high dead-unit fraction should be treated as suspicious rather than successful.
- Storage cadence is now explicit:
  - per-tick for raw trajectory context and critic predictions,
  - per-episode for behavioural and turn-execution summaries,
  - per-update for A2C optimisation and network-health signals,
  - per-run for Markdown synthesis and human diagnosis.
- Both raw and summarised data are needed. Raw data captures instability and spikes; summaries capture trends. One without the other is misleading.
- Failure signatures are now treated as first-class outputs at two layers:
  - section-specific insight bullets that explain what a section likely means,
  - per-episode heuristic crash-mode classification that can be aggregated into dominant failure patterns.
- A useful recent example of analytics value in this repository: chunk-windowed centreline and turn-execution reporting can now distinguish “stable local policy” from “actual corner-solving”, which was previously inferred manually from progress tables alone.
- This subsystem should stay focused on data capture and export orchestration, not on deriving environment truth.
- Tracker logic should remain append-only and idempotent per episode; duplicate records would invalidate chunk metrics quickly.
- Exporters should consume stable tracker records and derived metric modules only; they should avoid both live game-state queries and embedded business logic so new diagnostics can be added by extending `metrics/`.

## Discarded / Obsolete / No Longer Relevant

- The previous context folder had no canonical analytics document because this subsystem did not exist; that is now obsolete.
