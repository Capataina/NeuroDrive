# Immediate Implementation Plan — Analytics

## Header

- [ ] Status: Proposed, not yet executed.
- [ ] Scope: Make the analytics subsystem compile, produce correct run reports, and reflect the resulting reality in `context/`.
- [ ] Exit rule: stop when analytics compiles, exports are triggerable and verified, context matches reality, and this file is archived or removed.

## Implementation Structure

- [ ] Modules / files affected (expected):
  - `src/analytics/plugin.rs`
  - `src/analytics/trackers/episode.rs`
  - `src/analytics/exporters/json.rs`
  - `src/analytics/exporters/markdown.rs`
  - `src/analytics/metrics/chunking.rs`
  - `src/game/episode.rs`
  - `context/SYSTEM_ANALYTICS.md`
  - `context/SYSTEM_TELEMETRY.md`
- [ ] Responsibility boundaries:
  - analytics owns tracking, aggregation, and export orchestration.
  - game owns episode truth and reward truth.
  - exporters must consume tracked records, not live gameplay queries.
  - debug/HUD must remain independent from file export.
- [ ] Function inventory:
  - `on_exit_system` in `src/analytics/plugin.rs`
    - Inputs/outputs: consumes the app-exit signal and the tracker, then writes report files as a side effect.
    - Kind: orchestrator.
    - Called by: `AnalyticsPlugin` in `Update`.
  - `episode_tracker_system` in `src/analytics/trackers/episode.rs`
    - Inputs/outputs: reads `EpisodeState`, appends at most one record for the finished episode, mutates `EpisodeTracker`.
    - Kind: orchestrator with idempotence responsibility.
    - Called by: `AnalyticsPlugin` in `Update`.
  - `calculate_chunks` in `src/analytics/metrics/chunking.rs`
    - Inputs/outputs: maps raw episode records to chunk summary metrics.
    - Kind: helper, pure.
    - Called by: Markdown exporter.
  - `export_to_json` in `src/analytics/exporters/json.rs`
    - Inputs/outputs: serialises tracker records to a JSON file path.
    - Kind: helper with file-I/O side effects.
    - Called by: `on_exit_system`.
  - `export_to_markdown` in `src/analytics/exporters/markdown.rs`
    - Inputs/outputs: renders a Markdown summary report from tracker records to a file path.
    - Kind: helper with file-I/O side effects.
    - Called by: `on_exit_system`.
- [ ] Wiring summary:
  - episode finishes in `game::episode`
  - tracker observes the finished-episode snapshot and appends one record
  - exit handling detects app shutdown
  - exporters serialise raw records and derived chunk summaries

## Algorithm / System Sections

### Analytics plugin exit handling

The plugin is the orchestration boundary. Correct behaviour means the plugin compiles against the current Bevy API, triggers export exactly when the app exits, and does not own any episode truth itself.

The recommended default is to adapt exit handling to Bevy 0.18 messages and keep export-on-exit as the primary trigger. The alternative is to move export to an explicit keybind or command; that is safer to test interactively, but worse as the default because it makes persistence optional and easier to forget.

- [ ] Discovery (bounded):
  - [ ] Read `src/analytics/plugin.rs`.
  - [ ] Read one working Bevy 0.18 message usage example already in the repo, such as `src/game/collision.rs`.
  - [ ] Confirm whether `AppExit` is handled as a message in the current Bevy version used by `Cargo.toml`.
- [ ] Implementation playbook:
  - [ ] Replace the outdated exit reader type with the correct Bevy 0.18 reader API.
  - [ ] Keep export path construction inside the plugin so exporters remain format-focused.
  - [ ] Ensure one exit event leads to one export sequence.
  - [ ] Keep the “no records, no export” short-circuit.
- [ ] Stop & verify checkpoints:
  - [ ] Run `cargo check` as soon as the plugin compiles locally.
  - [ ] Start the app, close it after producing at least one completed episode, and confirm report files appear under `reports/`.
- [ ] Invariants / sanity checks:
  - [ ] Export must not panic when there are zero records.
  - [ ] Export must create the parent directory if it does not exist.
  - [ ] Exit handling must not duplicate files for a single shutdown.
- [ ] Minimal explicit test requirements:
  - [ ] Manual verification of one JSON file and one Markdown file written on exit.
  - [ ] At least one compile check after the API correction.

### Episode tracking correctness

The tracker is only trustworthy if it appends one record per completed episode and never duplicates a finished episode while `last_end_reason` remains set across frames. Correct behaviour means the stored values match the snapshot produced by `EpisodeState`.

The recommended default is to keep tracking logic idempotent by episode id. The alternative is to clear the game-side end marker immediately after tracking; that reduces tracker complexity, but couples analytics timing back into the game lifecycle and is worse now.

- [ ] Discovery (bounded):
  - [ ] Read `src/analytics/trackers/episode.rs`.
  - [ ] Read the episode finalisation path in `src/game/episode.rs`.
  - [ ] Verify how long `last_end_reason` remains populated across frames.
- [ ] Implementation playbook:
  - [ ] Confirm the finished episode id calculation matches the game-side increment timing.
  - [ ] Preserve append-only tracker behaviour.
  - [ ] If needed, add a small tracker-side guard field rather than changing game truth.
  - [ ] Keep the recorded schema minimal unless a clear analytics need justifies expansion.
- [ ] Stop & verify checkpoints:
  - [ ] Drive through at least two episode endings of different types and inspect the resulting tracker/export outputs.
  - [ ] Confirm there are no duplicate episode ids in the written JSON.
- [ ] Invariants / sanity checks:
  - [ ] `records` must be strictly ordered by `episode_id`.
  - [ ] One finished episode must yield exactly one record.
  - [ ] Tracker logic must not mutate `EpisodeState`.
- [ ] Minimal explicit test requirements:
  - [ ] Manual inspection of exported episode ids and end reasons.
  - [ ] Add a focused unit test only if the tracking edge case becomes non-obvious.

### Report content and chunk metrics

The exporters convert stored episode records into reusable artefacts. Correct behaviour means the JSON reflects raw tracker state and the Markdown reflects derived chunk summaries without silently dropping episodes or misreporting percentages.

The recommended default is to keep raw JSON as the source export and use Markdown as a thin presentation layer. The alternative is to export only Markdown; that is worse because it removes machine-readable post-processing.

- [ ] Discovery (bounded):
  - [ ] Read `src/analytics/exporters/json.rs`.
  - [ ] Read `src/analytics/exporters/markdown.rs`.
  - [ ] Read `src/analytics/metrics/chunking.rs`.
- [ ] Implementation playbook:
  - [ ] Verify percentage formatting is applied only at presentation time, not to stored values.
  - [ ] Verify chunking covers all records with no gaps or double-counting.
  - [ ] Keep Markdown generation deterministic for the same tracker contents.
  - [ ] If metadata is added, ensure both exporters use the same run identity.
- [ ] Stop & verify checkpoints:
  - [ ] Export a short run and inspect the first and last episode ids in both formats.
  - [ ] Confirm the Markdown chunk table matches the JSON totals for a simple manual spot-check.
- [ ] Invariants / sanity checks:
  - [ ] Raw records remain unmodified during export.
  - [ ] Chunk order follows episode order.
  - [ ] Empty tracker input must produce no misleading non-empty report.
- [ ] Minimal explicit test requirements:
  - [ ] Manual file inspection for one sample run.
  - [ ] Optional pure unit test for `calculate_chunks` if its behaviour changes.

## Integration Points

- [ ] Where it plugs into the existing pipeline:
  - `episode_loop_system` finalises episode state.
  - `episode_tracker_system` reads that state during `Update`.
  - `on_exit_system` writes reports during app shutdown.
- [ ] Order of execution and lifecycle placement:
  - simulate episode
  - finalise episode
  - track finished episode
  - exit app
  - export reports
- [ ] Pre-conditions:
  - `EpisodeState` must represent the last finished episode unambiguously.
  - `EpisodeTracker` must be initialised.
- [ ] Post-conditions:
  - tracker contains one record per completed episode.
  - report files exist and are readable after exit.

## Debugging / Verification

- [ ] Required logs, assertions, or inspection steps:
  - log when export starts and how many records are being written.
  - inspect whether the first written file path is under `reports/`.
  - inspect one JSON record and one Markdown chunk row for sanity.
- [ ] Common failure patterns:
  - outdated Bevy API causes compile failure in the exit system.
  - persistent `last_end_reason` causes duplicate records.
  - empty or malformed reports indicate exporter/schema drift rather than environment issues.

## Completion Criteria

- [ ] Functional correctness: analytics compiles and exports correct files.
- [ ] Integration correctness: episode tracking is one-record-per-episode and export triggers at shutdown.
- [ ] Tests passing: at minimum `cargo check`; add tests if analytics logic becomes non-trivial.
- [ ] Context documents updated to reflect reality.
- [ ] File archived or removed.
