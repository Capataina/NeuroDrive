# Plan — Codebase Cleanup

## Goal

Remove unused code, dead implementations, and stale structures that have accumulated through the rapid iteration on PPO, reward shaping, and analytics. The codebase currently generates ~20 dead-code warnings on every build.

## Motivation

Unused code creates noise: false warnings hide real issues, dead modules mislead readers about what's active, and stale structures consume mental overhead during navigation. A cleanup pass should be done periodically, especially after major architectural changes like the ReLU→Tanh switch and reward overhaul.

## Known Dead Code (from cargo warnings and code inspection)

### Brain / Common

- `Relu` struct in `mlp.rs` — unused since Tanh switch. Keep for reference but consider gating with `#[allow(dead_code)]` or removing entirely.
- `Brain` trait in `types.rs` — dead code, never used by the vectorised path. Remove.
- `glorot_uniform` in `math.rs` — still imported by `Linear::new` but `Linear::new` itself may be unused now that `new_orthogonal` is the default. Check call sites.

### Analytics Metrics

These modules compile but are never called from the current markdown report pipeline:

- `critic.rs` — `CriticStats`, `CriticDiagnostics`, `compute_critic_diagnostics` — all unused
- `inputs.rs` — `InputLearningChunk`, `InputSignalTrend`, `calculate_input_learning_chunks`, `summarize_input_signal_trends` — all unused
- `insights.rs` — `ReportInsights`, `build_report_insights` — unused
- `turns.rs` — `TurnExecutionSummary`, `summarize_turn_execution` — unused

Decision: these are potentially useful future diagnostics. Either wire them back into the report or remove them. Leaving dead code is the worst option.

### Analytics Models

- Several fields on `ChunkMetrics` are never read (progress_std, progress_median, etc.) — they're computed but not consumed by any exporter
- Several fields on `SectorDiagnosticsRow`, `EpisodeTimeSeries`, `UpdateTimeSeries`, `TrajectorySnapshotRow` are never read

### Game

- `wrap_angle` and `signed_angle_between` are duplicated across `observation.rs` and `episode.rs` — consolidate into a shared utility

### Empty Placeholders

- `src/brain/biological/` — empty directory
- `src/analytics/sessions/` — empty directory

Decision: keep as intentional placeholders or remove. If kept, add a README or module doc explaining their future purpose.

## Approach

1. Run `cargo check` and catalogue all warnings
2. For each warning, classify as: remove (truly dead), wire back in (useful but disconnected), or suppress (intentional placeholder)
3. Consolidate duplicated utilities
4. Remove dead code in one pass
5. Verify `cargo check` produces zero warnings (or only intentional suppressions)
6. Verify `cargo test` still passes

## Status

Planned. Not yet implemented. Low priority relative to reward/spawn/analytics work but should be done before the next major feature addition.
