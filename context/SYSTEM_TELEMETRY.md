# System — Telemetry and Observability

## Scope / Purpose

- Make environment and controller behaviour inspectable so visual plausibility is not mistaken for actual improvement.
- Separate immediate runtime observability from longer-horizon analytics export, while keeping both grounded in real episode data.

## Current Implemented System

- A runtime driving HUD exists and is toggled with `F3` (`src/debug/hud.rs`, `src/debug/overlays.rs`).
- The HUD shows progress, heading error, deaths, best progress with life index, current reward, current episode crash count, last end reason, and moving averages (`src/debug/hud.rs::update_driving_hud_text_system`).
- World-space overlays exist for centreline/projection geometry and for sensor raycasts (`src/debug/overlays.rs`).
- Informational logs exist for track spawn, car spawn, mode toggles, overlay toggles, collision reset, and analytics export attempts.
- Episode-level rolling means are computed in `EpisodeMovingAverages` and feed the HUD directly (`src/game/episode.rs`).

## Implemented Outputs / Artifacts (if applicable)

- Runtime HUD panel attached to the UI tree and populated from `DrivingHudStats`, `EpisodeState`, `EpisodeMovingAverages`, `TrackProgress`, and `SensorReadings` (`src/debug/hud.rs`).
- Runtime logs through Bevy logging macros across maps, game, brain, debug, and analytics modules.

## In Progress / Partially Implemented

- A separate analytics subsystem now persists episode records and A2C update-health records to JSON/Markdown reports on exit (`src/analytics/`).
- The HUD is still environment-centric; it does not yet expose A2C-specific diagnostics such as rollout size, update count, actor variance, or value loss.

## Planned / Missing / To Be Changed

- Reward decomposition remains coarse in the user-facing telemetry; the HUD still shows only the aggregate reward rather than per-term contributions.
- Learning-specific runtime telemetry is still missing, including policy statistics, update cadence, gradient/optimizer health, and mode-specific status.
- There is still no persisted run index, experiment metadata, or configuration snapshot attached to analytics outputs.

## Notes / Design Considerations (optional)

- The repository now has two observability layers with different jobs:
  - debug/HUD for immediate runtime inspection,
  - analytics/export for post-run summaries.
- These should stay separate: the HUD should stay lightweight and interactive, while analytics should be oriented around episode history and offline comparison.
- The telemetry story is therefore broader than the old single-file description and must be read together with `SYSTEM_ANALYTICS.md`.

## Discarded / Obsolete / No Longer Relevant

- The old assumption that telemetry was only an on-screen HUD is no longer accurate now that `src/analytics/` exists.
