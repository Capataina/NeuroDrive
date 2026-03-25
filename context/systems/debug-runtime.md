# System — Debug Runtime

## Scope / Purpose

- Provide live visual and textual inspection tools during manual driving and AI runs.
- Keep runtime diagnostics lightweight and interactive, separate from offline analytics export.

## Boundaries / Ownership

- `src/debug/overlays.rs` owns world-space gizmo overlays and the `F1`/`F2`/`F3` toggles.
- `src/debug/hud.rs` owns the diagnostics panel, recent-quarter summaries, and run assessment logic.
- `src/debug/plugin.rs` owns debug resource setup and scheduling.
- This subsystem reads runtime state from other systems but must not become a source of simulation truth.

## Current Implemented Reality

- `DebugOverlayState` controls three independent toggles:
  - geometry overlay,
  - sensor overlay,
  - telemetry/HUD visibility.
- Defaults are:
  - geometry on,
  - sensors off,
  - telemetry on.
- Geometry overlay (`F1`) draws:
  - centreline polyline,
  - current projection point,
  - tangent arrow,
  - car forward vector,
  - velocity vector,
  - lookahead preview markers and tangents.
- Sensor overlay (`F2`) draws ray segments and hit points from `SensorReadings`.
- The HUD (`F3`) shows:
  - current progress, lane offset, centreline gap, and heading error,
  - run-level episode/death/best-progress information,
  - moving averages from `EpisodeMovingAverages`,
  - a live A2C learning line when update stats exist,
  - recent-history quarter summaries with an “Improving/Mixed/Regressing/Warm-up” assessment.
- The HUD keeps its own recent episode history split into four real-time quarters for quick run judgement without waiting for exported reports.

## Key Interfaces / Data Flow

| Interface | Source | Debug use |
|---|---|---|
| `Track` and `TrackProgress` | maps/game | geometry overlay and progress display |
| `SensorReadings` | agent | sensor overlay and current HUD values |
| `EpisodeState` and `EpisodeMovingAverages` | game | run status, current reward, quarter summaries |
| `CollisionEvent` | game | death counting in HUD stats |
| `A2cTrainingStats` | brain | live learning-health line in HUD |

- The debug layer updates in two places:
  - fixed tick for stats accumulation,
  - normal `Update` for rendering and text refresh.

## Implemented Outputs / Artifacts

- Runtime resources:
  - `DebugOverlayState`
  - `DrivingHudStats`
  - `DrivingHudHistory`
  - `DrivingHudEpisodeAccumulator`
- Runtime UI:
  - `DrivingHudRoot`
  - fixed-column quarter summary grid
- Runtime world-space overlays via Bevy gizmos.

## Known Issues / Active Risks

- The HUD is informative but still summary-oriented; it does not expose full rollout-buffer internals or detailed per-layer update history.
- Overlay performance under very long AI runs has not been explicitly evaluated.
- The current assessment heuristic is intentionally lightweight and should not be treated as a substitute for offline analytics.

## Partial / In Progress

- The runtime debug layer has already expanded beyond simple environment overlays and now includes a small live learning-health view.
- It remains environment-heavy; direct world-space visualisation of policy mean/std or critic state does not exist yet.

## Planned / Missing / Likely Changes

- More brain-specific live inspection could be added later:
  - sampled versus mean action,
  - rollout size,
  - update cadence,
  - richer network-health display.
- A dedicated heading-error glyph or clearer lane-position visualisation may be worthwhile if geometry debugging becomes harder.

## Durable Notes / Discarded Approaches

- Keeping runtime debug separate from analytics export is the right split:
  - debug is for live plausibility and intervention,
  - analytics is for longer-horizon diagnosis after a run.

## Obsolete / No Longer Relevant

- Any note that still describes `F3` as future-facing is obsolete; it already controls a substantial diagnostics panel in the current runtime.
