# System — Telemetry and Observability

## Scope / Purpose

- Add lightweight, always-available metrics so “looks like learning” cannot be mistaken for actual improvement.
- Ensure metrics are meaningful even before learning exists (manual driving and heuristics should produce sensible numbers).

## Current Implemented System

- There is a minimal in-world driving-state HUD toggled by F3 (`src/debug/hud.rs`, `src/debug/overlays.rs::debug_overlay_toggle_system`).
- Logging exists for track spawn, car spawn, and collision reset events (`src/maps/monaco.rs`, `src/game/car.rs`, `src/game/collision.rs`).
- A continuous progress fraction is computed and shown in the HUD (`src/game/progress.rs::TrackProgress`, `src/debug/hud.rs::update_driving_hud_text_system`).
- Episode-level telemetry is computed in fixed update (`src/game/episode.rs`) including reward accumulation, crash counts, and moving averages.
- The HUD currently includes heading error, death count, best progress with life index, current reward, episode crash count, and moving-average summaries (`src/debug/hud.rs::update_driving_hud_text_system`).

## Implemented Outputs / Artifacts (if applicable)

- Logs: informational messages only, without numeric episode/progress metrics (`info!` calls in `src/maps/monaco.rs`, `src/game/car.rs`, `src/game/collision.rs`).
- On-screen metrics (F3): progress %, heading error, deaths, best progress + best life, reward, episode crash count, and moving averages.

## In Progress / Partially Implemented

- None tracked in `context/` yet.

## Planned / Missing / To Be Changed

- Reward decomposition detail is still coarse (currently progress delta + crash penalty + lap bonus; no extra terms yet).

## Notes / Design Considerations (optional)

- “Telemetry” should be defined before learning so it can be validated with manual control and deterministic replays.

## Discarded / Obsolete / No Longer Relevant

- None tracked in `context/` yet.
