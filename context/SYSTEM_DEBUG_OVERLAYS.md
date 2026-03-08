# System — Debug Visual Overlays

## Scope / Purpose

- Make geometry, measurement, and immediate driving state visually inspectable so environment correctness can be verified before debugging learning behaviour.
- Keep overlays independent and lightweight enough to toggle during live driving and AI runs.

## Current Implemented System

- Overlay state is managed by a single `DebugOverlayState` resource with independent booleans for geometry, sensors, and telemetry (`src/debug/overlays.rs`).
- `F1` toggles geometry overlays, `F2` toggles sensor overlays, and `F3` toggles the driving HUD (`src/debug/overlays.rs::debug_overlay_toggle_system`).
- Geometry overlay draws the centreline polyline, closest projection point, tangent arrow, line from car to projection point, car forward vector, and velocity vector (`src/debug/overlays.rs::draw_geometry_overlay_system`).
- Sensor overlay draws ray segments and hit markers using the same `SensorReadings` data consumed by the observation builder (`src/debug/overlays.rs::draw_sensor_overlay_system`).
- The telemetry toggle controls the visibility of the UI HUD rooted in `DrivingHudRoot` (`src/debug/hud.rs`).

## Implemented Outputs / Artifacts (if applicable)

- World-space gizmo overlays for geometry and sensors (`src/debug/overlays.rs`).
- A screen-space driving HUD panel for textual runtime state (`src/debug/hud.rs`).

## In Progress / Partially Implemented

- The overlay system remains environment-focused; it has not yet been extended for A2C-specific inspection such as sampled actions, policy mean/std, critic value, or rollout status.
- The telemetry toggle label in comments still reflects the old “learning telemetry” intent, but the actual HUD remains primarily driving-state telemetry.

## Planned / Missing / To Be Changed

- There is no dedicated world-space glyph for heading error, despite the HUD showing the scalar value.
- There is no overlay support yet for analytics summaries, episode chunk trends, or brain internals.
- Overlay performance and legibility have not yet been evaluated under long AI training runs.

## Notes / Design Considerations (optional)

- The overlay layer should remain diagnostic only and must not become a source of simulation truth.
- Because the brain subsystem can now drive the car, overlays are no longer just for manual testing; they are also needed to debug autonomous control failures.

## Discarded / Obsolete / No Longer Relevant

- The old assumption that `F3` was future-facing only is no longer correct; it already controls a real driving HUD in the current codebase.
