# System — Debug Visual Overlays

## Scope / Purpose

- Make all geometry and measurement quantities visually inspectable so correctness is provable before learning is introduced.
- Provide simple toggle controls (F1/F2/F3 per `README.md`) to keep the view usable.

## Current Implemented System

- Debug overlays are implemented using Bevy gizmos (`src/debug/plugin.rs`, `src/debug/overlays.rs`).
- F1 toggles a geometry overlay and F2 toggles a sensor overlay (`src/debug/overlays.rs::debug_overlay_toggle_system`).
- F3 toggles an on-screen driving-state HUD (`src/debug/overlays.rs::debug_overlay_toggle_system`, `src/debug/hud.rs`).
- The geometry overlay currently draws:
  - track centreline polyline
  - closest projection point and tangent vector (from `TrackProgress`)
  - line from car position to projection point
  - car forward vector
  - car velocity vector (`src/debug/overlays.rs::draw_geometry_overlay_system`).
- The sensor overlay currently draws:
  - raycast segments from car position
  - ray hit points (`src/debug/overlays.rs::draw_sensor_overlay_system`).
- The driving-state HUD currently displays:
  - progress percentage
  - heading error (degrees)
  - death count
  - best progress percentage and life index
  - reward and moving-average telemetry (`src/debug/hud.rs::update_driving_hud_text_system`).

## Implemented Outputs / Artifacts (if applicable)

- Visible (debug): centreline/projection gizmos (F1), sensor ray/hit gizmos (F2), and driving-state HUD (F3).

## In Progress / Partially Implemented

- None tracked in `context/` yet.

## Planned / Missing / To Be Changed

- Heading error is currently presented in the HUD, not as a dedicated world-space geometry glyph.

## Notes / Design Considerations (optional)

- Overlays should be independently toggleable so heavy debug views do not impact “normal driving” visibility.

## Discarded / Obsolete / No Longer Relevant

- None tracked in `context/` yet.
