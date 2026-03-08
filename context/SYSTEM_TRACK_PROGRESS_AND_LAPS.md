# System — Track Progress Metric and Lap Logic

## Scope / Purpose

- Provide a continuous progress measure for reward shaping, telemetry, and lap completion without exposing that measure as a privileged policy input.
- Keep lap logic deterministic, inspectable, and derived from track geometry rather than ad hoc counters.

## Current Implemented System

- A closed centreline polyline is derived from tile-grid connectivity and stored on the `Track` component (`src/maps/centerline.rs`, `src/maps/track.rs`, `src/maps/monaco.rs`).
- The car carries a `TrackProgress` component holding closest point, tangent, arc-length `s`, fraction, and distance to the centreline (`src/game/progress.rs`).
- Progress is recomputed every fixed tick by projecting the current car position onto the centreline (`src/game/progress.rs::update_track_progress_system`).
- Lap completion is detected in the episode loop through an armed wrap rule using `lap_arm_fraction`, `lap_wrap_from_fraction`, and `lap_wrap_to_fraction` (`src/game/episode.rs`).
- Reward shaping already consumes signed progress delta each tick, with wrap-aware handling to avoid false large jumps at the start/finish seam (`src/game/episode.rs`).

## Implemented Outputs / Artifacts (if applicable)

- Runtime `TrackProgress` component on the car (`src/game/progress.rs`).
- F1 geometry overlay visualising the centreline, closest projection point, tangent vector, car forward vector, and projection line (`src/debug/overlays.rs`).
- HUD progress percentage derived from `TrackProgress.fraction` (`src/debug/hud.rs`).

## In Progress / Partially Implemented

- Progress is now used in more places than before: reward accumulation, HUD telemetry, episode summaries, analytics records, and A2C reward collection all depend on it.
- The current implementation assumes one unbranched closed loop; that assumption is enforced by centreline construction but not yet surfaced as a richer validation/reporting layer.

## Planned / Missing / To Be Changed

- Finish-line geometry is still not used directly for crossing checks; lap completion remains progress-wrap based.
- There is still no dedicated regression test for progress continuity or lap detection under resets and edge cases.
- Multi-track support is missing, so the progress subsystem has not yet been validated against alternative layouts or branch-capable tiles.

## Notes / Design Considerations (optional)

- The centreline builder rejects ambiguous branches rather than guessing, which keeps progress meaning stable for the current single-loop environment.
- `TrackProgress` is environment truth, not observation truth, and that separation should be preserved as learning systems expand.

## Discarded / Obsolete / No Longer Relevant

- Earlier context that treated progress mainly as a debug metric is obsolete; it is now a central dependency for reward, telemetry, analytics, and controller training.
