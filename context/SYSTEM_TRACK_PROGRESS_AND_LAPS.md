# System — Track Progress Metric and Lap Logic

## Scope / Purpose

- Provide a continuous progress metric based on projecting the car onto a centreline parameterisation.
- Implement lap completion and episode termination conditions that are stable and visually verifiable.

## Current Implemented System

- A closed centreline polyline is derived from `TrackGrid` connectivity and stored on the `Track` component (`src/maps/centerline.rs`, `src/maps/track.rs`, `src/maps/monaco.rs::spawn_track`).
- The car has a `TrackProgress` component that stores closest projection point, tangent, arc-length `s`, and progress fraction (`src/game/progress.rs::TrackProgress`, `src/game/progress.rs::update_track_progress_system`).
- Lap completion is detected from progress wrap (armed threshold + wrap window) in the episode loop (`src/game/episode.rs::episode_loop_system`).
- A finish line stripe is still rendered as a visual marker (`src/maps/monaco.rs::render_finish_line`).

## Implemented Outputs / Artifacts (if applicable)

- Visible: finish line stripe (visual-only) (`src/maps/monaco.rs::render_finish_line`).
- Debug: F1 geometry overlay draws the centreline, projection point, and tangent arrow (`src/debug/overlays.rs`).

## In Progress / Partially Implemented

- None tracked in `context/` yet.

## Planned / Missing / To Be Changed

- Finish-line geometry is not yet directly used for crossing checks (lap logic is progress-wrap based).

## Notes / Design Considerations (optional)

- The current centreline is grid-derived and assumes a single closed-loop (degree-2) track; ambiguous branches are rejected rather than guessed (`src/maps/centerline.rs::TrackCenterline::build_closed_loop`).
- Corner tiles contribute quarter-circle arc samples (radius = half tile) so the centreline matches curved turns rather than cutting chords through the corner.

## Discarded / Obsolete / No Longer Relevant

- None tracked in `context/` yet.
