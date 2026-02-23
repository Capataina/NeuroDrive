# System — Game Environment (Car + Track + Collision + Episode Boundaries)

## Scope / Purpose

- Track the environment mechanics that define “what the agent acts in”, independent of any learning implementation.
- Separate “physical correctness” (movement and collision) from “measurement correctness” (progress, sensors, telemetry) so each can be validated in isolation.

## Current Implemented System

- The simulation spawns a single `Track` entity containing a `TrackGrid` and spawn pose (`src/maps/track.rs::Track`, `src/maps/monaco.rs::spawn_track`).
- The track is a tile grid with explicit road vs empty tiles and per-tile open-edge connectivity (`src/maps/parts/mod.rs::TilePart`, `src/maps/grid.rs::TrackGrid`).
- Road rendering is grid-derived (road surfaces + wall bars and arc walls for corner tiles) rather than spline/boundary geometry (`src/maps/grid.rs::render_tile_grid`).
- Fixed-timestep simulation is configured at 60 Hz (`src/main.rs`) and the core simulation systems run in `FixedUpdate`.
- Car motion is a simple “velocity + drag” integration driven by a stable action interface (`steering`, `throttle`) on the fixed timestep (`src/agent/action.rs`, `src/game/physics.rs`).
- Collision/off-track detection triggers when any corner of the rotated car sprite lies outside the driveable area (`src/game/collision.rs::collision_detection_system`).
- Reset behaviour resets the car transform and velocity to the track’s spawn pose on collision (no despawn/respawn) (`src/game/collision.rs::handle_collision_system`).
- Episode boundaries now include crash, timeout, and lap-complete in a fixed-tick episode loop (`src/game/episode.rs::episode_loop_system`).

## Implemented Outputs / Artifacts (if applicable)

- Visible: car sprite (`src/game/car.rs::spawn_car`) and track visuals (`src/maps/grid.rs::render_tile_grid`).
- Visible: finish line stripe (not currently used for lap detection) (`src/maps/monaco.rs::render_finish_line`).

## In Progress / Partially Implemented

- None tracked in `context/` yet.

## Planned / Missing / To Be Changed

- Action smoothing and/or actuator dynamics are not enabled by default (a simple first-order smoother exists but is currently disabled) (`src/agent/action.rs::ActionSmoothing`).
- Headless/fast-sim execution mode is missing (current app is always a windowed Bevy app in `src/main.rs`).

## Notes / Design Considerations (optional)

- The current `TrackGrid` model is suitable for early collision correctness and debug visualisation, and now also provides a grid-derived centreline for progress measurement (`src/maps/centerline.rs`).
- `TrackGrid::is_road_at` uses a wall-thickness inset so collision corresponds to the inner face of rendered walls (`src/maps/grid.rs::WALL_THICKNESS`, `src/maps/grid.rs::TrackGrid::is_road_at`).
- Runtime heading/forward computations now use quaternion-forward vectors rather than Euler decomposition, reducing 2D heading ambiguity across systems.

## Discarded / Obsolete / No Longer Relevant

- None tracked in `context/` yet.
