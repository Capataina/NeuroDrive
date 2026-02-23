# System — Sensors and Observation Vector

## Scope / Purpose

- Define and implement a stable, normalised observation vector that is sufficient for autonomous driving without privileged information.
- Provide a raycast sensor system with visual overlays to validate geometry and scaling.

## Current Implemented System

- A fixed-layout raycast sensor system exists and samples road-boundary distances against `TrackGrid::is_road_at()` each fixed tick (`src/agent/observation.rs::update_sensor_readings_system`).
- A stable fixed-size observation vector exists and is normalised/clamped per feature (`src/agent/observation.rs::ObservationVector`, `src/agent/observation.rs::build_observation_vector_system`).
- Heading and ray orientation are derived from quaternion forward vectors in world space (not Euler extraction), keeping 2D heading consistent with runtime physics.
- The only agent control path today is manual keyboard input mapped into the stable action interface (`src/agent/action.rs::keyboard_action_input_system`).

## Implemented Outputs / Artifacts (if applicable)

- Runtime components on the car:
  - `SensorReadings` (raw ray hits/distances, speed, heading error, angular velocity)
  - `ObservationVector` (normalised features) (`src/agent/observation.rs`).
- Current ray layout uses 11 rays including:
  - one direct-forward ray (0°)
  - front/side rays
  - two rear-oblique rays at approximately ±150° for drift/back-corner awareness (`src/agent/observation.rs::ObservationConfig`).
- Debug sensor overlay support exists (F2) and draws ray lines + hit points (`src/debug/overlays.rs::draw_sensor_overlay_system`).

## In Progress / Partially Implemented

- None tracked in `context/` yet.

## Planned / Missing / To Be Changed

- Sensor parameter tuning is still pending (ray count/angles/range and normalisation scales are defaults and not yet calibrated by task metrics).
- Observation contract versioning is missing (no explicit schema/version tag yet for replay/logging compatibility).

## Notes / Design Considerations (optional)

- A centreline projection and tangent now exist (`src/game/progress.rs::TrackProgress`), so heading-error features can be implemented once the observation vector is defined.

## Discarded / Obsolete / No Longer Relevant

- None tracked in `context/` yet.
