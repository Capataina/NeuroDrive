# System — Sensors and Observation Vector

## Scope / Purpose

- Define the stable observation contract consumed by controllers without exposing privileged environment state such as track progress.
- Provide enough visual and numerical instrumentation to verify sensor geometry before using the observation vector for learning.

## Current Implemented System

- A fixed-layout raycast sensor system samples road-boundary distances against `TrackGrid::is_road_at()` every fixed tick (`src/agent/observation.rs::update_sensor_readings_system`).
- The car stores raw `SensorReadings` and a normalised `ObservationVector` as components attached at spawn (`src/game/car.rs::spawn_car`).
- The observation contract is currently `11` ray distances plus speed, signed heading error, and angular velocity for a total input size of `14` (`src/agent/observation.rs`).
- Heading, signed heading error, and angular velocity are derived from world-space forward vectors and centreline tangent rather than Euler decomposition (`src/agent/observation.rs`, `src/game/progress.rs`).
- A stable `ObservationConfig` resource defines max ray range, ray march step, normalisation scales, and the fixed list of ray angles (`src/agent/observation.rs::ObservationConfig`).
- The brain subsystem already depends on this exact `14`-dimensional observation size when constructing the A2C model (`src/brain/a2c/mod.rs::A2cBrain::default`).

## Implemented Outputs / Artifacts (if applicable)

- `SensorReadings` holds per-ray hit distance, hit position, ray direction, scalar speed, heading error, angular velocity, and previous heading (`src/agent/observation.rs`).
- `ObservationVector` stores the fixed-size normalised feature array consumed by controllers (`src/agent/observation.rs`).
- The F2 sensor overlay draws rays and hit points for runtime visual verification (`src/debug/overlays.rs::draw_sensor_overlay_system`).

## In Progress / Partially Implemented

- The observation contract is no longer only a manual-driving aid; it is now used directly by the partial A2C controller, so stability matters more than the old context implied.
- Feature scaling is implemented, but there is still no explicit validation layer checking for saturation, dead rays, or distribution drift over a run.

## Planned / Missing / To Be Changed

- Sensor tuning remains pending; the current ray count, range, and angles are still hard-coded defaults rather than task-calibrated values.
- Observation contract versioning is still missing, which will matter once replay, snapshots, or offline analytics consume recorded observations.
- There is still no explicit schema/assertion layer ensuring the brain input dimension stays aligned with the observation producer.
- The current angle layout is manually enumerated rather than derived from a higher-level spread specification.

## Notes / Design Considerations (optional)

- `TrackProgress` exists in runtime state but is intentionally excluded from the observation vector to avoid privileged progress leakage into the policy input.
- Because the A2C code now consumes this vector directly, any future observation change must be coordinated with brain-model construction, snapshot compatibility, and analytics output.
- The current design keeps raw sensor measurements and the normalised policy input separate, which is useful for debugging and future exporter work.

## Discarded / Obsolete / No Longer Relevant

- The previous assumption that observations were only preparatory scaffolding for future learning is obsolete; they are now already part of a live, partial controller path.
