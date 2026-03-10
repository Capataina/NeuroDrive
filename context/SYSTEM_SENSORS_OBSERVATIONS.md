# System — Sensors and Observation Vector

## Scope / Purpose

- Define the stable observation contract consumed by controllers without exposing privileged environment state such as track progress.
- Provide enough visual and numerical instrumentation to verify sensor geometry before using the observation vector for learning.

## Current Implemented System

- A fixed-layout raycast sensor system samples road-boundary distances against `TrackGrid::is_road_at()` every fixed tick (`src/agent/observation.rs::update_sensor_readings_system`).
- The car stores raw `SensorReadings` and a normalised `ObservationVector` as components attached at spawn (`src/game/car.rs::spawn_car`).
- The observation contract now includes `11` ray distances, speed, signed lateral offset from the centreline, signed heading error, angular velocity, and `4` centreline lookahead samples with heading-delta and curvature features for a total input size of `23` (`src/agent/observation.rs`).
- Heading, signed heading error, and angular velocity are derived from world-space forward vectors and centreline tangent rather than Euler decomposition (`src/agent/observation.rs`, `src/game/progress.rs`).
- Signed lateral offset is computed from the car position relative to the closest centreline point using the centreline left-normal, giving the policy an explicit lane-placement signal rather than only direction-of-travel alignment (`src/agent/observation.rs`).
- A stable `ObservationConfig` resource defines max ray range, ray march step, lookahead distances, and normalisation scales, including the lateral-offset scale used for the new controlled observation experiment (`src/agent/observation.rs::ObservationConfig`).
- The brain subsystem now depends on `OBSERVATION_DIM` directly when constructing the A2C model, reducing hard-coded input-size drift risk (`src/agent/observation.rs`, `src/brain/a2c/mod.rs`).
- Sensor/observation rebuild now runs after episode finalisation, and episode resets explicitly resynchronise `TrackProgress`, so post-terminal observations align with the reset spawn state instead of stale crash state (`src/agent/plugin.rs`, `src/game/episode.rs`).

## Implemented Outputs / Artifacts (if applicable)

- `SensorReadings` holds per-ray hit distance, hit position, ray direction, scalar speed, signed lateral offset, heading error, angular velocity, and previous heading (`src/agent/observation.rs`).
- `ObservationVector` stores the fixed-size normalised feature array consumed by controllers (`src/agent/observation.rs`).
- The F2 sensor overlay draws rays and hit points for runtime visual verification (`src/debug/overlays.rs::draw_sensor_overlay_system`).

## In Progress / Partially Implemented

- The observation contract is no longer only a manual-driving aid; it is now used directly by the partial A2C controller, so stability matters more than the old context implied.
- The signed lateral-offset addition is being treated as a controlled experiment: the existing `11`-ray bundle remains unchanged so any behavioural change can be attributed to explicit lane-position information rather than simultaneous sensor simplification.
- Feature scaling is implemented, but there is still no explicit validation layer checking for saturation, dead rays, or distribution drift over a run.

## Planned / Missing / To Be Changed

- A centreline-first observation hierarchy is planned but not yet implemented; the intended direction is to treat centreline-relative features as primary and raycasts as safety support rather than the main representation.
- A reduced ray bundle experiment is planned after the signed-offset result is measured; the current candidate set is five rays (forward, two forward diagonals, and two side rays).
- A future action-space experiment may add an explicit brake channel while keeping it separate from this observation-only change.
- Sensor tuning remains pending; the current ray count, range, and angles are still hard-coded defaults rather than task-calibrated values.
- Observation contract versioning is still missing, which will matter once replay, snapshots, or offline analytics consume recorded observations.
- There is still no explicit schema/assertion layer ensuring the brain input dimension stays aligned with the observation producer.
- The current angle layout is manually enumerated rather than derived from a higher-level spread specification.

## Notes / Design Considerations (optional)

- `TrackProgress` exists in runtime state but is intentionally excluded from the observation vector to avoid privileged progress leakage into the policy input.
- The new lookahead features remain ego-relative and geometry-derived (heading/curvature ahead), which provides turn anticipation without exposing absolute track progress as a direct scalar.
- Signed lateral offset is considered acceptable observation truth because it describes current lane placement relative to the local centreline, not future progress or completion state.
- Because the A2C code now consumes this vector directly, any future observation change must be coordinated with brain-model construction, snapshot compatibility, and analytics output.
- The current design keeps raw sensor measurements and the normalised policy input separate, which is useful for debugging and future exporter work.

## Discarded / Obsolete / No Longer Relevant

- The previous assumption that observations were only preparatory scaffolding for future learning is obsolete; they are now already part of a live, partial controller path.
