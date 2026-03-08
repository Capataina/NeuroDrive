# NeuroDrive's Observation System

The observation system transforms the raw game world into a fixed-size numerical vector that the agent's neural network can consume. It operates in two phases: first collecting raw sensor readings from the environment, then normalising them into a compact `[f32; 14]` observation vector. This two-phase design separates physical measurement from mathematical preprocessing, making each independently testable and debuggable.

> **Prerequisites:** [Neural Network Architecture](../../concepts/core/neural_networks/architecture.md), [Car Physics Model](car_physics_model.md)

**Code location:** `src/agent/observation.rs`

---

## Two-Phase Design

**Phase 1 — SensorReadings (raw):** The car's sensors produce physical measurements in world units: ray distances in game units, speed in units/second, heading error in radians, angular velocity in radians/second. These values have different scales, different units, and different ranges.

**Phase 2 — ObservationVector (normalised):** The raw readings are mapped into a `[f32; 14]` array where every element has a bounded, consistent scale. This normalisation is critical — without it, the neural network would need to learn the relative scaling of its inputs purely from data, wasting capacity on a problem that is trivially solved by preprocessing.

The separation also aids debugging: raw sensor readings can be displayed in world-meaningful units (metres, degrees) while the normalised vector can be inspected for correct scaling without knowing the world's unit system.

---

## Ray-Based Distance Sensors

The car casts 11 rays at fixed angles relative to its heading:

`[-150°, -120°, -90°, -60°, -30°, 0°, 30°, 60°, 90°, 120°, 150°]`

These angles span a 300° arc, leaving only a 60° blind spot directly behind the car. Each ray measures the distance from the car to the nearest track boundary (the transition from road to grass).

### Ray Marching

Each ray is evaluated by stepping along it in increments of `step_size = 3.0` world units, up to a maximum range of `max_range = 375.0` units. At each step, the function `is_road_at()` tests whether the sample point falls on a road tile. The ray continues advancing as long as the point is on road.

When a step transitions from road to off-road, we know the boundary lies between the previous on-road point and the current off-road point. If no transition occurs within `max_range`, the ray reports its maximum distance.

### Binary Search Refinement

The coarse ray-march identifies an interval `[last_on_road, first_off_road]` of width `step_size = 3.0` units containing the exact boundary. A binary search then narrows this interval over 8 iterations, halving it each time:

`3.0 / 2^8 = 3.0 / 256 ≈ 0.012 units`

This gives sub-pixel precision for the boundary distance without requiring a tiny (and expensive) step size for the initial march. The combination of coarse marching plus binary refinement balances accuracy against computational cost — the march needs at most `375/3 = 125` steps per ray, and the refinement adds only 8 more steps.

With 11 rays, the total cost is at most `11 × (125 + 8) = 1,463` point-in-tile tests per observation, each of which is an O(1) lookup on the tile grid.

---

## Speed

Speed is computed as the Euclidean norm of the car's velocity vector:

`speed = √(vx² + vy²)`

This is a scalar magnitude — the observation does not encode the velocity direction separately. The heading and ray pattern already provide directional context, and including velocity components would add redundant information that correlates strongly with heading.

---

## Heading Error

Heading error measures the signed angular difference between the car's forward direction and the centreline tangent at the car's nearest projected point:

`heading_error = signed_angle_between(car_forward, centreline_tangent)`

The result is wrapped to `[-π, π]`. A heading error of zero means the car is pointing along the track. Positive values indicate the car is angled to the left of the track direction; negative values indicate rightward deviation.

This signal is essential for the agent to distinguish "facing the right way on a straight" from "facing sideways about to crash." Without it, the agent would need to infer alignment purely from the asymmetry of its ray pattern — possible but far harder to learn.

---

## Angular Velocity

Angular velocity measures how fast the car's heading is changing:

`angular_velocity = wrap_angle(heading - prev_heading) / dt`

The `wrap_angle` function ensures the difference is computed correctly across the ±π boundary. This signal tells the agent whether it is currently rotating and how fast — crucial for anticipating the effect of steering inputs and for stabilising cornering.

---

## Normalisation

All 14 observation elements are normalised to bounded ranges suitable for neural network input:

| Signal | Count | Raw range | Normalisation | Output range |
|---|---|---|---|---|
| Ray distances | 11 | `[0, 375]` | `÷ max_range` | `[0, 1]` |
| Speed | 1 | `[0, max_speed]` | `÷ max_speed` | `[0, 1]` |
| Heading error | 1 | `[-π, π]` | `÷ π` | `[-1, 1]` |
| Angular velocity | 1 | `[-8, 8]` rad/s | `÷ 8.0` | `[-1, 1]` |

The angular velocity divisor of `8.0` is a practical bound — the car's maximum rotation rate at full steering lock is `rotation_speed = 4.0 rad/s`, so `8.0` provides headroom for transient spikes without compressing the useful signal range.

---

## What Is Excluded: TrackProgress

The observation vector deliberately omits `TrackProgress` — the car's fractional position along the centreline (0.0 at start, 1.0 at finish). This exclusion is an intentional design choice to prevent **privileged information leakage**.

TrackProgress is computed from the centreline projection, which requires global knowledge of the track layout. A real autonomous agent has no access to its absolute position on a road — it can only see through its sensors. If the policy were given TrackProgress, it could learn to simply increase this number, which is trivially correlated with reward. The resulting policy would be brittle: it would fail on any track with a different geometry because its strategy depends on absolute position rather than local perception.

By restricting the observation to locally-observable quantities (ray distances, speed, heading error, angular velocity), the agent must learn a policy that generalises — reacting to the shape of the road ahead rather than memorising a position-indexed control sequence.

TrackProgress remains available to the *environment* for reward computation and episode termination (it is measurement, not observation), but it never enters the agent's observation vector.

---

## Design Considerations

The 14-dimensional observation vector is deliberately compact. Larger observation spaces (more rays, velocity components, previous actions) would provide more information but also increase the network size, the gradient computation cost, and the sample complexity. The current design encodes sufficient information for a competent driving policy while keeping the learning problem tractable for a small handwritten MLP.

The ray angles were chosen to provide dense coverage in the forward hemisphere (30° spacing from -90° to +90°) with sparser rear coverage. The 60° blind spot behind the car is acceptable because the car almost never needs to know what is behind it — it only moves forward.

> **See also:** [Car Physics Model](car_physics_model.md) for the state that feeds into observation, [Progress and Lap System](progress_and_lap_system.md) for how TrackProgress is computed, [Handwritten Neural Network](handwritten_neural_network.md) for how the observation vector is consumed.
