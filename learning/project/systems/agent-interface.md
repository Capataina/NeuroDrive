# The Agent Interface

## What This File Covers

The `agent/` module is the stable boundary between the environment and any controller. It owns exactly two things: the observation contract (what the policy sees) and the action contract (what the policy outputs). Understanding this interface in detail is essential for anyone adding a new controller, modifying what the policy observes, or trying to understand what information A2C is using to make decisions.

**Status:** Current implementation.

## Prerequisites

- `concepts/foundations/bevy-ecs-primer.md` — components, resources, systems
- `project/architecture/module-boundaries.md` — why agent/ is the stable boundary
- `concepts/domain-patterns/observation-design.md` — theory of the 23-dim vector

---

## The Two Contracts

### The Action Contract

Every controller in NeuroDrive — keyboard or neural network — communicates with the physics system through a single struct:

```rust
pub struct CarAction {
    pub steering: f32,   // [-1.0, 1.0]  left = negative
    pub throttle: f32,   // [ 0.0, 1.0]  0 = coast, 1 = full accelerate
}
```

This is stored in `ActionState`, a global singleton resource:

```rust
pub struct ActionState {
    pub desired: CarAction,   // what the controller wants
    pub applied: CarAction,   // what actually gets executed
}
```

**Why separate desired and applied?**

The `desired` vs `applied` split exists to allow action smoothing — a post-processing step that interpolates between the previous applied action and the new desired action. This dampens sudden steering reversals that could cause the car to oscillate.

Currently, smoothing is **disabled** — `applied` is set to exactly `desired`. But the separation means smoothing can be enabled without changing either the controller interface (which only writes `desired`) or the physics interface (which only reads `applied`). Both ends of the contract remain stable.

**Why no brake channel?**

The current action space has no explicit brake. Throttle at zero means coasting (drag decelerates the car), but there is no active braking. This is a known simplification. A brake channel could be added later as an action space extension.

### The Observation Contract

Every controller receives a fixed-size normalised observation vector:

```rust
pub const OBSERVATION_DIM: usize = 23;

pub struct ObservationVector {
    pub values: [f32; OBSERVATION_DIM],
}
```

The A2C model reads `OBSERVATION_DIM` directly, so the model dimensions and the observation vector are always in sync by construction.

---

## The 23-Dimensional Observation Vector

The observation vector encodes the car's sensory relationship to its environment. Every dimension is normalised to a roughly bounded range to make gradient-based learning stable.

### Feature Breakdown

| Indices | Feature | Normalisation | Description |
|---|---|---|---|
| 0–10 | Ray distances | ÷ max_ray_dist | 11 rays cast in a forward-hemisphere fan; each returns distance to the nearest non-road cell |
| 11 | Speed | ÷ max_speed | Current car speed |
| 12 | Signed lateral offset | ÷ half_track_width | Signed perpendicular distance from centreline; positive = left |
| 13 | Signed heading error | ÷ π | Angle between car forward vector and centreline tangent; positive = pointing left of tangent |
| 14 | Angular velocity | ÷ max_angular_velocity | Current yaw rate |
| 15–16 | Lookahead 1: (heading_delta, curvature) | — | Features at the first centreline lookahead point ahead |
| 17–18 | Lookahead 2: (heading_delta, curvature) | — | Features at the second lookahead point |
| 19–20 | Lookahead 3: (heading_delta, curvature) | — | Features at the third lookahead point |
| 21–22 | Lookahead 4: (heading_delta, curvature) | — | Features at the fourth lookahead point |

**Total: 23 dimensions**

### The 11 Rays

The rays are cast in a forward-hemisphere fan centred on the car's forward direction. The specific angles are manually specified in `ObservationConfig`. A hit means the ray reached a non-road cell at that distance; a miss means the ray reached its maximum range without hitting anything.

Rays give the policy direct "can I go straight ahead?" information. A car with tight clearance on the right side will see short rays on the right. A car on a straight with open space will see long rays in all forward directions.

**Dead ray problem:** If the car crashes into a wall, some rays may be very short. If all rays saturate at zero, the gradients through those features vanish. This is a known edge case; the policy should learn to avoid crash states partly to maintain non-saturated observations.

### Centreline-Relative Features

`signed_lateral_offset` and `signed_heading_error` are computed relative to the nearest centreline point and its tangent direction:

```
lateral_offset  = signed distance from car position to centreline
                  positive = car is to the left of centreline direction
heading_error   = angle from car forward to centreline tangent
                  positive = car is pointing left of the tangent direction
```

These two features are the most direct indicators of the car's lane-keeping quality. A policy with low lateral offset and low heading error is driving well on the centreline.

### Why TrackProgress Is Excluded

`TrackProgress.fraction` (how far around the lap the car is) is deliberately **not included** in the observation vector. The rationale:

- If the policy could see `fraction`, it could learn position-specific rules: "at fraction 0.3, always turn right"
- This would produce a brittle policy that memorised the current track layout rather than learning to drive
- Centreline-relative geometry features (heading error, lateral offset, lookahead curvature) give the policy everything it needs to drive well without revealing privileged position information

This is a design choice with trade-offs. A policy that does not know where it is on the track cannot preemptively anticipate an upcoming turn until the lookahead features start showing increasing curvature. For the current circuit, the four lookahead points provide enough anticipation distance, but this would need re-evaluation for more complex layouts.

### The Lookahead Samples

Each lookahead sample is a pair of features computed at a point some distance along the centreline ahead of the current projection:

| Feature | Meaning |
|---|---|
| `heading_delta` | Change in tangent direction between current centreline tangent and the lookahead point tangent |
| `curvature` | Approximate curvature of the centreline at the lookahead point |

The four lookahead points are spaced progressively further along the centreline, giving the policy a "preview" of upcoming track geometry. A turn that is still far ahead will appear as gradual curvature build-up; a sharp corner will show high curvature in the nearest lookahead point.

---

## Observation Production Pipeline

The observation is built in two systems inside `SimSet::Measurement`:

```
update_sensor_readings_system  →  build_observation_vector_system
```

### update_sensor_readings_system

Performs the actual world queries:
- Cast each ray from the car's current position using `TrackGrid.raycast()`
- Compute lateral offset and heading error from `TrackProgress`
- Sample lookahead points along `TrackCenterline`
- Collect angular velocity from car physics

Stores results in `SensorReadings` — a component attached to the car entity that holds the raw, un-normalised measurements.

### build_observation_vector_system

Reads `SensorReadings` and applies normalisation to produce `ObservationVector`. This separation is useful because:
- `SensorReadings` is used by the debug overlay (which needs raw values for display)
- `ObservationVector` is used by the A2C model (which needs normalised values for stable learning)

### Why These Run After the Episode System

Both observation systems run after `episode_loop_system`. If an episode ends this tick and the car resets to spawn, the sensor and observation update must operate on the **post-reset car position**, not the crash position.

If observations were built before the reset:
- the A2C brain would receive crash-state observations as the first observation of the new episode
- the rollout buffer would record a starting observation that does not correspond to the actual starting state

This subtle ordering requirement has caused bugs in earlier versions of the project. The current order is correct:

```
update_track_progress_system
    ↓
episode_loop_system  (reset happens here)
    ↓
update_sensor_readings_system
    ↓
build_observation_vector_system
```

---

## The Smoothing Layer

`ActionSmoothing` is a struct with:
- `enabled: bool` (default: `false`)
- `alpha: f32` — interpolation factor

When enabled:
```
applied = lerp(previous_applied, desired, alpha)
```

When disabled:
```
applied = desired
```

The smoothing system runs last in `SimSet::Input`, after all controllers have written their desired action. This ensures smoothing applies uniformly regardless of which controller (keyboard or A2C) wrote the desired action.

---

## Controller Mode Gate

`AgentMode` is a resource in `brain/types.rs` that is `Keyboard` or `Ai`.

The keyboard input system checks this at the start of each tick:

```rust
fn keyboard_action_input_system(mode: Res<AgentMode>, ...) {
    if *mode != AgentMode::Keyboard { return; }
    // ... write to ActionState.desired
}
```

Similarly, the A2C act system skips when mode is `Keyboard`. This mutual gate means exactly one controller writes to `desired` per tick.

**Note:** The `agent/` module imports `AgentMode` from `brain::types` only for this mode gate. This is the only direction in which `agent/` touches `brain/`. It references only the type, not any brain implementation.

---

## What the Agent Interface Guarantees

1. **Single observation format:** Any controller always receives the same 23-dimensional normalised vector. The observation contract does not change based on controller type.

2. **Single action format:** Any controller always writes to `ActionState.desired`. The physics system always reads `ActionState.applied`. No controller bypasses this contract.

3. **Post-reset consistency:** Observations always reflect the car's current position after any episode reset. There is no state where the observation and car position are misaligned.

4. **Dimension safety:** `OBSERVATION_DIM = 23` is a shared constant used by both the observation builder and the A2C model. The constant being shared is the only runtime protection against dimension mismatch. (There is no dedicated assertion that producers and consumers are aligned — this is a known gap.)

---

## Known Gaps and Future Directions

| Gap | Impact |
|---|---|
| No observation schema versioning | If the observation vector changes, old snapshots and rollout buffers become incompatible silently |
| No runtime dimension assertion | A mismatch between `OBSERVATION_DIM` and model input size would panic at runtime, not fail at compile time |
| No brake channel | Throttle-only deceleration limits aggressive cornering strategies |
| No saturation diagnostics | There is no runtime check for dead rays or saturated features |
| Manual ray layout | Ray angles are hard-coded rather than derived from a parametric spread specification |

---

## Related Files

- `concepts/domain-patterns/observation-design.md` — why these 23 features, theory of observation design
- `project/systems/environment-system.md` — how TrackProgress and TrackGrid are produced
- `project/systems/a2c-brain.md` — how ObservationVector is consumed by A2C
- `project/architecture/fixed-tick-pipeline.md` — ordering constraints on observation build
- `project/architecture/module-boundaries.md` — why agent/ is the stable boundary
