# Observation Design

## Why This Matters Here

The policy can only learn from what it can see. If the observation vector does not contain the information needed to make a good driving decision, no learning algorithm — however sophisticated — can compensate. Observation design is therefore one of the most important engineering decisions in this project.

NeuroDrive uses a carefully engineered 23-dimensional observation vector. Understanding why each feature exists, how it is computed, and what information it provides is essential for understanding the learning task.

**Status:** Current implementation.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — what an observation is in the MDP framework

## The Observation Vector

The full 23-dimensional vector (see `references/observation-vector-reference.md` for per-feature details):

```
[0..10]   — 11 raycast distances (normalised)
[11]      — speed (normalised)
[12]      — signed lateral offset from centreline (normalised)
[13]      — signed heading error relative to track tangent (normalised)
[14]      — angular velocity (normalised)
[15..22]  — 4 lookahead samples × 2 features (heading_delta, curvature)
```

`OBSERVATION_DIM = 23` is defined in `src/agent/observation.rs` and used directly by the A2C model constructor.

---

## Feature Categories

### 1. Proximity Sensors (Raycasts)

11 rays are cast from the car, typically in a fan covering forward and lateral directions. Each ray returns the normalised distance to the track boundary (1.0 = far from wall, 0.0 = touching wall).

**Why raycasts?**
- They encode *where the walls are* relative to the car's heading
- The pattern of short vs long distances tells the car whether it is in a straight, a curve, or near an edge
- 11 rays give enough angular resolution to distinguish narrow chicanes from wide straights

**Limitation:** Raycasts see only the near environment. They cannot tell the car what the track does 500ms from now. This is addressed by the lookahead features.

### 2. Speed

A single scalar: the car's current speed normalised to a reasonable maximum.

**Why speed?**
- High speed + large heading error is dangerous (motivates the heading-speed penalty)
- Low speed + straight stretch is sub-optimal (the car should accelerate)
- Speed provides context for all other features: the same lateral offset means different things at 10 km/h vs 100 km/h

### 3. Signed Lateral Offset

The signed distance from the car's position to the centreline, normalised.
- Positive: car is right of centreline
- Negative: car is left of centreline

**Why centralise?**
This feature directly captures whether the car is where it should be on track. A policy that cannot see its own lane position cannot learn to stay centred.

### 4. Signed Heading Error

The signed angle between the car's forward direction and the track tangent at the current centreline projection. Normalised to `[-1, 1]` (dividing by π).
- 0: car is pointing exactly along the track
- Positive: car is pointing right of track direction
- Negative: car is pointing left

**Why signed?**
Direction matters for recovery. If the car is pointing 30° right of the track, the correction is to steer left. An unsigned error cannot distinguish which way to correct.

### 5. Angular Velocity

The car's rotational speed. This captures momentum in the rotational degree of freedom — the car may be heading correctly now but spinning, which will cause a heading error in the next few ticks.

**Why angular velocity?**
Without angular velocity, the car cannot distinguish "pointing correct, stable" from "pointing correct, but rapidly rotating away." The angular velocity is necessary for smooth, stable control.

### 6. Lookahead Samples (4 × 2 features)

Four samples along the centreline ahead of the car, each providing:
- `heading_delta`: the change in track direction at that point relative to the car's current heading
- `curvature`: the local track curvature at that point

**Why lookahead?**
Raycasts and immediate geometry are reactive — they tell the car what is happening now. Lookahead is predictive — it tells the car what the track will require in the next few tenths of a second.

Without lookahead, a car entering a corner has no advance warning. It must react after it is already in the corner. With lookahead, a car can begin adjusting speed and heading *before* the corner — which is how human drivers navigate turns.

**Implementation:** Each lookahead sample is taken at a fixed arc-length distance ahead along the centreline. The four distances are configured in `ObservationConfig`.

---

## What Is Intentionally Excluded

`TrackProgress.fraction` (the normalised arc-length progress around the track) is **not** included in the observation vector. This is a deliberate design choice:

- Including raw progress would give the policy privileged information about where on the track it is — which is not available in the biological analogy (a brain does not have a GPS coordinate of its current position in a task).
- The combination of raycasts, heading error, and lookahead encodes *local geometry* without leaking global track position.
- This keeps the policy dependent on its sensory experience rather than its position in the lap.

---

## Normalisation

All observation features are normalised to approximately `[-1, 1]` or `[0, 1]`:

| Feature | Normalisation |
|---|---|
| Raycasts | Scaled by max range, clamped to [0, 1] |
| Speed | Divided by a max speed constant |
| Lateral offset | Divided by a half-track-width constant |
| Heading error | Divided by π |
| Angular velocity | Divided by a max angular velocity constant |
| Lookahead heading_delta | Divided by π (or similar) |
| Lookahead curvature | Divided by max expected curvature |

**Why normalise?**
Neural networks perform poorly when inputs have very different scales. A speed of 500 units and a heading error of 0.3 radians would produce wildly different neuron activations without normalisation, making learning difficult. Normalisation keeps all features on a comparable scale.

---

## Observation Design Principles

### Markov-Sufficient

The observation should be *approximately* Markov-sufficient for the task — meaning the agent should be able to make a good decision based on the current observation without needing memory of past observations. NeuroDrive's 23-dim vector includes:
- Current position (lateral offset)
- Current orientation (heading error)
- Current dynamics (speed, angular velocity)
- Immediate environment (raycasts)
- Short-horizon future (lookahead)

This is approximately sufficient for stable basic driving behaviour.

### Not Over-Compressed

Adding more features is generally better than fewer, up to the point of confusing the policy with irrelevant or redundant information. The current 23-dim vector is relatively compact but includes the critical geometry features.

### Grounded in Physics

Each feature is a physically meaningful quantity. This makes the observation interpretable: if something in the learning is going wrong, the observation vector can be read and diagnosed.

---

## Known Gaps and Future Considerations

- **No memory:** The current observation is purely instantaneous. A car with memory of the last few observations could handle more complex situations (e.g. recovering from a spin).
- **Single ray bundle:** The current 11-ray fan is manually enumerated. A more principled spread specification (with configurable density) would be easier to iterate on.
- **Centreline as primary geometry:** The lookahead and lateral offset features establish a centreline-relative frame. This is likely the right direction — further experiments could reduce the raycast count and emphasise centreline geometry.

---

## Related Files

- `references/observation-vector-reference.md` — feature-by-feature reference
- `project/systems/agent-interface.md` — where observations are built
- `exercises/core/trace-observation-vector.md` — trace observation construction for one tick
- `exercises/project/extend-observation-vector.md` — design and add a new observation feature
