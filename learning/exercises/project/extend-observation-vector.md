# Exercise: Extend the Observation Vector

## Context

The current 23-dimensional observation vector was designed to be compact and learnable. But for some driving situations — particularly tight corners and high-speed sections — additional features might help. This exercise asks you to design and plan a safe extension to the observation vector without implementing it, focusing on the interface, correctness, and dimension-alignment requirements.

**This is a design exercise, not an implementation exercise.** You should produce a written plan and identify every file that would need to change. You should not modify any code.

## Prerequisites

- `concepts/domain-patterns/observation-design.md` — feature design theory
- `project/systems/agent-interface.md` — the production pipeline and dimension constant
- `exercises/core/trace-observation-vector.md`
- `project/architecture/module-boundaries.md` — which module owns what

---

## The Task

Design an extension to the observation vector that adds **two features** representing the upcoming track curvature more explicitly:

1. **`curvature_gradient`:** The rate of change of curvature between the nearest and second lookahead points — how quickly is the track becoming more or less curved?
2. **`lookahead_distance_to_first_high_curvature`:** A normalised estimate of the distance to the next high-curvature point (sharp corner) along the centreline, or 1.0 if no high curvature is detected within the lookahead range.

These would add dimensions 23 and 24, making `OBSERVATION_DIM = 25`.

---

## Part 1: Identify Every File That Must Change

Before designing the features, identify the complete set of files that are affected by changing `OBSERVATION_DIM`.

Think through:
- Where is `OBSERVATION_DIM` defined?
- What uses `OBSERVATION_DIM` as an array size or model input?
- What systems read or write `ObservationVector`?
- What does the analytics trace capture that might need updating?

List every file path you believe would need to change, and what change would be needed in each.

---

## Part 2: Normalisation Design

For each new feature, specify:
- The raw value range (what are the minimum and maximum possible values?)
- The normalisation function to map to approximately [-1, 1] or [0, 1]
- What value the feature takes when the car is on a straight (no upcoming curvature)
- What value the feature takes when a sharp corner is immediately ahead

### Feature 1: `curvature_gradient`

The curvature at lookahead point k can be approximated from the existing lookahead data. The gradient is:
```
curvature_gradient ≈ (curvature_at_lookahead_2 - curvature_at_lookahead_1) / lookahead_spacing
```

Design the normalisation for this value.

### Feature 2: `lookahead_distance_to_first_high_curvature`

Define "high curvature" as curvature exceeding some threshold `κ_threshold`. The feature scans centreline points ahead until it finds one with `curvature > κ_threshold`, then returns the normalised distance to that point.

Design:
- What is a reasonable `κ_threshold`?
- How is "distance" measured (arc length along centreline)?
- What is the normalisation divisor (maximum possible lookahead distance)?

---

## Part 3: API Changes

The extension requires changes to the stable `agent/` interface. For each change, describe:

1. What changes in `SensorReadings` (if any)?
2. What changes in `ObservationVector`?
3. What changes in `update_sensor_readings_system`?
4. What changes in `build_observation_vector_system`?
5. What changes in `src/brain/a2c/model.rs` (the A2C model input size)?

---

## Part 4: Safety Analysis

For each new feature, identify at least one failure mode or edge case:

1. What if the car is near the end of the closed loop and the lookahead wraps around to the beginning? Does the curvature measurement remain valid?

2. What if `κ_threshold` is set too low? What if set too high?

3. If a new feature is always near 0.0 for most of the track, does it help the policy? Does it hurt (add noise without signal)?

4. What happens to the existing model if `OBSERVATION_DIM` changes from 23 to 25 and the model weights from a previously saved checkpoint are loaded? (Note: the current codebase has no model persistence, but design for it anyway.)

---

## Part 5: Regression Protection

If you implemented this extension, what tests would you add to verify:

1. The new feature values are in the expected normalised range during a test episode?
2. `OBSERVATION_DIM` is consistent between the observation builder and the model?
3. The analytics trace schema is updated to include the new features?

---

## Hints

<details>
<summary>Hint 1 (finding OBSERVATION_DIM usage)</summary>

Search for `OBSERVATION_DIM` in the source tree. It is used in at least:
- `src/agent/observation.rs` (definition and array construction)
- `src/brain/a2c/model.rs` (model input size)

It may also appear in any code that reads or allocates `ObservationVector.values`.

</details>

<details>
<summary>Hint 2 (normalisation for curvature gradient)</summary>

Curvature itself is already stored in the existing lookahead features. The gradient is simply the difference in curvature between two consecutive lookahead points, divided by the arc-length distance between them.

A reasonable normalisation: divide by the maximum expected curvature change per unit distance (which you can estimate from the sharpest corner on the current track). Clamp to [-1, 1] after normalisation.

</details>

<details>
<summary>Hint 3 (lookahead distance feature)</summary>

If no high-curvature point is found within the lookahead range, the feature should return 1.0 (maximum normalised distance, meaning "corner is far away or doesn't exist in range"). If a high-curvature point is found at distance d, the feature returns d / max_lookahead_distance. Clamp to [0, 1].

</details>

## Reflection Questions

After completing the design:

1. The current observation design includes 4 lookahead samples with heading_delta and curvature. Your new features aggregate information across those samples. What is the trade-off between raw lookahead samples (which the policy learns to interpret) versus aggregated higher-level features (which encode your prior knowledge about what matters)?

2. If you added these features and retrained A2C from scratch, would you expect learning to be faster? What evidence would you look for in the analytics to confirm or refute this?

3. The current `ObservationConfig` stores ray configuration. Should the new feature parameters (κ_threshold, max_lookahead_distance) also live in `ObservationConfig`? What are the trade-offs of hardcoding versus configuring these values?

## Related Files

- `concepts/domain-patterns/observation-design.md`
- `project/systems/agent-interface.md`
- `project/architecture/module-boundaries.md`
- `exercises/project/sketch-eligibility-traces.md` — the next project exercise
