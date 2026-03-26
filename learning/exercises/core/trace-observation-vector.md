# Exercise: Trace the Observation Vector

## Context

The observation vector is the only information the policy receives about the world. Understanding exactly what each dimension represents — why it is there, how it is normalised, and what the policy sees when the car is in various states — is essential for debugging, for extending the observation space, and for designing the biological architecture that will eventually replace A2C.

This exercise asks you to trace the full 23-dimensional observation vector for two specific car states.

## Prerequisites

- `concepts/domain-patterns/observation-design.md` — the 23-dim vector theory
- `project/systems/agent-interface.md` — observation production pipeline
- `project/systems/environment-system.md` — TrackProgress, SensorReadings

## The Task

For two car states described below, predict what the observation vector should contain. Then answer the diagnostic questions about what the policy "sees" in each state.

You do not need to run the code. Reason through what each feature value should be based on the car state description.

---

## State A: Car on the Centreline, Going Straight

**Car position:** Exactly on the centreline. Car heading matches centreline tangent exactly. Current section is a straight. Speed = 0.8 * max_speed.

**Predict the observation vector:**

| Index | Feature | Expected Value | Reasoning |
|---|---|---|---|
| 0–10 | Ray distances | ? | Car is on centreline in a straight — which rays should be long? Which short? |
| 11 | Speed (normalised) | ? | Speed = 0.8 * max_speed, normalised by max_speed |
| 12 | Signed lateral offset | ? | Car is exactly on centreline |
| 13 | Signed heading error | ? | Car heading matches tangent exactly |
| 14 | Angular velocity | ? | Car is going straight |
| 15–22 | Lookahead features | ? | Straight section ahead — what are heading_delta and curvature? |

Fill in the table with your predicted values before reading further.

---

## State B: Car Off-Centre in a Left Corner

**Car position:** 0.5 * half_track_width to the right of the centreline (car is between centreline and right wall). Car heading points 20 degrees left of the centreline tangent (car is angled into the corner too early). The section is a moderately tight left turn. Speed = 0.4 * max_speed.

**Predict the observation vector:**

| Index | Feature | Expected Value | Reasoning |
|---|---|---|---|
| 0–10 | Ray distances | ? | Car is offset right in a left corner — which rays hit walls sooner? |
| 11 | Speed (normalised) | ? | Speed = 0.4 * max_speed |
| 12 | Signed lateral offset | ? | Car is 0.5 * half_track_width to the right |
| 13 | Signed heading error | ? | Car heading is 20° left of tangent (car is turning into the corner) |
| 14 | Angular velocity | ? | Car is cornering |
| 15–22 | Lookahead features | ? | Left corner ahead — what should heading_delta and curvature show? |

---

## Sign Convention Questions

Answer these before looking at the code:

1. **Lateral offset sign:** The context file says "positive = left of centreline direction". In State B, the car is to the *right* of centreline. What sign does `signed_lateral_offset` have?

2. **Heading error sign:** The context file says "positive = pointing left of tangent". In State B, the car heading is 20° *left* of the tangent (car is already turning into the left corner). What sign does `signed_heading_error` have?

3. **Ray asymmetry:** In State B, with the car to the right of centreline in a left corner, which rays in the 11-ray bundle should be shorter on average — the left-pointing rays or the right-pointing rays?

---

## Part 2: What Does the Policy See?

For each state, answer:

1. **Is the policy in a "comfortable" state?** State A — the car is perfectly centred with matching heading. State B — the car is off-centre with non-zero heading error. What features distinguish these two states most strongly?

2. **What action would a well-trained policy likely output for each state?**
   - State A: throttle high? Steering neutral?
   - State B: steering correction needed? More or less throttle?

3. **Lookahead value in State A:** If the straight continues for 200m and the next corner is far ahead, the lookahead curvature values should be near zero. But the lookahead *heading delta* should also be near zero (the straight doesn't change direction). Verify this reasoning.

4. **Lookahead value in State B:** If the car is entering a left corner, the lookahead points ahead along the centreline should show positive heading_delta (centreline turns left = positive direction change). Does your State B table reflect this?

---

## Part 3: Saturation and Edge Cases

1. What happens to the ray observations if the car has just crashed and its nose is touching the wall? Which features become saturated (near max or min)?

2. If `angular_velocity` is normalised by `max_angular_velocity`, what value does it take when the car is stationary? What about during a high-speed turn?

3. The observation vector does **not** include `TrackProgress.fraction`. Design a thought experiment: if you added `fraction` as dimension 23, what failure mode might a naive policy learn? Why does excluding it avoid this failure?

---

## Hints

<details>
<summary>Hint 1 (ray bundle geometry)</summary>

The 11 rays are cast in a forward-hemisphere fan. They are symmetric around the car's forward direction. The leftmost and rightmost rays point approximately 90° left and right of forward. When the car is centred on a straight with equal walls on both sides, the outer rays should be roughly symmetric. When the car is offset to one side, the rays on that side will hit the wall sooner.

</details>

<details>
<summary>Hint 2 (lookahead samples)</summary>

Each lookahead sample is computed at a centreline point ahead of the car's projection. The `heading_delta` is the change in tangent direction from the current tangent to the lookahead point's tangent. For a straight section, all tangents are parallel, so heading_delta ≈ 0. For a left corner, the tangent rotates counterclockwise, producing a positive heading_delta (by convention, positive = left turn).

</details>

<details>
<summary>Hint 3 (sign conventions)</summary>

The unit test in the codebase verifies lateral offset sign. Briefly:
- If the car is to the right of the centreline direction (when looking in the direction of travel), `signed_lateral_offset` is negative.
- If the car heading is pointing left of the centreline tangent, `signed_heading_error` is positive.

This means in State B: `signed_lateral_offset < 0` (right of centre) and `signed_heading_error > 0` (angled left into corner).

</details>

## Reflection Questions

After completing the exercise:

1. How does the observation vector encode "danger"? What combination of feature values indicates that the car is in an immediately dangerous situation?

2. The 11 rays and 4 lookahead samples together take up 11 + 8 = 19 dimensions out of 23. Why so many ray dimensions relative to the 4 centreline-relative scalars (speed, lateral offset, heading error, angular velocity)?

3. A well-trained policy uses all 23 dimensions. But at test time, you want to understand what the policy is doing. Which 3-4 features do you think have the most influence on the action output? How would you test this hypothesis?

## Related Files

- `concepts/domain-patterns/observation-design.md` — the design theory
- `project/systems/agent-interface.md` — production pipeline
- `exercises/project/extend-observation-vector.md` — extension exercise that builds on this
