# Observation Vector Reference

A compact reference for the NeuroDrive 23-dimensional observation vector. Use this when you need to quickly look up a specific feature's index, normalisation, or semantics.

**Status:** Current implementation.

---

## Quick-Reference Table

| Index | Feature Name | Raw Value | Normalisation | Bounded Range | Notes |
|---|---|---|---|---|---|
| 0 | ray_0 | distance to wall | ÷ max_ray_dist | [0, 1] | Rightmost ray (~90° right) |
| 1 | ray_1 | distance to wall | ÷ max_ray_dist | [0, 1] | |
| 2 | ray_2 | distance to wall | ÷ max_ray_dist | [0, 1] | |
| 3 | ray_3 | distance to wall | ÷ max_ray_dist | [0, 1] | |
| 4 | ray_4 | distance to wall | ÷ max_ray_dist | [0, 1] | Right-of-forward ray |
| 5 | ray_5 | distance to wall | ÷ max_ray_dist | [0, 1] | Directly forward ray |
| 6 | ray_6 | distance to wall | ÷ max_ray_dist | [0, 1] | Left-of-forward ray |
| 7 | ray_7 | distance to wall | ÷ max_ray_dist | [0, 1] | |
| 8 | ray_8 | distance to wall | ÷ max_ray_dist | [0, 1] | |
| 9 | ray_9 | distance to wall | ÷ max_ray_dist | [0, 1] | |
| 10 | ray_10 | distance to wall | ÷ max_ray_dist | [0, 1] | Leftmost ray (~90° left) |
| 11 | speed | m/s or normalised | ÷ max_speed | [0, 1] | Current car speed |
| 12 | lateral_offset | m | ÷ half_track_width | [-1, 1] | + = left of centreline |
| 13 | heading_error | radians | ÷ π | [-1, 1] | + = facing left of tangent |
| 14 | angular_velocity | rad/s | ÷ max_angular_vel | [-1, 1] | Yaw rate |
| 15 | lookahead_1_heading_delta | radians | (scaled) | approx [-1, 1] | Turn at 1st lookahead point |
| 16 | lookahead_1_curvature | 1/m | (scaled) | approx [-1, 1] | Curvature at 1st point |
| 17 | lookahead_2_heading_delta | radians | (scaled) | approx [-1, 1] | Turn at 2nd lookahead point |
| 18 | lookahead_2_curvature | 1/m | (scaled) | approx [-1, 1] | Curvature at 2nd point |
| 19 | lookahead_3_heading_delta | radians | (scaled) | approx [-1, 1] | Turn at 3rd lookahead point |
| 20 | lookahead_3_curvature | 1/m | (scaled) | approx [-1, 1] | Curvature at 3rd point |
| 21 | lookahead_4_heading_delta | radians | (scaled) | approx [-1, 1] | Turn at 4th lookahead point |
| 22 | lookahead_4_curvature | 1/m | (scaled) | approx [-1, 1] | Curvature at 4th point |

**Total: 23 dimensions.**

---

## Ray Bundle Notes

The 11 rays (indices 0–10) are cast in a forward-hemisphere fan. The exact angular spread and spacing are configured in `ObservationConfig`. The fan is symmetric around the car's forward direction:

- Index 0: rightmost ray (approximately 90° right of forward)
- Index 5: directly forward ray
- Index 10: leftmost ray (approximately 90° left of forward)

**A value near 1.0** means the ray reached its maximum range without hitting a wall — the car has clear space in that direction.

**A value near 0.0** means the ray hit a wall very close — the car is near the wall in that direction.

---

## Sign Conventions

**Lateral offset (index 12):**
- Positive: car is to the LEFT of the centreline direction (when facing in the direction of travel)
- Negative: car is to the RIGHT of the centreline direction
- Zero: car is exactly on the centreline

**Heading error (index 13):**
- Positive: car is pointing to the LEFT of the centreline tangent
- Negative: car is pointing to the RIGHT of the centreline tangent
- Zero: car heading matches centreline tangent exactly

**Angular velocity (index 14):**
- Positive: car is rotating counterclockwise (turning left)
- Negative: car is rotating clockwise (turning right)
- Zero: car is not rotating

**Lookahead heading delta:**
- Positive: the centreline turns LEFT at this lookahead point (left corner ahead)
- Negative: the centreline turns RIGHT at this lookahead point (right corner ahead)
- Zero: the track is straight at this lookahead distance

---

## Typical Value Patterns

| Driving Situation | Notable Feature Values |
|---|---|
| Straight, centred, fast | rays ≈ 1.0; lateral_offset ≈ 0; heading_error ≈ 0; lookahead_curvature ≈ 0 |
| Left corner entry | heading_error > 0 (car starting to turn left); lookahead_heading_delta > 0 (left curve ahead) |
| Right corner entry | heading_error < 0; lookahead_heading_delta < 0 |
| Near right wall | rays on right (indices 0–4) short; rays on left (indices 6–10) long |
| Near left wall | rays on left (indices 6–10) short; rays on right (indices 0–4) long |
| Just crashed | at least one ray ≈ 0.0; lateral_offset may be large; episode reset imminent |
| After episode reset | all features reflect spawn pose: rays symmetric, heading_error ≈ 0 |

---

## What Is Not In the Observation

These values exist in the environment but are **deliberately excluded** from the observation vector:

| Excluded value | Reason for exclusion |
|---|---|
| `TrackProgress.fraction` | Would allow position-memorisation rather than geometry-based driving |
| `TrackProgress.s` (arc length) | Same reason |
| Episode counter or step count | Agent should drive the same regardless of how many episodes have elapsed |
| Crash history | Would allow policy to become more conservative after crashes |
| A2C internal state (rollout buffer) | Policy input must be state-only for the MDP formalism to apply |

---

## Constant Location

```rust
// src/agent/observation.rs
pub const OBSERVATION_DIM: usize = 23;
```

The A2C model input size is set from this constant:
```rust
// src/brain/a2c/model.rs
let in_features = OBSERVATION_DIM;
```

Any change to `OBSERVATION_DIM` must be synchronised with both files.

---

## Related Files

- `concepts/domain-patterns/observation-design.md` — design theory and feature rationale
- `project/systems/agent-interface.md` — production pipeline
- `exercises/core/trace-observation-vector.md` — practice exercise
- `exercises/project/extend-observation-vector.md` — extension design exercise
