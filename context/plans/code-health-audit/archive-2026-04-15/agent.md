# Agent — Code Health Findings

**Systems covered:** `src/agent/observation.rs`, `src/agent/action.rs`
**Finding count:** 2 findings (1 high, 1 medium)

---

## Algorithm Optimisation

### Adaptive Raycast Step Size for Long-Range Rays
- [x] Use a larger initial step size for raycasts that are far from boundaries, falling back to fine stepping near the boundary

**Category:** Algorithm Optimisation
**Severity:** High
**Effort:** Small
**Behavioural Impact:** Negligible (flagged — boundary distance may differ by up to half a coarse step, but binary refinement already handles this)

**Location:**
- `src/agent/observation.rs:287-313` — `raycast_to_road_boundary()`
- `src/agent/observation.rs:316-332` — `refine_boundary_distance()`

**Current State:**
Each raycast marches at a fixed step of 3.0 world units up to a max range of 375.0 units. That is up to 125 grid lookups per ray. With 11 rays per car and 8 cars, the worst case is 11,000 grid lookups per tick just for raycasting.

Many rays — particularly the rear-facing ones (±150°) and the side rays (±90°) — often detect boundaries well within the first 20-30 steps. But the forward-facing rays on a straight frequently march the full 125 steps before hitting the max range.

The grid lookup `is_road_at` is cheap (one world-to-cell conversion + one tile check + one edge check), but 11,000 calls per tick is still a substantial share of the observation system's cost.

**Proposed Change:**
Use a two-phase raycast:
1. **Coarse phase:** Step at 12.0 units (4x current step) until a boundary is detected or max range is reached.
2. **Fine phase:** When a boundary is detected during the coarse phase, fall back to the existing `refine_boundary_distance` binary search between the last safe position and the boundary position.

The binary refinement already exists and runs 8 iterations to narrow the boundary to within 0.02 units. The coarse step just gets to the approximate boundary location faster.

Alternatively, a simpler change: increase the march step to 6.0 units. The binary refinement already provides sub-step precision, so the coarse march only needs to detect *that* a boundary exists in the interval, not *where* exactly it is.

**Justification:**
The current 3.0 unit step provides precision that is immediately discarded by the 8-iteration binary refinement, which achieves precision of `3.0 / 2^8 ≈ 0.012` units regardless of step size. A 6.0 unit step with the same refinement achieves `6.0 / 2^8 ≈ 0.023` units of precision — still far below the 375.0 normalisation range. The observation system divides the distance by 375.0, so the maximum error from doubling the step size is `0.023 / 375.0 ≈ 0.00006` in normalised observation space — completely invisible to the policy.

Even with the two-phase approach (12.0 unit coarse step), the maximum error after binary refinement is `12.0 / 2^8 ≈ 0.047 units`, or `0.047 / 375 ≈ 0.00013` in observation space. The policy network cannot distinguish this from noise.

**Expected Benefit:**
Doubling the step to 6.0 halves the grid lookups from ~11,000 to ~5,500 per tick. The two-phase approach with a 12.0 coarse step would reduce it to ~2,750. This is a meaningful reduction in the observation system's per-tick cost.

**Impact Assessment:**
Negligible impact (flagged). The boundary distance may differ from the current value by up to half a coarse step before refinement, but after 8 binary-search iterations the precision is within 0.05 units for a 12.0 coarse step. Given that observations are normalised by 375.0, the difference is ~0.00013 in observation space — well below the noise floor of stochastic policy sampling. The policy's behaviour is identical in practice.

If exact reproducibility of observation values across step-size changes is important (e.g., for deterministic replay comparison against existing traces), this should be deferred. If the goal is just policy learning (current use case), the impact is genuinely negligible.

---

## Inconsistent Patterns

### `tangent_at_s` and `point_at_s` Duplicate Segment-Finding Logic
- [x] Extract a shared `find_segment_at_s` helper from `tangent_at_s` and `point_at_s` to eliminate the duplicated linear scan

**Category:** Inconsistent Patterns
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None (verified — same segment found, same output)

**Location:**
- `src/maps/centerline.rs:82-109` — `point_at_s()`
- `src/maps/centerline.rs:112-147` — `tangent_at_s()`

**Current State:**
Both `point_at_s` and `tangent_at_s` contain nearly identical segment-finding loops:
```rust
let s_wrapped = s.rem_euclid(self.total_length);
for i in 0..n {
    let seg_start = self.cumulative_lengths[i];
    let seg_end = if i + 1 < n {
        self.cumulative_lengths[i + 1]
    } else {
        self.total_length
    };
    let in_segment = (s_wrapped >= seg_start && s_wrapped < seg_end)
        || (i == n - 1 && (s_wrapped - self.total_length).abs() <= 1e-6);
    if in_segment { ... }
}
```

The only difference is what they do after finding the segment (interpolate a point vs. return a tangent). The segment-finding logic is duplicated verbatim.

**Proposed Change:**
Extract a private `fn find_segment_at_s(&self, s: f32) -> (usize, f32)` that returns the segment index and the local `t` parameter. Both `point_at_s` and `tangent_at_s` call this helper and then do their specific computation on the result.

This also prepares for the binary-search optimisation (finding #1 in environment.md): the binary search only needs to be implemented once in `find_segment_at_s`.

**Justification:**
Duplicated segment-finding logic means any optimisation (binary search) or bug fix must be applied in two places. Consolidation reduces maintenance surface and makes the binary-search optimisation a single-location change.

**Expected Benefit:**
Removes ~25 lines of duplicated code. Makes future optimisation (binary search on cumulative_lengths) a single-location change.

**Impact Assessment:**
Zero functional change. Same segment found, same interpolation parameter, same output.
