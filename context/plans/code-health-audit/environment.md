# Environment — Code Health Findings

**Systems covered:** `src/game/` (physics.rs, collision.rs, episode.rs, progress.rs), `src/maps/centerline.rs`
**Finding count:** 5 findings (2 high, 2 medium, 1 low)

---

## Algorithm Optimisation

### Cached-Hint Centreline Projection
- [x] Add a `last_segment_hint` to `TrackProgress` and use it to start the projection search at the most likely segment instead of scanning all segments

**Category:** Algorithm Optimisation
**Severity:** High
**Effort:** Small
**Behavioural Impact:** None (verified — projection finds the closest segment regardless of starting point; hint only reduces the search space)

**Location:**
- `src/maps/centerline.rs:181-221` — `TrackCenterline::project()`
- `src/game/progress.rs:38-56` — `update_track_progress_system()`

**Current State:**
`TrackCenterline::project()` performs a linear scan of **all** polyline segments (line 193: `for i in 0..n`) to find the closest point. The centreline has approximately 80-120 segments (8 arc samples per corner × number of corners, plus straight segments). This means 8 cars × 80-120 segment checks × every tick = 640-960 segment distance computations per tick in `update_track_progress_system`.

Additionally, `tangent_at_s` performs a similar linear scan (line 119: `for i in 0..n`) for every `tangent_at_s` call. The observation system calls `tangent_at_s` 12 times per car (once per lookahead sample) in `update_sensor_readings_system` (line 214). That is 96 linear scans of the centreline per tick just for lookahead tangents.

**Proposed Change:**
**For `project`:** Add a `last_segment: usize` field to `TrackProgress`. When projecting, start the search at `last_segment - 1` (wrapping) and search outward in both directions. Because cars move continuously and slowly relative to segment length, the closest segment is almost always `last_segment` or an adjacent one. Maintain the full-scan fallback if the hint search does not find a closer point within a small window (e.g., 5 segments in each direction).

**For `tangent_at_s`:** Replace the linear scan with a binary search on the `cumulative_lengths` array. Since `cumulative_lengths` is monotonically increasing, `partition_point` or `binary_search_by` finds the correct segment in O(log n) instead of O(n).

Both changes preserve the exact same output: `project` still returns the globally closest point (the hint just makes the search faster, with a fallback to full scan); `tangent_at_s` still returns the segment tangent at the given arc length.

**Justification:**
The centreline queries are the most frequently called spatial operations in the codebase:
- `project`: 8 calls/tick (one per car)
- `tangent_at_s`: 96 calls/tick (12 lookahead × 8 cars) + 8 calls/tick (once per car in episode loop for reward)

With 104 centreline queries per tick and ~100 segments per query, that is ~10,400 segment checks per tick. The hint-based approach would reduce `project` to ~2-10 segment checks per call, and binary search would reduce `tangent_at_s` from O(100) to O(7). Total segment checks would drop from ~10,400 to ~600.

**Expected Benefit:**
Approximately 10-15x reduction in centreline query work, from ~10,400 segment checks per tick to ~600. The actual wall-clock impact depends on the cost per segment check (one dot product + one distance = ~6 FLOPs), but at 10,400 checks/tick this represents a meaningful share of the Measurement phase.

**Impact Assessment:**
Zero functional change for both changes. `project` still finds the globally closest segment (hint is an optimisation, not a semantic change). `tangent_at_s` returns the exact same tangent because the cumulative_lengths array is searched for the same value — only the search algorithm changes from linear to binary.

---

### Pre-Compute Local Corner Array in Collision Detection
- [x] Hoist the `local_corners` array computation out of `collision_detection_system` since it is constant

**Category:** Performance Improvement
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None (verified — identical corner positions)

**Location:**
- `src/game/collision.rs:27-35` — `local_corners` array in `collision_detection_system`

**Current State:**
The `collision_detection_system` computes the 4 local corner offsets on every invocation:
```rust
let half_w = CAR_WIDTH * 0.5;
let half_h = CAR_HEIGHT * 0.5;
let local_corners = [
    Vec2::new(half_w, half_h),
    Vec2::new(half_w, -half_h),
    Vec2::new(-half_w, half_h),
    Vec2::new(-half_w, -half_h),
];
```

`CAR_WIDTH` and `CAR_HEIGHT` are constants, so this array is the same on every tick.

**Proposed Change:**
Make `local_corners` a `const` or `static` array defined at module level.

**Justification:**
Trivially constant computation recomputed every tick. The compiler may already optimise this away, but making it explicit improves readability and removes doubt.

**Expected Benefit:**
Negligible performance impact (compiler likely already hoists this). Improved code clarity.

**Impact Assessment:**
Zero functional change by construction.

---

## Known Issues and Active Risks

### Crash Classification Uses Debug Format for End Reason Matching
- [x] Replace `format!("{:?}", reason).contains("Crash")` with a direct pattern match on `EpisodeEndReason::Crash`

**Category:** Known Issues and Active Risks
**Severity:** Medium
**Effort:** Trivial
**Behavioural Impact:** None (verified — currently works because the Debug format of `EpisodeEndReason::Crash` contains "Crash", but fragile)

**Location:**
- `src/analytics/trackers/episode.rs:212` — `let is_crash = format!("{:?}", reason).contains("Crash");`

**Current State:**
The `episode_tracker_system` determines whether an episode ended in a crash by formatting the `EpisodeEndReason` enum via `Debug` and checking if the string contains "Crash":
```rust
let is_crash = format!("{:?}", reason).contains("Crash");
```

This works because `EpisodeEndReason::Crash` formats as `"Crash"`. But it is fragile — if a new variant were added whose Debug representation contained "Crash" (e.g., `NearCrash`), or if the variant were renamed, this would silently break.

**Proposed Change:**
Replace with a direct match:
```rust
let is_crash = reason == EpisodeEndReason::Crash;
```

The `EpisodeEndReason` enum already derives `PartialEq` and `Eq`.

**Justification:**
String-based enum matching via Debug formatting is a correctness risk. The direct comparison is simpler, faster (no allocation), and immune to naming changes.

**Expected Benefit:**
Eliminates a fragile string comparison and a `format!` allocation on every completed episode. More importantly, prevents a future silent bug if the enum changes.

**Impact Assessment:**
Zero functional change. `EpisodeEndReason::Crash` is the only variant whose Debug format contains "Crash", so the string check and the direct comparison produce identical results today.

---

## Complexity Hotspots

### `EpisodeState` Struct Has 34 Fields
- [x] Consider splitting `EpisodeState` into a `CurrentTickData` sub-struct and an `EpisodeAccumulator` sub-struct to reduce the cognitive load

**Category:** Complexity Hotspots
**Severity:** Medium
**Effort:** Medium
**Behavioural Impact:** None (verified — purely structural, same fields, same access patterns)

**Location:**
- `src/game/episode.rs:57-101` — `EpisodeState` struct definition

**Current State:**
`EpisodeState` has 34 fields mixing three concerns:
1. **Current-tick scratch data** (14 fields starting with `current_tick_`): reward, progress, speed, heading, forward, tangent, etc.
2. **Episode accumulators** (8 fields): `current_return`, `current_progress_reward_sum`, `current_crashes`, `current_best_progress_fraction`, etc.
3. **Last-episode summaries** (12 fields starting with `last_episode_`): stored after finalisation for downstream consumers.

The struct is a single Component, so all 34 fields are carried per car even though `current_tick_` fields are only meaningful within a single tick and `last_episode_` fields are only meaningful after episode end.

**Proposed Change:**
Group into sub-structs:
```rust
pub struct EpisodeState {
    pub episode_id: u32,
    pub ticks_in_episode: u32,
    pub tick: TickSnapshot,       // current_tick_* fields
    pub accum: EpisodeAccum,      // running sums and counters
    pub last: LastEpisodeSummary, // last_episode_* fields
}
```

All three sub-structs are plain data (no methods). The Component is still `EpisodeState` — callers access `episode_state.tick.reward` instead of `episode_state.current_tick_reward`. The struct remains a single Component on the entity.

**Justification:**
34 fields in a single flat struct is a readability and maintenance burden. The three groups have different lifetimes (per-tick, per-episode, cross-episode) and different consumers (tick data is consumed by analytics trace capture; accumulators are internal to the episode system; last-episode data is consumed by analytics episode tracker and HUD). Grouping them makes the ownership and lifecycle explicit.

This is not an architecture change — it is a within-component restructure. No system boundaries or data flow changes.

**Expected Benefit:**
Clearer field organisation. Easier to understand which fields are valid when. Reduces the chance of accidentally reading a `current_tick_` field outside the tick it was computed in.

**Impact Assessment:**
Zero functional change. Same fields, same data, same access patterns. Only the nesting depth of field access changes. All consumers (episode system, analytics trackers, HUD) would update their field paths but the data they receive is identical.
