# Analytics — Code Health Findings

**Systems covered:** `src/analytics/` (models.rs, trackers/episode.rs, trackers/trace.rs, trackers/action.rs, metrics/*, exporters/*)
**Finding count:** 4 findings (0 high, 2 medium, 2 low)

---

## Performance Improvement

### `TickTraceRecord` Uses `Vec<f32>` for Fixed-Size Arrays
- [ ] Replace `ray_distances: Vec<f32>`, `lookahead_heading_deltas: Vec<f32>`, and `lookahead_curvatures: Vec<f32>` with fixed-size arrays in `TickTraceRecord`

**Category:** Performance Improvement
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None (verified — same data, fixed-size arrays instead of heap-allocated vectors)

**Location:**
- `src/analytics/models.rs:122-124` — `TickTraceRecord` fields

**Current State:**
`TickTraceRecord` stores three fields as `Vec<f32>`:
```rust
pub ray_distances: Vec<f32>,
pub lookahead_heading_deltas: Vec<f32>,
pub lookahead_curvatures: Vec<f32>,
```

These always have exactly `NUM_RAYS` (11), `NUM_LOOKAHEAD_SAMPLES` (12), and `NUM_LOOKAHEAD_SAMPLES` (12) elements respectively. Every tick, for every car, the trace capture system creates a new `TickTraceRecord` with three `Vec` allocations — that is 24 heap allocations per tick (8 cars x 3 Vecs).

**Proposed Change:**
Replace with fixed-size arrays:
```rust
pub ray_distances: [f32; 11],
pub lookahead_heading_deltas: [f32; 12],
pub lookahead_curvatures: [f32; 12],
```

The serde `Serialize`/`Deserialize` derives work with fixed-size arrays up to length 32.

**Justification:**
The dimensions are compile-time constants. Using `Vec` for fixed-size data performs 24 unnecessary heap allocations per tick and adds 72 bytes of Vec overhead (pointer + len + capacity × 3) per record. Over a 30-second run at 60 Hz with 8 cars, that is 43,200 unnecessary allocations.

**Expected Benefit:**
Eliminates 24 heap allocations per tick (1,440 per second). Reduces per-record memory overhead. Records are contiguous in memory, improving cache locality during trace aggregation.

**Impact Assessment:**
Zero functional change. The arrays always have the same number of elements. JSON serialisation output is identical (arrays serialise the same way as Vecs of the same content).

Note: this requires updating the trace capture code in `src/analytics/trackers/trace.rs` to use array syntax instead of `Vec::from` or `.to_vec()`. The `SensorReadings` struct already uses fixed-size arrays (`[f32; NUM_RAYS]`, `[f32; NUM_LOOKAHEAD_SAMPLES]`), so the data source is already array-typed.

---

## Inconsistent Patterns

### Crash Type Stored as `String` Instead of Enum
- [ ] Replace `crash_type: Option<String>` in `EpisodeRecord` and `classify_crash` return type with a `CrashKind` enum

**Category:** Inconsistent Patterns
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None (verified — same crash types, strongly typed instead of stringly typed)

**Location:**
- `src/analytics/trackers/episode.rs:14-32` — `classify_crash()` returns `String`
- `src/analytics/models.rs:50` — `crash_type: Option<String>`

**Current State:**
The `classify_crash` function returns a `String` ("Stall", "Spin", "Slide", "Overshoot", "HeadOn"). The `EpisodeRecord` stores `crash_type: Option<String>`. The context documentation and system docs refer to a `CrashKind` enum with these exact variants, but no such enum exists in the code — the types are only represented as strings.

This means:
- Typos in crash type strings would be silent bugs.
- Pattern matching on crash types requires string comparison.
- The `classify_crash` function allocates a `String` on every crash (using `.to_string()`).

**Proposed Change:**
Define a `CrashKind` enum:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrashKind {
    Stall,
    Spin,
    Slide,
    Overshoot,
    HeadOn,
}
```

Place it in `src/analytics/models.rs`. Change `classify_crash` to return `CrashKind`. Change `EpisodeRecord.crash_type` to `Option<CrashKind>`. The JSON serialisation output changes from a raw string to the enum variant name, but since serde serialises unit enum variants as strings by default, the JSON output is identical.

**Justification:**
The context documentation already refers to `CrashKind` as an enum — the code should match. String-typed classification is fragile, allocates unnecessarily, and prevents exhaustive matching.

**Expected Benefit:**
Eliminates one `String` allocation per crash. Enables exhaustive `match` on crash types. Prevents typo bugs. Aligns code with documentation.

**Impact Assessment:**
Zero functional change. The same classification logic produces the same results, now typed as an enum instead of a string. JSON output is identical because serde serialises unit enum variants as their string names by default.

---

## Documentation Rot

### Buffer Comment Says "Old Single-Env Rollout Buffer. Retained Temporarily."
- [ ] Remove the stale comment at the top of `TrainerRolloutBuffer`

**Category:** Documentation Rot
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None

**Location:**
- `src/brain/ppo/buffer.rs:5` — `/// Old single-env rollout buffer. Retained temporarily.`

**Current State:**
The doc comment on `TrainerRolloutBuffer` says "Old single-env rollout buffer. Retained temporarily." followed by the actual description. The buffer is not old, not single-env, and not temporary — it is the active trainer-wide rollout buffer with per-env tagging. The first sentence is a leftover from when the buffer was being migrated from single-car to multi-car.

**Proposed Change:**
Remove the first sentence. The rest of the doc comment is accurate.

**Justification:**
Misleading documentation is worse than no documentation.

**Expected Benefit:**
Removes one misleading sentence.

**Impact Assessment:**
Zero functional change.

---

## Triage Needed

### `speed_std` Field Missing From `EpisodeRecord` Despite Being Computed in `TraceAggregates`
- [ ] Decide whether speed standard deviation should be added to `EpisodeRecord` for richer per-episode analytics

**Category:** Triage Needed
**Severity:** Low
**Effort:** Trivial

**Location:**
- `src/analytics/trackers/episode.rs:36-147` — `compute_trace_aggregates` computes `mean_speed` and `peak_speed` but not `speed_std`
- `src/analytics/models.rs:36-37` — `EpisodeRecord` has `mean_speed` and `peak_speed`

**Current State:**
The `EpisodeRecord` stores `mean_speed` and `peak_speed` but not `speed_std` (standard deviation of speed across the episode). The `TraceAggregates` struct also lacks a `speed_std` field. Speed variance within an episode would be useful for distinguishing "consistently fast" from "variable speed" driving patterns — a diagnostic that the 10-section markdown report cannot currently compute from episode-level data alone.

This is a triage item because adding the field would be a minor schema addition to `EpisodeRecord` (not a cleanup), and it is unclear whether the existing metric modules would benefit from it. The chunking module already computes speed stats per chunk, so the per-episode level may not add much.

No action required unless the analytics pipeline needs this metric.
