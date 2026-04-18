# Environment — Code Health Findings

**Systems covered:** `src/game/episode.rs`, `src/game/progress.rs`, `src/game/physics.rs`, `src/agent/observation.rs`, `src/maps/centerline.rs`, `src/maps/grid.rs`.

**Finding count:** 2 findings (0 high, 1 medium, 1 low).

**Context:** the 2026-04-15 audit added the cached-hint centreline projection, binary-search arc-length lookup, adaptive raycast step, and flat `[f32; OBSERVATION_DIM]` observation construction. What remains is modest: a small mismatch between the default `time_penalty_per_tick = -0.005` and the documented reward philosophy, and a cosmetic HashSet-in-closed-loop construction that runs once at startup.

---

## Known Issues and Active Risks

### Default `time_penalty_per_tick = -0.005` contradicts the documented reward philosophy
- [x] Either zero `EpisodeConfig::time_penalty_per_tick` to match `context/notes/reward-and-entertainment.md`, or update the notes to acknowledge this as a deliberate small nudge

**Category:** Known Issues and Active Risks
**Severity:** Medium
**Effort:** Trivial (set the default to 0.0) or Small (audit whether the nudge matters empirically and document)
**Behavioural Impact:** Possible (requires decision) — changing the default does affect reward shaping, which is a deliberate training decision

**Location:**
- `src/game/episode.rs:49` — `time_penalty_per_tick: -0.005` in the `Default for EpisodeConfig` impl.
- `src/game/episode.rs:266-300` — the penalty is applied every tick in `episode_loop_system` and accumulates into `accum.time_penalty_sum` and `tick.reward`.
- `README.md` §"Reward Structure" — explicitly lists the reward components as velocity projection + centreline proximity + crash penalty (0.0) + **no survival bonus / no time penalty**.
- `context/notes/reward-and-entertainment.md` §"Core Principle" — "No survival bonuses. A per-tick bonus for staying alive incentivises the policy to play safe..." The inverse — a per-tick *penalty* for existing — has a symmetric failure mode (policy is punished for episodes that last, incentivising fast crashing when progress gain is marginal).

**Current State:**
`time_penalty_per_tick` defaults to −0.005. At 60 Hz over a 30-second episode that is up to 1,800 ticks × −0.005 = **−9.0** worth of baseline reward pressure per episode, accumulating every tick regardless of progress. The velocity projection reward tops out roughly at `1.0` per tick when the car is moving at `speed_reward_reference = 200` along the tangent, so the time penalty is −0.5% of that per tick — not insignificant.

The reward philosophy file explicitly lists "No survival bonuses" as a core principle but does not explicitly list "No time penalty." The README's reward table also omits a time-penalty row. The live config includes one. This is configuration drift relative to the documented intent — either the documentation is wrong or the default is wrong.

Two plausible resolutions:

1. **If the penalty is intentional** (e.g. to nudge the policy away from circling at zero progress): document it in both the README reward table and the notes file with a concrete justification. The documentation should match the behaviour.
2. **If the penalty is unintentional** (e.g. inherited from an earlier shaping experiment that was not fully cleaned up): set `time_penalty_per_tick = 0.0` to match the stated philosophy.

Comparing against the reward design table in `README.md` §"Current Reward Structure," the current config mismatches the documented structure. The note file's "Failure Modes We've Hit" table shows only crash-penalty and braking experiments — there is no record of a time-penalty experiment, which suggests option 2 is more likely.

**Proposed Change:**
Either:
- Set `time_penalty_per_tick: 0.0` in the `Default for EpisodeConfig` impl and verify that existing training runs reported in recent analytics exports did not rely on the −0.005 nudge. This is a zero-effort config change but a **behaviour-altering** one and therefore needs a decision, not an automatic flip.
- Or add an explicit note to `context/notes/reward-and-entertainment.md` documenting why the time penalty exists and update the README's reward table to list it.

This finding is flagged as **requires decision** — the audit cannot unilaterally choose between the two resolutions without more context on whether recent training runs depended on the penalty.

**Justification:**
Direct evidence from the codebase (confidence: high) — the default is `-0.005` at `episode.rs:49` and the reward is applied every tick at `episode.rs:266,279,283`. Direct evidence from the notes + README (confidence: high) — neither documents the penalty. The gap is concrete; the resolution requires human judgment about whether to change the default or the documentation.

**Expected Benefit:**
- If the penalty is retired: removes a shaping signal the philosophy argues against and simplifies the reward structure toward the documented ideal (velocity projection + centreline proximity only).
- If the penalty is documented: removes drift between code and docs, reducing confusion on the next read.

**Impact Assessment:**
**Possible (requires decision).** Changing the default from −0.005 to 0.0 alters the effective per-tick reward by +0.005, which is within the per-tick magnitude range the policy optimises against. The change is not "free" in the audit's sense — it belongs in a deliberate reward-shaping decision, not in a cleanup pass. The audit's job is to surface the drift; the implementing engineer / project owner decides which side to correct.

Confidence: **high** that the drift exists; **requires decision** on resolution.

---

## Dead Code Removal (Low)

### Remove the `HashSet<(usize, usize)>` visited-set in `traverse_cells` or demote the check to a `debug_assert!`
- [x] In `centerline.rs::traverse_cells` replace the per-cell `HashSet::insert` visited tracking with a simple "we've come back to the start" check, or keep the set only under `debug_assertions`

**Category:** Dead Code Removal (the safety-net check is structurally dead on the production tile layout)
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None (verified — on every valid Monaco-style track the set never triggers a `NotClosedLoop` error; on malformed tracks the error still surfaces via the `DeadEnd` arm)

**Location:**
- `src/maps/centerline.rs:299-305, 323-325` — `let mut visited = std::collections::HashSet::<(usize, usize)>::new(); visited.insert(start_cell); … if !visited.insert(next) { return Err(CenterlineBuildError::NotClosedLoop); }`

**Current State:**
`traverse_cells` walks the tile grid following the unique open edge at each tile. The trap it guards against — revisiting a cell before closing — can only occur if the tile graph has a branch (which `choose_next_dir` already rejects via `AmbiguousBranch`) or if two disconnected loops share a tile (topologically impossible on a grid with degree-2 connectivity).

For NeuroDrive's Monaco track the set is populated with ~40 cells at startup and then dropped — it is one-shot cost at track build time, not a runtime hot path. The finding is **not** about performance; it is about the check being structurally unreachable on any well-formed track, and about the `HashSet` implying a safety concern that does not exist.

**Proposed Change:**
Either:
1. Delete the `visited` set entirely and trust `choose_next_dir` + the closing-back-to-start condition to terminate. The `NotClosedLoop` arm of `CenterlineBuildError` would need to be retained (still referenced from the doc comment about closed-loop rejection).
2. Keep the set but gate behind `#[cfg(debug_assertions)]` so release builds pay nothing for the safety net.

Option 2 is the safer and documented pattern ("validate aggressively in debug, assume in release"). Option 1 is more aggressive; if the track format ever grows to support branching, the visited set becomes load-bearing again.

**Justification:**
Analytical evidence (confidence: moderate — this is a cosmetic cleanup, not a correctness fix):
- `choose_next_dir` (`centerline.rs:439-476`) already rejects the only path by which a cell could be re-visited on a valid grid — an `AmbiguousBranch`.
- `traverse_cells` already has a terminating condition that does not depend on the visited set: the loop breaks when `next == start_cell`.
- The `HashSet` is allocated once at app startup and never again — zero runtime impact.

Research: not required for this finding per `detection-strategies.md` §"When Research Is Not Required" — the logic is simple enough to reason about directly.

**Expected Benefit:**
- Removes one startup allocation (negligible) and one conceptual layer from `traverse_cells`.
- Either clarifies that the check is a debug-only invariant (option 2) or proves the safety guarantee is already structural (option 1).

**Impact Assessment:**
Zero functional change on the current track format. Both options preserve the external error-reporting shape of `traverse_cells`.

Confidence: **moderate** (analytical). This is flagged as a low-severity cosmetic finding — not urgent, and the implementing engineer may legitimately keep the belt-and-braces guard as-is.

---

## Data Layout analysis applicability decision (required)

Applied to every system file in scope. The 2026-04-15 audit already closed the cached-hint centreline projection and adaptive raycast step — those were the two high-value Data Layout wins in the environment system. Current state shows no further Data Layout findings:

- `TrackProgress` is a plain POD struct; it is stored on each `Car` entity via Bevy's ECS component storage, which already gives struct-of-arrays layout at the storage level (Bevy archetypes).
- `ObservationVector` is `[f32; OBSERVATION_DIM = 43]` — inline fixed-size array, no pointer chase, optimal for reading.
- `SensorReadings` is similar.
- Centreline `points: Vec<Vec2>` is contiguous; the projection hint already exploits temporal locality.

No Data Layout finding for this system beyond what the prior audit closed.
