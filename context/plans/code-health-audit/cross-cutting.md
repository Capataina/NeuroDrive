# Cross-cutting — Code Health Findings

**Systems covered:** project-wide concerns (Cargo configuration, test infrastructure, analytics consistency).

**Finding count:** 2 findings (0 high, 1 medium, 1 low).

---

## Configuration Drift / Test Coverage Gap

### Add a thin `src/lib.rs` + `[lib]` Cargo entry so integration tests can import the crate's modules
- [x] Introduce `src/lib.rs` (module re-exports only — no new logic) and a `[lib]` section in `Cargo.toml` so `tests/*.rs` files can `use NeuroDrive::brain::...` and act as integration tests

**Category:** Test Coverage Gaps (blocks diagnostic-test writing in audits; discovered via this audit's inability to satisfy its diagnostic-test floor)
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None at runtime (the library target is built alongside the binary; no runtime path uses it). Flagged as "requires decision" because it changes the crate's build shape and is technically a production-source change.

**Location:**
- `Cargo.toml` (root) — no `[lib]` entry at present; only an implicit `[[bin]]`.
- `src/main.rs` — binary entry point.
- No `src/lib.rs` exists.
- No `tests/` directory exists.

**Current State:**
The project is configured as a pure binary crate. All existing tests (31 of them) live as `#[cfg(test)] mod tests` blocks inside their respective source files. This works for unit-level assertions but blocks three useful patterns:

1. **Integration tests** that cut across modules (e.g. a test that wires together a minimal `ActorCritic` + `TrainerRolloutBuffer` + `ppo_process_chunk` to assert end-to-end equivalence after a proposed refactor).
2. **Benchmarks or latency probes** that reference model types from outside the source tree.
3. **Audit-grade diagnostic tests** — the 2026-04-18 code-health-audit attempted to write a `tests/ppo_forward_hotpath_baseline.rs` to pin the action-selection latency as evidence for finding #1 in `brain-ppo.md`; it could not compile (`use NeuroDrive::brain::...` unresolved) and had to be retracted. The finding is recorded at "high analytical confidence" instead of "test-backed high confidence" as a direct result.

The minimum fix is a three-line `src/lib.rs` that re-exports the existing module tree:

```rust
pub mod agent;
pub mod analytics;
pub mod brain;
pub mod debug;
pub mod game;
pub mod maps;
pub mod profiling;
pub mod sim;
```

…plus a `[lib]` section in `Cargo.toml` naming the same paths. The binary `main.rs` continues to work unchanged. Nothing in the runtime path changes.

**Proposed Change:**
Add `src/lib.rs` with the module re-exports above (no other logic). Add a `[lib]` entry to `Cargo.toml`:

```toml
[lib]
name = "neurodrive"
path = "src/lib.rs"
```

Optionally also add a `[[bin]]` entry pointing at `src/main.rs` so the binary target is explicit.

**Justification:**
Evidence chain:
- Direct: `Cargo.toml` has no `[lib]` section and `src/lib.rs` does not exist — verified by `ls` and `cat`.
- Direct: `cargo test --test ppo_forward_hotpath_baseline` failed this session with `use of undeclared type NeuroDrive` — the test file was removed after the failure.
- Project principle: the README and `context/notes/performance-tuning-lessons.md` both emphasise data-driven iteration ("Learning must be measurable, not guessed"). A project with that philosophy benefits from being able to write test-backed evidence for performance claims.
- Counter: the runtime cost is zero. The `cargo build` artefact now includes a library `.rlib` in addition to the binary; release-build size is marginally larger but no hot-path code is affected.

Research mode: 1 — cross-referenced against Bevy project conventions (most Bevy example projects use either a pure binary or a lib+bin shape; bevy_best_practices on GitHub notes that test-heavy projects tend toward lib+bin).

**Expected Benefit:**
- Unblocks integration-test writing for future audits and for the implementing engineer's own PPO verification work.
- Preserves the exact binary target and runtime behaviour.
- Removes the obstacle that caused this audit's diagnostic-test obligation to be deferred on every PPO/environment finding.

**Impact Assessment:**
**Possible (requires decision)** — the crate grows a new target type. The runtime, main entry point, and all existing behaviour are preserved. The change is purely additive to the build graph. Flagged as "requires decision" because it is a build-shape change and because adding `src/lib.rs` technically creates a new production source file (which the audit itself cannot do under Rule 3, but the implementing engineer can).

Confidence: **high** that the gap exists and has current impact; the resolution itself is trivial once green-lit.

---

## Inconsistent Patterns (Low)

### The `EpisodeConfig::crash_penalty` field exists and is wired into reward accumulation even though both README and notes declare the penalty always zero
- [x] Decide whether to keep `crash_penalty` as a deliberately-zero configuration point, or remove the field and the associated `accum.crash_penalty_sum` accumulator as dead-configuration

**Category:** Inconsistent Patterns (config drift edge-case: the field exists but the documentation says it must never be non-zero)
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None if retained as-is. If removed, the behaviour is identical because the field is already 0.0 by default.

**Location:**
- `src/game/episode.rs:36` — field `pub crash_penalty: f32` on `EpisodeConfig`.
- `src/game/episode.rs:50` — `crash_penalty: 0.0` in `Default` impl.
- `src/game/episode.rs:272` — `terminal_reward += config.crash_penalty;` applied on crash.
- `src/game/episode.rs:298-299` — `episode_state.accum.crash_penalty_sum += config.crash_penalty;` accumulated.
- `src/game/episode.rs:104` — `pub crash_penalty_sum: f32` on `EpisodeAccum`.
- `README.md` §"Reward Structure" — "Crash penalty: 0.0 — Episode termination is the cost; no explicit penalty".
- `context/notes/reward-and-entertainment.md` §"Failure Modes We've Hit" — documents the outcome of adding a non-zero crash penalty ("Cars learned to stay still or brake constantly") as a lesson learned, with the explicit guidance "Never through reward penalties or bonuses that would make safe play optimal."

**Current State:**
The `crash_penalty` field, its `Default = 0.0`, its usage in `episode_loop_system`, and the paired `crash_penalty_sum` accumulator on `EpisodeAccum` and `LastEpisodeSummary` are all live. The documentation explicitly forbids non-zero crash penalties. The field exists as a configuration point, but the project's hard policy is that it must stay zero. This is configuration drift — not because the value will need tuning (it won't, per policy) but because the machinery exists to make a change the philosophy says should never be made.

Two reasonable resolutions:

1. **Keep it.** The field being zero-valued and explicitly documented as "the policy forbids non-zero" has educational value: a future engineer reading `EpisodeConfig` sees the field, reads the comment, and understands *why* it must stay zero. Add an inline comment on the field stating "Keep at 0.0; see reward-and-entertainment.md."
2. **Remove it.** Delete the field, the wiring in `episode_loop_system`, and the `crash_penalty_sum` accumulator. The `terminal_reward` mechanism still exists (`episode.rs:272`) and can be repurposed if a future terminal shaping signal is ever added. This reduces surface area.

Option 1 has the advantage that it keeps the analytics reporting of `crash_penalty_sum` meaningful (it is always 0.0, which is itself a useful assertion that the policy is enforced). Option 2 is cleaner but discards a field that could come back if the policy is ever revisited.

**Proposed Change:**
**Option 1** is recommended: add an inline doc comment on `pub crash_penalty: f32` explicitly pointing at the notes file and stating the invariant, so anyone reading the struct knows the field is "structurally present, policy-locked at zero" rather than "a knob someone forgot to turn."

```rust
/// Crash penalty applied once on crash episode end.
///
/// **Policy:** keep at `0.0`. See `context/notes/reward-and-entertainment.md`
/// — non-zero crash penalties produce "stay still / brake constantly" policies
/// and are incompatible with the entertainment-first reward philosophy.
pub crash_penalty: f32,
```

**Justification:**
Direct evidence from the codebase + the notes file + the README (confidence: high). This is straightforwardly documented drift.

**Expected Benefit:**
- Aligns code with documented policy (the policy becomes visible at the point where someone would otherwise change the value).
- Zero runtime cost.

**Impact Assessment:**
Zero functional change. Pure doc addition.

Confidence: **high** (direct).

---

## Data Layout analysis applicability decision (required)

Not applicable — these are cross-cutting configuration / test-infrastructure / documentation findings. There are no hot-path data structures in scope for this file.
