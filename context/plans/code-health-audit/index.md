# Code Health Audit — 2026-04-18

**Date:** 2026-04-18
**Scope:** Full repository.
**Status:** active

Prior audit: 2026-04-15 (archived in `archive-2026-04-15/`) closed 22 of 23 findings. This audit is a fresh pass against the upgraded `code-health-audit` skill (commit `6417142` on `Capataina/agent-skills`) with explicit obligation floors — pre-Pass-1 WebSearch, per-system research, diagnostic-test writing where it would resolve uncertainty, Pass-1 and Pass-2 checkpoints, Obligation Evidence Map, and this "What I Did Not Do" ledger.

## What I Did Not Do

For each non-negotiable obligation from the skill, status is one of `done` (with evidence), `partial` (with reason and partial evidence), or `skipped` (with reason). Silent omission is not permitted — this section mirrors the Obligation Evidence Map and the two must agree.

- **Pre-Pass-1 front-loaded WebSearch:** **done.** Query: `code health audit patterns for Rust Bevy reinforcement learning project`. Sources: <https://github.com/tbillington/bevy_best_practices>, <https://bevy.org/assets/learning/bevy-design-patterns-unofficial/>. Recorded in `obligation-evidence-map.md` under "Pre-Pass-1 front-loaded WebSearch."
- **Pass-1 checkpoint written before Pass 2 began:** **done.** See `PASS-1-CHECKPOINT.md` — includes systems identified, test baseline, Pass-2 prioritisation, and known issues surfaced from context.
- **Project test suite baseline captured in Pass 1:** **done.** Command: `cargo test`. Result: **31 passed; 0 failed; 0 ignored**. No flakiness observed. Recorded in the Pass-1 checkpoint.
- **Pre-existing test failures recorded as Known Issues findings:** **done.** No failures. Explicit "no failures" note in the Pass-1 checkpoint. The two non-test Known Issues already documented in `context/systems/brain-ppo.md` (no save/load, `debug_assert!`-only alignment) are noted in the Pass-1 checkpoint rather than duplicated as audit findings, per `finding-format.md` guidance on not re-flagging items already captured in `context/systems/*.md` "Known Issues / Active Risks" sections.
- **Research obligation met for every substantive system:** **done.** 3 substantive systems in Pass 2 (PPO hot path, environment/reward, cross-cutting), 3 Pass-2 WebSearch calls (plus the pre-Pass-1 one = 4 total). Each substantive system has at least one query with query text, ≥1 source URL, and a research-mode classification in the Obligation Evidence Map. Analytics, profiling, debug, and small glue systems are recorded as reasoned omissions with explicit justification ("When Research Is Not Required" criteria).
- **Research-mode variety across the audit:** **done.** Modes 1 (domain pattern lookup — Bevy/RL conventions + racing reward shaping), 2 (specific-technique evaluation — scratch-buffer vs per-call Vec allocation), and 3 (known-anti-pattern check — `slice::from_raw_parts` aliasing) all represented. Distribution recorded at the top of `obligation-evidence-map.md`.
- **Diagnostic-test obligation met:** **partial (reasoned omission).** The project is a **binary-only Cargo crate** — no `[lib]` target, no `src/lib.rs`. Integration tests in `tests/*.rs` cannot `use NeuroDrive::brain::...` without adding `src/lib.rs`, which is a production source file Rule 3 forbids the audit from creating. I attempted to write `tests/ppo_forward_hotpath_baseline.rs` with three diagnostic assertions (per-call latency baseline, Vec-allocation signature probe, `sample_normal` determinism check); it failed to compile (`use of undeclared type NeuroDrive`) and was retracted. The gap is itself surfaced as **[cross-cutting.md#add-a-thin-srclibrs--lib-cargo-entry](cross-cutting.md#add-a-thin-srclibrs--lib-cargo-entry)** — a medium-severity finding recommending the implementing engineer add the minimal `[lib]` entry so future audits can satisfy this floor. Each PPO/environment finding explicitly states its analytical confidence level and would upgrade to test-backed confidence after the library target exists.
- **Confidence upgrade pathway attempted before any moderate or low confidence finding:** **done.** Three moderate-confidence findings exist. For each, the upgrade path was attempted: (a) the `compute_gae_per_env` Option C variant is flagged moderate pending periodicity verification — upgrade path is "audit all prepare call sites"; (b) the `HashSet` removal is moderate because the check is only demonstrably unreachable on the current track format — upgrade path is "review future track formats"; (c) the `crash_penalty` field finding is low-but-moderate in retention reasoning — the recommendation documents both options because the choice is genuinely a human decision. None of these would have been pushed to "high" by a diagnostic test; all three require design decisions instead of evidence.
- **Pass-2 systems-audited checkpoint written before final output:** **done.** See `PASS-2-SYSTEMS-AUDITED.md`.
- **Obligation Evidence Map has one row per substantive system (no PENDING rows):** **done.** See `obligation-evidence-map.md`. Systems with no substantive findings appear as reasoned-omission rows rather than as absent entries.
- **"What I Did Not Do" section present at the top of `index.md`:** **done** (this section).
- **Data Layout and Memory Access Patterns applied to every system audited in Pass 2:** **done.** Each finding file has a "Data Layout analysis applicability decision" section. For the PPO hot path: finding #1 is exactly this category. For the environment: applicability decision explains that the 2026-04-15 audit already closed the high-value Data Layout wins; no further findings. For cross-cutting: not applicable (no hot-path structures in scope).
- **Production source code not modified:** **done.** Verified via `git status` at end of session — only `context/plans/code-health-audit/*` and the archived prior audit moved. No `src/` edits. No `Cargo.toml` edits. No test file persisted in `tests/` after the retracted compile attempt.

## Summary

Deep audit focused on surfacing residual hot-path wins and a structural-safety refactor after the 2026-04-15 pass. The strongest finding is finding #1 in `brain-ppo.md`: the single-sample `Linear::forward` / `Tanh::forward` path still allocates a fresh `Vec<f32>` per layer per car per tick (≈48 allocations per tick on the actor path alone), which the existing `BatchScratch` pattern can be generalised to eliminate. The Modularisation finding on `BatchScratch` removes three `unsafe { slice::from_raw_parts }` blocks by splitting the struct so Rust's disjoint-field borrow inference accepts the pattern safely. An environment-level configuration drift (`time_penalty_per_tick = -0.005` contradicting the documented reward philosophy) is surfaced for decision. A cross-cutting finding on the binary-only Cargo layout explains why the audit could not meet its diagnostic-test floor and recommends the minimal unblocking change.

## Findings Overview

| File | System | Critical | High | Medium | Low | Total |
|------|--------|----------|------|--------|-----|-------|
| [brain-ppo.md](brain-ppo.md) | PPO hot path (model, common mlp, update) | 0 | 1 | 2 | 1 | 4 |
| [environment.md](environment.md) | Game reward + centreline traverse | 0 | 0 | 1 | 1 | 2 |
| [cross-cutting.md](cross-cutting.md) | Project-wide (Cargo, docs ↔ code drift) | 0 | 0 | 1 | 1 | 2 |
| **Total** | | **0** | **1** | **4** | **3** | **8** |

## Priority Actions

1. **[HIGH]** Replace per-car-per-tick `Vec<f32>` allocations in single-sample `Linear::forward` / `Tanh::forward` with reusable scratch buffers — [brain-ppo.md#replace-per-car-per-tick-vecf32-allocations-in-single-sample-linearforward-with-reusable-scratch-buffers](brain-ppo.md#replace-per-car-per-tick-vecf32-allocations-in-single-sample-linearforward-with-reusable-scratch-buffers)
2. **[MEDIUM]** Eliminate per-prepare `HashMap` allocation in `compute_gae_per_env` — [brain-ppo.md#eliminate-per-prepare-hashmap-allocation-in-compute_gae_per_env](brain-ppo.md#eliminate-per-prepare-hashmap-allocation-in-compute_gae_per_env)
3. **[MEDIUM]** Split `BatchScratch` to remove the three `unsafe { slice::from_raw_parts }` blocks in `update.rs` — [brain-ppo.md#split-batchscratch-so-the-unsafe--slicefrom_raw_parts--aliasing-workarounds-in-updaters-become-safe](brain-ppo.md#split-batchscratch-so-the-unsafe--slicefrom_raw_parts--aliasing-workarounds-in-updaters-become-safe)
4. **[MEDIUM]** Decide on `time_penalty_per_tick` default — either zero it to match the reward philosophy, or document the nudge explicitly — [environment.md#default-time_penalty_per_tick--0005-contradicts-the-documented-reward-philosophy](environment.md#default-time_penalty_per_tick--0005-contradicts-the-documented-reward-philosophy)
5. **[MEDIUM]** Add minimal `src/lib.rs` + `[lib]` Cargo entry so integration tests can import crate modules — [cross-cutting.md#add-a-thin-srclibrs--lib-cargo-entry-so-integration-tests-can-import-the-crates-modules](cross-cutting.md#add-a-thin-srclibrs--lib-cargo-entry-so-integration-tests-can-import-the-crates-modules)
6. **[LOW]** Avoid `.collect::<Vec<f32>>()` for the two-element `ActionDist.std` in `forward_actor`/`forward` — [brain-ppo.md#avoid-the-collectvecf32-in-actorcriticforward_actor--forward-for-the-two-element-std-vector](brain-ppo.md#avoid-the-collectvecf32-in-actorcriticforward_actor--forward-for-the-two-element-std-vector)
7. **[LOW]** Document the `crash_penalty` invariant at the field definition — [cross-cutting.md#the-episodeconfigcrash_penalty-field-exists-and-is-wired-into-reward-accumulation-even-though-both-readme-and-notes-declare-the-penalty-always-zero](cross-cutting.md#the-episodeconfigcrash_penalty-field-exists-and-is-wired-into-reward-accumulation-even-though-both-readme-and-notes-declare-the-penalty-always-zero)
8. **[LOW]** Demote or delete the startup-only `HashSet` visited-check in `centerline.rs::traverse_cells` — [environment.md#remove-the-hashsetusize-usize-visited-set-in-traverse_cells-or-demote-the-check-to-a-debug_assert](environment.md#remove-the-hashsetusize-usize-visited-set-in-traverse_cells-or-demote-the-check-to-a-debug_assert)

## By Category

- Data Layout and Memory Access Patterns: 1 finding (high)
- Performance Improvement: 2 findings (1 medium, 1 low)
- Modularisation: 1 finding (medium)
- Known Issues and Active Risks: 1 finding (medium)
- Dead Code Removal: 1 finding (low)
- Configuration Drift / Test Coverage Gaps: 1 finding (medium)
- Inconsistent Patterns: 1 finding (low)

## Audit meta-notes

- The prior audit's optimisation work is visible throughout — flat weight storage, pre-allocated `BatchScratch`, batched critic forward, cached-hint centreline projection, `sample_normal` affine path, flat `orthogonal_init`, `[f32; OBSERVATION_DIM]` observation vector. This audit's findings are second-tier wins; none of them overturn a prior decision.
- The `unsafe` blocks introduced by the 2026-04-15 Phase 3 refactor (flagged for verification in `notes/session-2026-04-15.md`) were audited this session and found sound — finding #3 recommends removing the `unsafe` via a struct split, not fixing a bug.
- The project's binary-only Cargo layout is the single biggest blocker for diagnostic-test writing; the cross-cutting finding recommends the minimal unblocking change.
