# Obligation Evidence Map

Audit date: 2026-04-18
Stack: Rust 2024 edition, Bevy 0.18, custom handwritten PPO. Binary-only Cargo layout (no `[lib]` target).

## Pre-Pass-1 front-loaded WebSearch

| Query | Source URL | Mode | Purpose |
|---|---|---|---|
| "code health audit patterns for Rust Bevy reinforcement learning project" | <https://github.com/tbillington/bevy_best_practices>, <https://bevy.org/assets/learning/bevy-design-patterns-unofficial/> | 1 — domain pattern lookup | Establish the WebSearch floor before reference reading; identify Bevy + RL patterns |

## Research-mode distribution

| Mode | Count | Notes |
|---|---|---|
| 1 — domain pattern lookup | 2 | pre-Pass-1 Bevy/RL overview; Pass-2 RL racing reward shaping |
| 2 — specific-technique evaluation | 1 | scratch-buffer vs per-call Vec allocation for Rust MLP inference |
| 3 — known-anti-pattern check | 1 | `slice::from_raw_parts` aliasing hazards |

All three modes represented. Research floor satisfied.

## Per-system rows

| System | Research obligation (query + URL + mode) | Diagnostic-test obligation (path + assertion + result) | Findings emitted | Reasoned omissions |
|---|---|---|---|---|
| PPO hot path (`src/brain/ppo/`, `src/brain/common/`) | "reusable scratch buffer vs fresh Vec allocation per forward pass MLP Rust inference latency 2025" — <https://nnethercote.github.io/perf-book/heap-allocations.html>, <https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/>; Mode 2 (specific-technique evaluation). Plus "slice::from_raw_parts aliasing" — <https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html>, <https://doc.rust-lang.org/nomicon/borrow-splitting.html>; Mode 3. | **Reasoned omission** — see "Diagnostic-test deferral (binary-only crate)" note below. Baseline numbers from `context/notes/performance-tuning-lessons.md` cited in findings. | 4 findings (see [brain-ppo.md](brain-ppo.md)) | Diagnostic-test deferral recorded below (crate has no `[lib]` target; integration tests cannot import `brain::*` without editing production source to add `src/lib.rs`, which violates Rule 3). |
| Environment & reward (`src/game/episode.rs`, `src/game/progress.rs`, `src/agent/observation.rs`, `src/maps/centerline.rs`) | "2D racing environment reinforcement learning reward shaping centreline projection performance patterns" — <https://arxiv.org/html/2504.02420v2>, <https://arxiv.org/pdf/2103.10098>; Mode 1 (domain pattern lookup). | **Reasoned omission** — same binary-only-crate constraint. Findings rely on analytical evidence grounded in read code plus the prior audit's timing results. | 2 findings (see [environment.md](environment.md)) | Diagnostic-test deferral as above. Additionally: the centreline hint-path is already tested behaviourally by the existing 31-test suite passing. |
| Analytics pipeline (`src/analytics/trackers/`, `src/analytics/metrics/`) | Skipped — research not required per `detection-strategies.md` §"When Research Is Not Required" equivalent: runs on episode end and on exit only, not on the hot path; the fold patterns are straightforward Rust iterator code with no domain-specific optimisation frontier. | Not written — no uncertainty a test would resolve better than reading. | 1 finding (see [cross-cutting.md](cross-cutting.md)) | Research reasoned-omitted: off-hot-path batch work; general Rust code-health reasoning applies. |
| Cargo/dependency/build hygiene | Implicit from the pre-Pass-1 search (Bevy best practices referenced release-build flags). | Not written — dependency manifests are inspected directly. | 1 finding (see [cross-cutting.md](cross-cutting.md)) | — |
| `src/sim/`, `src/maps/parts/`, `src/brain/plugin.rs`, `src/game/plugin.rs`, small glue plugins | Research not required per §"When Research Is Not Required" — type definitions, plugin glue, no non-trivial logic. | Not written — no uncertainty a test would resolve. | 0 findings | Research + diagnostic-test obligations both reasoned-omitted as genuinely trivial. |

## Diagnostic-test deferral — binary-only crate (applies to all PPO/environment findings in this audit)

**Reason:** The NeuroDrive crate has no `[lib]` target in `Cargo.toml`. Integration tests in `tests/` cannot `use NeuroDrive::brain::...` without first adding `src/lib.rs` (a new production source file) or a `[lib]` entry in `Cargo.toml`. Both are production-source changes Rule 3 forbids the audit from making. Unit tests already exist as `#[cfg(test)] mod tests` blocks inside production source files; adding to those modules also requires editing production source, which Rule 3 forbids.

**Attempted upgrade path (recorded per evidence-and-justification §5 "Confidence Upgrade Pathway"):** I initially drafted `tests/ppo_forward_hotpath_baseline.rs` with latency baselines for `ActorCritic::forward_actor` and evidence probes for `Linear::forward`'s allocation signature. It failed to compile (the crate name `NeuroDrive` is not resolvable without a `[lib]` target). Removing the file left no test infrastructure the audit could safely extend without violating Rule 3.

**Impact on finding confidence:** Findings that would have been strengthened by direct timing numbers (e.g. "per-car `forward_actor` measured at X µs") are issued with **analytical** evidence grounded in concrete line-level reading and cross-referenced against timing observations already captured in `context/notes/performance-tuning-lessons.md` (the 426→2 stutter / 17.3→9.0ms mean-frame-time numbers from the prior optimisation pass). This is weaker than a test-backed measurement but still strong analytical evidence. Confidence level stated explicitly in each finding.

**Recommendation for the implementing engineer:** If a future audit wants test-backed evidence in this crate, adding a thin `src/lib.rs` (module re-exports only) and a `[lib]` entry in `Cargo.toml` is the minimal intervention. That is not zero-behavioural-change (the crate gains a library target), so it belongs in a separate planning decision, not in this audit.
