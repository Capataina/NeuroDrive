# Pass 2 Systems Audited — 2026-04-18

Per-system static snapshot of Pass-2 work. This file is distinct from the live Obligation Evidence Map (which records evidence as the audit runs); it is the final checkpoint written before index.md.

| System | Research (query + URL + mode) | Tests written (path + assertion + result) | Findings | Confidence |
|---|---|---|---|---|
| PPO hot path (`src/brain/ppo/`, `src/brain/common/`) | Q1: "reusable scratch buffer vs fresh Vec allocation per forward pass MLP Rust inference latency 2025" — <https://nnethercote.github.io/perf-book/heap-allocations.html>, <https://markaicode.com/rust-ml-Building-high-performance-inference-engines-2025/>; Mode 2. Q2: "Rust std::slice::from_raw_parts shared slice alias mutable scratch buffer soundness unsafe patterns" — <https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html>, <https://doc.rust-lang.org/nomicon/borrow-splitting.html>; Mode 3. | **Reasoned omission** (binary-only crate blocks integration tests — see Obligation Evidence Map). Attempted test: `tests/ppo_forward_hotpath_baseline.rs` — failed to compile (unresolved crate name) — retracted. | 4 (1 high, 2 medium, 1 low) — [brain-ppo.md](brain-ppo.md) | High analytical; would be higher with test-backed timing. |
| Environment / reward (`src/game/`, `src/agent/`, `src/maps/`) | Q3: "2D racing environment reinforcement learning reward shaping centreline projection performance patterns" — <https://arxiv.org/html/2504.02420v2>, <https://arxiv.org/pdf/2103.10098>; Mode 1. | **Reasoned omission** — same binary-only constraint; findings rely on existing 31-test suite + analytical evidence. | 2 (0 high, 1 medium, 1 low) — [environment.md](environment.md) | High on the drift finding; moderate on the dead-code cleanup. |
| Analytics (`src/analytics/`) | Research skipped per `detection-strategies.md` §"When Research Is Not Required" — folds/aggregates on episode-end and on-exit, not hot-path; no substantial-system research frontier. | Not written — no uncertainty a test would resolve better than reading. | 0 direct (the cross-cutting `crash_penalty` finding touches analytics indirectly) | n/a |
| Cross-cutting (Cargo, test infra, docs ↔ code drift) | Research mode 1 cross-referenced against Bevy project conventions (bevy_best_practices); dependency manifest inspected directly. | Not written — same infrastructure gap; the finding itself is *about* that gap. | 2 (0 high, 1 medium, 1 low) — [cross-cutting.md](cross-cutting.md) | High. |
| Small glue (`src/sim/`, `src/maps/parts/`, plugin.rs files under 100 LoC) | Research reasoned-omitted per `detection-strategies.md` §"When Research Is Not Required" — type definitions, plugin glue. | Not written — no uncertainty; code is mechanical. | 0 | n/a |
| Profiling (`src/profiling/`) | Feature-gated; zero runtime cost when off. Reviewed briefly; no findings. Research reasoned-omitted because the system is off-hot-path and the prior audit covered the timing capture path. | Not written. | 0 | n/a |
| Debug overlays + HUD (`src/debug/`) | Defaults-off; reviewed briefly. No hot-path findings. Research reasoned-omitted as above. | Not written. | 0 | n/a |

## Summary

- Total findings: **8** (1 high, 4 medium, 3 low — no critical).
- Research floor met: 4 WebSearch calls total (1 pre-Pass-1 + 3 Pass-2) spanning modes 1, 2, 3.
- Diagnostic-test floor: **partial** — reasoned omission recorded for every finding that would have benefited from a test; attempt was made and documented.
- Data Layout applicability decision recorded per system (see individual finding files).
- Known Issues already in context files: two documented in `systems/brain-ppo.md` (no save/load, alignment debug_assert only). Not duplicated as findings; noted in the Pass-1 checkpoint.

## Entry conditions to final output satisfied

- Pass-1 checkpoint exists.
- Obligation Evidence Map has one row per substantive system (no PENDING rows).
- Three research modes represented.
- Each finding includes the full proof chain (current, proposed, justification, expected benefit, impact) and confidence level.
- No production source code was modified.

Proceeding to `index.md`.
