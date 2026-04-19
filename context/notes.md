# Notes Index

Project notes capturing durable design decisions, codebase conventions, and lessons from prior work. Session-scoped work logs live in git history, not here.

| File | Summary |
|------|---------|
| `notes/biology-first-principle.md` | The project's guiding discipline: when we hit a problem, the answer comes from biology, not the ML toolkit. Rules out dropout / batch norm / experience replay as default answers unless they have a direct biological analogue. The thesis that makes NeuroDrive different. |
| `notes/brain-v1-design.md` | Concrete v1 design (Milestone 6): graph topology, three-factor plasticity with eligibility traces, raw-reward modulator (Option C — no critic), homeostasis, continual-backprop structural plasticity, tanh, reserved I/O neurons, cyclic connections with one-step propagation. |
| `notes/brain-v1-decisions.md` | Implementation decision log for Milestone 6. 20 numbered decisions covering the `AgentMode` → per-car `Controller` migration, side-by-side mode, graph storage, plasticity/homeostasis/structural wiring, analytics integration. Companion to `brain-v1-design.md`. |
| `notes/baseline-to-brain-inspired.md` | The transition framing: M1–M5 shipped (PPO validated), M6 next (brain-inspired v1). What carries forward, what changes, what the seven-paper research round settled. Cross-linked to the biology-first principle. |
| `notes/conventions.md` | Codebase conventions not enforced by tooling: per-car Component discipline, shared RNG seeding, `*Config` struct pattern, no Bevy events, `debug_assert!` usage, feature gating, RunContext/retention infrastructure, plugin-scoped system registration, normalisation state lives outside the model struct, disable flags for every training-time normaliser. |
| `notes/development-hardware.md` | Hardware constraints (M2 MacBook Air, Apple Silicon, no CUDA) and their implications for compute budget and optimisation strategy. |
| `notes/normalisation-layers.md` | The three orthogonal normalisations in the PPO stack (advantage / PopArt / observation), what each fixes, disable flags for ablations, and the common misattribution to advantage norm when critic target scale is the real problem. |
| `notes/performance-tuning-lessons.md` | Lessons from the 2026-04-18 dual-backend + batched-actor overhaul (21× frame-time improvement). Contributing factors, architectural patterns worth preserving, ruled-out options, and a "don't over-optimise" guiding principle for future work. |
| `notes/reward-and-entertainment.md` | Design philosophy: reward structure must produce entertaining driving behaviour, not just optimal behaviour — no crash penalty, no survival bonus / time penalty, velocity-projection reward, entertainment-first. Carries forward from PPO into the brain-inspired learner unchanged. |

## Active Work Areas

Forward-looking plans live in `context/plans/`:

- `plans/analytics-tui.md` — idea stage, terminal-based analytics explorer
- `plans/visual-overhaul.md` — idea stage, track/car rendering polish

Milestones 1–6 are all shipped and committed (M6 as six staged commits `6237aa7..c64ce9b` + wrap `4c5c7c5` + default/analytics fix `3a737d9`, pushed to origin/master). The brain-inspired v1 substrate, the full test suite (133 green), the analytics integration, and the side-by-side Fleet Comparison are all live.

The next active work area is **empirical validation of M6** — a real training run (AllBrain or SideBySide) to confirm the visible-learning acceptance bar. After that, M7 (brain visualisation) is the next explicit milestone. Training is running while this upkeep pass is in progress; no plan file has been drafted for validation because the work is "run it and read the report" rather than an implementation effort.

Research for Milestones 6–9 is captured in `context/references/brain-inspired-learning/` — start with `overview.md` which synthesises the seven deep-dive papers into one picture.
