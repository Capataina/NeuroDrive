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

No active implementation plans currently. Milestones 1–5 (PPO baseline + round-2 critic target-scaling) are complete and validated. Milestone 6 (brain-inspired v1) is the next active work area — a plan file will be drafted (`plans/brain-inspired-v1.md`) when implementation begins.

Research for Milestone 6 is captured in `context/references/brain-inspired-learning/` — start with `overview.md` which synthesises the seven deep-dive papers into one picture.
