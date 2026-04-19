# Notes Index

Project notes capturing durable design decisions, codebase conventions, and lessons from prior work. Session-scoped work logs live in git history, not here.

| File | Summary |
|------|---------|
| `notes/conventions.md` | Codebase conventions not enforced by tooling: per-car Component discipline, shared RNG seeding pattern, `*Config` struct naming, no Bevy events, `debug_assert!` usage, compile-time feature gating, `RunContext`/retention infrastructure, plugin-scoped system registration, normalisation state lives outside the model struct, disable flags for every training-time normaliser |
| `notes/development-hardware.md` | Hardware constraints (M2 MacBook Air, Apple Silicon, no CUDA) and their implications for compute budget and optimisation strategy |
| `notes/normalisation-layers.md` | The three orthogonal normalisations in the PPO stack (advantage / PopArt / observation), what each fixes, disable flags for ablations, and the common misattribution to advantage norm when critic target scale is the real problem |
| `notes/performance-tuning-lessons.md` | Lessons from the 2026-04-18 dual-backend + batched-actor overhaul (21× frame-time improvement). Contributing factors, architectural patterns worth preserving, ruled-out options, and a "don't over-optimise" guiding principle for future work |
| `notes/baseline-to-brain-inspired.md` | The transition from the PPO baseline (Milestone 1, complete) to brain-inspired local plasticity: why the baseline existed, what carries forward (environment, agent interface, analytics, reward philosophy), what was tried and reverted during the baseline, guiding principles for the upcoming phase |
| `notes/reward-and-entertainment.md` | Design philosophy: reward structure must produce entertaining driving behaviour, not just optimal behaviour — no crash penalty, no survival bonus / time penalty, velocity-projection reward, entertainment-first |

## Active Work Areas

Forward-looking plans live in `context/plans/`:

- `plans/analytics-tui.md` — idea stage, terminal-based analytics explorer
- `plans/visual-overhaul.md` — idea stage, track/car rendering polish

No active implementation plans currently. The round-2 critic target-scaling work landed across commits `c80d2ca` → `e86e737` and has been validated by `reports/analytics/run_1776556719.md`. Next phase (brain-inspired plasticity) is pending scope discussion before any `plans/` file lands.
