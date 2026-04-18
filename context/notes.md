# Notes Index

Project notes capturing durable design decisions, codebase conventions, and lessons from prior work. Session-scoped work logs live in git history, not here.

| File | Summary |
|------|---------|
| `notes/conventions.md` | Codebase conventions not enforced by tooling: per-car Component discipline, shared RNG seeding pattern, `*Config` struct naming, no Bevy events, `debug_assert!` usage, compile-time feature gating, `RunContext`/retention infrastructure, plugin-scoped system registration |
| `notes/development-hardware.md` | Hardware constraints (M2 MacBook Air, Apple Silicon, no CUDA) and their implications for compute budget and optimisation strategy |
| `notes/performance-tuning-lessons.md` | Lessons from PPO performance optimisation — wider critic cost, samples_per_tick as stutter knob, flat storage wins, bimodal frame pattern |
| `notes/reward-and-entertainment.md` | Design philosophy: reward structure must produce entertaining driving behaviour, not just optimal behaviour — no crash penalty, no survival bonus / time penalty, velocity-projection reward, entertainment-first |

## Active Work Areas

Forward-looking plans live in `context/plans/`:

- `plans/ppo-optimisation.md` — remaining PPO items (observation normalisation, LR annealing, log-std Adam extraction)
- `plans/performance-optimisation.md` — remaining non-PPO bottlenecks (raycasting, rendering, analytics capture, ECS scheduling); profile first
- `plans/testing-strategy.md` — test suite to expand beyond the current 31; `[lib]` target unblocks integration tests in `tests/*.rs`
- `plans/analytics-tui.md` — idea stage, terminal-based analytics explorer
- `plans/visual-overhaul.md` — idea stage, track/car rendering polish
