# Notes Index

Project notes capturing design decisions, preferences, conventions, and lessons from prior sessions.

| File | Summary |
|------|---------|
| `notes/development-hardware.md` | Hardware constraints (M2 MacBook Air, Apple Silicon, no CUDA), implications for compute budget and optimisation strategy |
| `notes/reward-and-entertainment.md` | Design philosophy: reward structure must produce entertaining driving behaviour, not just optimal behaviour — no crash penalty, velocity-projection reward, entertainment-first design constraint |
| `notes/performance-tuning-lessons.md` | Lessons from PPO performance optimisation: wider critic cost, samples_per_tick as stutter knob, flat storage wins, bimodal frame pattern |
| `notes/conventions.md` | Codebase conventions not enforced by tooling: per-car Component discipline, shared RNG seeding pattern, `*Config` struct naming, no Bevy events, `debug_assert!` usage, compile-time feature gating, shared `RunContext`/retention infrastructure, plugin-scoped system registration |
| `notes/session-2026-04-15.md` | Full code health audit implementation (22/23 findings), README rewrite, upkeep-context pass. `unsafe` in update.rs verified sound in the 2026-04-18 pass |
| `notes/session-2026-04-18.md` | Analytical obligations pass: inter-system relationships, dependency chain trace, coverage/gaps, convention capture all added to `context/`. No source changes |

## Active Work Areas

Context files most likely to be touched during ongoing Milestone 1 work:

- `systems/brain-ppo.md`, `systems/environment.md` — remain the active frontiers (corner-survival learning, critic capacity).
- `plans/ppo-optimisation.md`, `plans/performance-optimisation.md`, `plans/testing-strategy.md` — carry open work items for future sessions.
- `plans/code-health-audit/` — working tree currently shows uncommitted reorganisation from a separate skill run; owned by `code-health-audit`, not this upkeep skill.
