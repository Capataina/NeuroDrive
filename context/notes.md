# Notes Index

Project notes capturing design decisions, preferences, and lessons from prior sessions.

| File | Summary |
|------|---------|
| `notes/development-hardware.md` | Hardware constraints (M2 MacBook Air, Apple Silicon, no CUDA), implications for compute budget and optimisation strategy |
| `notes/reward-and-entertainment.md` | Design philosophy: reward structure must produce entertaining driving behaviour, not just optimal behaviour — no crash penalty, velocity-projection reward, entertainment-first design constraint |
| `notes/performance-tuning-lessons.md` | Lessons from PPO performance optimisation: wider critic cost, samples_per_tick as stutter knob, flat storage wins, bimodal frame pattern |
| `notes/session-2026-04-15.md` | Full code health audit implementation (22/23 findings), README rewrite, upkeep-context pass. Unsafe in update.rs to verify. |
