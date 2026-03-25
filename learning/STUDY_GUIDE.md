# Study Guide

This is a route selector, not a rigid curriculum. Pick the path that matches what you need right now, then combine paths as your understanding deepens.

## Choose A Route

- [ ] Start with `paths/project-architecture-path.md` if you want a top-down understanding of the current runtime first
- [ ] Start with `paths/reinforcement-learning-path.md` if your main goal is to understand the A2C baseline clearly
- [ ] Start with `paths/systems-and-simulation-path.md` if you care most about determinism, scheduling, physics, and environment truth
- [ ] Start with `paths/implementation-first-path.md` if you want the fastest route to making safe code changes

## Suggested Combinations

- [ ] Project architecture + systems and simulation
- [ ] Reinforcement learning + project A2C baseline
- [ ] Core concepts + project comparison of A2C versus the biological target
- [ ] Implementation-first path + project exercises

## Recommended Starting Points

- New to NeuroDrive but not new to Rust/Bevy:
  Read `project/architecture/runtime-architecture.md` first.
- Comfortable with RL, unsure about the repo:
  Read `project/systems/a2c-baseline.md` then `project/systems/environment.md`.
- Comfortable with games, unsure about RL:
  Read `concepts/core/actor-critic-and-gae.md` before touching the A2C system file.
- Interested in the long-term brain direction:
  Read `project/comparisons/a2c-baseline-vs-biological-target.md` after the architecture file.

## What This Learning System Assumes

- You can read basic Rust syntax.
- You want to understand both current implementation reality and future project direction.
- You prefer grounded explanations over hype or abstract roadmap prose.
