# Study Guide

This guide is not a single mandatory syllabus. It is a route selector.

NeuroDrive can be learned from several legitimate angles:

- as a Bevy/Rust simulation project,
- as an RL baseline-validation project,
- as a brain-inspired systems project in transition,
- as an observability and experiment-discipline problem,
- or as a future architecture exercise that has not yet reached its final form.

## Choose A Route

- [ ] Start with [paths/project-architecture-path.md](./paths/project-architecture-path.md) if you want a top-down understanding of the repository as it exists now.
- [ ] Start with [paths/implementation-first-path.md](./paths/implementation-first-path.md) if your immediate goal is to contribute safely to the codebase.
- [ ] Start with [paths/reinforcement-learning-path.md](./paths/reinforcement-learning-path.md) if you want to understand the current A2C baseline in detail.
- [ ] Start with [paths/neuroscience-path.md](./paths/neuroscience-path.md) if you care most about the README’s long-term brain-inspired direction.
- [ ] Start with [paths/debugging-and-observability-path.md](./paths/debugging-and-observability-path.md) if you care most about validating behaviour, measuring learning, and catching regressions.
- [ ] Start with [paths/foundations-path.md](./paths/foundations-path.md) if you want the theory scaffold before touching project files.

## Recommended Starting Advice By Background

If you are:

- **comfortable with Rust/Bevy but weaker on RL**
  Start with `foundations-path`, then `reinforcement-learning-path`, then `project-architecture-path`.
- **comfortable with RL but new to ECS/game-loop architecture**
  Start with `project-architecture-path`, then `debugging-and-observability-path`.
- **primarily interested in the project’s biological-learning ambition**
  Start with `neuroscience-path`, but do not skip the comparison files that explain why A2C exists right now.
- **trying to make safe implementation changes quickly**
  Start with `implementation-first-path`.

## Recommended Pairings

- [ ] `foundations-path` + `project-architecture-path`
- [ ] `reinforcement-learning-path` + `debugging-and-observability-path`
- [ ] `neuroscience-path` + `project/comparisons/current-baseline-vs-target-biological-system.md`
- [ ] `implementation-first-path` + `exercises/project/inspect-the-observation-pipeline.md`
- [ ] `project-architecture-path` + `exercises/project/debug-a2c-rollout-alignment.md`

## Suggested Milestones

- [ ] Milestone 1: I can explain the repository’s current runtime pipeline from observation to action to reward to update.
- [ ] Milestone 2: I understand why the current system is both useful and incomplete as a baseline.
- [ ] Milestone 3: I can explain the difference between present A2C implementation reality and the README’s eventual local-plasticity goal.
- [ ] Milestone 4: I can inspect a proposed change and predict which subsystems, docs, analytics, and exercises it should touch.

## What To Read If You Have Limited Time

If you only have one short session:

1. `LEARNING_MAP.md`
2. `project/architecture/runtime-overview.md`
3. `project/comparisons/current-baseline-vs-target-biological-system.md`
4. `project/evolution/project-state-and-next-pressure-points.md`

If you have one focused evening:

1. `paths/project-architecture-path.md`
2. `project/systems/environment.md`
3. `project/systems/agent-interface.md`
4. `project/systems/a2c-baseline.md`
5. `project/systems/analytics.md`
6. `exercises/project/inspect-the-observation-pipeline.md`

## What To Do Next After A Route

After finishing any major path, you should do one of:

- a project exercise,
- a comparison file,
- a materials guide for deeper external reading.

The archive is designed to alternate between explanation and active reasoning. If you only read passively, you will understand the vocabulary but not the system.
