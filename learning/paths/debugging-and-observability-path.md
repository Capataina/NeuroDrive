# Debugging And Observability Path

## Who This Path Is For

Use this path if you care most about detecting regressions, interpreting behaviour, and deciding whether the learner is actually improving.

## Recommended Sequence

- [ ] `concepts/core/determinism-and-fixed-timestep-simulation.md`
- [ ] `project/architecture/data-flow-and-schedule.md`
- [ ] `project/systems/environment.md`
- [ ] `project/systems/a2c-baseline.md`
- [ ] `project/systems/analytics.md`
- [ ] `project/systems/debug-runtime.md`
- [ ] `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
- [ ] `exercises/project/debug-a2c-rollout-alignment.md`
- [ ] `exercises/project/extend-the-analytics-schema.md`
- [ ] `materials/rust-bevy-and-game-loop-engineering.md`

## What You Should Learn

You should finish this path able to answer three questions:

1. where runtime truth is created,
2. where it is only observed and summarised,
3. which current gaps make training results easier to misread than they should be.
