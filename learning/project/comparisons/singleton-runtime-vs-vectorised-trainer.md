# Singleton Runtime Versus Vectorised Trainer

## Why This Comparison Matters

One of the clearest next implementation pressures in NeuroDrive is the gap between today’s singleton runtime and the proposed 25-car vectorised A2C trainer.

## Current Singleton Reality

Today the runtime assumes:

- one car,
- singleton action state,
- singleton episode truth,
- singleton-oriented debug views,
- analytics shaped around one active driving instance at a time.

This keeps the code simpler and the visible scene easy to reason about.

## Proposed Vectorised Direction

The plan in `context/plans/vectorised-a2c-visual-trainer.md` argues for:

- many visible cars,
- one shared policy,
- per-car environment truth,
- trainer-level ranking,
- cohort-aware analytics,
- visual highlighting of the best current performer.

## Why The Difference Is Structural

This is not just "spawn more cars". It changes core assumptions across:

- environment state ownership,
- action and observation storage,
- episode bookkeeping,
- analytics schemas,
- debug focus logic,
- rollout alignment rules.

## Why The Vectorised Direction Is Attractive

- more A2C-faithful synchronous collection,
- better batch efficiency,
- less reliance on a single trajectory stream,
- more visually informative training runs.

## Main Hidden Costs

- removal of many singleton assumptions,
- more complex debug and analytics ownership,
- higher risk of state-alignment bugs,
- potential visual clutter,
- more demanding verification needs.

## Takeaway

The vectorised trainer is a strong next systems direction, but it should be treated as a real architecture project, not a convenience feature.

## Related Files

- `project/systems/environment.md`
- `project/systems/a2c-baseline.md`
- `project/evolution/project-state-and-next-pressure-points.md`
- `exercises/project/design-the-vectorised-trainer-boundaries.md`
