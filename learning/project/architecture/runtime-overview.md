# Runtime Overview

## Status

Current in the project runtime.

## Why This Architecture Matters

NeuroDrive is already large enough that you should not think of it as "a car and a learner". It is a structured runtime with clear subsystem roles:

- maps,
- game/environment,
- agent interface,
- brain,
- analytics,
- debug,
- shared simulation ordering.

Understanding that split is the fastest way to avoid confused edits.

## High-Level Shape

```text
Track + centreline
    -> car physics and episode truth
    -> observation/action interface
    -> brain acts through stable control boundary
    -> analytics records behaviour and updates
    -> debug renders live interpretation
```

## Subsystem Boundaries

### `maps`

Owns:

- track construction,
- grid occupancy,
- centreline derivation,
- spawn pose,
- some visual geometry.

Why foundational:
Every later subsystem depends on spatial truth from here.

### `game`

Owns:

- the car entity,
- deterministic dynamics,
- collision truth,
- progress measurement,
- reward shaping,
- episode reset logic.

This subsystem is where environment truth is created.

### `agent`

Owns:

- stable action semantics,
- optional action smoothing,
- sensor derivation,
- observation vector construction.

This is the contract layer between environment and controller.

### `brain`

Owns:

- controller mode selection,
- current A2C baseline,
- rollout and update logic.

It consumes observation and reward truth but does not define them.

### `analytics`

Owns:

- run tracking,
- tick and episode record building,
- derived metrics,
- export to JSON and Markdown.

It observes runtime truth; it should not mutate it.

### `debug`

Owns:

- live world overlays,
- HUD summaries,
- quick run-health interpretation.

Again, it is an observer, not a truth source.

## Repository-Wide Architectural Story

The current implementation stage can be summarised like this:

- the environment is already substantial,
- the controller interface is now stable enough to matter,
- the A2C baseline is real,
- analytics and debug infrastructure are not roadmap-only extras,
- the biological-learning path is still ahead of the runtime rather than inside it.

That is why the project feels transitional rather than unfinished.

## Pressure Points

The current architecture is coherent, but several tensions remain:

- singleton-car assumptions limit vectorised training,
- A2C reproducibility is weaker than environment determinism,
- run metadata is thinner than a serious experiment workflow needs,
- the biological `src/brain/biological/` direction is still not implemented.

## Related Files

- `project/architecture/data-flow-and-schedule.md`
- `project/systems/environment.md`
- `project/comparisons/current-baseline-vs-target-biological-system.md`
