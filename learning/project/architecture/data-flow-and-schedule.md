# Data Flow And Schedule

## Status

Current in the project runtime.

## Why This File Matters

If you misunderstand the tick ordering in NeuroDrive, you will misunderstand the whole runtime.

This project is intentionally schedule-sensitive:

- actions must be selected before physics,
- collision truth must exist before episode finalisation,
- reward and terminal state must exist before A2C reward collection,
- observation rebuild should reflect reset state after terminal handling.

## Current FixedUpdate Flow

```text
SimSet::Input
  keyboard input
  A2C act
  action smoothing

SimSet::Physics
  car physics
  action stats capture

SimSet::Collision
  off-track detection

SimSet::Measurement
  progress update
  episode loop
  sensor update
  observation vector build
  trace capture
  A2C reward collection
  HUD fixed-tick stats
```

## Why This Exact Order Exists

### Input first

The runtime needs a single desired action before physics runs. Both keyboard and AI paths write into the same control surface.

### Physics before collision

Collision truth is about the post-step vehicle state, not about the previous tick’s state.

### Collision before episode truth

Episode logic must know whether the current tick ended in a crash.

### Episode truth before observation rebuild

If a terminal event causes a reset, post-reset observations should describe the reset position, not the crash frame. This is subtle but important for training correctness.

### Observation rebuild before A2C reward collection bootstrap usage

When a rollout truncates without terminal completion, the next observation may be needed for bootstrap value estimation.

## Update And Last Schedules

Outside `FixedUpdate`, the repository also uses:

- `Update` for mode toggling, HUD text refresh, and overlay rendering,
- `Last` for A2C flush-on-exit and analytics export.

This division is sensible:

- simulation truth lives in fixed tick,
- UI interaction and app-level controls live in regular update,
- final export lives on shutdown.

## Design Lesson

A lot of reliability in learning systems does not come from "better neural nets". It comes from clear ownership of temporal truth.

NeuroDrive’s schedule design is already part of its engineering quality. It is also why future refactors should be conservative about moving systems between sets without a strong reason.

## Related Files

- `project/systems/environment.md`
- `project/systems/a2c-baseline.md`
- `project/systems/analytics.md`
- `project/systems/debug-runtime.md`
