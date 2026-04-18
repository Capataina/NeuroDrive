# Performance Optimisation — Remaining Work

## Vision

Smooth 60 FPS with 16+ cars and all debug overlays enabled. Stuttering breaks the "watchable learning" promise of the project.

## Status

PPO hot-path work is largely done. Prior audits closed the expensive wins: flat weight storage, pre-allocated `SampleScratch` + `BatchScratch`, batched forward/backward, cache-friendly loop order, per-sample `Vec` allocation removal, `HashMap` → `Vec` on GAE env grouping, `unsafe` eliminated via borrow split. Profiling is live behind `--features profiling`, producing per-system timings and auto-exit reports.

Car count is 8. The last benchmarked run moved mean frame time from ~17 ms to ~9 ms with the batched critic path and dropped stutters from 426 to 2. Those numbers predate the 2026-04-18 `SampleScratch` refactor — a fresh profiling pass would confirm the current baseline.

The remaining bottlenecks are **outside the PPO hot path**.

## Remaining Work

### Raycasting / Observations
- 11 rays × per-step marching × 8 cars × 60 Hz remains the next-largest per-tick cost
- `is_road_at` queries the grid every step; cache locality may still be poor despite the adaptive step win
- Sensor system runs every tick even when readings barely change (temporal coherence unused)
- **Candidates:** precomputed boundary-cell spatial index; conditional rebuild only when a car has moved more than ε

### Rendering / Debug
- Gizmo drawing runs every frame regardless of overlay visibility
- Leaderboard + HUD text updates every frame even when state is unchanged
- **Candidates:** skip draw passes when overlays are toggled off; dirty-flag HUD text so only changed fields rebuild

### Analytics capture
- Per-tick trace allocates and pushes to vectors every tick
- Episode tracker folds every frame in Update
- **Candidates:** fixed-capacity ring buffers; fold only on episode boundary, not every frame

### ECS scheduling
- Some FixedUpdate systems could potentially run in parallel but are sequenced by `SimSet` membership
- **Candidates:** audit which systems genuinely need strict ordering (physics → collision → measurement) vs which were sequenced defensively

### Memory / allocation
- `obs_stack` `Vec` in `ppo_act_all_cars_system` is allocated fresh each tick — could be a resource
- Rollout buffer capacity is lost on `std::mem::take` rotation; a double-buffered pattern would preserve it

## Priority Order

Profile first, then optimise. Run `cargo run --features profiling`, inspect the generated report, and tackle whichever subsystem dominates. Without a fresh profiling pass after the 2026-04-18 PPO work, any guess at the current top bottleneck is stale.
