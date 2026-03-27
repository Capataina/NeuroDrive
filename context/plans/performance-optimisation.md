# Idea — Performance Optimisation

## Vision

The entire point of NeuroDrive is watching cars learn in real time. With just 3 cars, the simulation already stutters noticeably. This needs to be fixed comprehensively — not just PPO compute, but everything: rendering, physics, raycasting, memory allocation, ECS scheduling, and the analytics capture pipeline.

The target is smooth 60 FPS with 16+ cars and all debug overlays enabled.

## Why This Matters

- Stuttering breaks the "watchable learning" promise of the project
- Higher car counts (8–16) dramatically improve PPO sample efficiency, but only if the simulation can keep up
- Performance headroom enables future features: more complex tracks, visual effects, live TUI streaming
- The biological brain (Milestone 2) will add per-synapse computation — the system needs to be lean before that arrives

## Known Bottleneck Areas (brainstorm)

### PPO / Brain
- Per-sample forward pass during updates (should be batched matrix ops)
- Vec allocations on every `Linear::forward` and `Relu/Tanh::forward` call
- Input cache cloning (`input.to_vec()`) on every forward pass
- Buffer `clone()` when preparing PPO updates (should be swap)
- No SIMD — inner loops are scalar f32 dot products
- GAE computation iterates per-env with HashMap lookups

### Raycasting / Observations
- 11 rays × per-step marching with binary-search refinement × N cars × 60 Hz
- `is_road_at` queries the grid every step — cache locality may be poor
- Sensor system runs every fixed tick even when readings haven't meaningfully changed

### Rendering / Debug
- Gizmo drawing for all overlays happens every frame regardless of visibility
- Leaderboard and HUD text updates every frame
- Sprite rendering for N cars (minor, but scales)

### Analytics
- Per-tick trace capture allocates and pushes to vectors every tick
- Episode tracker folds on every frame in the Update schedule
- Action accumulators run per-tick per-car

### Memory / Allocation
- Frequent small Vec allocations throughout the hot path
- No object pooling for temporary computation buffers
- Rollout buffer grows unbounded then clones and clears

### ECS Scheduling
- System ordering may cause unnecessary pipeline stalls
- Some systems could potentially run in parallel but are sequenced by set membership

## Potential Approaches (not committed)

- Pre-allocated scratch buffers for all neural network operations
- Batched matrix multiply for PPO update chunks
- SIMD intrinsics or a lightweight linalg crate (e.g., `ultraviolet`, `glam` for matrix ops)
- Spatial grid optimisation for raycasting (precomputed boundary cells)
- Conditional overlay rendering (skip draw calls when overlays are toggled off)
- Double-buffered rollout buffer (swap instead of clone+clear)
- Profile-guided optimisation pass with `cargo flamegraph`
- Bevy system parallelism audit

## Open Questions

- What is the actual frame time breakdown? Need a profiling pass (flamegraph) before committing to specific optimisations.
- Which bottleneck dominates: PPO updates, raycasting, or rendering?
- Is the 60 Hz fixed timestep the right target, or should we consider decoupling render and sim rates?
- How much does car count actually affect frame time? Need to benchmark 3 vs 8 vs 16.

## Status

Idea stage. Should be preceded by a profiling pass to identify the actual bottlenecks rather than guessing. Revisit after the PPO optimisation plan is underway.
