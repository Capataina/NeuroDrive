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

### PPO / Brain (largely addressed)
- ~~Per-sample forward pass during updates (should be batched matrix ops)~~ **Done** — batched forward/backward passes
- ~~Vec allocations on every `Linear::forward` and `Tanh::forward` call~~ **Done** — pre-allocated `BatchScratch` buffers
- ~~Input cache cloning (`input.to_vec()`) on every forward pass~~ **Done** — flat weight storage with in-place operations
- ~~Buffer `clone()` when preparing PPO updates (should be swap)~~ **Done** — swap instead of clone
- No SIMD — inner loops are scalar f32 dot products (but iterator chains enable LLVM auto-vectorisation)
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

## Extensive Performance Profiling and Live Monitoring

### Vision

Build a comprehensive, engine-grade performance profiling and monitoring system — inspired by the profiling tools in Unity and Unreal Engine. The system has two components:

1. **Performance export** — the running NeuroDrive app instruments its own frame loop and writes structured timing data to a file (or streams it via a socket/shared-memory channel), similar to how the analytics pipeline exports episode data to `reports/`. This captures per-frame and per-system timing breakdowns: how long each ECS system takes, frame budget utilisation, allocation counts, PPO update cost, raycasting cost, etc.

2. **Live TUI performance viewer** — a separate terminal process that reads the performance stream in real time and renders a live dashboard showing frame time breakdown, system hotspots, per-tick budget charts, and historical trends. You run the NeuroDrive app in one terminal and the TUI viewer in another — the viewer shows what the app is doing without affecting its performance.

### Architecture Sketch

```
┌──────────────────┐         shared file / socket / mmap         ┌──────────────────┐
│   NeuroDrive     │ ──── writes timing data each frame ────────►│  perf-viewer     │
│   (Bevy app)     │                                             │  (TUI binary)    │
│                  │                                             │                  │
│  FrameTimer      │         e.g. reports/perf-stream.bin        │  ratatui panels  │
│  SystemTimers    │         or localhost:9999 UDP                │  frame budget    │
│  AllocCounter    │         or /tmp/neurodrive-perf.mmap         │  system heatmap  │
│                  │                                             │  sparklines      │
└──────────────────┘                                             └──────────────────┘
```

### What to Instrument

- **Per-system timing**: wrap each FixedUpdate / Update system with timing guards, record min/max/mean per system per second
- **Frame budget**: total frame time vs 16.67ms target, stutter detection
- **PPO update cost**: time per chunk, time per epoch, amortised cost per tick
- **Raycasting cost**: time per car, time per ray, total per tick
- **Memory pressure**: allocation counts per frame (using a custom allocator or counters)
- **Buffer sizes**: rollout buffer length, trace accumulator sizes, HUD history length
- **Car count scaling**: how each metric changes as car count increases

### Export Format

A binary or MessagePack stream of per-frame records, compact enough to write at 60 Hz without measurable overhead. Each record contains system timings as a flat struct — no heap allocations in the hot path. The file rotates or caps at a configurable size.

Post-run, a static report can be generated from the exported data (similar to the Markdown analytics report) — a performance summary with worst-frame analysis, system hotspot ranking, and scaling projections.

### TUI Viewer

A separate Rust binary (same workspace, different `[[bin]]` target) using `ratatui` or `crossterm`. Reads the performance stream and renders:

- Frame time sparkline (last N frames)
- System timing breakdown (stacked bar or heatmap)
- Per-system min/mean/max table
- Stutter counter and worst-frame highlight
- PPO update cost overlay
- Car count and current episode

The viewer is read-only — it cannot affect the running app. It connects on launch, shows "waiting for data" if the app hasn't started, and reconnects if the app restarts.

### Open Design Questions

- **Transport**: file-based (simplest, slight latency), UDP socket (low-latency, fire-and-forget), or memory-mapped file (zero-copy, platform-specific)? File-based is the pragmatic starting point.
- **Bevy integration**: Bevy 0.18 has `bevy_diagnostic` with `FrameTimeDiagnosticsPlugin` — can we hook into this, or is a custom system-timing approach better for per-system granularity?
- **Overhead budget**: the instrumentation itself must cost <0.5ms per frame. Timing guards using `std::time::Instant` are cheap; allocation tracking is harder.
- **Workspace structure**: should the TUI viewer be a separate binary in the same Cargo workspace, or a separate project?

## Status

Partially implemented. The profiling system (per-system timing, auto-exit, JSON + Markdown reports) is now live behind `--features profiling`. Several PPO performance optimisations have been completed (flat weight storage, pre-allocated scratch buffers, batched forward/backward, iterator-based loops, swap instead of clone, Adam bias correction precomputation). The live TUI viewer remains idea-stage. Car count has been bumped to 8. Revisit for raycasting, rendering, and analytics optimisations.
