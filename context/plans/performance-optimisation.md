# Performance Optimisation — Remaining Work

## Status — 2026-04-18

**The PPO hot path is effectively done.** After the dual-backend GEMM + batched actor + `samples_per_tick=32` work:

- Mean frame time 15.7 ms → 0.735 ms (21×)
- PPO Epoch 13.5 ms → 0.45 ms (30×)
- Action selection 1.98 ms → 0.13 ms (16×)
- Budget utilisation 94% → 4.4%
- 0 stutters, 0% frames over budget

The simulation now uses ~5% of the 16.67 ms frame budget. See `notes/performance-tuning-lessons.md` for the contributing factors and `reports/performance/perf_1776539216.md` for the latest measured profile.

**Performance is no longer the dominant constraint.** Further work should be driven by concrete bottleneck measurements, not speculation.

## Remaining Work — Only If Measured Need Appears

### Sensor raycasting spatial index
Current cost: 0.168 ms / tick (11 rays × 8 cars, adaptive marching against flat grid). A precomputed spatial index would halve this at best. Not worth the structural change while we have ~15 ms of headroom.

### Analytics trace-capture sampling
Current cost: ~0.19 ms / tick combined across Trace Capture + Trace Snapshot + PPO Reward Collection. Could gate per-tick capture behind a sampling rate or switch to fixed-capacity ring buffers. Not worth touching while we have headroom.

### Bevy ECS parallelism audit
Some FixedUpdate systems could theoretically run in parallel (analytics snapshots don't depend on HUD updates, etc.). But every candidate pair is sub-millisecond. Expected gain <0.3 ms. Defer until something in the pipeline grows by an order of magnitude.

### Accelerate rectangular-shape rounding
Apple Accelerate's AMX peaks at matrix sizes that are multiples of 8. Our 64×43 layer (obs → actor hidden) hits tile-quantisation overhead. Padding `in_dim` to 48 or 64 at layer construction would claw back a few percent at worst. Only worth it if a specific benchmark shows the 0.45 ms PPO Epoch needs to be ~0.4 ms for some reason.

## Priority Order

**Profile first, then optimise.** If a `cargo run --release --features profiling` report shows a new dominant bottleneck (>2 ms mean), investigate that. Otherwise leave performance alone — the project now has ample headroom for Milestone 2+ work (brain-inspired plasticity, structural growth, replay systems) without further tuning.
