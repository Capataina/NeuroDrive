# Performance Tuning Lessons

## Core Constraint

The simulation must run at 60 Hz with 8 cars and feel smooth to watch. Performance isn't about hitting benchmarks — it's about the game being enjoyable to observe while cars learn.

## State of Play — 2026-04-18

After the dual-backend + batched-actor performance overhaul, the simulation runs in **4.4% of the frame budget** (0.735 ms mean frame time, zero stutters, 0% frames over budget). **Performance is no longer the dominant constraint** on what the project can add. See `reports/performance/perf_1776539216.md` for the full breakdown.

Previous baseline was 15.7 ms mean, 94% budget utilisation, 50% of frames over budget. The 21× overall improvement is substantially larger than the research papers projected (8–20× was the estimate for the dominant PPO Epoch alone).

## What Worked

### Contributing factors to the 2026-04-18 transformation

1. **Apple Accelerate backend** (`cblas_sgemm` → AMX) — biggest contributor. Dropped PPO Epoch from 13.5 ms to 0.45 ms (30×). Evidence: meekolab benchmarks show Accelerate hits ~700 GFLOP/s at N=64 on M1/M2 vs ~80 GFLOP/s for OpenBLAS. Calling it was ~15 lines of Rust FFI via the `cblas` crate.
2. **Batched multi-car actor forward** (`forward_actor_batch`) — mirrors the pre-existing `forward_critic_batch`. Stacks all 8 car observations into `batch_io.obs_batch`, runs one mat-mat through the actor. Dropped action-selection from 1.98 ms to 0.13 ms (16×). No new dependencies.
3. **`samples_per_tick: 64 → 32`** — trivial config change that halves per-tick PPO Epoch work. On its own it's just amortisation (no FLOP reduction), but combined with the GEMM backend speedup it eliminated any residual stutter risk.
4. **`VECLIB_MAXIMUM_THREADS=1` pin** — prevented Accelerate's default thread pool from spawning workers that compete with Bevy's render thread at our small matrix sizes. Invisible unless you measure it, but the report would otherwise show jitter spikes around PPO Epoch ticks.

### Earlier lessons that still apply

- **Flat `Vec<f32>` weight storage** remains load-bearing — the switch from `Vec<Vec<f32>>` was the single biggest win on the path to today's state (the 30-March audit caught ~43× cache-miss penalty from the nested structure). Every subsequent backend (scalar loop, matrixmultiply, Accelerate) depends on contiguous row-major memory.
- **Pre-allocated scratch buffers** (`BatchIo`, `BatchScratch`, `SampleScratch`) are still the pattern for the hot path — Accelerate doesn't obviate the need to avoid per-call heap allocations.
- **Disjoint-field borrow splitting** let us eliminate three `unsafe { slice::from_raw_parts }` blocks: by making inputs (`BatchIo`) and intermediates (`BatchScratch`) sibling fields on `ActorCritic`, Rust's borrow checker accepts `&mut self.scratch` and `&self.batch_io.*` simultaneously via disjoint-field inference.

## What Did Not Work / Ruled Out

- **Hand-rolled NEON intrinsics** were evaluated in the 2026-04-18 research and ruled out. The `matrixmultiply` crate already uses NEON microkernels under the hood; writing our own would produce at best 1.5-2.5× over scalar (not the naive 4×) and is dominated by either matrixmultiply or Accelerate. Maintenance cost is disproportionate for a "baseline" PPO that will be retired.
- **f16/bfloat16 quantisation** ruled out by the numerical-correctness constraint — PPO gradient stability needs f32 precision.
- **GPU offload (Metal)** ruled out — the M2 Air has no discrete GPU and at sub-millisecond kernels the CPU↔GPU synchronisation overhead dominates. Bevy's render loop already contends for the GPU.
- **`faer-rs`** ruled out — positioned as a medium/large-matrix library, not a fit for 64×128 GEMMs.
- **Wider critic on top of 128** — a larger critic (say 2×256) would fit the budget now that we have 95% spare, but the 2×128 width is sufficient per the 30 March analysis. No reason to widen further without a concrete capacity-limited symptom.

## Architectural Patterns Worth Preserving

- **Three-way backend dispatch behind a single stable API** — `gemm_backend::sgemm/sgemm_nt/sgemm_tn` is the shape, with compile-time `#[cfg]`-selected implementations. Call sites know nothing about which backend is active. This pattern extends cleanly to other hardware-specific optimisations (e.g., SIMD raycasting) if ever needed.
- **Platform auto-default with opt-in override** — default Cargo features pick the platform-optimal backend without the user doing anything; `--features force-*` exists for A/B testing and CI. This is the right balance between "works out of the box" and "verifiable across backends".
- **Mandatory backend disclosure in performance reports** — the `### Build` section in every perf artefact records which backend produced the numbers. Makes cross-run comparisons straightforward and catches accidental configuration drift.

## The Bimodal Frame Pattern

Resolved. Before today, training-ticks ran PPO Epoch at 13-32 ms (over budget), non-training ticks at ~2-3 ms. The pattern caused the "feels choppy" complaint even though the profile-report stutter count was often low. After today, training-ticks run PPO Epoch at ~0.5-1 ms, non-training ticks stay at ~0.5 ms. The bimodal pattern is structurally still present but both modes are well under budget, so it's invisible.

## Next Optimisations Only If Needed

These are documented for completeness but **not worth pursuing** unless a specific workload creates a new bottleneck:

- Sensor raycasting spatial index — current cost is 0.168 ms, would halve at best. Not worth the structural change.
- Analytics trace-capture sampling — 0.19 ms per tick. Could gate per-tick capture behind a sampling rate. Not worth it while we have 15+ ms of headroom.
- Bevy ECS parallelism audit — FixedUpdate systems are currently sequenced by SimSet. Some could theoretically run in parallel. But all candidate pairs (e.g. HUD updates vs analytics snapshots) are already sub-millisecond each; parallelising saves at most ~0.3 ms.
- Accelerate rectangular-shape rounding — 64×43 is not a multiple of 8 and may under-utilise AMX. Unmeasured. If it matters, pad `in_dim` to 48 or 64 at layer construction. Not pursued because measured performance is already so far under budget.

## Guiding Principle

**Optimise when you hit a concrete wall, not on speculation.** Today's work was justified because the user saw visible stutter and over-budget frames. Every recommendation in the 2026-04-18 research papers was grounded in measured data. Future optimisation decisions should follow the same discipline — a profiling report showing a new bottleneck, not a hunch about what might be slow.
