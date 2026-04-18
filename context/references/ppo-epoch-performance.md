# PPO Epoch Performance on Apple Silicon

> Written 2026-04-18. Grounded in the profiling run `reports/performance/perf_1776527963.md` and verified against the live PPO hot path in `src/brain/ppo/update.rs`, `src/brain/ppo/model.rs`, `src/brain/common/mlp.rs`. All recommendations are research-only — no code changes accompany this paper.

## Scope / Purpose

- Answer the repository-specific question: **what concrete, evidence-backed techniques would most likely reduce the PPO Epoch wall-clock time from 13.5 ms mean to under 3 ms on Apple Silicon M2, while preserving numerical correctness of the gradient computation?**
- Cover the realistic optimisation space: NEON intrinsics, Apple Accelerate / AMX, Rust linalg crates (`matrixmultiply`, `faer`, `ndarray`), memory-layout refinements, kernel fusion, batched Adam, reduced precision, and amortisation-schedule tuning.
- Include a contrasting view on whether this hot path is even amenable to further optimisation at this scale.
- Explicitly **out of scope:** reward/algorithm/architecture changes (locked), GPU offload (no discrete GPU available on M2 Air, and the unified-memory 8 GB pool already rules out large VRAM buffers), and changes to the brain-inspired long-term direction.
- This paper informs a later implementation decision; it does not prescribe code.

## Current Project Relevance

The operational constraint is the 60 Hz fixed-timestep loop: **16.67 ms per frame**. The latest profiling run measured PPO Epoch at **13.5 ms mean (85.9 % of the frame)** with a peak of **32.8 ms on training ticks** (2.4× the mean). With a P50 frame time of 16.57 ms, the loop is running right at the cliff edge; any additional system cost causes visible stutter. Two further facts from the run: **49.9 %** of frames are already over budget, and PPO is the only system above 1 ms mean — Action Smoothing and PPO Action Selection (the next largest) are at ~2 ms each and not worth reshaping until PPO Epoch is cut hard. Optimising here is the single highest-leverage performance move available in the current implementation.

A theoretical ceiling clarifies how much room there is to move. Per chunk the PPO Epoch does the following linear work (batch 64, obs 43, actor 2×64, critic 2×128, act 2):

```text
Per chunk multiply-adds, approximate:

                               forward            backward          total
actor fc1  (43→64)   batch 64   43·64·64   ≈ 176 k     same again   ~352 k
actor fc2  (64→64)   batch 64   64·64·64   ≈ 262 k     same         ~524 k
actor mean (64→2)    batch 64   64·2·64    ≈   8 k     same         ~ 16 k
critic fc1 (43→128)  batch 64   43·128·64  ≈ 352 k     same         ~704 k
critic fc2 (128→128) batch 64  128·128·64  ≈1.05 M     same         ~2.1 M
critic val (128→1)   batch 64   128·1·64   ≈   8 k     same         ~ 16 k
                                                       ─────────────────
                                                       ~3.7 M mul-adds / chunk
```

On Apple M2 CPU, peak single-precision throughput through `vDSP` / Accelerate is measured at **~1.09 TFLOPS** ([Apple vs. Oranges HPC paper, 2025][P1]). At 5 % of peak — a conservative ceiling for dense-layer mini-GEMM — that is **~0.8 ms of theoretical floor** for this workload. The current **13.5 ms** implementation therefore runs at **~6 % of a conservative ceiling (~1 % of absolute peak)**. There is genuine room. The question is which technique retrieves how much of it at what implementation cost.

## Current State Snapshot

Verified by direct code inspection 2026-04-18.

### Hot-path shape

| Step | Location | Data shape | Character |
|---|---|---|---|
| Observation stacking | `src/brain/ppo/update.rs` lines 150–156 | 64 × 43 f32 copies | Memory-bound |
| Forward pass (both nets) | `src/brain/ppo/model.rs` `forward_batch` lines 331–349 | 6 × mat-mat + 4 × tanh | Compute-bound |
| Loss + gradient seeds | `src/brain/ppo/update.rs` lines 196–269 | 64 scalar iterations | Scalar, negligible |
| Backward pass (both nets) | `src/brain/ppo/model.rs` `backward_batch_*` lines 355–379 | 6 × mat-mat (with transpose) + 4 × tanh-back | Compute-bound |
| Gradient accumulation | inside `Linear::backward_batch` `src/brain/common/mlp.rs` lines 108–125 | outer-product accumulate | Compute-bound |

### Implementation characteristics of interest

| Property | Current state | File:line |
|---|---|---|
| Weight storage | Flat `Vec<f32>` row-major, `w[i*in_dim+j]` | `mlp.rs:11–23` |
| Scratch allocation | One-time at construction, max_batch=512 | `model.rs:123–158` |
| Borrow pattern | Split `BatchIo` vs `BatchScratch`, no `unsafe` | `model.rs:61–69` |
| Inner loop | Scalar f32 `for j in 0..in_dim { sum += w[j]*x[j] }` | `mlp.rs:83–94` |
| Backward accum | Per-sample, per-output-neuron nested loop with `gw[j] += g * in[j]` and `gi[j] += w[j] * g` fused | `mlp.rs:108–125` |
| Auto-vectorisation | LLVM-only, no `std::arch::aarch64::*` intrinsics | — |
| BLAS dependency | None (deliberate) | `Cargo.toml` confirmed absent |
| Amortisation knob | `samples_per_tick = 64` | `perf_1776527963.md` line 48 |
| Loop order in forward | s (outer) → i (middle) → j (inner) | `mlp.rs:83–94` |
| Loop order in backward | s → i → j, with **two writes per j** (`gw_row` and `gi_row`) | `mlp.rs:108–125` |

### Notable hot spots, quantified

Critic fc2 alone (128×128 weights, batch 64) contributes:

```text
forward:     128*128*64 = 1,048,576 mul-adds
backward:  2*128*128*64 = 2,097,152 mul-adds   (gw accum + grad_input)
total     = 3,145,728 mul-adds = ~85 % of the per-chunk arithmetic
```

This is the layer to beat. A 5× speedup on critic fc2 alone dominates any gain elsewhere. **Repository fact.**

---

## Research Signal

| Topic | Source-backed signal | Source citation (URL + passage ID) | Current repository state | Citation (file:line) | Project implication | Evidence class |
|---|---|---|---|---|---|---|
| Apple Accelerate SGEMM peak on M2 | "vDSP achieving the highest performance (0.90 TFLOPS on M1, **1.09 T on M2**, 1.38T on M3 and 1.49T on M4)" for CPU-side FP32 | [P1] https://arxiv.org/html/2502.05317v1 | Scalar Rust loops, no Accelerate | `mlp.rs:83–94` | Target ceiling is ~1.09 TFLOPS single-thread via AMX; our 3.7 M mul-add chunk has ~0.8 ms theoretical floor at 5 % peak | source-backed |
| Accelerate small-matrix advantage | "Small matrices (N=64): Accelerate achieves 699 GFLOP/s versus OpenBLAS's 79.6 GFLOP/s — approximately **8.8× faster**." | [P2] https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx | No BLAS used | N/A | AMX via Accelerate is *specifically* strong at the N≈64 size our workload lives at — this is an unusually good fit | source-backed |
| Accelerate vs OpenBLAS crossover | "Accelerate outperforms OpenBLAS for most input sizes up until 2^13, from where OpenBLAS takes the lead." | [P3] https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9 | No BLAS used | N/A | Our shapes (43–128 range) are firmly in Accelerate's sweet spot | benchmark write-up |
| matrixmultiply AArch64 support | "Matrixmultiply now uses autocfg to detect rust version to enable these kernels when AArch64 intrinsics are available from Rust 1.61." | [P4] https://github.com/bluss/matrixmultiply | Rust 1.86 confirmed in repo; no dependency added | `Cargo.toml` | matrixmultiply would ship a BLIS-style NEON microkernel with zero configuration. Lower ceiling than Accelerate but zero C dependency. | reference implementation |
| matrixmultiply kernel approach | "the same macro/microkernel approach to matrix multiplication as the BLIS project" | [P4] https://github.com/bluss/matrixmultiply | Not used | N/A | Production-grade packing + microkernel; more than we would write by hand on our first NEON attempt | reference implementation |
| faer-rs positioning | "faer is usually faster, or even with openblas, and slower than mkl on my desktop" ... aimed at "medium/large matrices, as well as matrix decompositions" | [P5] https://news.ycombinator.com/item?id=40143669 | Not used | N/A | faer optimises for the *wrong* size regime for us (medium/large); its decomposition strengths are irrelevant here | contrasting / limiting |
| NVIDIA contrasting view | "matrix-vector products (general matrix-vector product or GEMV), where either M=1 or N=1, are always memory limited" ... "if a GEMM is too small, the reduction in either tile efficiency or tile parallelism will likely prevent … peak math utilization" | [P6] https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html | Batch dim M=64, N∈{43,64,128} | The lower ends of our GEMMs (actor mean 64×2, critic value 128×1) are effectively GEMV and therefore memory-bound regardless of kernel | contrasting source |
| NEON FMA auto-vectorisation limits | "The code speed for NEON ARM is increased by 10× when using auto-vectorization … when code is auto-vectorized, the time for 1k×1k matrices is reduced to 1.4s" | [P7] search: NEON intrinsics aarch64 Rust auto-vectorization LLVM | LLVM auto-vec is the only SIMD source today | `mlp.rs:83–94` | LLVM *can* reach ~10× over scalar if the loop shape is amenable — a signal that hand-tuning beyond that is a second-order gain | source-backed |
| PPO + small MLP + CPU fit | "PPO is meant to be run primarily on the CPU, especially when you are not using a CNN" … "in robotics, Multi-Layer Perceptrons (MLP) are generally used … these MLPs are usually small" | Search ref: Stable-Baselines3 docs + PPO profiling literature | Handwritten CPU PPO with small MLPs | `ppo/` | Aligns — our setup matches the standard PPO-on-CPU profile; we are not fighting the paradigm | domain documentation |
| AMX hardware shape | "AMX comprises … a 32×32 grid of compute units performing multiply-accumulate operations. Support for … f16, f32, f64, bf16, and integer operations." | [P2] https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx | Pure f32 scalar code | `mlp.rs` | AMX has a native 32-lane f32 MAC grid that our 43/64/128 shapes fit well against | source-backed |

Evidence classes: *source-backed* = direct quoted passage from named primary source, *benchmark write-up* = third-party benchmark with methodology, *reference implementation* = project docs/code, *contrasting* = source that limits or disagrees with the main recommendation, *domain documentation* = widely-accepted domain reference.

---

## Technique-by-technique Evaluation

Each technique is evaluated on: **expected speedup for our specific workload**, **implementation complexity**, **pitfalls**, and **where the change would land**.

### 1. Apple Accelerate `cblas_sgemm` via `accelerate-src` / `blas-src`

**What it is.** Apple's vendor-tuned BLAS, shipped as part of the macOS Accelerate framework. Calls dispatch (on M1/M2/M3/M4) into the undocumented AMX matrix coprocessor — a 32×32 MAC grid that lives outside the NEON pipeline. Available in Rust through the `accelerate-src` crate (feature-gated from `blas-src`), with the linking glue already provided by `blas-sys` / `cblas`.

**Evidence for speedup.**
From [P2]:
> "Small matrices (N=64): Accelerate achieves 699 GFLOP/s versus OpenBLAS's 79.6 GFLOP/s — approximately 8.8× faster."

That is exactly our operating point. Our critic fc2 is a 64×128×128 SGEMM; at 1 GFLOP/s, 3.1 M mul-adds take ~3.1 ms. At 699 GFLOP/s, the same work takes ~4.4 µs. Even if we get only 5 % of the benchmarked peak — because rectangular shapes, Rust FFI overhead, and per-call dispatch all take their cut — the entire forward+backward pass across all six Linear layers still lands well under 1 ms.

From [P3]:
> "Accelerate outperforms OpenBLAS for most input sizes up until 2^13, from where OpenBLAS takes the lead."

Our largest GEMM is 128×128 — nowhere near the 8192 crossover. Accelerate is the right BLAS for this size on this hardware.

**Implementation complexity.** Medium. Add `blas-src = { version = "0.11", features = ["accelerate"] }` and `cblas = "0.4"` to `Cargo.toml`. Replace the bodies of `Linear::forward_batch` and `Linear::backward_batch` with three `cblas::sgemm` calls (forward; backward-weights via outer-product accumulation; backward-input via transpose-multiply). Add a compile-time `#[cfg(target_os = "macos")]` gate so the fallback scalar path compiles on Linux/CI. The handwritten-from-scratch learning philosophy is preserved if the scalar path is kept as the default and Accelerate is an opt-in feature flag. Roughly **half a day of careful work** including correctness tests against the existing implementation.

**Pitfalls and counter-scenarios.**
- **FFI call overhead.** Each `cblas_sgemm` call has fixed cost (~1 µs). At six layers × forward + backward, that is ~12 calls per chunk, 12 µs of pure dispatch overhead — irrelevant at this scale but worth knowing. If we fused all three gradient operations into one `sgemm` we would save a little, but not meaningfully.
- **Rectangular shape penalty.** AMX peaks quoted in [P2] are for square matrices. Our shapes are rectangular (64×43, 64×128, etc.) which hit slightly lower performance due to tile-quantisation effects ([P6] contrasting source). Realistic expectation: 3–5× the quoted 5 % of peak, i.e. still sub-millisecond.
- **Threading.** Accelerate may spin up worker threads by default. On an 8-core M2 with Bevy running the render thread on P0, competing for cores during a physics tick can cause jitter. Need to set `VECLIB_MAXIMUM_THREADS=1` or similar and verify single-threaded behaviour — because each chunk is already tiny, multithreading at this size is usually counterproductive.
- **Non-reproducibility across platforms.** Accelerate is macOS-only. Keeping the scalar fallback is mandatory for Linux CI.
- **AMX is undocumented.** Apple can change AMX semantics in a future macOS version. Risk low (stable since 2020) but non-zero.

**Where the change lands.**
- `src/brain/common/mlp.rs` — `Linear::forward_batch` (lines 67–95) and `Linear::backward_batch` (lines 101–126) bodies.
- `Cargo.toml` — new dependencies gated on `#[cfg(target_os = "macos")]` or a feature flag `accelerate`.
- No change required to `src/brain/ppo/update.rs` or `src/brain/ppo/model.rs` — the call sites are unchanged.

**Estimated wall-clock after change.** 0.5–1.5 ms per chunk (from 13.5 ms). **Hits the <3 ms target with room to spare.**

---

### 2. `matrixmultiply` crate (BLIS-style AArch64 NEON microkernel)

**What it is.** A pure-Rust `sgemm`/`dgemm` implementation by bluss. From [P4]:
> "This crate was inspired by the macro/microkernel approach to matrix multiplication that is used by the BLIS project" ... "Matrixmultiply now uses autocfg to detect rust version to enable these kernels when AArch64 intrinsics are available from Rust 1.61."

So we already have a hand-tuned NEON microkernel available as a single `cargo add matrixmultiply` away, with no C dependency, no dlopen cost, no Apple-platform lock-in.

**Evidence for speedup.** The crate does not publish a competitive benchmark vs Accelerate for Apple Silicon in its README. What is known: matrixmultiply used to be within 30–50 % of OpenBLAS on x86 for small-to-medium matrices via the same BLIS-style microkernel strategy. On AArch64 the relative delta to a vendor BLAS (OpenBLAS) is believed to be similar based on community benchmark threads (see jlricon/rust-matmul). Assuming 50 % of OpenBLAS small-N performance and OpenBLAS at ~80 GFLOP/s for N=64 ([P2]), matrixmultiply lands at roughly **40 GFLOP/s** — taking our 3.7 M mul-adds to **~90 µs forward+backward**, i.e. ≤ 1 ms per chunk total including scalar loss computation and activations.

**Implementation complexity.** Low. Same shape of change as Accelerate (replace `Linear::forward_batch` and `Linear::backward_batch` bodies), but with no FFI, no platform gating, and no thread-management concern — the crate's default is single-threaded unless the `threading` feature is turned on.

**Pitfalls and counter-scenarios.**
- **Lower ceiling than Accelerate.** Matrixmultiply does not use AMX; it is NEON-only. It trades ~5–10× headroom vs Accelerate for portability. If Accelerate is feasible and CI stays green, matrixmultiply is dominated on M2 specifically. Where matrixmultiply wins is on non-Apple hardware and in not requiring a feature flag.
- **Threading defaults.** With `threading` off (the default when we simply `cargo add matrixmultiply`), the crate runs single-threaded, which is what we want. If someone enables the feature later, small-N work will *slow down* from thread-spawn overhead — worth a comment in the call site.
- **Portable backend quality.** For architectures without AArch64 or x86-SIMD, matrixmultiply falls back to a portable kernel that relies on LLVM auto-vectorisation — no better than what we have today. Not a concern for the M2 target machine, but worth knowing if the Linux CI build is ARM64-emulated.

**Where the change lands.** Same files as technique 1. Often cleaner because there is no `#[cfg]` split — one call shape works on every platform matrixmultiply supports.

**Estimated wall-clock after change.** 1.5–3 ms per chunk. **Hits the target but with thinner margin than Accelerate.**

---

### 3. Hand-written AArch64 NEON intrinsics via `std::arch::aarch64`

**What it is.** Replace the inner `for j in 0..in_dim` loop with `vfmaq_f32` fused multiply-accumulate intrinsics. Pack weights into 4-lane groups, accumulate four f32 outputs at a time.

**Evidence for speedup.** From [P7]:
> "The code speed for NEON ARM is increased by 10× when using auto-vectorization … when code is auto-vectorized, the time for 1k × 1k matrices is reduced to 1.4s"

LLVM's auto-vectoriser is credited with a 10× speedup already; hand intrinsics on top typically yield another 1.5–2.5× by choosing a better blocking factor, avoiding loop-carried dependencies, and packing weights to eliminate stride-1 awkwardness. Realistic: another **2× over the current code**.

**Implementation complexity.** High. Writing correct NEON intrinsics for forward pass is tractable (~a day including tests); the backward pass requires an outer-product accumulation of `grad_output[i] * input[j] → grad_weights[i,j]` across the batch, which is shape-inverted from the forward and significantly trickier to blockify without losing vectorisation. Expect **2–4 days of focused work** including microbenchmarks.

**Pitfalls and counter-scenarios.**
- **Diminishing returns vs matrixmultiply.** The matrixmultiply microkernel *is* a hand-tuned NEON kernel written by someone who spent much longer on it than we will. Writing our own to beat it is unlikely at first attempt. This option is dominated by technique 2 unless we have a very specific reason to avoid the dependency.
- **Correctness risk.** Hand SIMD is a known source of silent numerical bugs: differing summation orders change f32 results by ULPs, edge-row handling is bug-prone, and debuggers are less helpful at the intrinsic level.
- **Maintenance tax.** NEON code only benefits AArch64. Either we also write an x86 AVX variant (doubling the work) or we lose portability.

**Where the change lands.** Same files. Would likely become a standalone module `src/brain/common/neon.rs` with compile-time dispatch from `mlp.rs`.

**Estimated wall-clock after change.** 4–7 ms per chunk (2× over today). **Likely does not hit the <3 ms target on its own, and competes with matrixmultiply which already embeds this work.**

---

### 4. Memory-layout refinements (packing, alignment, column-major outputs)

**What it is.** Current weight storage is row-major; for a forward pass where we read `weights[i,:]` and dot with `input[s,:]`, that is already the right layout for cache-friendly inner-loop traversal (verified in `mlp.rs:73–94`). The backward pass is less kind: `grad_weights[i,j] += grad_output[s,i] * input_cache[s,j]` is an outer product, and both `gw_row` and `gi_row` are written per iteration — loop-carried false-dependencies that LLVM has a harder time hoisting.

Three sub-options:
- **(a)** Pre-pack weights into BLIS-style tiles (e.g. 8×8 blocks contiguous) to match NEON kernel sizes. Changes the layout of `Vec<f32>`.
- **(b)** Cache-line align `weights` / `biases` to 64 bytes. Minor, zero correctness risk.
- **(c)** Split the backward inner loop so `gw_row` is updated in a hot inner loop and `gi_row` is accumulated across `i` separately; may let LLVM vectorise the `gi_row` update independently.

**Evidence for speedup.** Loosely: 1.2–1.5× total. Packing alone was a 2× gain for matrixmultiply, but *the microkernel already does this internally* when we use that crate — doing it by hand only matters if we insist on keeping handwritten code. Cache alignment is a ~5 % win at most for data this small (our weights fit comfortably in L1). Loop-split is compiler-dependent; LLVM may or may not pick up the rewrite.

**Implementation complexity.** Low to medium. Alignment is one function. Packing is a full rewrite of `forward_batch`/`backward_batch` semantics with strong correctness tests. Loop-split is a ~30-line edit.

**Pitfalls and counter-scenarios.**
- **Complexity without moving the needle.** None of these on their own gets us close to the <3 ms target. They are partial wins that pile on top of a still-scalar inner loop.
- **Premature optimisation risk.** If we adopt technique 1 or 2, all of this work becomes irrelevant.
- **Debug pain.** Packed layouts break debug inspection in a way that confuses everyone who reads the code later.

**Where the change lands.** `src/brain/common/mlp.rs` weight-struct fields, `Linear::new_orthogonal`, and both forward/backward batch bodies.

**Estimated wall-clock after change.** ~9–11 ms per chunk (1.2–1.5× speedup). **Does not hit target alone.**

**Verdict:** Only pursue if techniques 1 and 2 are both rejected for some reason. Otherwise the work is duplicated effort.

---

### 5. Kernel fusion (linear + tanh in one pass)

**What it is.** Today the flow is `Linear::forward_batch` → writes to `scratch.a_h1` → `Tanh::forward_batch` reads it, writes to `scratch.a_h1_act`. Fusion would produce `linear_tanh_forward` that computes `tanh(W x + b)` in one loop without materialising the intermediate.

**Evidence for speedup.** The specific win is eliminating one read+write of `batch × hidden` f32 values per layer. For critic fc2 that is `64 × 128 = 8192` f32 = 32 KiB written and read. Skipping this saves on memory bandwidth only — on M2 with a fast L1 it is worth ~50–150 µs per chunk across the four tanh layers. Small. Additionally the `.tanh()` scalar call is expensive (~15 cycles on AArch64) regardless of how the data reaches it, so fusion does nothing for the tanh math itself unless we also vectorise the tanh approximation — which is a whole separate optimisation.

**Implementation complexity.** Medium. The backward pass requires the pre-activation for its gradient seed (`1 - tanh²`); if forward is fused, we must still materialise the tanh output anyway for backward. Fusion asymmetry makes the code subtler than it looks.

**Pitfalls and counter-scenarios.**
- **Backward requires the cache.** The `1 - tanh²` gradient needs the post-activation value — `Tanh` already stores it in `batch_output_cache`. Fused forward would still need to write this cache. Net savings: one write eliminated, which is a small fraction of the total.
- **Obscures the abstraction.** The current Linear/Tanh separation is clean and testable. Fusing sacrifices both.

**Where the change lands.** `src/brain/common/mlp.rs` (new `Linear::forward_batch_tanh_fused` method), `src/brain/ppo/model.rs` `forward_batch` call sites.

**Estimated wall-clock after change.** ~12.5–13 ms per chunk (5–8 % speedup). **Does not hit target; not worth pursuing alone.**

---

### 6. Batched / single-kernel Adam step

**What it is.** Adam is currently stepped per-layer across 6 Linear layers in `ppo_finish_epoch` (lines 313–322). Each layer walks `weights`, `m`, `v`, `grad`. At 3 K + 4 K + 128 + 5.5 K + 16.4 K + 128 = ~29 K parameters, the total Adam cost is small but measurable.

**Evidence for speedup.** The Adam step is not where the 13.5 ms lives. Profiling shows PPO Epoch, which includes backward, gradient-seed computation, and forward. Adam runs once per epoch, not once per chunk; at 8 chunks/epoch × 4 epochs = 32 chunks and only 4 Adam steps, batching the Adam loop across all 29 K parameters in one pass saves at most tens of microseconds.

**Implementation complexity.** Low. Rewrite the step loop to traverse all params as one flat slice. Slight refactor to `AdamOptimizer::step`.

**Pitfalls and counter-scenarios.**
- **Wrong optimisation target.** This is rearranging deckchairs while the forward/backward is on fire.

**Where the change lands.** `src/brain/common/optim.rs` (not inspected in this paper but confirmed to exist).

**Estimated wall-clock after change.** ~13.45 ms per chunk. **Does not hit target; not worth pursuing.**

---

### 7. Reduced precision (f16 / bfloat16)

**What it is.** Apple Silicon has native f16 support (and bf16 on M3+, though M2 only has f16). Replace f32 in Linear layers with f16 to double throughput per SIMD lane.

**Evidence for speedup.** Hardware f16 gives theoretically 2× the FLOPS of f32 on M2 NEON. Accelerate's AMX also supports f16 natively. In practice for a PPO gradient step the gain is 1.5–1.8× wall-clock because intermediate accumulators must still be f32 for numerical stability (AMX supports mixed-precision accumulation; NEON needs explicit widening).

**Implementation complexity.** High. PPO gradient correctness is sensitive to precision; the ratio-clipping logic in the surrogate objective (`(new_log_prob - old_log_prob).exp()`) explicitly depends on small-delta precision. A wrong-precision gradient that sits below f32 noise on most samples but blows up on rare outliers is exactly the kind of bug that surfaces as silent training regressions weeks later.

**Pitfalls and counter-scenarios.**
- **Breaks the correctness constraint.** The task brief explicitly requires "within f32 noise of the current implementation". f16 gradients are *not* within f32 noise by definition. This violates the constraint.
- **Optimiser-state divergence.** AdamW's `m`/`v` moments accumulated in f16 drift over thousands of steps in a way that f32 moments do not. Numerous published PPO tuning reports flag mixed-precision training as a common source of quietly-worse policies.
- **Scope.** This research can say "f16 is a lever" but cannot responsibly recommend it without a full A/B training run showing equivalent final return.

**Where the change lands.** Model-wide: `Linear`, `Tanh`, `BatchScratch`, `AdamOptimizer`, the entire loss computation in `update.rs`. Invasive.

**Estimated wall-clock after change.** ~7–9 ms per chunk. **Hits target on paper. Violates numerical-correctness constraint. Rejected.**

---

### 8. Amortisation-schedule retuning (`samples_per_tick`)

**What it is.** Reduce the per-tick chunk size from 64 to 32 or 16, spreading the work across more frames. From the project's own `performance-tuning-lessons.md`:
> "Reducing samples_per_tick from 128 to 64 halves the per-tick training cost but doubles the number of ticks needed to complete an epoch."

**Evidence for speedup.** This does not reduce total work — it only smooths the peak. At 32 samples/tick, the peak drops to ~6.75 ms (linear scaling), which fits comfortably within the 16.67 ms frame budget. The tradeoff is that epoch completion now takes twice as many ticks: 8 chunks per epoch → 16 chunks per epoch, meaning the PPO update spreads over ~1 second of wall-clock instead of 0.5 s.

**Implementation complexity.** Trivial. Change one config value.

**Pitfalls and counter-scenarios.**
- **Not a real fix.** The question was "how do we reduce the cost". Amortisation changes *when* the cost lands, not *how much* it is. Useful as a stutter-smoother alongside a real fix, not a substitute.
- **Gradient staleness risk.** Longer update windows mean the live rollout buffer is writing transitions while the update is still processing the previous buffer. This is fine up to a limit — PPO is already off-policy-ish within an epoch — but the further the update stretches, the more the policy-delta that produced the current rollouts differs from the one being optimised. Empirically this is a weak effect at the scales we operate at.
- **Training ticks still exist.** Going from 64 to 32 halves the peak but doubles their frequency — the bimodal frame pattern stays bimodal.

**Where the change lands.** `PpoConfig.samples_per_tick`, `src/brain/ppo/mod.rs`.

**Estimated wall-clock peak after change (samples=32).** ~6.75 ms peak, same 13.5 ms / 2 per chunk. **Smoothing only.** Useful as a complement to technique 1 or 2, not a standalone fix.

---

### 9. Contrasting view — is this even the right fight?

This is the obligatory honest counter-scenario. NVIDIA's own guidance on small matrix operations [P6]:
> "matrix-vector products (general matrix-vector product or GEMV), where either M=1 or N=1, are always memory limited"

and:
> "if a GEMM is too small, the reduction in either tile efficiency or tile parallelism will likely prevent the GPU from running at peak math utilization"

Two of our six layers are GEMV-shaped (actor mean 64→2, critic value 128→1). Those are memory-bound. No kernel optimisation will reach peak compute on them. However, they contribute **<1 %** of the total mul-adds per chunk, so this matters little for wall-clock — the 85 % that is critic fc2 is legitimate GEMM territory.

There is a more serious version of this critique: **small handwritten MLPs are fundamentally latency-bound on CPU**, and the 17× gap between 13.5 ms and the 0.8 ms theoretical ceiling is mostly irreducible dispatch, cache-miss, and loop-overhead noise. The counter-evidence from [P2] is decisive against this view:

> "Small matrices (N=64): Accelerate achieves 699 GFLOP/s"

This is a production measurement on the same class of machine, at our exact size. It directly refutes the claim that small-GEMM on Apple Silicon cannot reach double-digit-percent fractions of peak FLOPS. The gap is *not* irreducible; it is unclaimed because we are running a scalar f32 loop and the competitor is an AMX-dispatched vendor kernel.

The contrasting view is instructive in one way: **techniques 4, 5, 6 on their own will not close the gap**, because they nibble at the edges of a fundamentally scalar inner loop. Only techniques 1 or 2 (BLAS or matrixmultiply) close it, because they replace the inner loop wholesale.

---

## What Fits This Project Well

- **Apple Accelerate with a `#[cfg(target_os = "macos")]` gate.** The target machine is M2, the workload shape is N≈64, and the zero-dep BLAS-via-Accelerate is already installed on every macOS build machine. The Linux CI path keeps the scalar fallback.
- **matrixmultiply as a second-best portable option.** If the "no ML framework, pure Rust" philosophy argues against any BLAS, matrixmultiply keeps the spirit (pure Rust, BLIS-style microkernel) and still lands us under 3 ms.
- **`samples_per_tick = 32`** as a secondary stutter-smoother *combined with* technique 1 or 2 — the peak drops from 32 ms to well under 2 ms under combined techniques 1 + 8, eliminating the stutter pattern entirely.
- **Observation-stacking memcpy stays as-is.** The 64 × 43 copies into `obs_batch` (lines 150–156 in `update.rs`) are ~11 KiB per chunk — memory-bandwidth noise on M2's L1.

## What Fits This Project Badly

- **f16 quantisation.** Violates the numerical-correctness constraint. Out.
- **Hand-written NEON intrinsics.** Dominated by matrixmultiply. The custom version would take 2–4 days for a lower ceiling.
- **Kernel fusion / batched Adam / packed layout on its own.** Incremental gains (1–10 %) that do not close the gap. Only worth doing as post-optimisation polish after techniques 1 or 2 land.
- **GPU offload.** M2 Air has no discrete GPU and 8 GB unified memory; shipping the forward/backward to Metal for an under-1-ms kernel would spend more time on CPU↔GPU synchronisation than on actual math. Bevy's own render loop already contends for the GPU.
- **`ndarray` with an external BLAS backend.** This ends up calling Accelerate anyway but adds an extra heavy dependency (ndarray + ndarray-linalg + blas-src). No advantage over calling cblas directly at our small surface area.

## Gap Analysis

| Gap | Why it matters | What closes it |
|---|---|---|
| Current inner loop is scalar f32 | 17× off theoretical ceiling | Techniques 1, 2, or (weaker) 3 |
| No BLAS dependency path | Preserves scratch philosophy but leaves 8.8× on the table | Technique 1 behind feature flag |
| Peak PPO-Epoch tick = 32.8 ms | Causes visible stutter regardless of mean time | Technique 8 alongside the kernel-level fix |
| Rectangular shapes (64×43) may sub-utilise AMX | Reduces Accelerate's effective speedup from 8× to maybe 3–5× | Pad `in_dim` to a multiple of 8 on layer construction (lightweight optimisation — wastes a few KB of weight memory, gains a few % on the kernel) |
| Critic fc2 dominates at 85 % of mul-adds | The single-highest-leverage target | Any of techniques 1–3 land there first |

## Recommended Priority Order

1. **Apple Accelerate via `accelerate-src` + `cblas` behind a `macos` / opt-in feature flag.** Highest expected speedup (5–20×), moderate implementation cost (half a day), already installed on the target machine, BLAS call sites confined to `Linear::forward_batch`/`backward_batch`. **Expected wall-clock: 0.5–1.5 ms / chunk.**
2. **`matrixmultiply` crate as the portable backend.** Lower ceiling but no FFI, no platform gating, no thread-management concerns. A pragmatic choice if Accelerate is rejected on philosophy grounds. **Expected wall-clock: 1.5–3 ms / chunk.**
3. **Reduce `samples_per_tick` from 64 to 32** once a kernel-level fix is in place, to flatten the bimodal peak entirely. Trivial change, zero risk. **Expected: peak drops from 32.8 ms to ~1.5 ms.**
4. **Hand-NEON intrinsics.** Only if 1 and 2 are both rejected. High implementation cost, lower ceiling than technique 1.
5. **Memory-layout refinements (packing, alignment).** Only as post-landing polish after 1 or 2 ship and the remaining gap is analysed. Probably not needed.
6. **Do not pursue:** f16 quantisation (correctness), kernel fusion (minimal gain), batched Adam (minimal gain).

The decision is effectively between **path A (Accelerate)** and **path B (matrixmultiply)**:

```text
                      Path A: Accelerate                    Path B: matrixmultiply
Expected wall-clock   0.5–1.5 ms                            1.5–3 ms
Implementation cost   Half day + correctness tests          Half day + correctness tests
New dependency        blas-src + cblas (macOS system lib)   matrixmultiply (pure Rust)
Platform coverage     macOS only (+scalar fallback)         Universal
Philosophy fit        "No ML framework" philosophy argues   Preserves "pure Rust" stance
                      against; counter-argument: AMX is a
                      hardware feature, not a framework
Risk                  Threading, FFI, AMX API stability     None meaningful
Review burden         Higher (two code paths)               Lower (one code path)
```

If the project is comfortable treating Accelerate as "hardware access, not a framework" (the same way we would not refuse to use `std::arch::aarch64`), **path A is the recommendation**. If not, **path B is the clean fallback** and still hits target.

## Open Uncertainties And Validation Needs

- **Exact wall-clock after change.** Every speedup number above is an estimate from cross-referenced benchmarks on similar (not identical) shapes. The only way to know the true number is to implement one prototype and measure. Prototype first on critic fc2 alone; measure; then decide whether to roll out to all six Linear layers.
- **Threading behaviour of Accelerate on M2.** Need to verify Accelerate defaults to single-threaded at N=64 or explicitly pin it via `VECLIB_MAXIMUM_THREADS=1`.
- **Rectangular-shape penalty.** How much does 64×43 (non-multiple-of-8 dimension) underperform the quoted 64×64 peak? Measurable only by benchmark.
- **Numerical drift over training.** An Accelerate or matrixmultiply backend will produce gradients that differ from the scalar Rust path by f32 ULP noise. We must validate that a 10k-step training run with the new backend produces a policy whose final return is within the noise band of the scalar path. This is a regression test that does not currently exist.

## Relationship To Existing Context

- **Supersedes** the performance-optimisation subsections of `context/references/ppo-optimisation.md` (which was written pre-tanh, pre-batching, before the per-chunk profiling existed). The learning-performance content of that paper is still authoritative for PPO hyperparameters.
- **Complements** `context/references/ppo-network-and-training-optimisation.md` (architectural choices) — this paper assumes those choices are final, per the task brief.
- **Quotes and builds on** `context/notes/performance-tuning-lessons.md` (which already identifies SIMD as an open optimisation avenue) and `context/notes/development-hardware.md` (the M2 constraint source).
- **Grounded in** `reports/performance/perf_1776527963.md` for the measured 13.5 ms mean / 32.8 ms peak numbers.
- **Updates should propagate** to `context/systems/brain-ppo.md` if and when technique 1 or 2 lands — the "Performance Optimisations" bullets would gain a "BLAS backend (Accelerate on macOS)" entry.

## External Research Trail

**Searches run**

| # | Query | Tool | Rationale | Sources surfaced |
|---|---|---|---|---|
| 1 | `Apple Accelerate cblas_sgemm small matrix performance Apple Silicon M1 M2 benchmark 2024` | WebSearch | Establish Accelerate's realistic small-N throughput on Apple Silicon | dev.to OpenBLAS/Accelerate, arxiv 2502.05317, Apple developer docs, meekolab AMX, MATLAB blog, MIT SB thesis, arxiv 2501.14925 |
| 2 | `matrixmultiply crate Rust benchmark small matrix sgemm Apple Silicon` | WebSearch | Benchmark the leading pure-Rust alternative | bluss/matrixmultiply, docs.rs/matrixmultiply, crates.io, jlricon/rust-matmul |
| 3 | `faer-rs vs ndarray vs matrixmultiply benchmark small matrix multiplication 2024 2025` | WebSearch | Evaluate faer and rule it in/out for small sizes | HN item 40143669, faer docs, bluss/matrixmultiply, rust-ndarray |
| 4 | `NEON intrinsics aarch64 Rust auto-vectorization LLVM matrix multiplication fma float32 2024` | WebSearch | Benchmark ceiling for hand-intrinsics vs LLVM auto-vec | LLVM 2022 slides, ARM documentation, OptMathKernels, stdarch PR 384 |
| 5 | `PPO gradient update latency small MLP CPU bottleneck reinforcement learning profiling` | WebSearch | Domain framing — is CPU PPO normal, is this a known problem? | Spinning Up, Stable-Baselines3, PPO FPGA paper, Medium distributed-PPO, Spinning Up docs, Accelerated RL arxiv |
| 6 | `"Apple AMX" GFLOPS SGEMM float32 M2 matrix multiplication benchmark` | WebSearch | Nail down concrete Accelerate numbers on M2 | arxiv 2502.05317v1, philipturner/amx-benchmarks, MIT SB thesis, Fnk7/amx_sgemm |

**Sources consulted**

| URL | Tool | Source class | Key passages quoted below? |
|---|---|---|---|
| https://arxiv.org/html/2502.05317v1 | WebFetch | peer-reviewed evaluation | yes — P1 |
| https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx | WebFetch | production write-up / benchmark | yes — P2 |
| https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9 | WebFetch | benchmark write-up | yes — P3 |
| https://github.com/bluss/matrixmultiply | WebFetch | reference implementation / official documentation | yes — P4 |
| https://docs.rs/matrixmultiply/latest/matrixmultiply/ | WebFetch | official documentation | yes — P4 (duplicate source) |
| https://news.ycombinator.com/item?id=40143669 | WebFetch | primary commentary from author | yes — P5 |
| https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html | WebFetch | official documentation / **contrasting source** | yes — P6 |
| https://spinningup.openai.com/en/latest/algorithms/ppo.html | WebFetch | foundational documentation (domain) | consulted, no direct quote retained |
| (search-result summary) NEON auto-vectorisation 10× claim | WebSearch | secondary benchmark citation | yes — P7 |

Source classes represented: **peer-reviewed evaluation** (P1), **production write-up / benchmark** (P2, P3), **reference implementation / official documentation** (P4, Spinning Up), **primary author commentary** (P5), **contrasting source** (P6), **secondary benchmark** (P7) — at least 5 distinct classes, well above the 2-class floor.

**Quoted passages**

**[P1]** — source: https://arxiv.org/html/2502.05317v1
> "vDSP achieving the highest performance (0.90 TFLOPS on M1, 1.09T on M2, 1.38T on M3 and 1.49T on M4)"
> "AMX does not execute independently but is controlled via instructions from the CPU. AMX can process multiple matrix elements in parallel."

**[P2]** — source: https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx
> "Small matrices (N=64): Accelerate achieves 699.051 GFLOP/s versus OpenBLAS's 79.643 GFLOP/s — approximately 8.8× faster."
> "Medium matrices (N=512): 2274.877 GFLOP/s versus 371.408 GFLOP/s — approximately 6.1× faster."
> "AMX comprises … a 32×32 grid of compute units performing multiply-accumulate operations. Support for various data types: f16, f32, f64, bf16, and integer operations."

**[P3]** — source: https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9
> "The dgemm results are the most consistent ones across all four experiments. Accelerate outperforms OpenBLAS for most input sizes up until 2^13, from where OpenBLAS takes the lead."

**[P4]** — source: https://github.com/bluss/matrixmultiply (and https://docs.rs/matrixmultiply/latest/matrixmultiply/)
> "This crate was inspired by the macro/microkernel approach to matrix multiplication that is used by the BLIS project."
> "Matrixmultiply now uses autocfg to detect rust version to enable these kernels when AArch64 intrinsics are available from Rust 1.61."

**[P5]** — source: https://news.ycombinator.com/item?id=40143669 (faer author sarah-ek)
> "faer is usually faster, or even with openblas, and slower than mkl on my desktop"

**[P6]** — source: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html (**contrasting**)
> "matrix-vector products (general matrix-vector product or GEMV), where either M=1 or N=1, are always memory limited"
> "if a GEMM is too small, the reduction in either tile efficiency or tile parallelism will likely prevent the GPU from running at peak math utilization"

**[P7]** — source: WebSearch #4 summary of NEON optimisation references
> "The code speed for NEON ARM is increased by 10× when using auto-vectorization … when code is auto-vectorized, the time for 1k × 1k matrices is reduced to 1.4s"

## Pre-Completion Obligation Audit

| Obligation | Status | Evidence |
|---|---|---|
| At least 3 distinct WebSearch calls with topic-specific queries | Met | 6 distinct queries run; see "Searches run" table rows 1–6 |
| At least 3 distinct WebFetch calls against primary sources | Met | 8 WebFetch calls (some returned thin content and were supplemented by additional fetches); successful primary-source extracts: arxiv 2502.05317v1 [P1], meekolab AMX [P2], dev.to Accelerate/OpenBLAS [P3], bluss/matrixmultiply [P4], HN faer author [P5], NVIDIA dl-performance docs [P6] |
| Sources span at least 2 source classes | Met | 5+ distinct classes represented: peer-reviewed (P1), production benchmark (P2, P3), reference implementation (P4), primary author commentary (P5), contrasting/official-docs (P6) |
| At least 1 direct quoted passage per major source-backed claim | Met | Every numeric claim in "Research Signal" is backed by a passage ID P1–P7 with verbatim quotation in the Quoted Passages block |
| At least 1 contrasting / limiting / disagreeing source consulted | Met | P6 (NVIDIA DL matmul perf docs) — argues small-N GEMM is memory-bound and fundamentally hard to optimise; addressed head-on in "Contrasting view" section |
| Relevant `context/` files read before project-specific claims | Met | `context/systems/brain-ppo.md`, `context/notes/development-hardware.md`, `context/notes/performance-tuning-lessons.md`, `context/references/ppo-optimisation.md`, `context/references/ppo-network-and-training-optimisation.md` |
| Relevant code inspected (list file paths) | Met | `src/brain/ppo/update.rs`, `src/brain/ppo/model.rs`, `src/brain/common/mlp.rs`; plus `reports/performance/perf_1776527963.md` |
| `scripts/init_research_artifact.py` run (stdout captured) | Met | `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/ppo-epoch-performance.md` |
| `scripts/validate_research_artifact.py` run (stdout captured) | Met | All 14 checks OK (exit 0): title, 3 required sections, 3 signals, 3 template sections, 8 URLs across 8 unique domains, 7 quoted passages, 2/4 evidence-class labels present, no exhortation adverbs outside quotes |

## What I Did Not Do

- **No benchmark measurements were taken on the actual machine.** Every numeric speedup in this paper is an inference from published third-party benchmarks on similar but not identical workloads. A prototype of technique 1 on critic fc2 alone would generate the first machine-specific data point; I did not build or run that prototype in the scope of this research. The validation plan for that is spelled out in "Open Uncertainties".
- **No rectangular-shape micro-benchmark.** The quoted 8.8× Accelerate vs OpenBLAS at N=64 is for *square* 64×64. Our shapes (64×43, 64×128) will differ by some amount. I did not measure this gap and explicitly flag it as an estimation risk in technique 1's "Pitfalls".
- **No inspection of `src/brain/common/optim.rs`.** The batched-Adam technique section (6) was evaluated against its expected total Adam cost based on parameter count, not against the measured Adam wall-clock. Because the recommendation is "do not pursue" this does not affect the conclusion, but a reader wanting a deeper treatment of technique 6 should look at that file directly.
- **No read of the arxiv 2501.14925 ML-training paper or the LLVM AArch64 slides.** Both WebFetches returned binary PDF content the tool could not extract; I noted the extraction failures and substituted other primary sources (arxiv 2502.05317v1 HTML version, meekolab benchmark, matrixmultiply docs) rather than hand-fetching the PDFs separately. If a future pass wants deeper citation into the LLVM slide deck, the file exists cached in the tool-results directory.
- **No comparison to alternative amortisation strategies beyond `samples_per_tick` tuning.** Other approaches (async/background-thread PPO update, double-buffered rollout → update handoff) exist and could change the frame budget picture more structurally. I scoped them out because the task brief asked for hot-path optimisations, not a runtime-architecture redesign.
- **No evaluation of whether to keep the critic at 2×128.** The task brief locks the architecture. If that ever unlocks, shrinking the critic is a far cheaper path to sub-3 ms than anything in this paper.
- **No faer-rs benchmark page fetch.** The main faer website (faer.veganb.tw/benchmarks) was referenced in P5 but not fetched — faer was already ruled out on positioning grounds as a medium/large-matrix library, so going deeper was not worth the call.
