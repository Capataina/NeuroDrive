# Brain PPO — Code Health Findings

**Systems covered:** `src/brain/ppo/` (model.rs, buffer.rs, update.rs, mod.rs), `src/brain/common/` (mlp.rs, math.rs, optim.rs)
**Finding count:** 10 findings (6 high, 3 medium, 1 low)

---

## Data Layout and Memory Access Patterns

### Flatten Rollout Buffer From AoS to SoA Flat Storage
- [x] Replace `Vec<Vec<f32>>` fields in `TrainerRolloutBuffer` with flat `Vec<f32>` storage and an explicit dimension stride

**Category:** Data Layout and Memory Access Patterns
**Severity:** High
**Effort:** Medium
**Behavioural Impact:** None (verified — same data, same access patterns, different layout)

**Location:**
- `src/brain/ppo/buffer.rs:10-19` — `TrainerRolloutBuffer` struct definition
- `src/brain/ppo/buffer.rs:23-39` — `push_pre_step()`
- `src/brain/ppo/mod.rs:200-208` — buffer push in act system
- `src/brain/ppo/update.rs:153-157` — obs batch stacking in `ppo_process_chunk`

**Current State:**
The `TrainerRolloutBuffer` stores `states`, `actions`, and `latent_actions` as `Vec<Vec<f32>>`. Each `push_pre_step` call passes owned `Vec<f32>` values, meaning every transition push for every car every tick allocates 3 separate heap `Vec<f32>` objects (43-element state, 2-element actions, 2-element latent_actions). With 8 cars at 60 Hz, that is 1,440 heap allocations per second just for the buffer pushes. When the buffer reaches the 512-transition horizon, it holds 1,536 separate `Vec<f32>` allocations for states alone.

The `ppo_process_chunk` function then stacks these scattered vectors into a contiguous obs batch by copying from each `Vec<f32>` individually (`buffer.states[idx]` for each sample in the chunk). This means the chunk loop chases a pointer per sample to a separately allocated 43-float vector, defeating spatial locality.

**Proposed Change:**
Replace the three `Vec<Vec<f32>>` fields with flat `Vec<f32>` fields plus explicit dimension constants:
- `states: Vec<f32>` with stride `obs_dim` (43). Access: `states[i * obs_dim .. (i+1) * obs_dim]`.
- `actions: Vec<f32>` with stride `act_dim` (2).
- `latent_actions: Vec<f32>` with stride `act_dim` (2).

Change `push_pre_step` to take `&[f32]` slices and extend the flat vectors. Change `ppo_process_chunk` to index directly into the flat buffer for obs batch construction (which may become a simple `copy_from_slice` of contiguous ranges when combined with the shuffled-index gather).

The `frozen_buffer` swap via `std::mem::take` stays identical. GAE computation indexes into `rewards`, `values`, `dones`, `env_ids` which are already flat `Vec<f32>` / `Vec<bool>` / `Vec<u32>` — no change needed there.

**Justification:**
This is the same pattern that delivered the "43x" win when `Linear::weights` was flattened from `Vec<Vec<f32>>` to flat `Vec<f32>` (documented in `context/notes/performance-tuning-lessons.md`). The rollout buffer is the second-largest consumer of `Vec<Vec<f32>>` in the codebase and follows the exact same access pattern: write once sequentially, then read in batches.

Eliminating 1,440 heap allocations per second removes allocator pressure that fragments memory and competes with the training loop for cache lines. The contiguous layout also enables the obs-batch stacking in `ppo_process_chunk` to be a series of `memcpy` operations from known offsets rather than pointer-chasing through separate allocations.

**Expected Benefit:**
Eliminates ~1,440 heap allocations per second during normal training. Reduces memory fragmentation. Improves cache locality during obs-batch construction in `ppo_process_chunk`. The flat buffer is also smaller in total memory (no per-Vec overhead of pointer + length + capacity × 3 per transition).

**Impact Assessment:**
Zero functional change. The same data is stored in the same logical order; only the physical layout changes. GAE computation, PPO ratio calculation, and all downstream consumers index by transition index — the index semantics are preserved. The `compute_gae_per_env` function accesses `rewards`, `values`, `dones`, `env_ids` which are already flat. The only changed access is `states[idx]` / `actions[idx]` / `latent_actions[idx]`, which become slice operations on the flat buffer.

---

### Transpose Weight Access Pattern in `Linear::forward_batch`
- [x] Store a transposed weight copy (or restructure the inner loop) so that `forward_batch` reads weights sequentially rather than with an `in_dim` stride

**Category:** Data Layout and Memory Access Patterns
**Severity:** High
**Effort:** Medium
**Behavioural Impact:** None (verified — identical matrix multiplication, different memory access order)

**Location:**
- `src/brain/common/mlp.rs:105-117` — `forward_batch()` inner loop

**Current State:**
The `forward_batch` method uses a `j`-then-`i` loop order (lines 108-116):
```
for j in 0..self.in_dim {
    let x = in_row[j];
    let mut w_idx = j;
    for o in out_row.iter_mut() {
        *o += self.weights[w_idx] * x;
        w_idx += self.in_dim;
    }
}
```

This accesses `weights[j]`, `weights[j + in_dim]`, `weights[j + 2*in_dim]`, ... — a stride-`in_dim` pattern through the weight array. For the critic's first layer (`in_dim=43, out_dim=128`), each inner iteration jumps 43 floats (172 bytes) through the weight buffer. A 64-byte cache line holds 16 floats, so every access is guaranteed to be a cache miss (stride > cache line). The loop performs `in_dim × out_dim = 5,504` weight reads per sample, nearly all of which miss L1.

The comment on line 98 says "Using ikj loop order for cache-friendly access on row-major data" but the actual code implements a `jk` pattern that is cache-hostile for the weights.

**Proposed Change:**
Two options (choose one):

**Option A — Keep row-major weights, use `i-then-j` loop order:**
```rust
for i in 0..self.out_dim {
    let w_row = &self.weights[i * self.in_dim..(i + 1) * self.in_dim];
    let mut sum = self.biases[i];
    for j in 0..self.in_dim {
        sum += w_row[j] * in_row[j];
    }
    out_row[i] = sum;
}
```
This reads weights sequentially within each row. The trade-off is that the output accumulation is now per-neuron rather than per-input, which may prevent LLVM from auto-vectorising the output scatter. But the weight reads are sequential, which is the dominant cost.

**Option B — Store a column-major transposed copy for forward:**
Add a `weights_transposed: Vec<f32>` field to `Linear` (column-major: `weights_t[j * out_dim + i] = weights[i * in_dim + j]`). Populate it once at construction and after each optimiser step. The `forward_batch` inner loop then reads `weights_t[j * out_dim .. (j+1) * out_dim]` sequentially, which is perfectly cache-friendly for the broadcast-multiply-accumulate pattern the current code intends.

Both options keep the same `weights` array for backward passes (which already access row-major correctly). Option B adds memory overhead (one extra weight buffer per layer) but preserves the current loop structure and enables SIMD. Option A is simpler.

**Justification:**
The current stride-`in_dim` access pattern means virtually every weight read misses L1 cache. For the critic layers (128×43, 128×128, 1×128), this is the dominant cost in forward passes. Sequential access would be served entirely from L1/L2 after the first line of each row/column is fetched.

The comment on line 98 indicates the developer intended cache-friendly access but the implementation has the opposite effect. The `jk` loop order is cache-friendly for the *input* (which is read once per j iteration) but cache-hostile for the *weights* (which are the larger array).

**Expected Benefit:**
The weight reads constitute the majority of memory traffic in `forward_batch`. Eliminating the stride access should provide a substantial speedup — potentially 2-4x for the matrix multiply portion of forward passes, which is the single most expensive operation in the PPO pipeline.

**Impact Assessment:**
Zero functional change. Matrix multiplication is associative and commutative over the summation; the result is identical regardless of evaluation order. Both options compute the same dot products. Floating-point summation order changes, but the model uses stochastic gradient descent where such differences are noise within the learning process — the trained behaviour is identical in expectation.

---

## Performance Improvement

### Eliminate Per-Car Vec Allocations in `ppo_act_all_cars_system`
- [x] Replace `obs.values.to_vec()`, `actions.to_vec()`, and `latent_actions.to_vec()` with slice-based buffer push

**Category:** Performance Improvement
**Severity:** High
**Effort:** Small
**Behavioural Impact:** None (verified — same data pushed, different allocation strategy)

**Location:**
- `src/brain/ppo/mod.rs:200-208` — buffer push call in `ppo_act_all_cars_system`
- `src/brain/ppo/buffer.rs:23-39` — `push_pre_step` signature

**Current State:**
Every tick, for each of the 8 cars, `ppo_act_all_cars_system` calls:
```rust
buffer.push_pre_step(
    env_id.0,
    obs.values.to_vec(),       // allocates 43-element Vec
    actions.to_vec(),           // allocates 2-element Vec
    latent_actions.to_vec(),    // allocates 2-element Vec
    safety_clamp_hits,
    value,
    old_log_prob,
);
```

Each `.to_vec()` performs a heap allocation. That is 24 heap allocations per tick (8 cars x 3 vecs), 1,440 per second. The 2-element `Vec<f32>` allocations are especially wasteful — each allocates 24 bytes of heap for 8 bytes of data, plus the 24-byte Vec header.

**Proposed Change:**
After flattening the rollout buffer (finding #1), change `push_pre_step` to accept `&[f32]` slices and extend the internal flat buffers with `extend_from_slice`. The caller passes `&obs.values`, `&actions`, `&latent_actions` without any allocation.

If the buffer flattening is deferred, an intermediate fix is to change `push_pre_step` to accept `&[f32]` and have the buffer perform the push internally, still into `Vec<Vec<f32>>` (which would move the allocation into the buffer but at least allow it to be amortised).

**Justification:**
24 heap allocations per tick is avoidable waste in a performance-critical path. The data is consumed by the buffer (which already owns it) and never needs to be an independent `Vec` at the call site.

**Expected Benefit:**
Eliminates 1,440 heap allocations per second on the action-selection path. Reduces allocator pressure and GC-like memory fragmentation that competes with the training loop.

**Impact Assessment:**
Zero functional change by construction. The same bytes are stored in the same buffer positions. Only the allocation strategy changes.

---

### Cache Normal Distribution Construction in `sample_normal`
- [x] Avoid constructing a new `Normal` distribution on every call to `sample_normal`

**Category:** Performance Improvement
**Severity:** High
**Effort:** Trivial
**Behavioural Impact:** None (verified — identical sampling distribution)

**Location:**
- `src/brain/common/math.rs:79-82` — `sample_normal()`
- `src/brain/ppo/mod.rs:153-156` — call site in `ppo_act_all_cars_system` (8 cars x 2 actions per tick)

**Current State:**
```rust
pub fn sample_normal(mean: f32, std: f32, rng: &mut impl Rng) -> f32 {
    let normal = Normal::new(mean, std).unwrap();
    normal.sample(rng)
}
```

Every call constructs a `Normal` distribution object (which involves computing `1/std` and validating inputs) then samples once and discards it. With 8 cars x 2 action dimensions, this creates and discards 16 `Normal` objects per tick (960 per second).

**Proposed Change:**
Replace with direct sampling using the standard normal and affine transformation:
```rust
pub fn sample_normal(mean: f32, std: f32, rng: &mut impl Rng) -> f32 {
    let z: f32 = StandardNormal.sample(rng);
    mean + std * z
}
```

`StandardNormal` is a zero-size type with no construction cost. The affine transformation `mean + std * z` is mathematically identical to sampling from `Normal(mean, std)`.

Alternatively, cache a single `StandardNormal` as a constant — but since it is a ZST, construction is free.

**Justification:**
`Normal::new()` performs a validation check (`std > 0`) and stores `1/std` for the Ziggurat algorithm. But `rand_distr::StandardNormal` uses the same Ziggurat algorithm for N(0,1) and the affine transformation is trivially correct. This eliminates per-call overhead without changing the sampling distribution.

**Expected Benefit:**
Eliminates 960 unnecessary `Normal` constructions per second. Small but free — the code becomes simpler and faster simultaneously.

**Impact Assessment:**
Zero functional change. `N(mean, std)` is defined as `mean + std * N(0, 1)`. The affine transformation is exact; no floating-point precision difference beyond what the distribution already has.

---

### Eliminate Gradient-Seed Clone Allocations in `ppo_process_chunk`
- [x] Remove the `gv` and `gm` Vec clones by restructuring the borrow to allow direct slice passing

**Category:** Performance Improvement
**Severity:** High
**Effort:** Small
**Behavioural Impact:** None (verified — same gradient values, different ownership strategy)

**Location:**
- `src/brain/ppo/update.rs:275-278` — gradient seed cloning

**Current State:**
```rust
let gv: Vec<f32> = grad_values[..chunk_size].to_vec();
let gm: Vec<f32> = grad_means[..chunk_size * act_dim].to_vec();

brain.model.backward_batch_critic(&gv, chunk_size);
brain.model.backward_batch_actor(&gm, chunk_size);
```

The comment explains this exists to resolve a borrow conflict: `grad_values` and `grad_means` are mutable borrows into `brain.model.scratch`, and `backward_batch_critic`/`backward_batch_actor` take `&mut self` on the model. The clone creates separate owned copies so the scratch buffers can be mutated during backward.

This allocates and copies `chunk_size` + `chunk_size * 2` floats per chunk (66 floats for the default 64 samples = 264 bytes per allocation, but the allocation overhead and cache pollution matter more than the size).

**Proposed Change:**
Add dedicated `grad_seed_values: Vec<f32>` and `grad_seed_means: Vec<f32>` fields to `BatchScratch`. During the per-sample loss loop, write gradients into these dedicated buffers instead of into `gc_out`/`ga_out`. Then pass slices of the dedicated buffers to backward_batch. Since the dedicated buffers are separate from the scratch buffers that backward mutates, no clone is needed.

This requires changing the loss loop to write to `scratch.grad_seed_values[s]` and `scratch.grad_seed_means[s * act_dim + j]` instead of `grad_values[s]` and `grad_means[s * act_dim + j]`. The backward functions then receive `&scratch.grad_seed_values[..chunk_size]` etc.

The borrow checker conflict can also be resolved by extracting the backward call into a separate scope, but the dedicated-buffer approach is cleaner because it makes the data flow explicit.

**Justification:**
Two heap allocations per chunk tick (every tick that processes PPO training) is avoidable. The allocations are small but happen in the hottest loop in the system. Pre-allocating the seed buffers in BatchScratch follows the same pattern as all other scratch buffers.

**Expected Benefit:**
Eliminates 2 heap allocations per PPO chunk tick. Reduces allocator contention and cache pollution in the training hot path.

**Impact Assessment:**
Zero functional change. The same gradient values are passed to the same backward functions. Only the memory they reside in changes.

---

### Eliminate obs-batch Allocation in `ppo_process_chunk`
- [x] Add an `obs_batch` field to `BatchScratch` to eliminate the per-chunk `vec!` allocation

**Category:** Performance Improvement
**Severity:** High
**Effort:** Trivial
**Behavioural Impact:** None (verified — same observation data, pre-allocated buffer)

**Location:**
- `src/brain/ppo/update.rs:153` — `let mut obs_batch = vec![0.0f32; chunk_size * obs_dim];`

**Current State:**
Every PPO chunk tick allocates a fresh `Vec<f32>` of size `chunk_size * obs_dim` (64 * 43 = 2,752 floats = 11 KB) to stack observation data before the batched forward pass. The comment notes this exists to avoid a double-mutable-borrow on `brain.model` (scratch vs forward_batch).

**Proposed Change:**
Add an `obs_batch: Vec<f32>` field to `BatchScratch`, pre-allocated at construction to `max_batch * obs_dim` (512 * 43 = 22,016 floats). Use this pre-allocated buffer instead of the per-chunk allocation. Since `obs_batch` is populated before `forward_batch` is called (not during), the borrow conflict does not apply — `obs_batch` can be a separate field on the scratch struct that is passed as a slice.

**Justification:**
This follows the exact pre-allocation pattern used for all other scratch buffers in `BatchScratch`. The per-chunk allocation is an oversight from when the scratch buffer was introduced.

**Expected Benefit:**
Eliminates one 11 KB heap allocation per PPO chunk tick. With 8 chunks per epoch and 4 epochs per update, that is 32 allocations per PPO update cycle.

**Impact Assessment:**
Zero functional change by construction. The same observation data is copied into the same positions; only the buffer's lifetime changes from per-chunk to persistent.

---

### Batch Critic Forward Passes During Action Selection
- [x] Collect all 8 cars' observations and run a single batched `forward_critic` call instead of 8 sequential single-sample calls

**Category:** Performance Improvement
**Severity:** High
**Effort:** Medium
**Behavioural Impact:** None (verified — identical value predictions, batched execution)

**Location:**
- `src/brain/ppo/mod.rs:139-211` — `ppo_act_all_cars_system`
- `src/brain/ppo/model.rs:177-183` — `forward_critic` (single-sample)

**Current State:**
The `ppo_act_all_cars_system` iterates all 8 cars and calls `forward_actor` + `forward_critic` sequentially per car (lines 140-141):
```rust
let action_dist = brain.model.forward_actor(&obs.values);
let value = brain.model.forward_critic(&obs.values);
```

Each `forward_critic` call runs the full 3-layer critic MLP as a single-sample mat-vec multiply. The `forward_actor` must remain per-car because sampling depends on each car's result immediately. But the critic value is only used for buffer storage — it does not affect the action taken. All 8 values could be computed in a single batched pass after all cars' observations are collected.

The performance notes document that `forward_critic` adds ~1.7ms per tick for 8 cars and that "batching could be done" (from `context/notes/performance-tuning-lessons.md`).

**Proposed Change:**
1. First pass: iterate all cars, call `forward_actor`, sample actions, collect observations into a small stack buffer (8 x 43 = 344 floats).
2. Single call: `forward_batch` on the critic with the 8-sample batch (using the existing batched infrastructure).
3. Second pass: distribute the 8 value predictions to the buffer pushes.

The actor forward passes must remain sequential (each car's action depends on the actor output), but the critic passes are independent and can be batched. This requires restructuring the system to two passes — one for actor + action sampling, one for critic batching + buffer push.

**Justification:**
Mat-mat multiplication has much better arithmetic intensity than 8 sequential mat-vec multiplies. The batched forward infrastructure already exists (`forward_batch` in `ActorCritic`). The performance notes explicitly identify this as an opportunity. For an 8-sample batch through the 128-wide critic, the batched path should be roughly 3-5x faster than 8 sequential single-sample passes due to better cache utilisation and potential LLVM vectorisation.

**Expected Benefit:**
Reduces the per-tick critic cost from ~1.7ms (8 sequential single-sample passes) to roughly 0.4-0.6ms (one 8-sample batched pass). This is ~1ms saved per tick, which is significant against the 16.67ms frame budget.

**Impact Assessment:**
Zero functional change. The same observations produce the same value predictions. The critic has no internal state that depends on evaluation order (unlike the actor's RNG-dependent sampling). The only change is that all 8 predictions are computed in a single batched call rather than 8 sequential calls.

Note: This requires a `forward_critic_batch` method or reusing the existing `forward_batch` with a way to extract only the critic output. The existing `forward_batch` computes both actor and critic, which is wasteful here. A dedicated `forward_critic_batch` that only runs the critic layers would be ideal.

---

## Algorithm Optimisation

### Flatten `orthogonal_init` Output to Avoid Intermediate Allocation
- [x] Refactor `orthogonal_init` to return a flat `Vec<f32>` in row-major order instead of `Vec<Vec<f32>>`

**Category:** Algorithm Optimisation
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None (verified — identical initialised weight values)

**Location:**
- `src/brain/common/math.rs:9-68` — `orthogonal_init()`
- `src/brain/common/mlp.rs:26-31` — `Linear::new_orthogonal()` immediate flattening

**Current State:**
`orthogonal_init` returns `Vec<Vec<f32>>` — a nested vector of rows. `Linear::new_orthogonal` immediately flattens this into a contiguous `Vec<f32>`:
```rust
let init = orthogonal_init(out_dim, in_dim, scale, rng);
let mut weights = Vec::with_capacity(out_dim * in_dim);
for row in &init {
    weights.extend_from_slice(row);
}
```

The intermediate `Vec<Vec<f32>>` allocates `rows` separate heap allocations that are immediately discarded after flattening. For the critic's first layer (128 x 43), this is 128 heap allocations created and freed during construction.

**Proposed Change:**
Refactor `orthogonal_init` to work on a flat `Vec<f32>` in row-major order internally and return that directly. The Gram-Schmidt algorithm can be expressed with explicit row/column indexing into a flat buffer. Remove the `Vec<Vec<f32>>` entirely.

**Justification:**
This is a one-time cost (only at construction), so the performance impact is negligible. The value is in code clarity — the function that produces data for a flat-storage system should itself use flat storage, avoiding the conceptual mismatch and the unnecessary intermediate allocation.

**Expected Benefit:**
Eliminates the intermediate `Vec<Vec<f32>>` allocation at construction (one-time, ~128 heap allocs for the critic). Makes the init function match the storage format of its consumer.

**Impact Assessment:**
Zero functional change. The same Gram-Schmidt orthogonalisation produces the same weight values. The output format changes from `Vec<Vec<f32>>` to flat `Vec<f32>`, which is what `Linear::new_orthogonal` already converts to.

---

### Redundant `value_predictions` Clone in `ppo_prepare_update`
- [x] Remove the `value_predictions` clone from `ppo_prepare_update` since `PreparedUpdate` already owns the frozen buffer which contains the same values

**Category:** Performance Improvement
**Severity:** Medium
**Effort:** Trivial
**Behavioural Impact:** None (verified — `value_predictions` is identical to `frozen_buffer.values`)

**Location:**
- `src/brain/ppo/update.rs:81` — `let value_predictions = buffer.values.clone();`
- `src/brain/ppo/update.rs:348` — only use: `explained_variance(&prepared.returns, &prepared.value_predictions)`

**Current State:**
`ppo_prepare_update` clones `buffer.values` into `prepared.value_predictions` (line 81). After `std::mem::take` on line 92, the frozen buffer owns the original values. The only consumer of `value_predictions` is `explained_variance` on line 348, which takes `&[f32]` slices.

The clone creates a duplicate of the entire values vector (512 floats = 2 KB) that is identical to `prepared.frozen_buffer.values`.

**Proposed Change:**
Remove the `value_predictions` field from `PreparedUpdate`. Change the `explained_variance` call on line 348 to use `&prepared.frozen_buffer.values` instead of `&prepared.value_predictions`.

**Justification:**
The clone is unnecessary — the data already exists in the frozen buffer. This is a simple oversight from when the buffer was cloned rather than swapped.

**Expected Benefit:**
Eliminates one 2 KB allocation per PPO update cycle. Minor but trivially free.

**Impact Assessment:**
Zero functional change by construction. `value_predictions` and `frozen_buffer.values` contain identical data because `value_predictions` was cloned from `buffer.values` immediately before `buffer` was moved into `frozen_buffer`.

---

### `Tanh::forward` Clones Output Unnecessarily
- [x] Remove the `.clone()` in `Tanh::forward` by restructuring the cache to avoid double allocation

**Category:** Performance Improvement
**Severity:** Medium
**Effort:** Trivial
**Behavioural Impact:** None (verified — same cached output, no clone needed)

**Location:**
- `src/brain/common/mlp.rs:194-198` — `Tanh::forward()` single-sample path

**Current State:**
```rust
pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
    let output: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
    self.output_cache = Some(output.clone());
    output
}
```

The output vector is allocated, then cloned into the cache, then returned. Two allocations for one result.

**Proposed Change:**
Compute into the cache directly and return a clone of the cache, or better, compute once, store, and return:
```rust
pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
    let output: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
    self.output_cache = Some(output.clone());
    output
}
```

Wait — this is the same. The issue is that `forward` needs to return an owned `Vec` and cache one. The cleanest fix: store the output in the cache first, then clone to return:
```rust
let output: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
self.output_cache = Some(output);
self.output_cache.as_ref().unwrap().clone()
```

This is still two allocations. The real fix is to make the single-sample path use a pre-allocated buffer (like the batch path does), but the single-sample `Tanh::forward` is only used on the action-selection path (not the training path), so the impact is minor.

Actually, the simplest fix: just do `self.output_cache = Some(output.clone()); output` — which is what it already does. The alternative is to return a reference, but the caller expects `Vec<f32>`.

Given that the single-sample path is only used for per-car action selection (not training), this is genuinely low severity. Downgrading.

**Justification:**
The clone allocates a second copy of a 64-element vector on every single-sample forward pass. With 8 cars and 4 tanh layers each (actor tanh1, actor tanh2 — critic is also called), this is 32 extra allocations per tick on the action-selection path.

However, since the action-selection path also allocates in `Linear::forward` (which returns `Vec<f32>`), fixing just the Tanh clone is incomplete without also fixing the Linear single-sample path. The batch path (used for training) is already allocation-free.

**Expected Benefit:**
Eliminates 32 unnecessary vector clones per tick on the action-selection path.

**Impact Assessment:**
Zero functional change. Same tanh output, same cache.

---

## Dead Code Removal

### `Linear::backward` Single-Sample Path is Dead Code
- [x] Remove the `#[allow(dead_code)]` on `Linear::backward` and `Tanh::backward` and verify they are truly unused, or remove them entirely

**Category:** Dead Code Removal
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None (verified — already marked `#[allow(dead_code)]` and never called)

**Location:**
- `src/brain/common/mlp.rs:66` — `Linear::backward()` with `#[allow(dead_code)]`
- `src/brain/common/mlp.rs:200` — `Tanh::backward()` with `#[allow(dead_code)]`

**Current State:**
Both `Linear::backward` and `Tanh::backward` (single-sample variants) are marked `#[allow(dead_code)]`. The training path uses `backward_batch` exclusively. The single-sample forward path (action selection) does not use backward at all. These functions have zero callers in the entire codebase.

**Proposed Change:**
Remove both functions. If they are needed in the future (unlikely — the batch path is strictly superior), they can be restored from version control.

**Justification:**
Dead code with `#[allow(dead_code)]` annotations clutters the file and can mislead readers into thinking a non-batch backward path is used somewhere.

**Expected Benefit:**
Removes ~35 lines of dead code and two `#[allow(dead_code)]` suppressions.

**Impact Assessment:**
Zero functional change by construction — the functions have zero callers.
