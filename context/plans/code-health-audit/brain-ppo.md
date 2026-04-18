# Brain / PPO — Code Health Findings

**Systems covered:** `src/brain/ppo/` (act, collect, epoch, flush; `model.rs`, `update.rs`, `buffer.rs`); `src/brain/common/` (`mlp.rs`, `math.rs`, `optim.rs`).

**Finding count:** 4 findings (1 high, 2 medium, 1 low).

**Context:** the 2026-04-15 audit already closed the highest-leverage items (AoS→SoA buffer flattening, pre-allocated `BatchScratch`, batched forward/backward, cache-friendly `forward_batch` loop order, flat `orthogonal_init`, batched critic forward pass for action selection, Tanh-clone fix, `sample_normal` affine-only path). The findings below are the residual wins this deeper pass surfaced plus one structural-safety refactor that removes the `unsafe` blocks.

---

## Data Layout and Memory Access Patterns

### Replace per-car-per-tick `Vec<f32>` allocations in single-sample `Linear::forward` with reusable scratch buffers
- [x] Replace the fresh-`Vec`-per-layer pattern in `Linear::forward` and `Tanh::forward` with a scratch-buffer design so that `ActorCritic::forward_actor`, `forward_critic`, and `forward` no longer allocate on the action-selection path

**Category:** Data Layout and Memory Access Patterns
**Severity:** High
**Effort:** Medium
**Behavioural Impact:** None (verified by equivalence reasoning — identical output layout, only allocation strategy changes)

**Location:**
- `src/brain/common/mlp.rs:47-56` — `Linear::forward`: allocates `let mut output = vec![0.0; self.out_dim]` and returns it by value.
- `src/brain/common/mlp.rs:171-175` — `Tanh::forward`: `.map(|&x| x.tanh()).collect()` into a fresh `Vec<f32>`, caches it, and returns a `.clone()` of the cache.
- `src/brain/ppo/model.rs:178-186` — `ActorCritic::forward_actor`: issues five `forward` calls per car (`a_fc1`, `a_tanh1`, `a_fc2`, `a_tanh2`, `a_mean`), each allocating, plus a `.collect()` for `std = a_log_std.iter().map(|&ls| ls.exp()).collect()`.
- `src/brain/ppo/mod.rs:207` — `ppo_act_all_cars_system` calls `forward_actor(&obs.values)` for every car every tick.
- `src/brain/ppo/mod.rs:355` — `forward_critic` called per non-terminal car on every update prepare (one additional allocation cascade: `c_fc1`, `c_tanh1`, `c_fc2`, `c_tanh2`, `c_value`).

**Current State:**
On the action-selection hot path (`SimSet::Input`, every tick, every car), the single-sample forward pipeline allocates **one fresh `Vec<f32>` per layer** plus one `Vec<f32>` for the `std` field of `ActionDist`. For the actor alone that is: 1 × `fc1.out_dim=64` + 1 × `tanh1=64` + 1 × `fc2.out_dim=64` + 1 × `tanh2=64` + 1 × `mean.out_dim=2` + 1 × `std=act_dim=2` = **6 allocations per car per tick**. With the default 8-car trainer, that is 48 allocations per tick on the actor path. At 60 Hz that is 2,880 allocations per second from `forward_actor` alone. The critic uses the same allocation pattern on the bootstrap path inside `ppo_collect_rewards_all_cars_system` (one additional 6-allocation cascade per non-terminal car when `reached_horizon || reached_terminal_batch`).

`Tanh::forward` carries a second inefficiency: it allocates the output vec via `.collect()`, stores it in `output_cache: Option<Vec<f32>>`, and then returns `self.output_cache.as_ref().unwrap().clone()` (`mlp.rs:172-175`). The clone is a redundant second allocation — the caller only needs ownership to pass into the next `forward`, and a `&[f32]` would do.

The existing batched path (`forward_batch`, `BatchScratch`) already proves the scratch-buffer pattern works in this codebase; the single-sample path was left with per-call allocations because the comment on `Linear::forward` (line 46) says "allocation is acceptable for the low-frequency action-selection path." At 1 car that comment was defensible; at 8 cars × 60 Hz × 6 allocations/call, it is no longer accurate.

**Proposed Change:**
Add a `SampleScratch` struct (analogous to the existing `BatchScratch`) that owns the six per-layer intermediates as flat `Vec<f32>`s sized for the actor and critic hidden dims and the action dim, pre-allocated at `ActorCritic::new`. Refactor:

1. `Linear::forward(&mut self, input: &[f32]) -> Vec<f32>` → `Linear::forward_into(&mut self, input: &[f32], output: &mut [f32])` (writes into a pre-allocated slice, returns unit; keeps `input_cache` behaviour unchanged).
2. `Tanh::forward(&mut self, input: &[f32]) -> Vec<f32>` → `Tanh::forward_into(&mut self, input: &[f32], output: &mut [f32])` (writes in place into `output`, updates `output_cache` via `.copy_from_slice`). This also eliminates the internal `.clone()` of the cache.
3. `ActorCritic::forward_actor` / `forward_critic` / `forward` take `&mut self` and write into the scratch buffers directly; they return a thin view (either `ActionDistView<'a>` borrowing from scratch, or the existing owned `ActionDist` filled from the scratch buffer via a single `Vec::from_slice` at the leaf — one allocation per call instead of six).
4. `std` — because `act_dim = 2` and `a_log_std: Vec<f32>` has length 2, the `.collect()` in `model.rs:184` and `:205` can be replaced by writing into a `[f32; 2]` on the stack or into a two-element scratch slice; the size is statically known.

The old `forward` signature can be retained as a thin wrapper calling `forward_into` with a freshly-allocated Vec, so existing call sites outside the hot path (tests, diagnostics) do not need to change.

**Justification:**
Analytical evidence (confidence: high):
- Line-level inspection confirms `Linear::forward` returns `vec![0.0; self.out_dim]` on every call (`mlp.rs:49`). This is a heap allocation with an allocator call, zeroing, and a deallocation when the temporary is dropped at the end of `forward_actor`.
- Per-call allocation count for `forward_actor` is structurally six (five `forward` results + `std.collect()`).
- The prior audit's performance measurements (`context/notes/performance-tuning-lessons.md`) show the action-selection cost rose from ~1.7ms to ~3.3ms when the critic widened from 64 to 128 and ran sequentially per car. After the 2026-04-15 audit's batched critic switch, the critic portion of action selection is one mat-mat rather than eight sequential mat-vecs. The actor forward is still sequential per car, which makes this allocation pattern the largest remaining source of hot-path allocator engagement.
- Research support: the Rust Performance Book §"Heap Allocations" and the Markaicode high-performance inference write-up both list "pre-allocated tensor memory" as the primary pattern for latency-sensitive inference in Rust. The existing `BatchScratch` is the same pattern applied to the batched update path — generalising it to the per-sample action-selection path is the obvious completion.

Research mode: 2 (specific-technique evaluation).

**Expected Benefit:**
- Eliminates ~48 heap allocations per tick (8 cars × 6 per forward_actor) from the hot path — roughly 2,880 allocations/second at 60 Hz, many of which currently land on cold pages because the allocator has moved on between frames.
- Removes the redundant `Tanh::forward` output-cache clone (one extra allocation per tanh call, affecting every tanh forward on both actor and critic paths in every variant — single-sample, single-sample critic, and full forward).
- Should reduce the actor portion of the per-tick action-selection cost by a meaningful fraction. Exact number requires a benchmark; the prior audit's 17.3ms→9.0ms end-to-end improvement came from this family of changes on the training path.

**Impact Assessment:**
Zero functional change. The forward arithmetic, tanh output, and `ActionDist.mean`/`std` values are identical — only the backing storage strategy changes. `input_cache` and `output_cache` semantics are preserved for the single-sample backward path (unchanged). The `ActionDist` struct fields are unchanged externally; only the internal path that populates them differs. Floating-point operations are not reordered.

Edge case considered: concurrency. None of the hot-path fixed-update systems here run in parallel — Bevy's fixed-update schedule serialises systems that share `ResMut<PpoBrain>`. Scratch buffers live inside the mutably-borrowed `PpoBrain` resource, so aliasing is impossible by Rust's borrow checker.

Confidence: **high** (analytical). A diagnostic benchmark would have pushed this to "strongest," but see the diagnostic-test deferral in the Obligation Evidence Map for why one could not be written in this audit without violating Rule 3.

---

## Performance Improvement

### Eliminate per-prepare `HashMap` allocation in `compute_gae_per_env`
- [x] Replace the `HashMap<u32, Vec<usize>>` env-grouping pass in `compute_gae_per_env` with a single-pass scheme keyed on the dense contiguous `env_id` range

**Category:** Performance Improvement
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None (verified by equivalence reasoning — same GAE values, same advantage/return vectors)

**Location:**
- `src/brain/ppo/buffer.rs:111-151` — `TrainerRolloutBuffer::compute_gae_per_env`
- `src/brain/ppo/mod.rs:195-203` — `TrainerConfig` default car count is 8 (env_ids are `0..8`)

**Current State:**
Every time a PPO update is prepared, `compute_gae_per_env` does:
1. Allocates `advantages = vec![0.0; n]` and `returns = vec![0.0; n]` — required for the result (fine).
2. Allocates a fresh `HashMap<u32, Vec<usize>>` keyed by `env_id`, and walks the buffer pushing each index into the matching bucket. This allocates one `Vec<usize>` per unique env_id, each of which grows internally.
3. Iterates the HashMap with `for (eid, indices) in &env_indices` — HashMap iteration order is non-deterministic across runs even with a fixed seed, though in this code the per-env GAE values are independent so the visit order does not affect the output.

With `max_steps = 512` buffer transitions across 8 envs, each env has ~64 indices, so this is 8 `Vec<usize>` allocations each growing through 2-3 capacity doublings (to 64) = roughly 24+ allocator calls per PPO update, plus the `HashMap` backing-table allocation (typically one power-of-two bucket array). PPO updates occur every ~512/8 = 64 ticks per env (~1s of wall time at 60 Hz), so this cost is amortised but is still unnecessary.

**Proposed Change:**
Because `env_id` values are dense and small (allocated `0..num_envs` by the trainer), group indices into a pre-allocated `Vec<Vec<usize>>` of length `num_envs` (or a flat `Vec<u32>` run-length encoding). Two shapes work:

**Option A (simplest, same shape):** Replace `HashMap<u32, Vec<usize>>` with `Vec<Vec<usize>>` of length `max_env_id + 1` (computed from `self.env_ids.iter().max()` on the first pass; a single pre-allocated sub-vec per env). Reuses capacity across prepares if the buffer owns the structure.

**Option B (allocation-free):** Store on `TrainerRolloutBuffer` a reusable `env_indices: Vec<Vec<usize>>` that is `.clear()`-ed per call (capacity preserved), not reallocated.

**Option C (preferred, data-layout win):** Because env transitions are interleaved in a predictable pattern (the act system always runs all 8 cars in order each tick), the `env_ids` vector is structurally periodic: `[0,1,2,…,7,0,1,2,…,7,…]`. For such a buffer, per-env indices can be generated arithmetically without any intermediate storage: for env `e`, indices are `{e, e + num_envs, e + 2·num_envs, …}`. If the audit confirms that periodicity holds (it does for default operation — see `ppo_act_all_cars_system` which iterates `car_query` in Bevy-entity-order every tick, and `ppo_collect_rewards_all_cars_system` does the same), the HashMap can be eliminated entirely. The safeguard is a `debug_assert!` that the pattern holds; a fallback path handles the (currently impossible) non-periodic case.

Start with Option B (zero-risk, preserves behaviour exactly); escalate to Option C only after a separate verification pass confirms periodicity at every prepare site.

**Justification:**
Analytical evidence (confidence: high for B; moderate for C pending periodicity verification):
- The HashMap allocation is unambiguous from `buffer.rs:122` (`let mut env_indices: HashMap<u32, Vec<usize>> = HashMap::new()`).
- The call site runs at PPO prepare time, which under default config is ~1× per second per training run — not the tightest hot path, but free to remove.
- Research support: generic Rust performance guidance consistently recommends replacing "HashMap keyed by small integer" with "`Vec` indexed by integer" when the key range is dense (Rust Performance Book §"Heap Allocations" and the Markaicode inference write-up both list this).

Research mode: 2 (specific-technique evaluation, shared with finding #1 above).

**Expected Benefit:**
- Removes 8-9 heap allocations per PPO update (1 HashMap table + 8 per-env `Vec<usize>`s).
- Makes env iteration order deterministic (HashMap iteration order depends on hasher state — not a correctness issue here but a minor determinism improvement).
- Option C specifically would make `compute_gae_per_env` fully zero-allocation except for the output vectors.

**Impact Assessment:**
Zero functional change. Per-env GAE values are computed inside independent loops whose iteration order over envs does not affect the output (each env's GAE is path-independent of other envs'). The output `(advantages, returns)` vectors are returned by value with identical shapes and identical numeric content. The existing `single_env_gae_matches_flat_gae` and `multi_env_gae_isolates_envs` unit tests (`src/brain/ppo/buffer.rs:161`, `:200`) exercise this function and will catch any behaviour drift.

Confidence: **high** (for Option B); **moderate** (for Option C, pending the periodicity audit noted in the proposal).

---

## Modularisation

### Split `BatchScratch` so the `unsafe { slice::from_raw_parts }` aliasing workarounds in `update.rs` become safe
- [x] Reshape `BatchScratch` into two sub-structs (e.g. `ForwardScratch` and `GradientScratch`) so the borrow checker sees disjoint fields and the two `unsafe` blocks at `update.rs:162` and `update.rs:291-292` can be deleted

**Category:** Modularisation
**Severity:** Medium
**Effort:** Medium
**Behavioural Impact:** None (verified — purely a type-level refactor; no runtime logic changes)

**Location:**
- `src/brain/ppo/update.rs:156-162` — first `unsafe { std::slice::from_raw_parts(obs_batch_ptr, obs_slice_len) }` for the batched forward pass input alias.
- `src/brain/ppo/update.rs:282-292` — second and third `unsafe { ... }` blocks for `grad_seed_values` and `grad_seed_means` backward inputs.
- `src/brain/ppo/model.rs:13-54` — `BatchScratch` struct that currently holds forward intermediates, backward intermediates, and the three "gradient-seed" buffers in one flat struct.

**Current State:**
`ppo_process_chunk` needs to pass `obs_batch` (a shared slice) into `brain.model.forward_batch(&mut self, …)` while `self` already mutably borrows the same scratch struct that `obs_batch` lives in. Rust's borrow checker (correctly) treats `brain.model.scratch` as a single borrow unit and rejects the combined `&[f32]` + `&mut self` pattern. The author worked around this by taking a raw pointer to `obs_batch` and rebuilding a shared slice via `std::slice::from_raw_parts`. The same pattern is repeated for `grad_seed_values` and `grad_seed_means` in the backward call.

Soundness audit result (performed this session):
- `forward_batch` reads from `obs_batch` and writes into `self.scratch.a_h1`, `a_h1_act`, `a_h2`, `a_h2_act`, `a_out`, and the critic analogues. None of these are `obs_batch` itself. `Linear::forward_batch` internally writes to `self.batch_input_cache` — but that cache is owned by the individual `Linear` layer (not by `BatchScratch`), so it does not alias the `obs_batch` slice either.
- `backward_batch_critic` / `backward_batch_actor` read `grad_seed_values` / `grad_seed_means` and write to `gc_*` / `ga_*` intermediates — different fields of `BatchScratch`.

Both `unsafe` blocks are **sound today**. But they are load-bearing guarantees that any future edit to `forward_batch` or `backward_batch_*` must preserve — and there is no compiler check that the invariant still holds. The risk is that a later refactor that looks innocuous (e.g. adding a debug-logging write to `obs_batch` at the top of `forward_batch`, or reusing `obs_batch` as a working buffer inside the loop) silently turns the code into undefined behaviour.

Research support (Rust nomicon §"Splitting Borrows", `std::slice::from_raw_parts` documentation): the canonical Rust pattern for "I need to pass some fields as shared while mutably borrowing others" is to split the struct into independently-borrowable sub-structs. `slice::from_raw_parts` is a last resort when that is infeasible.

**Proposed Change:**
Split `BatchScratch` into three cooperating sub-structs:

```
struct BatchInputs      // owned inputs passed as shared references
    obs_batch
    grad_seed_values
    grad_seed_means

struct ForwardIntermediates  // actor + critic forward scratch
    a_h1, a_h1_act, a_h2, a_h2_act, a_out
    c_h1, c_h1_act, c_h2, c_h2_act, c_out

struct BackwardIntermediates  // actor + critic backward scratch
    ga_out, ga_h2_act, ga_h2, ga_h1_act, ga_h1, ga_input
    gc_out, gc_h2_act, gc_h2, gc_h1_act, gc_h1, gc_input
```

(Or a simpler two-way split if that is cleaner: `inputs` and `intermediates`.)

Store them as three named fields on `ActorCritic` (or a combined `Scratch` whose fields are accessed by name). Then the borrow-split in `ppo_process_chunk` becomes:

```rust
let obs = &brain.model.inputs.obs_batch[..chunk_size * obs_dim];
brain.model.forward_batch_with(obs, &mut brain.model.intermediates, chunk_size);
```

Rust's disjoint-field borrow inference accepts this because `inputs` and `intermediates` are separate struct fields. Both `unsafe` blocks disappear.

**Justification:**
Analytical evidence (confidence: high):
- The soundness audit confirms no field-level aliasing exists today, so the refactor is a direct simplification — not a bug fix.
- Rust nomicon §"Splitting Borrows" documents this exact pattern as the idiomatic fix for the borrow-checker shape the current code works around.
- The long-term safety argument is that removing `unsafe` reduces the surface area where future edits can silently introduce UB. This is a hygiene improvement, not a runtime improvement.

Research mode: 3 (known-anti-pattern check). Source: `slice::from_raw_parts` docs, Rust nomicon borrow-splitting.

**Expected Benefit:**
- Deletes three `unsafe` blocks and one `SAFETY:` comment block.
- Removes a load-bearing invariant from the mental model required to edit `forward_batch` / `backward_batch_*`.
- Makes the struct shape self-documenting — it becomes obvious from the type which buffers are inputs and which are working space.

**Impact Assessment:**
Zero functional change. Purely a structural reshape of where fields live; the bytes, operations, and order of operations are all preserved. The public API of `ActorCritic` (forward_batch, backward_batch_*, zero_grad) is unchanged at the call-site level — only internal field access patterns shift. The runtime characteristics (allocation count, memory layout of individual buffers) are unchanged; the sub-struct boundaries are compile-time only.

Confidence: **high** (analytical). The soundness audit of the current `unsafe` is itself the strongest evidence — it confirms the refactor is a pure simplification.

---

## Performance Improvement (Low)

### Avoid the `collect::<Vec<f32>>()` in `ActorCritic::forward_actor` / `forward` for the two-element `std` vector
- [x] Return `[f32; 2]` for `ActionDist.std` (or write into a stack array) instead of calling `.iter().map(...).collect()` on every per-car forward

**Category:** Performance Improvement
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None (verified — same two values, stack vs heap only)

**Location:**
- `src/brain/ppo/model.rs:184` — `let std = self.a_log_std.iter().map(|&ls| ls.exp()).collect();` (inside `forward_actor`)
- `src/brain/ppo/model.rs:205` — same in `forward`
- `src/brain/ppo/model.rs:6-9` — `ActionDist` definition: `pub struct ActionDist { pub mean: Vec<f32>, pub std: Vec<f32> }`

**Current State:**
The action dimension is `act_dim = 2` and is fixed by the environment (steering + throttle). `ActionDist.std` is therefore always a 2-element `Vec<f32>`, but it is allocated via `.collect()` on every single-sample forward. For 8 cars × 60 Hz, that is 480 unnecessary heap allocations per second — each for a two-element Vec.

A matching observation applies to `ActionDist.mean`, which comes from `a_mean.forward(&a2_r)` and is a 2-element Vec. This one is partially addressed by finding #1 above (the broader scratch-buffer refactor); this finding addresses the specifically trivial `std` field.

**Proposed Change:**
Either:
1. Change `ActionDist.std` to `[f32; 2]` (matching the known `act_dim = 2`). This requires updating the two call sites in `forward_actor` / `forward` and the consumers in `ppo_act_all_cars_system` (`mod.rs:213-217`), which already index with `[i]` — a plain array supports identical indexing.
2. If the project wants to keep `act_dim` generic (looking at `ActorCritic::new`, `act_dim` is a parameter but the only caller uses `2`), introduce a `smallvec::SmallVec<[f32; 4]>` — but this adds a dependency, violating Rule 2. Prefer option 1.

**Justification:**
Analytical evidence (confidence: high):
- Line-level reading confirms the `.collect()` on a known-length-2 iterator.
- `ActionDist.mean` is already handled by finding #1's broader refactor; this finding stands alone because converting `std` to `[f32; 2]` is a one-line change that lands independently of the larger refactor.
- The cost per call is tiny (a 2-element alloc is maybe 24-32 bytes) but the call count is 480/sec and the avoidance is free.

**Expected Benefit:**
- Eliminates one heap allocation per `forward_actor` / `forward` call.
- Makes the `act_dim = 2` assumption explicit in the type.

**Impact Assessment:**
Zero functional change. The two `std` values are unchanged; only storage shifts from heap to stack. Callers already index `.std[i]` — array indexing has identical semantics.

Confidence: **high** (analytical).

---

## Data Layout analysis applicability decision (required per Obligations checklist)

- Applied to every file listed under this system. The highest-leverage Data Layout finding (`forward_actor` scratch buffers) is finding #1 above. No further Data Layout wins are surfaced in this system — the prior audit's flat `Vec<f32>` weight storage, `BatchScratch` pre-allocation, and column-major weight access pattern cover the major categorical bases. Finding #3 above (the `BatchScratch` split) is a Modularisation finding, not a Data Layout one, because the bytes don't move — only the type fencing around them does.

---

## Notes on residual claims not promoted to findings

- The `a_log_std_grad` accumulation across chunks within an epoch was audited and verified correct: `zero_grad` (`mlp.rs:130`, called via `model.zero_grad` at `update.rs:109` when `sample_offset == 0`) resets the gradient at the start of each epoch; subsequent chunks within the same epoch accumulate into it; `ppo_finish_epoch` consumes it. No finding.
- The `unsafe` block soundness audit came back clean — no bug found, only the Modularisation finding #3 above (remove the need for `unsafe` rather than fix it).
- `compute_gae_per_env` uses `HashMap::entry(…).or_default().push(i)` — covered by finding #2.
- `orthogonal_init` at `math.rs:12-65` was fully rewritten by the prior audit to produce a flat row-major `Vec<f32>` directly; the inner Gram-Schmidt loops are sound and appropriately scoped.
