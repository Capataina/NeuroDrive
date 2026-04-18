# PPO Optimisation — Remaining Work

## Status — 2026-04-18

All performance items are complete. All architecture items (tanh, orthogonal init, asymmetric actor-critic, AdamW on critic, log_std floor, per-minibatch advantage normalisation with Fisher-Yates shuffle, amortised updates, 8-car vectorisation, three-backend GEMM dispatch, batched multi-car action selection, `samples_per_tick=32`, `time_penalty_per_tick=0.0`) are implemented and verified across all four backend variants.

Learning-state snapshot from 30 March: cars learn full throttle and crash at the first corner. The architectural fixes (wider critic + AdamW + log_std floor) are in place but haven't been run long enough to confirm they change the throttle-collapse pattern. **A full training run under current code is the natural next validation** — performance is no longer an obstacle to it.

## Remaining Work — Learning-Quality Items

### Running Observation Normalisation

**Files:** new `src/agent/normalisation.rs`, `src/agent/observation.rs`.

Welford's online algorithm for mean/variance per observation feature. Clip normalised values to `[-10, 10]`. Warmup window of ~1000 ticks during which statistics accumulate without being applied.

**Why it matters:** raw observations span wildly different magnitudes (rays in `[0,1]`, curvatures in `[−1,1]`, speed deltas unbounded). Even with orthogonal init, a feature whose scale drifts over training causes the critic's early-layer weights to chase moving targets. This is the single biggest missing item from the 2026-03 research pass.

### Linear LR Annealing

**Files:** `src/brain/ppo/mod.rs`, `src/brain/common/optim.rs`.

Decay actor and critic learning rates from their initial values to 0 over `total_timesteps` steps. Simplest path: add `set_learning_rate(lr)` on `AdamOptimizer`, call it from `ppo_finish_epoch` with the current fraction.

**Why it matters:** after ~500k ticks the policy should commit to a local optimum rather than oscillate around it. Fixed LR keeps the policy hopping.

### Extract log-std Adam into shared optimiser

**Files:** `src/brain/ppo/update.rs`, `src/brain/common/optim.rs`.

`ppo_finish_epoch` has ~14 lines of inlined Adam for `a_log_std` that duplicates `AdamOptimizer::step()`. Extend `AdamOptimizer` to handle scalar params, or extract a helper. Low priority — correctness hygiene, not a learning or performance win.

**Why it matters:** if the canonical Adam hyperparams drift (e.g. bias-correction term is updated in one place and not the other), the log-std optimiser would silently diverge.

## Reference

- `context/systems/brain-ppo.md` — verified current architecture (updated 2026-04-18)
- `context/references/ppo-network-and-training-optimisation.md` — research backing the critic-capacity fixes
- `context/references/ppo-optimisation.md` — durable PPO research (pre-overhaul snapshot; read for principles, not state)
- `context/references/ppo-epoch-performance.md` — the 2026-04-18 research that drove the GEMM backend work
- `context/references/ppo-action-selection-performance.md` — the 2026-04-18 research that drove the batched-actor refactor
