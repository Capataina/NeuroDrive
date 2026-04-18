# PPO Optimisation — Remaining Work

## Status

Phase 0, Phase 1, and most of Phase 3 are complete. Prior audits closed the biggest items: tanh activation, orthogonal init with output-head scaling, per-minibatch advantage normalisation with Fisher-Yates shuffle, batched forward/backward, amortised updates at 64 samples/tick, asymmetric actor-critic (actor 2×64, critic 2×128), AdamW weight decay on the critic, `log_std` floor at −1.0, per-sample and per-batch scratch buffers, per-env GAE without `HashMap`, 8-car vectorisation, `time_penalty_per_tick = 0.0`.

Learning-state snapshot from 30 March: cars learn full throttle, crash at the first corner. The architectural fixes (wider critic + AdamW + log_std floor) are in place but have not been run long enough to confirm they change the throttle-collapse / critic-discrimination pattern. A training session under current code is the natural next validation.

## Remaining Work

### Running Observation Normalisation

**Files:** new `src/agent/normalisation.rs`, `src/agent/observation.rs`.

Welford's online algorithm for mean/variance per observation feature. Clip normalised values to `[-10, 10]`. Warmup window of ~1000 ticks during which statistics accumulate without being applied (so the first few hundred updates aren't trained against wildly different feature scales).

**Why it still matters:** raw observations span wildly different magnitudes (rays in `[0,1]`, curvatures in `[−1,1]`, speed deltas unbounded). Even with orthogonal init, a feature whose scale drifts over training causes the critic's early-layer weights to chase moving targets. This is the single biggest missing item from the 2026-03 research pass.

### Linear LR Annealing

**Files:** `src/brain/ppo/mod.rs`, `src/brain/common/optim.rs`.

Decay actor and critic learning rates from their initial values to 0 over `total_timesteps` steps. Simplest path: add `set_learning_rate(lr)` on `AdamOptimizer`, call it from `ppo_finish_epoch` with the current fraction.

**Why it still matters:** after ~500k ticks the policy should commit to a local optimum rather than oscillate around it. Fixed LR keeps the policy hopping.

### Extract log-std Adam into shared optimiser

**Files:** `src/brain/ppo/update.rs`, `src/brain/common/optim.rs`.

`ppo_finish_epoch` has ~14 lines of inlined Adam for `a_log_std` that duplicates `AdamOptimizer::step()`. Extend `AdamOptimizer` to handle scalar params, or extract a helper. Low priority — this is correctness hygiene, not a performance or learning win.

**Why it still matters:** if the canonical Adam hyperparams drift (e.g. bias-correction term is updated in one place and not the other), the log-std optimiser would silently diverge.

## Reference

- `context/systems/brain-ppo.md` — verified current architecture
- `context/references/ppo-network-and-training-optimisation.md` — research backing the critic-capacity fixes
- `context/references/ppo-optimisation.md` — durable PPO research (pre-overhaul snapshot; read for principles, not state)
