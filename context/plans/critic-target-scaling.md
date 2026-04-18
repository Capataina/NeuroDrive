# Critic Target Scaling — Round 2 Training Setup

## Purpose

Ship the four interventions the research round 1 converged on:

1. **γ: 0.99 → 0.995** — fixes credit-horizon / anticipation-horizon mismatch
2. **PopArt on `c_value`** — fixes unbounded target scale driving `c_fc2` saturation
3. **Observation running mean/var normaliser** — Andrychowicz's #1 PPO recommendation
4. **Target-KL early stop** — cheap guardrail against policy overshoot while the critic adapts

This is a **single integrated change set** because the four interventions are orthogonal and mutually supportive — each addresses a different aspect of the critic-training pipeline. Shipping them together means round 2 measures the whole stack at once; shipping them serially would require four separate training runs, which is not what's wanted.

## Context from round-1 research

The three relevant references:

- `context/references/ppo-critic-architecture.md` — saturation diagnosis, ranked interventions A–G
- `context/references/value-target-normalisation.md` — PopArt derivation + Rust sketch
- `context/references/ppo-tuning-knobs-racing.md` — γ, target-KL, entropy / log_std analysis

Shared diagnosis: the critic's training signal is damped in two independent ways (unbounded target scale, too-short γ horizon) and the observation input is unnormalised. The three independent research threads converged on exactly the fixes in this plan.

## Falsification criteria

Round 2 research will measure against these predictions. If any of them fail to move in the expected direction, the diagnosis was wrong somewhere — the round 2 prompt will carry the baseline vs round 2 comparison for the research agent to judge.

| Metric | Baseline (run_1776543971) | Post-fix expectation | Meaning if not met |
|---|---|---|---|
| `c_fc2` saturation | 68.5% | < 40% | Target scale was not the bottleneck — escalate to LayerNorm or widen critic |
| `c_value` weight L2 | 19.22 | stable (no longer chasing) | Weight norm still climbing means returns still growing unboundedly — PopArt µ/σ not tracking |
| Crash vs average value gap | 46.9 / 80.9 (42%) | < 30 / ~80 (> 60% gap) | Critic still cannot distinguish dangerous from safe states |
| Overshoot crash share (chunk 10) | 92% | < 60% | Anticipatory throttle-off not emerging — γ change was insufficient |
| Route consistency | 0.001 | > 0.05 | Fleet still not converging on the solution |
| Fleet members completing > 50% progress | ~1 | ≥ 3 | One-car story rather than broad learning |
| Explained variance | 0.71 | stable or higher | PopArt predictions have broken |
| Approx KL | 0.00352 | < 0.02 per update | Target-KL early stop should cap this |

## What changes, concretely

### 1. γ change

`src/brain/ppo/mod.rs:60` — `PpoConfig::default().gamma` from `0.99` to `0.995`.

That's the whole change. GAE already reads `brain.config.gamma` so no downstream edits are needed.

### 2. PopArt on `c_value`

New state on `PpoBrain`:

```rust
pub struct ValueNorm {
    pub mu: f32,       // running mean of returns
    pub sigma: f32,    // running std of returns
    pub beta: f32,     // EMA decay per update (start 1e-4 per update)
    pub epsilon: f32,  // min sigma floor
}
// initial state: mu=0, sigma=1 — identity transform until first update
```

Four integration points:

- **`PpoBrain` field** — add `value_norm: ValueNorm` resource-level state.
- **`ppo_prepare_update`** — before the first chunk, compute batch mean/std of `returns`, EMA-update `mu, sigma`, and apply POP rescale to `c_value.weights` and `c_value.biases[0]`. The rescale formula:
  - `W' = W * (old_sigma / new_sigma)`
  - `b' = (old_sigma * b + old_mu - new_mu) / new_sigma`
  - This preserves existing predictions `sigma*z+mu` across the stats update.
- **Loss computation** (`ppo_process_chunk`, around line 210) — the critic's raw output `z = c_out[s]` is now interpreted as the **normalised** value prediction. Target becomes `ret_norm = (ret - mu) / sigma`. Huber loss runs on `z - ret_norm` (normalised residual, `delta = 1.0` stays fixed). Gradient seed into `c_value` backward is the normalised Huber gradient.
- **Inference paths** — three callsites need denormalisation:
  - `ActorCritic::forward_critic` (single-sample bootstrap) — multiply raw output by `sigma`, add `mu`.
  - `ppo_act_all_cars_system` pass 3 — the `value` written to `PolicyOutput.value_prediction` and `push_pre_step(…, value, …)` must be denormalised (GAE consumes values in reward units).
  - On-exit flush path — same as bootstrap.

**Important:** the buffer's `values` field holds denormalised values (reward units). GAE returns `returns` in reward units. `explained_variance(&returns, &values)` stays valid because both are in the same units. The only point of normalisation is inside the training loss computation and the critic's raw linear output.

### 3. Observation running mean/var normaliser

Resource: `ObservationNormalizer { mu: [f32; OBSERVATION_DIM], m2: [f32; OBSERVATION_DIM], count: u64 }` — Welford online stats.

- **Where it lives:** a new Bevy `Resource` in `src/agent/observation.rs`.
- **Where it's updated:** inside `build_observation_vector_system`, after the raw-feature assembly and before writing to `observation.values`. Update stats per-car per-tick with Welford's algorithm.
- **Where it's applied:** same system. After stats update, normalise each dim: `(x - mu) / (sqrt(m2/count) + eps)`, clip to `[-10, 10]` per SB3 convention.
- **Warmup:** for the first 1000 samples (~2 s of 8-car training), pass observations through unchanged (stats are too noisy to normalise against). The `count >= 1000` check is cheap.
- **Persistence:** stats live in the resource across episodes — not reset on episode boundary. This is intentional; the full training run is a single distribution from the normaliser's point of view.

### 4. Target-KL early stop

New field on `PpoConfig`: `target_kl: Option<f32>` (default: `Some(0.03)`). NeuroDrive's reward scale is larger than MuJoCo's so 0.03 is a reasonable starting value; the SB3 convention is "early-stop when `approx_kl > 1.5 * target_kl`" which gives an effective threshold of 0.045.

Integration in `ppo_epoch_system`:

- Extract the KL aggregation from `ppo_finish_epoch` so it computes at end of every epoch (not only the final one).
- After an epoch completes, if `approx_kl_this_epoch > 1.5 * target_kl`, set `prepared.epochs_remaining = 0` (stops further epochs). Always call `ppo_finish_epoch(..., is_final=true)` on that epoch so stats get written.
- New stat field: `stats.epochs_actually_completed` (distinct from `config.ppo_epochs`) to surface early-stop events in analytics.

## Out of scope for this plan

These round-1 candidates are **deliberately excluded** from this batch:

- log_std weight decay (`ppo-tuning-knobs-racing.md` rec B.3) — ship only if round 2 shows entropy still drifting
- `ent_coef = 0.0` (rec B.2) — changes the exploration regime, confounds the measurement
- Throttle `a_mean` initial bias (rec A.2) — actor-side tweak, wait until the critic stabilises
- `log_std_ceil` tightening (rec B.4) — belt-and-braces, not yet needed
- Re-introducing brake (rec A.3) — only if critic fix leaves overshoot dominant
- LayerNorm on critic (ppo-critic rec C) — only if PopArt leaves `c_fc2` saturation > 40%
- Widen critic (ppo-critic rec E) — diagnosis says capacity is not the bottleneck
- ReLU/GELU critic (ppo-critic rec F) — research explicitly recommends against
- Extending lookahead > 650 units (observation-horizon verdict) — only relevant if cars sustain > 600 u/s

These remain documented in the references. If round 2 research surfaces them as needed, we implement them in a round-3 batch.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| PopArt rescale math wrong → critic's predictions silently drift | Medium | Unit test: after POP step, forward pass on a fixed input must return the same value as before the step |
| γ = 0.995 amplifies centreline reward bias → cars become slow centreline trackers | Low-medium | Watch chunk-1–3 mean speed; if < 200 u/s, revert γ first before touching reward coefs |
| Observation normalisation changes the meaning of saturation baselines | Certain | Expected; round 2 research will be told both configurations explicitly |
| Target-KL early stop fires on every update (LR too high for new reward scale) | Low-medium | Stat field `epochs_actually_completed` surfaces this; lower actor LR to 1e-4 if persistent |
| PopArt µ/σ becomes unstable during the growing-return phase | Low | β = 1e-4 per update is conservative; the EMA can track ~6× growth over 60 updates |

## Analytics requirements satisfied by `analytics-round2.md`

Round 2 needs to discriminate "fix worked" from "fix failed" at a glance. The analytics enhancements in the companion plan (`analytics-round2.md`) capture the specific signals each falsification criterion above needs — saturation timeseries, return distribution per update, PopArt µ/σ evolution, fleet per-car progress breakdown, critic prediction error distribution, pre-crash throttle-release analysis.

## Commit plan

One commit for this plan, after analytics-round2 has landed:

- Title: `ppo: round-2 intervention batch — γ=0.995, PopArt on c_value, obs normaliser, target-KL early stop`
- Body: reference the three research artefacts, list the four fixes, restate the falsification criteria.

## Success criterion for this plan

Plan is complete when:

- [x] All four changes compile and `cargo test` passes with existing tests
- [x] New tests added for PopArt POP invariance, observation normaliser, target-KL early-stop trigger
- [x] `context/systems/brain-ppo.md` updated to reflect the new architecture
- [x] `context/architecture.md` updated to mention PopArt and obs normalisation
- [x] One end-to-end training run (≥ 5k episodes) completes without panic or explained-variance collapse
- [x] Round-2 research agents have been dispatched with baseline + post-fix reports and the diff

User runs the training between steps 5 and 6.
