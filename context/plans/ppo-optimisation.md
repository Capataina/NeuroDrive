# Plan — PPO Optimisation and Improvement

## Status — Phases 0–1 Complete, Phase 2 Partially Complete, Phase 3 Partially Complete

Phases 0 and 1 are fully complete. Phase 2 has key items done. Phase 3.3 (more envs) is done — car count bumped to 8. Remaining items are valid future work.

| Item | Status | Notes |
|------|--------|-------|
| Phase 0.1: ReLU → Tanh | **Done** | |
| Phase 0.2: Per-tick progress reward | **Done** | Superseded by velocity projection reward (`dot(velocity, tangent)`) in the reward-and-spawn-overhaul. |
| Phase 1.1: Orthogonal init + output head scaling | **Done** | |
| Phase 1.2: Minibatch shuffle + per-minibatch adv norm | **Done** | |
| Phase 1.3: Centreline proximity reward | **Done — re-added** | Originally removed as exploitable, then re-added with `coef = 0.3` alongside velocity projection reward. |
| Phase 2.1: Running observation normalisation | Not started | |
| Phase 2.2: Linear LR annealing | Not started | |
| Phase 2.3: Adam ε → 1e-5 | **Done** | |
| Phase 2.4: Increase crash penalty | **Superseded** | Crash penalty now set to `0.0` — the velocity projection reward provides sufficient signal without penalising crashes. |
| Phase 2.5: Extract log-std Adam into shared optimiser | Not started | The `ppo_finish_epoch` function has 14 lines of inlined Adam for `a_log_std` that duplicates `AdamOptimizer::step()`. Extend `AdamOptimizer` to handle scalar params, or extract a helper. Low priority but prevents divergence if hyperparams change. |
| Phase 3.3: More vectorised environments | **Done** | Car count bumped to 8 |
| Phase 3: Remaining items | Not started | |

The reward structure has been fundamentally redesigned in `context/plans/reward-and-spawn-overhaul.md`. The finish-line removal and analytics overhaul are both complete. Remaining Phase 2–3 items (observation normalisation, LR annealing, more envs, value clipping) remain valid future work under the new reward paradigm.

---

## Goal

Make the PPO baseline capable of completing at least one full lap of the track. The current implementation plateaus at 11–15% progress with 100% crash rate due to two root causes: dead ReLU neurons starving the actor of capacity, and a reward structure that stops providing gradient signal once the car hits the same crash point repeatedly.

This plan is grounded in the research paper at `context/references/ppo-optimisation.md` and verified against the current codebase.

## Success Criteria

- At least one car achieves 100% progress (cumulative forward arc-length equivalent to track length)
- Dead neuron rate drops below 5% (tanh effectively eliminates this)
- Average progress across a 100-episode window exceeds 50%
- Crash rate in a 100-episode window drops below 80%

---

## Phase 0 — Critical Unblock (P0)

These two changes address the root causes directly. They should be implemented together and tested before anything else.

### 0.1 Switch ReLU → Tanh

**Files:** `src/brain/common/mlp.rs`, `src/brain/a2c/model.rs`

**Steps:**

1. Add a `Tanh` activation struct in `mlp.rs`:
   - `forward(input) → input.iter().map(|x| x.tanh()).collect()`
   - `backward(grad_output) → zip(cache, grad).map(|(x, g)| g * (1.0 - x.tanh().powi(2)))`
   - Cache the input (same pattern as `Relu`)
2. In `model.rs`, replace all four `Relu::new()` instances with `Tanh::new()`:
   - `a_relu1`, `a_relu2` → `a_tanh1`, `a_tanh2`
   - `c_relu1`, `c_relu2` → `c_tanh1`, `c_tanh2`
3. Update field names throughout `model.rs` (forward, zero_grad references)
4. Update dead neuron tracking in `update.rs` — tanh doesn't have "dead" neurons in the ReLU sense. Replace with a saturation metric: fraction of activations where `|output| > 0.99` (saturated tanh). This is the tanh equivalent of a health diagnostic.
5. Update `A2cLayerHealth` field name from `dead_relu_fraction` to something activation-agnostic like `activation_health_fraction`, or keep it and document the semantic change.

**Verification:** Run for ~200 episodes. Confirm:
- No dead neuron reports (or near-zero saturation)
- Progress should already show improvement from restored capacity

### 0.2 Per-Tick Progress Reward

**Files:** `src/game/episode.rs`

**Steps:**

1. In `episode_loop_system`, change the progress gain calculation from:
   ```rust
   let progress_gain = (progress.fraction - previous_best_progress).max(0.0);
   ```
   to:
   ```rust
   let progress_gain = (progress.fraction - episode_state.previous_progress_fraction).max(0.0);
   ```
   This rewards any forward movement this tick, not just exceeding the episode's all-time best.

2. Keep `current_best_progress_fraction` tracking for analytics — it's still useful for reporting, just no longer used for reward.

3. Consider whether `progress_reward_scale` (currently 140.0) needs adjustment. Per-tick progress will produce many small rewards rather than occasional larger ones. The total reward per episode may increase substantially. Start with the same scale and observe — if returns become very large, reduce the scale to ~80–100.

**Verification:** Run for ~200 episodes. Confirm:
- Episodes where the car doesn't beat its best still show positive progress reward
- The reward plateau at chunk 4–5 disappears
- Combined with tanh, max progress should break past the 15% ceiling

### Phase 0 Checkpoint

After both 0.1 and 0.2, run a full 500+ episode session and compare against the baseline run (run_1774569725). Key metrics to compare:

| Metric | Baseline | Target |
|--------|----------|--------|
| Max progress | 15.7% | >25% |
| Avg progress (last 100 eps) | ~13% | >18% |
| Dead ReLU / saturation | 36–45% | <5% |
| Crash rate | 100% | Still high but with more variation |
| Dominant crash sector | Sector 3 only | More distributed |

If Phase 0 alone doesn't break past 25%, proceed to Phase 1. If it does, skip to Phase 2 (stability) and add Phase 1 items opportunistically.

---

## Phase 1 — Training Signal Quality (P1)

These changes improve the quality of the gradient signal the policy receives. Implement in order.

### 1.1 Orthogonal Initialisation with Output Head Scaling

**Files:** `src/brain/common/mlp.rs`, `src/brain/common/math.rs`, `src/brain/a2c/model.rs`

**Steps:**

1. Add an `orthogonal_init(rows, cols, scale, rng)` function in `math.rs`:
   - Generate a random matrix
   - Compute QR decomposition (or use the iterative Gram-Schmidt approach for simplicity)
   - Multiply by `scale`
   - For a simple 64×23 or 64×64 matrix, a good approximation: generate random Gaussian matrix, iterate Gram-Schmidt orthogonalisation a few times, then scale
2. In `ActorCritic::new`, use orthogonal init with:
   - Hidden layers (`a_fc1`, `a_fc2`, `c_fc1`, `c_fc2`): scale = √2 (compensates for tanh's output range)
   - Actor mean output (`a_mean`): scale = 0.01 (near-zero initial policy → uniform exploration)
   - Critic value output (`c_value`): scale = 1.0

**Why the output head scaling matters:** With Glorot init, the `a_mean` layer produces non-trivial initial outputs, which biases the initial policy toward specific actions. With 0.01× scale, initial outputs are near zero, and `tanh(near_zero) ≈ near_zero`, so the initial policy is roughly centred and exploratory.

### 1.2 Minibatch Shuffling and Per-Minibatch Advantage Normalisation

**Files:** `src/brain/a2c/update.rs`, `src/brain/a2c/buffer.rs`

**Steps:**

1. Add a `shuffled_indices: Vec<usize>` field to `PreparedUpdate`
2. In `ppo_prepare_update`, populate it with `(0..n).collect()` then shuffle using the brain's RNG
3. At the start of each epoch (`sample_offset == 0`), re-shuffle the indices
4. In `ppo_process_chunk`, index into the buffer through `shuffled_indices[i]` instead of `i` directly
5. Remove the global advantage normalisation from `compute_gae_per_env`
6. At the start of each chunk in `ppo_process_chunk`, compute the mean and std of advantages for the current chunk's indices, and normalise on the fly:
   ```
   let chunk_indices = &prepared.shuffled_indices[offset..end];
   let chunk_adv_mean = chunk_indices.iter().map(|&i| advantages[i]).sum::<f32>() / chunk_size;
   let chunk_adv_std = ...;
   // Then when using: let adv = (advantages[idx] - chunk_adv_mean) / (chunk_adv_std + 1e-8);
   ```

### 1.3 Centerline Proximity Reward

**Files:** `src/game/episode.rs`

**Steps:**

1. Add to `EpisodeConfig`:
   ```rust
   pub centerline_reward_coef: f32,    // e.g. 0.02
   pub centerline_reward_max_dist: f32, // e.g. 50.0 (world units)
   ```
2. In `episode_loop_system`, after computing `progress.distance`, add:
   ```rust
   let cl_ratio = (progress.distance / config.centerline_reward_max_dist).min(1.0);
   let centerline_reward = config.centerline_reward_coef * (1.0 - cl_ratio * cl_ratio);
   ```
3. Add `centerline_reward` to the tick reward sum
4. Add tracking fields to `EpisodeState` for analytics: `current_tick_centerline_reward`, `current_centerline_reward_sum`
5. Update analytics capture to include the new reward component

**Tuning note:** Start with `coef = 0.02`. This is small relative to progress reward but provides continuous signal even when progress is zero. Increase if the car still doesn't track the line; decrease if it becomes too conservative (hugging centreline at the expense of progress).

---

## Phase 2 — Optimisation Stability (P2)

### 2.1 Running Observation Normalisation

**Files:** new `src/agent/normalisation.rs`, `src/agent/observation.rs`, `src/agent/mod.rs`

**Steps:**

1. Create a `RunningMeanStd` resource with:
   - `mean: [f32; OBSERVATION_DIM]`
   - `var: [f32; OBSERVATION_DIM]`
   - `count: f64`
   - `update(batch: &[[f32; OBSERVATION_DIM]])` — Welford's online algorithm
   - `normalise(obs: &[f32]) -> [f32; OBSERVATION_DIM]` — `(x - mean) / sqrt(var + 1e-8)`, clipped to `[-10, 10]`
2. Initialise with the current static scaling ranges as starting estimates (so the first few hundred ticks aren't wildly different)
3. In `build_observation_vector_system`, after building the raw normalised vector, apply the running normalisation
4. Add a warmup flag: for the first 1000 ticks, only accumulate statistics without applying normalisation (use static scaling during warmup)

### 2.2 Linear LR Annealing

**Files:** `src/brain/a2c/mod.rs`, `src/brain/common/optim.rs`

**Steps:**

1. Add to `A2cBrain`:
   ```rust
   pub total_timesteps: usize,  // e.g. 500_000
   pub timesteps_elapsed: usize,
   ```
2. In `a2c_collect_rewards_all_cars_system`, after incrementing step counter, compute the LR fraction:
   ```rust
   let frac = 1.0 - (brain.timesteps_elapsed as f32 / brain.total_timesteps as f32).min(1.0);
   ```
3. Either: update the optimizer's LR directly before each `step()` call, or pass the fraction as a multiplier. The simplest approach is to add a `set_learning_rate(lr)` method to `AdamOptimizer` and call it before each epoch's `ppo_finish_epoch`.

### 2.3 Adam ε → 1e-5

**Files:** `src/brain/common/optim.rs`

**Change:** `epsilon: 1e-8` → `epsilon: 1e-5` in `AdamOptimizer::new` default.

### 2.4 Increase Crash Penalty

**Files:** `src/game/episode.rs`

**Change:** `crash_penalty: -5.0` → `crash_penalty: -10.0` in `EpisodeConfig::default()`.

**Rationale:** At 13% progress with scale 140, the episode earns ~18 progress reward. A -5 crash penalty is only 28% of this. At -10, it's 55% — much more salient.

---

## Phase 3 — Scale and Polish (P3)

Only pursue after Phases 0–2 have been tested and the car is making meaningful progress past the first corner.

### 3.1 Reward/Return Normalisation

Add running standard deviation tracking for discounted returns. Divide rewards by `running_std(returns)` (without subtracting mean). This auto-scales the value function's target range.

### 3.2 Value Function Clipping

Clip value predictions to `[V_old - clip_ε, V_old + clip_ε]` during PPO updates, mirroring the policy clip. Use the same `clip_epsilon = 0.2`.

### 3.3 More Vectorised Environments

Increase `num_envs` from 3 to 8–16. This requires verifying spawn position spacing and potentially adjusting `spawn_lateral_spread`. More envs = more diverse transitions per rollout = better gradients.

### 3.4 Rollout Horizon Tuning

If per-tick progress reward makes episodes much shorter or longer, the `max_steps = 512` horizon may need adjustment. Monitor average episode length and ensure the buffer typically contains 2–4 complete episodes before updating.

---

## Implementation Order Summary

```text
Phase 0 (critical, do first)
  ├── 0.1  ReLU → Tanh
  ├── 0.2  Per-tick progress reward
  └── ── CHECKPOINT: 500-episode test run ──

Phase 1 (signal quality)
  ├── 1.1  Orthogonal init + head scaling
  ├── 1.2  Minibatch shuffle + per-minibatch adv norm
  ├── 1.3  Centerline proximity reward
  └── ── CHECKPOINT: 500-episode test run ──

Phase 2 (stability)
  ├── 2.1  Running observation normalisation
  ├── 2.2  Linear LR annealing
  ├── 2.3  Adam ε → 1e-5
  ├── 2.4  Increase crash penalty
  └── ── CHECKPOINT: 500-episode test run ──

Phase 3 (scale — only if needed)
  ├── 3.1  Reward/return normalisation
  ├── 3.2  Value function clipping
  ├── 3.3  More vectorised envs
  └── 3.4  Horizon tuning
```

## Open Questions

- Should `progress_reward_scale` be reduced when switching to per-tick progress? Start at 140, but if total episode returns exceed ~50, consider reducing to 80–100.
- Is `rotation_speed = 4.0` physically sufficient for the track's corners at learned speeds? If the car still can't corner after Phase 0+1, investigate the physics constraint before adding more PPO complexity.
- Should the centerline reward replace or supplement the heading-speed penalty? Start by supplementing — remove only if the two signals conflict.

## Reference

- Research paper: `context/references/ppo-optimisation.md`
- Architecture: `context/architecture.md`
- Baseline run: `reports/run_1774569725.md` (681 episodes, 15.7% max progress, 100% crash rate)
