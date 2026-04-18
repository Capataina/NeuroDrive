# Plan — Testing Strategy

## Goal

Establish a test suite that catches mathematical errors, contract violations, and integration regressions before they reach a training run. The existing 31 tests cover analytics sparklines, phase detection, physics determinism, GAE correctness, and HUD assessment — they do not verify the reward computation, neural network math, PPO update correctness, observation contract, or random-spawn logic.

Every time we change a reward term, activation function, or initialisation scheme, we should know immediately if the math is wrong rather than discovering it 20 minutes into a confusing run.

The 2026-04-18 audit added a `[lib]` target to the crate — integration tests under `tests/*.rs` can now `use neurodrive::brain::...`, which unblocks the higher-leverage test categories below (category 7 especially).

## What We're NOT Doing

- No mocking frameworks or heavy test infrastructure
- No end-to-end simulation tests (too slow, too fragile)
- No snapshot/golden-file tests for reports (format changes too often)
- No coverage targets — just the tests that actually catch bugs

---

## Test Categories

### 1. Unit Tests — Neural Network Primitives

**Location:** `src/brain/common/mlp.rs`, `src/brain/common/math.rs`, `src/brain/common/optim.rs`

Pure functions with known mathematical properties. Easy to test, high value.

| Test | What it verifies |
|------|-----------------|
| `tanh_forward_output_range` | Outputs ∈ (−1, 1) for arbitrary inputs |
| `tanh_forward_zero_is_zero` | tanh(0) = 0 |
| `tanh_backward_gradient_check` | Numerical ≈ analytical gradient at several x |
| `linear_forward_into_shape_and_zero` | `forward_into` writes `out_dim` values, with zero input producing bias |
| `linear_backward_batch_gradient_check` | Numerical vs analytical for a small (3→2) layer |
| `linear_zero_grad_clears_all` | After `zero_grad()`, all grad values are 0.0 |
| `orthogonal_init_columns_unit_norm` | Column norms ≈ 1 before scale |
| `orthogonal_init_columns_orthogonal` | Dot products between distinct columns ≈ 0 |
| `orthogonal_init_scale_applied` | Column norms ≈ scale |
| `adam_step_reduces_loss_on_quadratic` | Several Adam steps converge toward target on `(w − target)²` |

### 2. Unit Tests — Reward Computation

**Location:** `src/game/episode.rs`

The reward logic is the most frequently changed code and the most consequential if wrong. Use a helper that constructs minimal state and calls the reward computation directly.

| Test | What it verifies |
|------|-----------------|
| `velocity_projection_aligned_with_tangent_is_max` | `dot(velocity, tangent) / speed_reference × scale` hits expected peak when velocity is parallel to the tangent at `speed_reference` |
| `velocity_projection_perpendicular_is_zero` | Velocity perpendicular to tangent → projection = 0 → reward component = 0 |
| `velocity_projection_reverse_is_negative` | Moving backward along the tangent → negative reward (no clamp to 0) |
| `centreline_reward_at_zero_distance_is_coef` | On-centreline → reward = `centreline_reward_coef` |
| `centreline_reward_at_max_distance_is_zero` | At `centreline_reward_max_distance` → reward = 0 |
| `centreline_reward_monotone_decreasing` | Further from centreline → smaller reward |
| `time_penalty_default_is_zero` | Regression guard — `time_penalty_per_tick == 0.0` in `EpisodeConfig::default()` |
| `crash_penalty_default_is_zero` | Regression guard — `crash_penalty == 0.0` in `EpisodeConfig::default()` |
| `total_reward_is_sum_of_components` | `tick.reward == velocity_projection + centreline + time_penalty + terminal` |

### 3. Unit Tests — PPO Update Math

**Location:** `src/brain/ppo/update.rs`

| Test | What it verifies |
|------|-----------------|
| `squashed_gaussian_log_prob_finite` | No NaN/Inf for reasonable inputs |
| `squashed_gaussian_log_prob_symmetry` | log_prob(x, mean) ≈ log_prob(−x, −mean) for steering component |
| `ppo_ratio_no_clip_when_small_change` | ratio ≈ 1 when new ≈ old log_prob |
| `ppo_ratio_clips_when_large_change` | Far-from-1 ratio gets clipped to [1−ε, 1+ε] |
| `advantage_normalisation_per_chunk` | Per-chunk mean ≈ 0, std ≈ 1 after normalisation |
| `shuffled_indices_are_permutation` | Every index present exactly once after Fisher-Yates |
| `saturated_tanh_detection` | Outputs with |value| > 0.99 counted as saturated |
| `value_huber_loss_quadratic_region` | Error < δ → loss = 0.5 × error² |
| `value_huber_loss_linear_region` | Error > δ → loss = δ × (|error| − 0.5δ) |
| `gradient_clip_scales_correctly` | norm > max → all grads scaled by max/norm |
| `gradient_clip_noop_when_small` | norm ≤ max → grads unchanged |

### 4. Unit Tests — GAE Computation

**Location:** `src/brain/ppo/buffer.rs`

Already has 2 tests. Extend with:

| Test | What it verifies |
|------|-----------------|
| `gae_terminal_state_zero_bootstrap` | `done == true` → no value bootstrap leaks through |
| `gae_returns_equal_advantages_plus_values` | `returns[i] == advantages[i] + values[i]` |
| `gae_single_step_episode` | One transition with done → advantage = reward − value |
| `env_grouping_capacity_reused_across_calls` | Buckets' capacity does not regrow when `compute_gae_per_env` runs twice in sequence |

### 5. Unit Tests — Observation Contract

**Location:** `src/agent/observation.rs`

| Test | What it verifies |
|------|-----------------|
| `observation_dim_matches_constant` | `OBSERVATION_DIM = 43` and matches the field-count sum |
| `ray_normalisation_in_range` | Ray features ∈ [0, 1] |
| `v_forward_v_lateral_normalisation` | Both components ∈ [−1, 1] (using `speed_reward_reference` as the normaliser) |
| `lateral_offset_normalisation_in_range` | Centreline lateral offset ∈ [−1, 1] |
| `heading_error_normalisation_in_range` | Heading error ∈ [−1, 1] |
| `previous_action_recorded` | `previous_steering` / `previous_throttle` propagated correctly between ticks |

### 6. Unit Tests — Random Spawn

**Location:** `src/game/plugin.rs`, `src/game/episode.rs`

| Test | What it verifies |
|------|-----------------|
| `random_spawn_position_on_track` | Sampled position via `point_at_s` is inside the drivable grid |
| `random_spawn_heading_matches_tangent` | Spawn rotation = `atan2(tangent.y, tangent.x)` |
| `random_spawn_covers_full_track` | 100 samples cover at least 80% of the track length in 10%-buckets |

### 7. Integration Tests — Reward + PPO Pipeline

Now unblocked by the `[lib]` target (`tests/*.rs` can import `neurodrive::brain::...`).

| Test | What it verifies |
|------|-----------------|
| `one_episode_produces_nonzero_gradients` | After one rollout + PPO update, at least some weight gradients are non-zero |
| `positive_advantage_increases_action_probability` | Sample with positive advantage → gradient direction increases log-prob |
| `reward_scale_affects_gradient_magnitude` | Doubling `velocity_reward_scale` roughly doubles gradient norms |
| `forward_actor_allocation_free` | Instrumented allocator sees 0 heap allocs across 1000 `forward_actor` calls (2026-04-18 `SampleScratch` regression guard) |

### 8. Regression Guards

| Test | Motivated by |
|------|-------------|
| `no_dead_relu_fields_in_analytics` | ReLU → tanh migration left stale references in the past |
| `no_unsafe_in_ppo_update` | 2026-04-18 audit removed three `unsafe` blocks via borrow split; regression guard prevents reintroduction |
| `episode_config_defaults_match_reward_philosophy` | `time_penalty_per_tick == 0.0` and `crash_penalty == 0.0` are policy-locked per `notes/reward-and-entertainment.md` |

---

## Implementation Priority

```text
High value, low effort (do first)
  ├── Reward computation unit tests (category 2)
  ├── Tanh forward/backward gradient checks (category 1)
  ├── PPO ratio and clipping tests (category 3)
  └── GAE extensions (category 4)

Medium value, low effort
  ├── Orthogonal init property tests (category 1)
  ├── Observation range tests (category 5)
  └── Regression guards (category 8)

Medium value, medium effort
  ├── Random spawn tests (category 6)
  └── Linear backward gradient check (category 1)

High value, higher effort (do later)
  └── Integration tests (category 7)
```

## Conventions

- Unit tests live next to the code they test (`#[cfg(test)] mod tests`) — not in a separate directory
- Integration tests live in `tests/*.rs` (newly possible since the 2026-04-18 `[lib]` target)
- Test names describe the property being verified, not the method
- Use `assert!((actual - expected).abs() < epsilon)` for float comparisons; `epsilon = 1e-4` unless tighter precision is needed
- Gradient checks use `epsilon = 1e-3` for the finite-difference step and `tolerance = 1e-2` for the comparison (f32 precision limits)

## Status

Plan stage. None of the tests in this plan have been implemented. The existing 31 tests remain unchanged.
