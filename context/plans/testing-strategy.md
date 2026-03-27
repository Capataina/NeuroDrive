# Plan — Testing Strategy

## Goal

Establish a test suite that catches mathematical errors, contract violations, and integration regressions before they reach a training run. The current 31 tests mostly cover analytics sparklines, phase detection, and physics determinism — they don't verify the reward computation, neural network math, PPO update correctness, observation contract, or spawn logic.

Every time we change a reward term, activation function, or initialisation scheme, we should know immediately if the math is wrong rather than discovering it 20 minutes later in a confusing run report.

## What We're NOT Doing

- No mocking frameworks or heavy test infrastructure
- No end-to-end simulation tests (too slow, too fragile)
- No snapshot/golden-file tests for reports (format changes too often)
- No coverage targets or metrics — just the tests that actually catch bugs

---

## Test Categories

### 1. Unit Tests — Neural Network Primitives

**Location:** `src/brain/common/mlp.rs`, `src/brain/common/math.rs`, `src/brain/common/optim.rs`

These are pure functions with known mathematical properties. Easy to test, high value.

| Test | What it verifies |
|------|-----------------|
| `tanh_forward_output_range` | All outputs ∈ (-1, 1) for arbitrary inputs |
| `tanh_forward_zero_is_zero` | tanh(0) = 0 |
| `tanh_backward_gradient_check` | Numerical gradient matches analytical: `(tanh(x+ε) - tanh(x-ε)) / 2ε ≈ backward gradient` for several x values |
| `tanh_backward_at_zero` | Gradient at 0 is 1.0 |
| `linear_forward_shape` | Output dimension matches bias dimension |
| `linear_backward_gradient_check` | Numerical gradient vs analytical for a small (3→2) layer |
| `linear_zero_grad_clears_all` | After `zero_grad()`, all grad values are 0.0 |
| `orthogonal_init_columns_unit_norm` | Each column of the weight matrix has approximately unit norm (before scaling) |
| `orthogonal_init_columns_orthogonal` | Dot product between distinct columns ≈ 0 |
| `orthogonal_init_scale_applied` | Column norms ≈ scale value |
| `orthogonal_init_different_seeds_different_weights` | Two calls with different RNG states produce different matrices |
| `adam_step_reduces_loss_on_quadratic` | A few Adam steps on `w² - target` converge toward target |
| `adam_epsilon_is_1e5` | Verify the constant hasn't been accidentally changed |
| `glorot_uniform_variance_bound` | Weights within expected ±√(6/(fan_in+fan_out)) range |

### 2. Unit Tests — Reward Computation

**Location:** `src/game/episode.rs`

The reward logic is the most frequently changed code and the most consequential if wrong. These tests should use a helper that constructs minimal state and calls the reward computation directly.

| Test | What it verifies |
|------|-----------------|
| `speed_weighted_progress_zero_speed_zero_reward` | progress_delta > 0 but speed = 0 → reward = 0 |
| `speed_weighted_progress_zero_progress_zero_reward` | speed > 0 but progress_delta = 0 → reward = 0 |
| `speed_weighted_progress_positive_case` | Known delta × speed / reference × scale = expected value |
| `speed_weighted_progress_high_speed_multiplier` | Speed > reference → multiplier > 1.0 → reward > base |
| `backward_progress_gives_zero` | progress_delta < 0 → clamped to 0 → reward = 0 |
| `time_penalty_is_flat` | Every tick gets exactly `time_penalty_per_tick`, no heading component |
| `crash_penalty_magnitude` | Terminal reward on crash = crash_penalty exactly |
| `lap_bonus_magnitude` | Terminal reward on lap complete = lap_bonus exactly |
| `total_reward_is_sum_of_components` | tick_reward = progress_reward + time_penalty + terminal_reward |

### 3. Unit Tests — PPO Update Math

**Location:** `src/brain/a2c/update.rs`

The PPO math is subtle and easy to break. These tests verify the core gradient computation.

| Test | What it verifies |
|------|-----------------|
| `squashed_gaussian_log_prob_finite` | No NaN/Inf for reasonable inputs |
| `squashed_gaussian_log_prob_symmetry` | log_prob(x, mean) ≈ log_prob(-x, -mean) for steering (component 0) |
| `ppo_ratio_no_clip_when_small_change` | ratio ≈ 1.0 when new_log_prob ≈ old_log_prob → no clipping |
| `ppo_ratio_clips_when_large_change` | ratio far from 1.0 → clipped to [1-ε, 1+ε] |
| `advantage_normalisation_per_chunk` | Per-chunk mean ≈ 0, std ≈ 1 after normalisation |
| `shuffled_indices_are_permutation` | All indices present exactly once after shuffle |
| `saturated_tanh_detection` | Outputs > 0.99 counted as saturated, others not |
| `value_huber_loss_quadratic_region` | Error < δ → loss = 0.5 × error² |
| `value_huber_loss_linear_region` | Error > δ → loss = δ × (|error| - 0.5δ) |
| `gradient_clip_scales_correctly` | When norm > max, all grads scaled by max/norm |
| `gradient_clip_noop_when_small` | When norm ≤ max, grads unchanged |

### 4. Unit Tests — GAE Computation

**Location:** `src/brain/a2c/buffer.rs`

Already has 2 tests. Extend with:

| Test | What it verifies |
|------|-----------------|
| `gae_terminal_state_zero_bootstrap` | Done=true → no value bootstrap leaks through |
| `gae_advantages_not_globally_normalised` | Raw advantages are NOT zero-mean (since we removed global normalisation) |
| `gae_returns_equal_advantages_plus_values` | returns[i] = advantages[i] + values[i] for all i |
| `gae_single_step_episode` | One transition with done=true → advantage = reward - value |

### 5. Unit Tests — Observation Contract

**Location:** `src/agent/observation.rs`

| Test | What it verifies |
|------|-----------------|
| `observation_dim_matches_constant` | OBSERVATION_DIM = NUM_RAYS + 4 + NUM_LOOKAHEAD_SAMPLES × 2 |
| `ray_normalisation_in_range` | Ray features ∈ [0, 1] |
| `speed_normalisation_in_range` | Speed feature ∈ [0, 1] |
| `lateral_offset_normalisation_in_range` | Lateral offset ∈ [-1, 1] |
| `heading_error_normalisation_in_range` | Heading error ∈ [-1, 1] |
| `curvature_normalisation_in_range` | Curvature features ∈ [-1, 1] |

### 6. Unit Tests — Random Spawn

**Location:** `src/game/episode.rs` or `src/game/plugin.rs`

| Test | What it verifies |
|------|-----------------|
| `random_spawn_position_on_track` | Sampled position via `point_at_s` is within road boundaries |
| `random_spawn_heading_matches_tangent` | Spawn rotation = atan2(tangent.y, tangent.x) |
| `random_spawn_covers_full_track` | 100 samples cover at least 80% of the track length (within 10% buckets) |
| `car_zero_always_at_canonical_spawn` | Car 0 spawn position matches track.spawn_position |

### 7. Integration Tests — Reward + PPO Pipeline

These test that the reward flows correctly through the full pipeline: episode → buffer → GAE → PPO update. They don't run the full simulation but wire up the minimal components.

| Test | What it verifies |
|------|-----------------|
| `one_episode_produces_nonzero_gradients` | After one rollout + PPO update, at least some weight gradients are nonzero |
| `positive_advantage_increases_action_probability` | For a sample with positive advantage, the gradient direction would increase the action's log-prob |
| `reward_scale_affects_gradient_magnitude` | Doubling progress_reward_scale roughly doubles gradient norms |

### 8. Regression Tests — Things That Broke Before

| Test | What it verifies | Motivated by |
|------|-----------------|-------------|
| `no_dead_relu_fields_in_analytics` | No source file in `src/` contains "dead_relu" | ReLU→Tanh migration left stale references |
| `episode_config_crash_penalty_reasonable` | crash_penalty ∈ [-20, 0] | Accidentally set too high, caused car paralysis |
| `no_centerline_reward_in_episode` | episode.rs doesn't contain "centerline_reward_coef" | Centreline reward was farmable |

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
  └── Regression tests (category 8)

Medium value, medium effort
  ├── Random spawn tests (category 6)
  ├── Linear backward gradient check (category 1)
  └── Adam convergence test (category 1)

High value, higher effort (do later)
  └── Integration tests (category 7)
```

## Conventions

- Tests live next to the code they test (`#[cfg(test)] mod tests` in each file) — not in a separate `tests/` directory
- Test names describe the property being verified, not the method name: `speed_weighted_progress_zero_speed_zero_reward` not `test_progress_reward`
- Each test should be independent — no shared mutable state between tests
- Use `assert!((actual - expected).abs() < epsilon)` for float comparisons, with `epsilon = 1e-4` unless tighter precision is needed
- Gradient checks use `epsilon = 1e-3` for the finite-difference step and `tolerance = 1e-2` for the comparison (f32 precision limits)

## Open Questions

- Should we add a CI step? Currently tests are run manually. Even a simple `cargo test` pre-commit hook would catch most regressions.
- Should integration tests (category 7) use the real Bevy ECS or test the pure functions in isolation? Pure functions are faster and more stable but don't catch system wiring bugs.
- How do we handle tests for code that depends on `Track` entities? Either extract the pure logic into free functions (preferred) or use minimal Bevy test worlds.

## Status

Plan stage. No tests from this plan have been implemented yet. The existing 31 tests remain unchanged.
