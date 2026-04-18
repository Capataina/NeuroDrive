# Plan — Testing Strategy

## Status — 2026-04-18

**Most of this plan is implemented.** The suite went from 31 tests to 99 in the 2026-04-18 expansion, covering every category originally scoped except the reward-computation unit tests (which require Bevy ECS plumbing) and a long-running training-convergence integration test.

Remaining work is narrow and non-urgent — see the Remaining Work section at the bottom.

## What's Already in Place (2026-04-18)

### Unit tests

| File | Tests covering |
|------|----------------|
| `src/brain/common/mlp.rs` | Linear forward batch (shape, bias, handcrafted oracle, cache, finite-difference gradient check, zero_grad, L2 norm); Tanh forward/backward including gradient check and range bounds |
| `src/brain/common/math.rs` | `orthogonal_init` (row norms, scale, shape, seed variance); `normal_log_prob` (peak, symmetry, finiteness); `sample_normal` (empirical mean/std recovery over 5k samples, determinism); `normal_entropy` (positivity, monotonic in std) |
| `src/brain/common/optim.rs` | Adam convergence on quadratic loss; zero-grad noop; AdamW weight decay shrink; `weight_decay=0` equivalence to pure Adam |
| `src/brain/common/gemm_backend.rs` | Handcrafted 2×2 GEMM oracle; alpha/beta accumulate; sgemm_nt / sgemm_tn transposition semantics; overwrite-vs-accumulate; rectangular shapes; backend name validity |
| `src/brain/ppo/buffer.rs` | Original GAE tests retained; extensions for empty buffer, returns = adv + value identity, single-step episode, terminal-state bootstrap guard, `EnvGrouping` capacity preservation, deterministic iteration order |
| `src/brain/ppo/update.rs` | `squashed_gaussian_log_prob` finite + symmetric for steering; `clip_linear_gradients` scale/noop/zero-max; PPO ratio clip bounds; Huber loss quadratic/linear regions |
| `src/game/episode.rs` | `time_penalty_per_tick == 0.0` and `crash_penalty == 0.0` regression guards (policy-locked defaults per `notes/reward-and-entertainment.md`); other default-value guards; `push_with_limit` + `mean` helper behaviours |

### Integration tests (tests/*.rs, enabled by the [lib] target)

| File | Tests covering |
|------|----------------|
| `tests/gemm_correctness.rs` | Cross-validates whichever GEMM backend is active against an inline scalar reference for every matrix shape PPO uses in production — actor hidden SGEMM, forward_batch NT, backward weight-grad TN, critic training shape (128×128), alpha/beta accumulate, rectangular shapes. Tolerance 5e-3 accommodates f32 ULP drift between backend summation orders. |
| `tests/ppo_pipeline.rs` | Linear forward+backward produces finite non-zero gradients; Linear→Tanh chain is stable; Adam step after backward moves weights; varying batch sizes across calls don't corrupt state; Tanh preserves sign at non-zero inputs. |

### Verified across all backend variants

```
cargo test                                                   →  99 passed, 0 warnings
cargo test --no-default-features --features force-scalar     →  99 passed
cargo test --no-default-features --features force-matrixmultiply →  99 passed
cargo test --no-default-features --features force-accelerate →  99 passed
cargo check --release                                        →  clean
cargo check --features profiling                             →  clean
```

## Conventions in Use

- Unit tests live next to the code they test (`#[cfg(test)] mod tests` in each file) — not in a separate directory.
- Integration tests live in `tests/*.rs` and import the crate via `use neurodrive::...`.
- Test names describe the property being verified, not the method.
- Float comparisons use `assert!((actual - expected).abs() < epsilon)` with `epsilon = 1e-4` by default.
- Gradient checks use `epsilon = 1e-3` for the finite-difference step and `tolerance = 1e-2` for the comparison (f32 precision limits).

## Remaining Work

### Reward-computation unit tests (not yet written)

The reward logic inside `episode_loop_system` runs against Bevy ECS state (`Query`, `Res`, components). Pure unit tests would require extracting the reward computation into free functions that take plain arguments — a refactor, not a test. **Status:** deferred until the reward logic needs to change again, at which point the extract-then-test path becomes worth it.

Default-value regression guards ARE in place (`default_time_penalty_is_zero`, `default_crash_penalty_is_zero`, `default_reward_weights_match_documented_values`) — they catch the kind of policy-drift that actually occurred in the 2026-04-18 audit.

### Random-spawn unit tests (not yet written)

Same story — `spawn_car` is tied into Bevy ECS. Could be unit-tested after extracting the geometry calculation (sampled arc-length → spawn position + heading) into a free function. **Status:** deferred; the existing `deterministic_replay_same_seed_same_actions_identical_trajectory` test in `src/game/physics.rs` covers the most important spawn-path property indirectly.

### Long-running training-convergence integration test

Would run ~10k training steps and assert final policy quality (mean progress, crash rate) is within some tolerance of a known-good reference. Catches silent regressions in training behaviour — e.g. if someone changes a gradient-clipping threshold and the policy stops converging.

**Status:** not implemented. Cost: 30-60 seconds per run, which is too slow for regular `cargo test`. Would be marked `#[ignore]` and run manually or in a nightly CI pass.

### Cross-backend numerical-drift test over multiple optimiser steps

Would run N Adam steps on a synthetic loss with each backend, verify the final weight vectors are within ULP tolerance of each other. Deeper than the single-step correctness test in `tests/gemm_correctness.rs`. **Status:** not urgent — single-step equivalence is verified, and the rest of the test suite passing on all backends gives us strong empirical evidence that multi-step drift is within bounds.

## Guiding Principle

The test suite's job is to catch real regressions, not chase coverage metrics. Every test added in 2026-04-18 can be traced to a specific concrete risk: either the 2026-04-18 refactor broke something subtle (backend swap, GAE env_grouping capacity), or a prior drift mattered (time_penalty policy-lock regression guards), or a known failure mode wasn't previously guarded (Adam convergence on quadratic).

Future additions should pass the same test: "what concrete bug does this test prevent, and is that bug realistic given the code as it stands?"
