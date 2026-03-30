# Performance Tuning Lessons

## Core Constraint

The simulation must run at 60 Hz with 8 cars and feel smooth to watch. Performance isn't about hitting benchmarks — it's about the game being enjoyable to observe while cars learn.

## What We Learned (30 March 2026)

### Wider critic = more compute

Widening the critic from 64 to 128 hidden roughly doubled the PPO training cost (7ms → 13ms per chunk) and doubled the action selection cost (1.7ms → 3.3ms for 8 cars). The critic forward pass runs once per car per tick, so wider critics scale linearly with car count.

The split into `forward_actor`/`forward_critic` avoids running both networks when only one is needed, but during action selection both are still required (actor for actions, critic for value estimates stored in the rollout buffer for GAE).

### samples_per_tick is the stutter knob

Reducing samples_per_tick from 128 to 64 halves the per-tick training cost but doubles the number of ticks needed to complete an epoch. This trades peak stutter severity for more frequent but smaller overhead. Currently at 64, which is still causing noticeable choppiness with the 128-wide critic.

### The bimodal frame time pattern

Most ticks are fast (~4ms) because they only do simulation + action selection. A subset of ticks (~23%) also do PPO training and cost 50-60ms. This bimodal pattern is inherent to amortised PPO — the question is whether the training ticks are fast enough to stay within budget.

### Vec<Vec<f32>> was catastrophic

The original Vec<Vec<f32>> weight storage caused ~43× worse performance than theoretical due to cache misses. Switching to flat Vec<f32> row-major storage was the single biggest win. Always use contiguous memory for matrix data.

## Open Performance Problems

- The 128-wide critic is still too expensive for smooth 8-car training. Options: reduce to 96 hidden, further reduce samples_per_tick, or implement SIMD intrinsics for the matrix multiply hot path.
- Action selection forward_critic adds ~1.7ms per tick for 8 cars that could potentially be batched (run all 8 critic forwards as one batch) rather than 8 sequential single-sample forwards.
