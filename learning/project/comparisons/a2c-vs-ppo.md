# Comparison: A2C vs PPO

## Overview

A2C (Advantage Actor-Critic) and PPO (Proximal Policy Optimisation) are both on-policy actor-critic algorithms. They are closely related, and PPO is widely considered the current default baseline for continuous control RL. NeuroDrive uses A2C. This file explains what the differences actually are, where they matter, and why A2C was the right choice for this project at this stage.

**Status:** A2C is implemented. PPO is not implemented. This is a design comparison, not an implementation guide.

## Prerequisites

- `concepts/core/policy-gradient-methods.md` — policy gradient, entropy
- `concepts/core/advantage-estimation.md` — GAE
- `concepts/core/actor-critic-architecture.md` — the shared model structure
- `project/decisions/a2c-as-baseline.md` — why A2C was chosen

---

## What A2C and PPO Share

Before examining differences, it is worth noting that A2C and PPO are more similar than different:

- Both are on-policy: they collect a rollout, use it for updates, discard it
- Both use GAE for advantage estimation
- Both have a policy (actor) and value function (critic)
- Both use entropy bonuses to encourage exploration
- Both clip or constrain gradients in some way
- Both update the same parameters for actor and critic (potentially with separate learning rates)

The fundamental algorithmic structure is the same. PPO adds a specific mechanism on top of this shared base.

---

## The Core Difference: Policy Update Constraints

### A2C Update

A2C performs a single gradient step on the policy gradient loss:

```
loss_policy = -mean( log_prob_t * A_norm_t )
```

No constraint is placed on how large this update can be. If the gradient is large (e.g. a single high-advantage action near the start of training), the weights can change dramatically in one step.

### PPO's Clipped Objective

PPO introduces a **probability ratio** between the current policy and the old policy (the policy that collected the rollout):

```
r_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

And clips this ratio in the objective:

```
loss_ppo = -mean( min( r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t ) )
```

With `ε ≈ 0.2`, the policy is not allowed to change so much that the new probabilities are more than 20% higher or lower than the old probabilities. If the ratio goes outside `[0.8, 1.2]`, the clipped objective stops contributing gradient for that sample.

This constraint is called the **trust region** — the policy update stays "close" to the policy that collected the data, preventing large destabilising jumps.

### Why Does PPO Need a Trust Region?

The policy gradient theorem assumes that the gradient estimate is valid for the current policy. If the update moves the policy far from where the data was collected, the gradient estimate becomes stale — it was computed under `π_old` but is being used to update `π_new`. Large updates amplify this off-policy error.

In practice, without a trust region, policy gradient updates can occasionally produce a catastrophically bad update (a large step in a wrong direction) that collapses the policy. Recovery from such collapses is slow. PPO's clipping prevents the worst of these.

---

## Practical Performance Comparison

### When Does PPO Outperform A2C?

PPO's advantage is most pronounced when:
1. **Long rollouts** — with long episodes, the data collected is used for a mini-batch update multiple times. PPO's clipping keeps the importance ratio valid across multiple passes through the same data.
2. **Multiple update epochs** — PPO typically updates for 4–10 epochs per rollout. A2C typically does one pass. Multiple epochs make better use of collected data.
3. **Unstable training dynamics** — environments where a bad update can cause a catastrophic policy collapse benefit more from PPO's stability guarantee.

### When is the Difference Small?

The difference between A2C and PPO shrinks when:
- Episodes are short and rollouts are collected frequently
- Only one update epoch is done per rollout (removing PPO's main efficiency advantage)
- The gradient is naturally well-behaved (small, smooth)
- The environment is simple enough that collapsed policies recover quickly

NeuroDrive's current regime — short episodes (60-second timeout, often much shorter on crashes), single-car rollout collection, low-dimensional observations — is not the regime where PPO's trust region provides the most benefit.

---

## Implementation Complexity Comparison

For a **handwritten implementation from scratch in Rust**:

| Component | A2C | PPO |
|---|---|---|
| Rollout collection | ✓ Identical | ✓ Identical |
| GAE computation | ✓ Identical | ✓ Identical |
| Policy loss | Single term: `-logprob * A` | Clipped ratio + additional bookkeeping |
| Old log-probs | Not needed | Must be stored in rollout buffer |
| Ratio computation | Not needed | `exp(new_logprob - old_logprob)` |
| Clip threshold | Not needed | Additional hyperparameter ε |
| Multiple epochs | Not needed | Inner loop over rollout data |

PPO is not *much* more complex than A2C, but each additional component is another place where a handwritten implementation can go wrong. The old log-probs must be computed and stored correctly. The ratio must be computed numerically stably. The clipping must be applied symmetrically. The multi-epoch inner loop must reshuffle data correctly.

For a baseline whose primary job is to validate the environment contract, the simpler implementation is more appropriate. A bug in the PPO clip logic would introduce training instability that might look like an environment problem.

---

## When Should NeuroDrive Upgrade to PPO?

If A2C successfully validates the environment but then hits a performance ceiling that prevents further progress, upgrading to PPO is a natural next step. The upgrade would be purely inside `brain/a2c/` — the environment, agent, and analytics layers would be unchanged.

Signals that PPO might help:
- A2C learning becomes unstable after initial progress (policy collapses, then slowly recovers)
- Sample efficiency is the bottleneck and multiple epochs per rollout would help
- Hyperparameter sensitivity is high (PPO's trust region reduces sensitivity to learning rate choices)

PPO should not be added speculatively — it would complicate the implementation for uncertain benefit while the current priority is validating the environment contract.

---

## Brief Notes on Other Algorithms

For completeness, two other common choices in continuous control:

### SAC (Soft Actor-Critic)

**Type:** Off-policy, maximum-entropy.

SAC uses a replay buffer and trains on past experience. It is generally more sample-efficient than A2C or PPO. However:
- Off-policy requires a replay buffer (much more infrastructure)
- SAC has automatic entropy tuning (another moving part)
- The interaction model (asynchronous collection from buffer) does not fit the fixed-tick single-car model naturally

SAC would be the right choice if sample efficiency were the primary concern and implementation complexity were acceptable. It is not appropriate as a first validation baseline.

### TD3 (Twin Delayed DDPG)

**Type:** Off-policy, deterministic.

TD3 uses a deterministic policy (not stochastic) and two critic networks to reduce overestimation bias. Less commonly used now that SAC exists. Deterministic policies cannot naturally express exploration as part of the policy itself, which is a disadvantage for the early-training stage.

---

## Summary Table

| Dimension | A2C | PPO | SAC |
|---|---|---|---|
| Policy type | Stochastic | Stochastic | Stochastic |
| On/off-policy | On-policy | On-policy | Off-policy |
| Trust region | No | Yes (clipping) | N/A |
| Multiple epochs per rollout | No | Yes | N/A |
| Implementation complexity | Low | Moderate | High |
| Sample efficiency | Low | Low-moderate | High |
| Stability | Moderate | High | High |
| Appropriate for handwritten validation | **Yes** | Possible | No |

---

## Related Files

- `project/decisions/a2c-as-baseline.md` — the full reasoning for choosing A2C
- `project/systems/a2c-brain.md` — the implementation
- `concepts/core/policy-gradient-methods.md` — the shared foundation
- `concepts/core/advantage-estimation.md` — GAE, shared by both A2C and PPO
