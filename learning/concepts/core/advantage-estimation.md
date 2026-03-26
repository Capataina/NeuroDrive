# Advantage Estimation

## Why This Matters Here

GAE (Generalised Advantage Estimation) is the central calculation in every A2C update NeuroDrive performs. The `compute_gae` function in `src/brain/a2c/buffer.rs` computes the advantages used in the policy gradient and the returns used in the critic's value loss. If you understand GAE deeply, you understand the core of the update step.

Getting GAE wrong is one of the most common sources of subtle A2C bugs. This file builds the concept from first principles and works through a concrete numerical example.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — returns, discount, Bellman equations, advantage function
- `concepts/core/policy-gradient-methods.md` — why advantages reduce variance

## Notation

| Symbol | Meaning |
|---|---|
| `r_t` | Reward at timestep `t` |
| `V_t` | Critic's value estimate at state `s_t` |
| `γ` | Discount factor |
| `λ` | GAE lambda (controls bias-variance trade-off) |
| `δ_t` | TD error at timestep `t` |
| `Â_t` | GAE advantage estimate |
| `R_t` | Return target for the critic (value target) |
| `done_t` | Terminal flag: 1 if episode ended at `t`, 0 otherwise |

---

## The Problem: High-Variance Returns

In REINFORCE, the advantage is estimated as:

```
A_t ≈ G_t - V(s_t)
```

where `G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...` is the actual Monte Carlo return from `t` onwards.

The problem: `G_t` is noisy. Every single future reward from `t` to the end of the episode contributes, and each reward has randomness from all the stochastic action choices after `t`. For a 30-second episode at 60 Hz, `G_t` at the first timestep sums ~1800 future rewards.

Even if the policy is good, the variance of this sum is enormous. High-variance gradients slow learning and destabilise training.

---

## The One-Step TD Error

At the other extreme, we can estimate the advantage using only one step of information:

```
δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
```

This is the **temporal difference (TD) error**. It measures "how much better was this step than the critic expected?"

- If `δ_t > 0`: the step was better than expected
- If `δ_t < 0`: the step was worse than expected
- `(1 - done_t)` is the terminal mask: no future value if the episode ended

**Low variance but high bias:** the one-step estimate depends only on one reward and one critic prediction. The variance is low. But if the critic is wrong (biased), the advantage estimate inherits that bias.

---

## GAE: Interpolating Between Monte Carlo and TD

**Generalised Advantage Estimation** (Schulman et al., 2016) introduces a parameter `λ ∈ [0, 1]` that blends between:
- `λ = 0`: pure one-step TD (low variance, high critic-bias)
- `λ = 1`: pure Monte Carlo (high variance, unbiased if not bootstrapping)

The GAE formula is built as an exponentially weighted sum of n-step TD errors:

First define the k-step advantage:

```
Â_t^(1) = δ_t                                           (1-step)
Â_t^(2) = δ_t + γ * δ_{t+1}                            (2-step)
Â_t^(k) = Σ_{l=0}^{k-1} (γ)^l * δ_{t+l}               (k-step)
```

GAE is the λ-weighted blend:

```
Â_t^GAE = (1-λ) * Σ_{k=1}^∞  λ^{k-1} * Â_t^(k)
```

After algebraic simplification, this reduces to the elegant recurrence:

```
Â_t = δ_t + γ * λ * (1 - done_t) * Â_{t+1}
```

This is computed backwards from the end of the rollout.

---

## The Recurrence

Starting from the last timestep `T`:

```
Â_T = δ_T   (no future GAE available)
Â_{T-1} = δ_{T-1} + γ * λ * (1 - done_{T-1}) * Â_T
Â_{T-2} = δ_{T-2} + γ * λ * (1 - done_{T-2}) * Â_{T-1}
...
```

The terminal mask `(1 - done_t)` is crucial: it zeroes out the future GAE contribution when an episode terminates at `t`. This is what prevents advantage estimates from spanning across episode boundaries.

---

## Value Targets (Returns)

GAE also produces value targets for the critic:

```
R_t = Â_t + V(s_t)
```

This is the "return" target — the critic is trained to minimise `(V(s_t) - R_t)²`. This target blends Monte Carlo returns and bootstrapped values according to `λ`.

---

## The Bootstrap at the End of a Rollout

NeuroDrive collects rollouts of a fixed horizon (512 steps or when a terminal state is reached). If the rollout ends before an episode terminates, the last state is not terminal, and we need to bootstrap:

```
next_value = V̂(s_{T+1})    (critic's estimate of the state after the rollout ends)
```

This bootstrap value is used as the "future value" for the last step's TD error:

```
δ_T = r_T + γ * next_value * (1 - done_T) - V_T
```

If the rollout ends at a terminal state (`done_T = true`), `next_value = 0`.

In `src/brain/a2c/buffer.rs`:

```rust
let next_val = if t + 1 < self.rewards.len() {
    self.values[t + 1]         // use stored critic value
} else {
    next_value                 // bootstrap from caller
};
let mask = if self.dones[t] { 0.0 } else { 1.0 };

let delta = self.rewards[t] + gamma * next_val * mask - self.values[t];
gae = delta + gamma * lambda * mask * gae;
```

---

## Advantage Normalisation

After computing the batch of advantages, NeuroDrive normalises them:

```rust
let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
let variance: f32 = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
let std = (variance + 1e-8).sqrt();

for a in &mut advantages {
    *a = (*a - mean) / std;
}
```

This standardises advantages to roughly mean 0 and std 1 across the batch.

**Why normalise?** The magnitude of raw advantages depends on the scale of the rewards. Without normalisation, a run with large rewards produces large advantages and therefore large policy gradient steps, potentially destabilising training. Normalisation decouples the gradient step size from the reward scale.

**Caution:** Normalising advantages changes their relative values. An advantage of `+2σ` is very good; `−2σ` is very bad. But the absolute value of `2σ` depends on the batch's reward distribution. This is usually fine for learning but can obscure diagnostic information in analytics.

---

## Worked Example: One Rollout

Suppose a 5-step rollout with γ = 0.99, λ = 0.95:

```
Rewards:       r = [0.5, 0.5, 0.5, -5.0, 0.0]
Values:        V = [2.0, 2.1, 2.2,  2.0, 0.1]
Dones:         d = [F,   F,   F,    T,   F  ]
Next value:    next_value = 0.0  (episode ended)
```

Step 1: Compute TD errors (backwards from t=4):

```
t=4: next_val = next_value = 0.0 (end of buffer, use bootstrap)
     mask = 1 - done[4] = 1 - 0 = 1
     δ₄ = 0.0 + 0.99 * 0.0 * 1 - 0.1 = -0.1

t=3: next_val = V[4] = 0.1
     mask = 1 - done[3] = 1 - 1 = 0   ← episode ended
     δ₃ = -5.0 + 0.99 * 0.1 * 0 - 2.0 = -5.0 - 2.0 = -7.0

t=2: next_val = V[3] = 2.0
     mask = 1
     δ₂ = 0.5 + 0.99 * 2.0 * 1 - 2.2 = 0.5 + 1.98 - 2.2 = 0.28

t=1: next_val = V[2] = 2.2
     mask = 1
     δ₁ = 0.5 + 0.99 * 2.2 * 1 - 2.1 = 0.5 + 2.178 - 2.1 = 0.578

t=0: next_val = V[1] = 2.1
     mask = 1
     δ₀ = 0.5 + 0.99 * 2.1 * 1 - 2.0 = 0.5 + 2.079 - 2.0 = 0.579
```

Step 2: Compute GAE backwards (starting gae = 0):

```
t=4: gae = δ₄ + 0.99 * 0.95 * 1 * 0 = -0.1 + 0 = -0.1
     Â₄ = -0.1,   R₄ = -0.1 + 0.1 = 0.0

t=3: gae = δ₃ + 0.99 * 0.95 * 0 * (-0.1)   ← mask = 0
         = -7.0 + 0 = -7.0
     Â₃ = -7.0,   R₃ = -7.0 + 2.0 = -5.0

t=2: gae = δ₂ + 0.99 * 0.95 * 1 * (-7.0)
         = 0.28 + (-6.5835) = -6.3035
     Â₂ = -6.3035, R₂ = -6.3035 + 2.2 = -4.1035

t=1: gae = δ₁ + 0.99 * 0.95 * 1 * (-6.3035)
         = 0.578 + (-5.9293) = -5.3513
     Â₁ = -5.3513, R₁ = -5.3513 + 2.1 = -3.2513

t=0: gae = δ₀ + 0.99 * 0.95 * 1 * (-5.3513)
         = 0.579 + (-5.033) = -4.454
     Â₀ = -4.454,  R₀ = -4.454 + 2.0 = -2.454
```

**Interpretation:** The crash at t=3 (reward -5.0) propagates backward through the GAE recurrence, making the advantages at t=0 through t=2 negative. The policy will be pushed to make actions at those timesteps *less likely*. If the policy had good steering going into that crash, it will now be penalised — which is correct, because the trajectory ended badly.

After normalisation, the relative sign and magnitude of advantages is preserved but scaled to unit variance.

---

## NeuroDrive Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `gamma` | 0.99 | Rewards 100 steps away are worth 37% of current |
| `gae_lambda` | 0.95 | Strongly biased toward multi-step TD over pure MC |
| `max_steps` | 512 | Maximum rollout length before update |
| `min_update_steps` | 128 | Minimum batch size before a terminal-triggered update |

The combination of `γ = 0.99` and `λ = 0.95` is a very common default in on-policy RL. It produces advantage estimates that are slightly biased toward the critic (due to λ < 1) but with substantially lower variance than pure Monte Carlo.

---

## Common Misunderstandings

❌ "GAE is just a fancy return computation"
✅ GAE is a principled interpolation between Monte Carlo and TD estimation. The λ parameter has a direct interpretation in terms of the bias-variance trade-off.

❌ "Advantage normalisation is always safe"
✅ Normalisation changes the absolute scale of gradients. It is generally safe for training stability but should be understood, not blindly applied.

❌ "The terminal mask (done flag) is optional"
✅ Omitting the terminal mask causes advantage estimates to span episode boundaries — propagating value information from one episode's future into a different episode's past. This silently biases the gradients.

❌ "The bootstrap value is the critic's output for the last stored state"
✅ The bootstrap is the critic's estimate for the state *after* the rollout ends — the next state that would follow the last stored state. If the rollout ends at terminal, the bootstrap is 0.

---

## Related Files

- `concepts/core/reinforcement-learning.md` — return, discount, value functions
- `concepts/core/policy-gradient-methods.md` — why advantages are used
- `exercises/core/implement-gae.md` — implement GAE from scratch
- `project/systems/a2c-brain.md` — the live `compute_gae` function
