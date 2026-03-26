# Policy Gradient Methods

## Why This Matters Here

Policy gradient methods are the mathematical engine behind A2C. Understanding the policy gradient theorem — what it says, why it is correct, and what it implies for implementation — is essential for understanding why the NeuroDrive update code is structured the way it is. Every term in `a2c_update.rs` exists because of this theorem.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — MDP, return, value functions, Bellman equations
- `concepts/foundations/probability-and-distributions.md` — log-probability, Gaussian distribution

## Notation

| Symbol | Meaning |
|---|---|
| `θ` | Policy network parameters |
| `π_θ(a\|s)` | Policy: probability of action `a` in state `s` under `θ` |
| `J(θ)` | Policy objective: expected return |
| `∇_θ J(θ)` | Policy gradient |
| `τ` | A trajectory (s₀, a₀, r₀, s₁, a₁, r₁, ...) |
| `G_t` | Return from timestep `t` |
| `b(s)` | Baseline function (to reduce variance) |

---

## Core Idea

In supervised learning, you know the correct output for each input and you minimise the error between prediction and target. In RL, there is no "correct action" to supervise on — only reward signals that say whether the outcomes were good or bad.

Policy gradient methods solve this by directly differentiating the expected return with respect to the policy parameters. You do not need to know the correct action; you just need to know which actions led to good outcomes and push their log-probabilities up.

---

## The Policy Objective

We want to maximise:

```
J(θ) = E_τ ~ π_θ [ G(τ) ]
```

where `G(τ)` is the total return from trajectory `τ`.

We need the gradient `∇_θ J(θ)` to update `θ`. The problem is that the expectation is over a distribution that depends on `θ` itself (through `π_θ`), which makes differentiation non-trivial.

---

## The Log-Derivative Trick

The key mathematical move is the log-derivative identity:

```
∇_θ π_θ(a|s)  =  π_θ(a|s) * ∇_θ log π_θ(a|s)
```

This follows from `∇_θ log f = (∇_θ f) / f`.

Using this identity, we can rewrite the gradient of the expectation as an expectation of a product:

```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t | s_t) * G_t ]
```

This is the **REINFORCE** policy gradient estimator (Williams, 1992).

### Why This Is Powerful

We no longer need to differentiate through the environment dynamics (`P(s'|s,a)`) — those are unknown in general. We only need to differentiate `log π_θ(a|s)`, which is a quantity we can compute exactly because we own the policy network.

### Intuition

The gradient `∇_θ log π_θ(a_t | s_t)` is the direction in parameter space that increases the probability of action `a_t` in state `s_t`. Multiplying by `G_t` scales this by how good the trajectory was.

In plain terms: **make actions that led to good outcomes more probable, and make actions that led to bad outcomes less probable.**

---

## The Policy Gradient Theorem

The formal statement (Sutton et al., 1999):

For any differentiable policy `π_θ`:

```
∇_θ J(θ) ∝ Σ_s μ^π(s) * Σ_a Q^π(s, a) * ∇_θ π_θ(a|s)
```

where `μ^π(s)` is the on-policy state distribution (how often we visit state `s` under `π`).

Rewritten using the log-derivative trick and the on-policy expectation:

```
∇_θ J(θ) = E_{s ~ μ^π, a ~ π_θ} [ Q^π(s, a) * ∇_θ log π_θ(a|s) ]
```

### Key Insight: Causality

The `Q^π(s_t, a_t)` term should only include rewards *at or after* timestep `t`, not before. Actions taken at `t` cannot affect rewards that happened before `t`. This causality constraint means we can replace `G(τ)` with `G_t` (return from `t` onwards):

```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t | s_t) * G_t ]
```

---

## Baselines and Variance Reduction

The REINFORCE estimator has high variance because `G_t` is a noisy estimate of the return — it depends on every random action taken throughout the episode.

To reduce variance without introducing bias, we subtract a **baseline** `b(s_t)` that does not depend on the action:

```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t | s_t) * (G_t - b(s_t)) ]
```

**This is still unbiased.** Any baseline that depends only on state, not on action, can be subtracted without changing the expected value of the gradient (because `E_a[∇ log π * b] = b * E_a[∇ log π] = b * 0 = 0`).

The best possible baseline (in the mean-squared sense) is the value function `V^π(s_t)`. Subtracting it gives the **advantage**:

```
A^π(s_t, a_t) = G_t - V^π(s_t)
```

A positive advantage means action `a_t` led to better-than-average return. A negative advantage means it led to worse-than-average return.

---

## The REINFORCE Update in Practice

For each collected trajectory, the policy gradient update is:

```
θ ← θ + η * Σ_t ∇_θ log π_θ(a_t | s_t) * A_t
```

or, equivalently, minimising the loss:

```
L_policy = -Σ_t log π_θ(a_t | s_t) * A_t
```

(Note the negative sign: gradient ascent on J is gradient descent on -J.)

---

## Why Pure REINFORCE is Insufficient

REINFORCE as described above has several practical problems:

1. **High variance:** Monte Carlo returns `G_t` have high variance because they sum many random reward signals across the full episode.
2. **Sample inefficiency:** each trajectory can only be used once (the policy that collected the data is the policy being updated — "on-policy").
3. **No value network:** we need some way to estimate `V^π(s)` to compute advantages.

These limitations motivate the actor-critic framework, where a separate **critic** learns `V^π(s)` concurrently with the actor learning `π_θ`.

---

## From REINFORCE to A2C

The progression from REINFORCE to A2C:

| Step | Idea | What it improves |
|---|---|---|
| REINFORCE | Use full episode return `G_t` | Baseline |
| Baseline subtraction | Subtract `V(s_t)` from return | Variance reduction |
| Actor-critic | Learn `V(s_t)` with a separate network | Reduces variance, enables online updates |
| GAE | Interpolate between MC and TD | Controls bias-variance trade-off |
| A2C | Synchronous multi-actor batched updates | Gradient stability |

---

## How This Appears in NeuroDrive

In `src/brain/a2c/update.rs`, for each transition `i` in the rollout:

```rust
let log_prob = squashed_gaussian_log_prob(latent, squashed, mean, std, j);
let d_loss_d_mean = -adv * d_lp_d_mean;     // -advantage * ∂(log π)/∂mean
policy_loss_sum += -adv * log_prob;
```

The policy loss is `- Σ advantage * log π`. The negative sign is because we minimise the loss (which is maximising the policy gradient objective).

The advantage `adv` comes from GAE: `advantages = compute_gae(...)`.

The gradient of `log π` with respect to `μ` and `log σ` is computed analytically from the Gaussian formula — this is what `d_lp_d_mean` and `d_lp_d_log_std` represent.

---

## The Entropy Bonus

A standard addition to the policy loss:

```
L = L_policy - entropy_coef * H(π)
```

(Minimising this maximises return AND maximises entropy.)

Entropy regularisation prevents premature collapse of the policy to a near-deterministic strategy. In NeuroDrive with `entropy_coef = 0.01`, this keeps the standard deviations from collapsing too early.

---

## Common Misunderstandings

❌ "We are teaching the policy which action is correct"
✅ We are scaling the log-probability of the action taken by how much better/worse than average it was. We do not have a "correct" label.

❌ "Policy gradients are unique to deep learning"
✅ REINFORCE predates deep learning by decades (Williams 1992). The policy gradient theorem (Sutton et al. 1999) works for any differentiable parameterisation.

❌ "Baseline subtraction biases the gradient"
✅ State-dependent baselines do not bias the gradient estimate. The proof relies on the fact that `E_a[∇_θ log π_θ(a|s)] = 0`.

❌ "A2C is always better than REINFORCE"
✅ A2C adds a learned value function and uses batch updates, which typically reduces variance and improves learning speed. But a badly trained critic can produce worse advantage estimates than raw Monte Carlo returns. The critic must be well-calibrated.

---

## Related Files

- `concepts/core/reinforcement-learning.md` — MDP, returns, value functions
- `concepts/core/advantage-estimation.md` — GAE for computing advantages
- `concepts/core/actor-critic-architecture.md` — the actor-critic structure
- `concepts/foundations/probability-and-distributions.md` — log-probability of the Gaussian policy
- `project/systems/a2c-brain.md` — the full NeuroDrive update code
