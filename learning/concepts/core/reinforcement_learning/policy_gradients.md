# Policy Gradients

## Prerequisites

- Markov Decision Processes — `markov_decision_processes.md` (states, actions, returns, policies)
- Calculus: partial derivatives, chain rule — `concepts/foundations/calculus_and_gradients.md`
- Probability: Gaussian distribution, log-probabilities — `concepts/foundations/probability_and_distributions.md`

## Target Depth for This Project

**Level 4** — Can derive and implement. You should be able to write out the policy gradient theorem from memory, compute a REINFORCE update by hand, and explain why NeuroDrive uses log-probabilities and baselines.

---

## Core Concept

### Step 1: Why Optimise the Policy Directly?

In value-based methods (like Q-learning), we learn a value function and derive a policy from it (e.g. "pick the action with the highest Q"). This works well for discrete actions but becomes awkward for continuous control: NeuroDrive's agent must output a steering angle in [−1, 1] and a throttle in [0, 1] — there are infinitely many possible actions.

**Policy gradient** methods sidestep this by parameterising the policy directly as a function π_θ(a | s) — a probability distribution over actions given a state, controlled by parameters θ (the neural network weights). We then optimise θ to maximise expected return.

### Step 2: The Objective Function

We want to find the parameters θ that maximise the expected return:

J(θ) = E_{τ ∼ π_θ} [G(τ)]

where τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...) is a trajectory sampled by following policy π_θ, and G(τ) is the total discounted return of that trajectory.

The challenge: J(θ) is an expectation over trajectories, which depend on θ through the policy. We cannot compute J(θ) exactly — we must estimate it from sampled trajectories.

### Step 3: The Policy Gradient Theorem

The policy gradient theorem tells us that:

∇_θ J(θ) = E_{τ ∼ π_θ} [ Σ_{t=0}^{T} ∇_θ log π_θ(a_t | s_t) · G_t ]

This is remarkable. It says: to compute the gradient of the expected return, we do not need to differentiate through the environment's transition dynamics. We only need:

1. The gradient of the log-probability of each action under the current policy
2. The return following that action

**Intuition**: ∇ log π_θ(a | s) points in the direction that increases the probability of action a. Multiplying by G_t scales this: if the return was high, we push the policy harder towards that action. If the return was low (or negative), we push away from it.

### Step 4: The Log-Probability Trick

Why do we use log π rather than π directly? Consider the gradient of the probability of a trajectory:

∇_θ P(τ | θ) = P(τ | θ) · ∇_θ log P(τ | θ)

This is the "log-derivative trick" — a direct application of ∇ log f(x) = ∇f(x) / f(x). It converts a gradient of a product (the probability of a trajectory is a product of many terms) into a sum of log-gradients, which is far more numerically stable and computationally tractable.

For a trajectory, log P(τ | θ) = Σ_t log π_θ(a_t | s_t) + Σ_t log P(s_{t+1} | s_t, a_t). The transition terms do not depend on θ, so they vanish when we differentiate with respect to θ. This is why policy gradients do not require a model of the environment.

### Step 5: REINFORCE

The simplest policy gradient algorithm is **REINFORCE** (Williams, 1992):

1. Collect a complete trajectory τ = (s_0, a_0, r_0, ..., s_T) using the current policy π_θ
2. For each timestep t, compute the return G_t = Σ_{k=t}^{T} γ^{k-t} r_k
3. Compute the gradient estimate: ∇_θ J ≈ (1/T) Σ_t ∇_θ log π_θ(a_t | s_t) · G_t
4. Update: θ ← θ + α · ∇_θ J

REINFORCE is correct in expectation but suffers from extremely high variance. In practice, the gradient estimates are so noisy that learning is unstable.

### Step 6: The Variance Problem and Baselines

The variance problem arises because G_t can be large and variable. If all returns are positive (say, between 50 and 100), then *every* action gets reinforced, just by different amounts. This wastes gradient signal.

A **baseline** b(s) is a state-dependent scalar subtracted from the return:

∇_θ J ≈ Σ_t ∇_θ log π_θ(a_t | s_t) · (G_t − b(s_t))

Crucially, subtracting any baseline that does not depend on the action leaves the gradient *unbiased* (provably: E[∇ log π · b] = 0 when b does not depend on a). The optimal baseline turns out to be close to V(s) — the state-value function. This is exactly where the *critic* enters the picture (see `value_functions_and_critics.md`).

When we use V(s) as the baseline, the quantity (G_t − V(s_t)) is the **advantage** — how much better this specific action was compared to the average. This connects policy gradients to actor-critic methods.

---

## Mathematical Foundation

### Worked Example: A 2-State, 2-Action Problem

Consider a tiny environment:

- **States**: S = {s_A, s_B}
- **Actions**: A = {left, right}
- **Policy**: a softmax policy with one parameter θ per state-action pair

Let the policy be:

π_θ(left | s_A) = exp(θ_1) / (exp(θ_1) + exp(θ_2))
π_θ(right | s_A) = exp(θ_2) / (exp(θ_1) + exp(θ_2))

Start with θ_1 = 0.5, θ_2 = −0.3.

**Step 1: Compute action probabilities**

exp(0.5) = 1.6487, exp(−0.3) = 0.7408

π(left | s_A) = 1.6487 / (1.6487 + 0.7408) = 1.6487 / 2.3895 = 0.6900
π(right | s_A) = 0.7408 / 2.3895 = 0.3100

**Step 2: Compute log-probabilities**

log π(left | s_A) = ln(0.6900) = −0.3711

**Step 3: Compute ∇_θ₁ log π(left | s_A)**

For a softmax policy, the gradient of the log-probability with respect to the logit of the chosen action is:

∇_{θ_1} log π(left | s_A) = 1 − π(left | s_A) = 1 − 0.6900 = 0.3100

This is a well-known softmax gradient identity: ∂ log π(a_i) / ∂ θ_i = 1 − π(a_i).

**Step 4: Suppose the return was G = 3.0**

The gradient update for θ_1 is:

Δθ_1 = α · ∇ log π(left | s_A) · G = α · 0.3100 · 3.0 = 0.93α

With learning rate α = 0.1: Δθ_1 = 0.093

New θ_1 = 0.5 + 0.093 = 0.593

The probability of choosing "left" in s_A has increased — because it led to a positive return.

### Gaussian Policy for Continuous Actions

NeuroDrive uses a Gaussian policy for continuous control. The actor network outputs a mean μ(s; θ) for each action dimension, and σ is parameterised via a learnable log-standard-deviation.

For a single action dimension, the policy is:

π_θ(a | s) = (1 / (σ√(2π))) · exp(−(a − μ)² / (2σ²))

The log-probability is:

log π_θ(a | s) = −(a − μ)² / (2σ²) − ln(σ) − 0.5 ln(2π)

**Gradient with respect to the mean μ:**

∂ log π / ∂μ = (a − μ) / σ²

**Intuition**: if the sampled action a was greater than the mean μ, the gradient is positive — pushing the mean towards a. The magnitude is scaled by 1/σ²: when the policy is confident (small σ), a given deviation has a larger gradient.

**Gradient with respect to log σ** (NeuroDrive parameterises in log-space):

Let log_std = ln(σ), so σ = exp(log_std).

∂ log π / ∂ log_std = (a − μ)² / σ² − 1

If the sampled action was far from the mean (large (a − μ)²/σ²), the gradient is positive, pushing σ larger (more exploration). If the action was close to the mean, the gradient is negative, pushing σ smaller (more exploitation).

### Worked Example: Gaussian Policy Gradient Step

Suppose: μ = 0.3, σ = 0.5, sampled action a = 0.7, advantage A = 2.0

**Log-probability:**
log π = −(0.7 − 0.3)² / (2 · 0.25) − ln(0.5) − 0.5 ln(2π)
log π = −0.16 / 0.5 + 0.6931 − 0.9189
log π = −0.32 + 0.6931 − 0.9189 = −0.5458

**Gradient w.r.t. μ:**
∂ log π / ∂μ = (0.7 − 0.3) / 0.25 = 1.6

**Policy gradient for μ (one sample):**
∂L / ∂μ = −A · ∂ log π / ∂μ = −2.0 · 1.6 = −3.2

(The negative sign is because we *minimise* the negative expected return, i.e. the loss is −A · log π.)

**Gradient w.r.t. log_std:**
∂ log π / ∂ log_std = (0.4)² / 0.25 − 1 = 0.64 − 1 = −0.36

**Policy gradient for log_std:**
∂L / ∂ log_std = −A · (−0.36) − entropy_coef = 0.72 − 0.01 = 0.71

The entropy regularisation term (−entropy_coef) is subtracted to discourage premature shrinking of σ.

This is precisely the computation in NeuroDrive's `update.rs`:

```rust
let d_lp_d_mean = (a - mean) / (std * std + 1e-8);
let d_lp_d_log_std = ((a - mean).powi(2) / (std * std + 1e-8)) - 1.0;
let d_loss_d_mean = -adv * d_lp_d_mean;
let d_loss_d_log_std = -adv * d_lp_d_log_std - entropy_coef;
```

---

## How NeuroDrive Uses This

NeuroDrive's A2C implementation uses the policy gradient theorem with the following specifics:

1. **Policy representation**: The actor MLP (14→64→64→2) outputs a mean vector μ for two action dimensions (steering, throttle). Two learnable log-std parameters (one per dimension) define the standard deviations σ = exp(log_std). Actions are sampled from N(μ, σ²).

2. **Log-std clamping**: log_std is clamped to [−2.0, 0.5], corresponding to σ ∈ [0.135, 1.649]. This prevents the policy from collapsing to a near-deterministic distribution (too little exploration) or exploding to an extremely wide one (random noise).

3. **Advantage as the weighting signal**: Instead of using raw returns G_t, NeuroDrive uses GAE-computed advantages (see `advantage_estimation_gae.md`). This dramatically reduces variance compared to REINFORCE.

4. **Gradient accumulation**: Gradients are accumulated across all 2048 steps in the rollout buffer before a single Adam optimiser step. This is full-batch gradient descent within each rollout, not stochastic mini-batching. The mean is taken by dividing each gradient by the batch size.

5. **Entropy regularisation**: An entropy bonus (coefficient 0.01) is added to prevent premature convergence. The gradient of the entropy with respect to log_std is always −1 (since ∂H/∂log_std = 1 for a Gaussian), which appears as the `- entropy_coef` term in the log_std gradient.

6. **Manual backpropagation**: The mean gradient ∂L/∂μ is backpropagated through the actor MLP using the cached forward-pass activations. Each Linear layer computes weight and bias gradients and passes the input gradient to the previous layer. This chain of `.backward()` calls is the handwritten equivalent of autograd.

---

## Common Misconceptions

1. **"Policy gradients require differentiating through the environment."** They do not. The policy gradient theorem is remarkable precisely because the transition dynamics P(s′ | s, a) cancel out. We only need to differentiate log π_θ(a | s) with respect to θ, which is entirely within the neural network. The environment is treated as a black box that returns rewards and next states.

2. **"Subtracting a baseline introduces bias."** Any baseline b(s) that depends only on the state (not the action) provably leaves the expected gradient unchanged. The variance reduction is free — no bias is introduced. The optimal baseline is approximately V(s), which is why we train a critic.

3. **"A single trajectory gives a good gradient estimate."** A single trajectory gives an *unbiased* estimate, but with enormous variance. REINFORCE with one trajectory per update is theoretically correct but practically useless for anything beyond toy problems. This is why NeuroDrive collects 2048 steps per update and uses GAE with a learned baseline (the critic).

---

## Glossary

| Term | Definition |
|------|-----------|
| **Policy gradient** | A gradient of the expected return w.r.t. policy parameters — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **REINFORCE** | The simplest policy gradient algorithm using complete episode returns |
| **Log-probability trick** | The identity ∇ log f = ∇f / f, used to convert trajectory probability gradients into tractable sums |
| **Baseline** | A state-dependent scalar subtracted from returns to reduce variance without introducing bias |
| **Entropy regularisation** | An additive bonus that rewards policy randomness to maintain exploration — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Gaussian policy** | A policy parameterised as a normal distribution N(μ, σ²) over continuous actions — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Log-std** | The natural logarithm of the standard deviation, used as the learnable parameter to ensure σ > 0 |

---

## Recommended Materials

1. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — Chapter 13 (Policy Gradient Methods). Sections 13.1–13.4 cover the policy gradient theorem and REINFORCE. Section 13.5 covers REINFORCE with baseline.

2. **David Silver's RL Course** — Lecture 7: Policy Gradient Methods. Clear derivation of the policy gradient theorem with visual aids. ~90 minutes.

3. **Andrej Karpathy, "Deep Reinforcement Learning: Pong from Pixels"** — Blog post (karpathy.github.io). An accessible, code-driven walkthrough of REINFORCE applied to Atari Pong, with excellent intuition-building.

4. **Lilian Weng, "Policy Gradient Algorithms"** — Blog post at lilianweng.github.io. Comprehensive survey covering REINFORCE through PPO, with consistent notation throughout. The Gaussian policy derivation is particularly well-presented.
