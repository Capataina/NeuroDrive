# Advantage Actor-Critic (A2C)

## Prerequisites

- Markov Decision Processes — `markov_decision_processes.md` (states, actions, returns, discount factor)
- Policy Gradients — `policy_gradients.md` (∇ log π, REINFORCE, baselines, Gaussian policy)
- Value Functions and Critics — `value_functions_and_critics.md` (V(s), TD errors, MSE loss, critic architecture)
- Generalised Advantage Estimation — `advantage_estimation_gae.md` (TD residuals, GAE-λ, normalisation, recursive computation)
- Neural Networks — `concepts/core/neural_networks/forward_pass_and_layers.md` and `backpropagation.md` (forward pass, backward pass, gradient accumulation)
- Optimisers — `concepts/core/neural_networks/weight_initialisation_and_optimisers.md` (Adam)

## Target Depth for This Project

**Level 4** — Can derive and implement. You should be able to walk through a complete A2C training iteration from rollout collection through gradient update, explain every loss term, and read NeuroDrive's `src/brain/a2c/` code with full comprehension.

---

## Core Concept

### Step 1: What Is A2C?

**Advantage Actor-Critic** (A2C) is an on-policy reinforcement learning algorithm that combines:

- An **actor** (policy network) that learns *what to do* — mapping states to action distributions
- A **critic** (value network) that learns *how good things are* — estimating expected future return from each state

The actor is updated using the policy gradient theorem, with the critic's value estimates used to compute advantages (see [GLOSSARY.md → Actor-Critic](../../../GLOSSARY.md)). The "A" in A2C stands for "advantage" — the policy gradient is weighted by GAE-computed advantages rather than raw returns, which dramatically reduces variance.

A2C is the synchronous variant of A3C (Asynchronous Advantage Actor-Critic, Mnih et al. 2016). NeuroDrive uses a single environment with no parallelism, making A2C the natural choice.

### Step 2: On-Policy vs Off-Policy

A2C is **on-policy**: it collects data using its current policy, performs one update, and then discards the data. The next update uses freshly-collected data from the (now slightly different) policy.

This stands in contrast to **off-policy** methods (like DQN, SAC) that store past experiences in a replay buffer and can reuse them for many updates.

| Property | On-Policy (A2C) | Off-Policy (SAC/DQN) |
|----------|----------------|---------------------|
| Data efficiency | Lower — data used once | Higher — data reused |
| Stability | Generally more stable | Can be unstable without target networks |
| Complexity | Simpler (no replay buffer) | More complex (replay buffer, target networks) |
| Staleness | No stale data | Must correct for behavioural policy mismatch |
| Implementation | Straightforward | Requires more infrastructure |

NeuroDrive uses A2C deliberately: it is the simplest algorithm that validates learnability. Complexity is the enemy of debuggability, and the goal is to prove the environment works before transitioning to brain-inspired plasticity (see [GLOSSARY.md → On-Policy](../../../GLOSSARY.md)).

### Step 3: The Architecture

NeuroDrive uses **separate actor and critic networks** — no shared layers.

**Actor** (learns the policy):
```
obs (14) → Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2) → mean μ
+ 2 learnable log_std parameters → σ = exp(log_std)
→ π(a|s) = N(μ, σ²)
```

**Critic** (learns the value function):
```
obs (14) → Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → scalar V(s)
```

The 14-dimensional input is NeuroDrive's observation vector: 11 normalised ray distances, speed, heading error, and angular velocity (see [GLOSSARY.md → Observation Vector](../../../GLOSSARY.md)). The 2 output dimensions are steering and throttle.

### Step 4: The Three Loss Components

A2C minimises a combined loss with three terms:

**Policy loss** (actor):
L_policy = −(1/N) Σ_t A_t · log π_θ(a_t | s_t)

This is the REINFORCE gradient but with advantages A_t (from GAE) instead of raw returns. The negative sign is because we *maximise* expected return, which means *minimising* the negative of it.

**Value loss** (critic):
L_value = (1/2N) Σ_t (V_φ(s_t) − G_t)²

Standard MSE regression — the critic learns to predict the returns G_t = A_t + V_old(s_t).

**Entropy bonus** (actor):
L_entropy = −c_ent · (1/N) Σ_t H[π(· | s_t)]

For a Gaussian with standard deviation σ, the entropy is H = 0.5 + 0.5 ln(2π) + ln(σ). The entropy bonus rewards policies that maintain randomness, preventing premature convergence to a suboptimal deterministic policy. NeuroDrive uses c_ent = 0.01.

### Step 5: The Complete Algorithm

```
Initialise: actor π_θ, critic V_φ, rollout buffer B, step counter k = 0
Repeat:
  1. COLLECT ROLLOUT
     For t = 0 to max_steps − 1:
       Observe s_t
       Forward pass: (μ, σ) = actor(s_t), v = critic(s_t)
       Sample a_t ~ N(μ, σ²), clamp to valid range
       Compute log π(a_t | s_t)
       Store (s_t, a_t, log_prob_t, v_t) in B
       Execute a_t in environment, observe r_t, done_t
       Store (r_t, done_t) in B
       If done_t: reset environment

  2. COMPUTE ADVANTAGES
     If last step not terminal:
       next_value = critic(s_{last+1})
     Else:
       next_value = 0
     (advantages, returns) = GAE(B, next_value, γ, λ)
     Normalise advantages

  3. UPDATE NETWORKS
     Zero all gradients
     For each (s_t, a_t, A_t, G_t) in B:
       Forward: (μ, σ) = actor(s_t), v = critic(s_t)
       Accumulate actor gradients: ∂/∂θ [−A_t · log π(a_t|s_t) − c_ent · H]
       Accumulate critic gradients: ∂/∂φ [0.5 · (v − G_t)²]
     Step actor optimiser (Adam, LR = 3e-4)
     Step critic optimiser (Adam, LR = 1e-3)
     Step log_std optimiser (Adam, LR = 3e-4)
     Clamp log_std to [−2.0, 0.5]

  4. CLEAR BUFFER
     Discard all collected data (on-policy: data is stale after update)
```

---

## Mathematical Foundation

### Walked-Through Training Iteration with Concrete Numbers

Let us trace through a *tiny* update with 4 steps (NeuroDrive uses 2048, but we use 4 for clarity).

**Collected rollout data:**

| t | obs (simplified) | action [steer, throttle] | reward | done | V(s_t) |
|---|-----------------|-------------------------|--------|------|--------|
| 0 | [0.8, 0.5, ...] | [0.2, 0.6] | +0.4 | false | 0.50 |
| 1 | [0.7, 0.4, ...] | [−0.1, 0.7] | +0.3 | false | 0.45 |
| 2 | [0.3, 0.2, ...] | [0.5, 0.4] | −2.0 | true | 0.60 |
| 3 | [0.9, 0.6, ...] | [0.0, 0.8] | +0.5 | false | 0.30 |

The last step was not terminal, so we bootstrap: next_value = critic([next_obs]) = 0.40.

**Phase 1: TD residuals (γ = 0.99)**

δ_3 = 0.5 + 0.99 · 1.0 · 0.40 − 0.30 = 0.5 + 0.396 − 0.30 = **0.596**
δ_2 = −2.0 + 0.99 · 0.0 · V(s_3) − 0.60 = −2.0 + 0 − 0.60 = **−2.600** (done=true)
δ_1 = 0.3 + 0.99 · 1.0 · 0.60 − 0.45 = 0.3 + 0.594 − 0.45 = **0.444**
δ_0 = 0.4 + 0.99 · 1.0 · 0.45 − 0.50 = 0.4 + 0.4455 − 0.50 = **0.3455**

**Phase 2: GAE backwards (λ = 0.95)**

GAE_3 = 0.596 + 0 = **0.596** (last step, gae starts at 0)
GAE_2 = −2.600 + 0.99 · 0.95 · 0.0 · 0.596 = **−2.600** (done=true, chain cut)
GAE_1 = 0.444 + 0.9405 · (−2.600) = 0.444 − 2.445 = **−2.001**
GAE_0 = 0.3455 + 0.9405 · (−2.001) = 0.3455 − 1.882 = **−1.537**

Advantages: [−1.537, −2.001, −2.600, 0.596]
Returns: [−1.537 + 0.50, −2.001 + 0.45, −2.600 + 0.60, 0.596 + 0.30] = [−1.037, −1.551, −2.000, 0.896]

**Phase 3: Normalise advantages**

mean = (−1.537 − 2.001 − 2.600 + 0.596) / 4 = −5.542 / 4 = −1.3855
var = [(−0.1515)² + (−0.6155)² + (−1.2145)² + (1.9815)²] / 4 = [0.0230 + 0.3788 + 1.4750 + 3.9263] / 4 = 1.4508
std = √(1.4508 + 1e-8) = 1.2045

A_norm = [−0.126, −0.511, −1.008, 1.645]

**Phase 4: Gradient computation (one sample, t=0)**

Suppose actor outputs μ = [0.15, 0.55], σ = [0.50, 0.50] for obs at t=0.
Action was [0.2, 0.6]. Normalised advantage A = −0.126.

For steering (j=0):
∂ log π / ∂μ_0 = (0.2 − 0.15) / (0.25) = 0.2
∂L / ∂μ_0 = −(−0.126) · 0.2 / 4 = 0.0063

For throttle (j=1):
∂ log π / ∂μ_1 = (0.6 − 0.55) / (0.25) = 0.2
∂L / ∂μ_1 = −(−0.126) · 0.2 / 4 = 0.0063

The positive gradient means the actor should push its mean *towards* these actions slightly — but only slightly, because the advantage is close to zero (this was a near-average action).

For t=3 (A = +1.645), the gradient would be much larger and negative — strongly pushing the mean towards the successful action.

The critic gradient for t=0:
∂L / ∂V = (V(s_0) − G_0) / N = (0.50 − (−1.037)) / 4 = 1.537 / 4 = 0.384

This large positive gradient will push V(s_0) down towards the target −1.037 (the critic was overestimating).

**Phase 5: Adam step**

After accumulating gradients across all 4 samples, the Adam optimiser updates:
1. Update first-moment estimate: m ← β₁ · m + (1 − β₁) · g
2. Update second-moment estimate: v ← β₂ · v + (1 − β₂) · g²
3. Bias-correct: m̂ = m/(1 − β₁^t), v̂ = v/(1 − β₂^t)
4. Update parameters: θ ← θ − lr · m̂ / (√v̂ + ε)

The actor uses lr = 3e-4, the critic uses lr = 1e-3.

---

## How NeuroDrive Uses This

### Specific Implementation Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rollout length | 2048 steps | Long enough for multiple episodes; provides stable gradient estimates |
| γ (discount) | 0.99 | Long planning horizon (~100 effective steps) suitable for track completion |
| λ (GAE lambda) | 0.95 | Moderate bias-variance trade-off; proven default |
| Actor LR | 3e-4 | Standard for policy optimisation; slow enough for stability |
| Critic LR | 1e-3 | Faster than actor — regression is easier than policy optimisation |
| Entropy coef | 0.01 | Small enough not to dominate policy loss, large enough to prevent collapse |
| log_std clamp | [−2.0, 0.5] | σ ∈ [0.135, 1.649] — prevents degenerate distributions |
| Hidden dim | 64 | Two hidden layers; sufficient for 14-dim input, small enough for fast iteration |
| Actor dims | 14→64→64→2 | 2 outputs: steering mean, throttle mean |
| Critic dims | 14→64→64→1 | 1 output: scalar state-value V(s) |
| Adam β₁, β₂ | 0.9, 0.999 | Standard Adam hyperparameters |
| Gradient update | Full-batch | All 2048 steps accumulated before one optimiser step |

### Execution Flow in Bevy

NeuroDrive integrates A2C into the ECS fixed-timestep pipeline:

1. **`a2c_act_system`** (SimSet::Input): reads `ObservationVector`, runs actor forward pass, samples action, stores (state, action, log_prob, value) in rollout buffer, writes clamped action to `ActionState.desired`.

2. **Physics, collision, measurement systems** run in subsequent SimSets.

3. **`a2c_collect_reward_system`** (SimSet::Measurement, after episode_loop): reads `EpisodeState.current_tick_reward` and `current_tick_end_reason`, appends reward and done flag to the buffer. If the buffer reaches `max_steps` (2048), triggers `a2c_update()`.

4. **`a2c_update()`** (in `update.rs`): computes GAE, iterates over all buffered transitions to accumulate gradients, steps both Adam optimisers, updates log_std with its own Adam-style update, clamps log_std, records health metrics, and clears the buffer.

### Manual Backpropagation

NeuroDrive does not use an autograd library. Each layer caches its input during `.forward()`, and `.backward()` uses that cache to compute:
- Weight gradients: `grad_weights[i][j] += grad_output[i] * input[j]`
- Bias gradients: `grad_biases[i] += grad_output[i]`
- Input gradients (passed to previous layer): `grad_input[j] += weights[i][j] * grad_output[i]`

Gradients **accumulate** (note the `+=`) across all 2048 samples before one `.step()` call. This is equivalent to full-batch gradient descent.

### Action Clamping

The Gaussian policy can sample actions outside the valid range (steering ∈ [−1, 1], throttle ∈ [0, 1]). NeuroDrive handles this by:

1. Sampling from N(μ, σ²)
2. Clamping to the valid range via `CarAction::clamped()`
3. Recomputing the log-probability using the *clamped* action

This is an approximation — a principled approach would use a Beta distribution or squashed Gaussian. The clamped-action fraction is tracked as a diagnostic in `A2cTrainingStats`.

### Learnable log_std

The standard deviations σ are not output by the actor network. Instead, they are **free parameters** (one per action dimension) stored in `a_log_std`, shared across all states. This simplification means exploration is state-independent — the policy has the same spread everywhere. The log_std parameters have their own Adam-style update with the same learning rate as the actor (3e-4).

---

## Common Misconceptions

1. **"A2C is outdated and PPO is always better."** PPO adds a clipped surrogate objective to prevent destructively large policy updates. This is valuable when policies are fragile or environments are challenging. But PPO adds complexity (clipping ratio, multiple epochs over the same data). For NeuroDrive's purpose — validating learnability — A2C is the right tool. Adding PPO's complexity before the baseline works would obscure debugging.

2. **"The entropy bonus makes the agent random."** The entropy coefficient (0.01) is deliberately small. It provides a gentle nudge towards exploration that prevents the policy from collapsing to a deterministic strategy too early. As the agent improves and advantages become more decisive, the policy gradient signal overwhelms the entropy bonus, and the policy naturally becomes more concentrated. The entropy term is a regulariser, not a randomiser.

3. **"On-policy methods waste data because they throw it away."** On-policy methods use each sample *correctly* — the data was generated by the current policy, so the gradient estimate is unbiased. Off-policy methods reuse data but must correct for the mismatch between the data-generating policy and the current policy (via importance sampling or other techniques). The "waste" in on-policy methods is the price of simplicity and stability. For NeuroDrive, where the environment runs at 60 Hz and data collection is cheap, this trade-off is favourable.

---

## Glossary

| Term | Definition |
|------|-----------|
| **A2C** | Advantage Actor-Critic — an on-policy RL algorithm using separate actor and critic networks — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Actor** | The policy network that maps states to action distribution parameters — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Critic** | The value network that estimates V(s) — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Policy loss** | −(1/N) Σ A_t · log π(a_t\|s_t) — drives the actor to favour high-advantage actions — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Value loss** | (1/2N) Σ (V(s_t) − G_t)² — drives the critic towards accurate return predictions |
| **Entropy bonus** | A regularisation term that rewards policy randomness to maintain exploration — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Rollout buffer** | Storage for on-policy transitions (s, a, r, done, log_prob, V) — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Full-batch update** | Accumulating gradients over all samples before one optimiser step (no mini-batching) |
| **On-policy** | Using only data from the current policy; discarding after update — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **log_std** | Learnable log-standard-deviation parameters for the Gaussian policy |

---

## Recommended Materials

1. **Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)** — The original A3C paper (arXiv: 1602.01783). Section 4 describes the advantage actor-critic variant. NeuroDrive's A2C is the synchronous single-worker case of this algorithm.

2. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — Chapter 13.5 (Actor-Critic Methods). Provides the theoretical foundation for combining policy gradients with learned baselines.

3. **OpenAI Spinning Up — "Vanilla Policy Gradient"** — spinningup.openai.com. A clean, well-documented A2C implementation in PyTorch with detailed inline commentary. The implementation structure closely mirrors NeuroDrive's approach.

4. **Andrej Karpathy, "Deep Reinforcement Learning: Pong from Pixels"** — Blog post at karpathy.github.io. While it covers REINFORCE rather than A2C, it provides essential intuition about policy gradients that transfers directly. The "gradient as a supervised learning signal" framing is particularly helpful.
