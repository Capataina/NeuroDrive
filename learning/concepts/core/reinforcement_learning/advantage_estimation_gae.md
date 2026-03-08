# Generalised Advantage Estimation (GAE)

## Prerequisites

- Markov Decision Processes — `markov_decision_processes.md` (returns, discount factor γ)
- Value Functions and Critics — `value_functions_and_critics.md` (V(s), TD errors, Bellman equation)
- Policy Gradients — `policy_gradients.md` (why advantages matter for variance reduction)

## Target Depth for This Project

**Level 4** — Can derive and implement. You should be able to compute GAE by hand for a given trajectory, explain the bias-variance trade-off controlled by λ, and trace through NeuroDrive's `buffer.rs` implementation line by line.

---

## Core Concept

### Step 1: The Problem GAE Solves

Policy gradients require weighting ∇ log π(a|s) by some measure of "how good was this action?" Two extremes exist:

**Monte Carlo advantage** (use the full return):
A_t^MC = G_t − V(s_t)

This is **unbiased** — G_t is the true return. But it has **high variance** because G_t includes the randomness of every future action and transition. A single lucky or unlucky trajectory can produce a wildly misleading gradient.

**TD(0) advantage** (use a one-step bootstrap):
A_t^TD = r_t + γ V(s_{t+1}) − V(s_t) = δ_t

This has **low variance** (it only depends on one reward and two value estimates) but **high bias** — V(s_{t+1}) is only an approximation of the true future return, and any error in V propagates into the advantage.

We need a method that can smoothly trade off between these extremes. GAE provides exactly this.

### Step 2: TD Residuals — The Building Block

The **TD residual** at step t is:

δ_t = r_t + γ · (1 − done_t) · V(s_{t+1}) − V(s_t)

The (1 − done_t) factor is crucial: if the episode terminates at step t (done_t = true), there is no next state, so the bootstrap term vanishes and δ_t = r_t − V(s_t).

Each δ_t is a one-step advantage estimate: "what I got (r_t + γV(s_{t+1})) minus what I expected (V(s_t))."

### Step 3: n-Step Returns and n-Step Advantages

We can extend beyond one step. An **n-step advantage** uses n actual rewards before bootstrapping:

A_t^(1) = δ_t (1-step, TD(0))
A_t^(2) = δ_t + γ δ_{t+1} (2-step)
A_t^(3) = δ_t + γ δ_{t+1} + γ² δ_{t+2} (3-step)
...
A_t^(∞) = Σ_{l=0}^{∞} γ^l δ_{t+l} (infinite-step, equivalent to Monte Carlo)

As n increases, bias decreases (more real rewards, less bootstrapping) but variance increases (more randomness included).

### Step 4: GAE-λ — The Exponentially Weighted Average

**GAE-λ** (Schulman et al., 2016) takes an exponentially-weighted average of all n-step advantages:

GAE_t = (1−λ) [A_t^(1) + λ A_t^(2) + λ² A_t^(3) + ...]

After algebraic simplification, this becomes:

GAE_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

The parameter **λ ∈ [0, 1]** controls the trade-off:

| λ Value | Behaviour |
|---------|-----------|
| λ = 0 | GAE_t = δ_t (pure TD(0), low variance, high bias) |
| λ = 1 | GAE_t = G_t − V(s_t) (pure Monte Carlo, zero bias, high variance) |
| λ = 0.95 | A practical middle ground used by NeuroDrive |

### Step 5: Recursive Computation

The infinite sum has an elegant recursive form that makes implementation trivial:

GAE_t = δ_t + γ · λ · (1 − done_t) · GAE_{t+1}

We compute this **backwards** from the last timestep to the first:
1. Start with GAE_T = 0 (or bootstrap from V(s_T) if the episode did not terminate)
2. For t = T−1, T−2, ..., 0: compute GAE_t = δ_t + γλ(1−done_t) · GAE_{t+1}

This is a single backward pass — O(T) time, O(T) space. NeuroDrive implements this in `buffer.rs` as the variable `gae` accumulated in a reverse loop.

### Step 6: Advantage Normalisation

After computing GAE for all timesteps, NeuroDrive **normalises** the advantages:

A_normalised = (A − mean(A)) / (std(A) + ε)

where ε = 1e-8 prevents division by zero. This ensures advantages have zero mean and unit variance across the batch, which:

1. Prevents the gradient magnitude from depending on the absolute scale of rewards
2. Ensures roughly half the advantages are positive (reinforced) and half negative (suppressed)
3. Stabilises training when the reward scale changes over the course of learning

---

## Mathematical Foundation

### Complete Worked Example: 5-Step Trajectory

Consider a 5-step trajectory with the following observed data:

| t | s_t | r_t | done_t | V(s_t) |
|---|-----|-----|--------|--------|
| 0 | s_0 | +0.5 | false | 1.2 |
| 1 | s_1 | +0.3 | false | 0.8 |
| 2 | s_2 | +0.7 | false | 1.5 |
| 3 | s_3 | −3.0 | true | 0.9 |
| 4 | s_4 | +0.4 | false | 0.6 |

**Parameters**: γ = 0.99, λ = 0.95

The V(s_t) values are the critic's predictions *at collection time* (stored in the rollout buffer). The next_value (V(s_5) for bootstrapping after t=4) is 0.7.

#### Phase 1: Compute TD residuals δ_t

For each timestep, we need V(s_{t+1}). For the last step (t=4), we use next_value = 0.7.

**t = 4**: not terminal (done=false), next_val = 0.7
δ_4 = r_4 + γ · (1−0) · V(s_5) − V(s_4) = 0.4 + 0.99 · 0.7 − 0.6 = 0.4 + 0.693 − 0.6 = **0.493**

**t = 3**: terminal (done=true), so the bootstrap term vanishes
δ_3 = r_3 + γ · (1−1) · V(s_4) − V(s_3) = −3.0 + 0.0 − 0.9 = **−3.900**

**t = 2**: not terminal, next_val = V(s_3) = 0.9
δ_2 = r_2 + γ · 1.0 · 0.9 − V(s_2) = 0.7 + 0.891 − 1.5 = **0.091**

**t = 1**: not terminal, next_val = V(s_2) = 1.5
δ_1 = r_1 + γ · 1.0 · 1.5 − V(s_1) = 0.3 + 1.485 − 0.8 = **0.985**

**t = 0**: not terminal, next_val = V(s_1) = 0.8
δ_0 = r_0 + γ · 1.0 · 0.8 − V(s_0) = 0.5 + 0.792 − 1.2 = **0.092**

Summary of TD residuals:

| t | δ_t |
|---|-----|
| 0 | 0.092 |
| 1 | 0.985 |
| 2 | 0.091 |
| 3 | −3.900 |
| 4 | 0.493 |

#### Phase 2: Compute GAE backwards

**t = 4** (last step, initialise with gae = 0):
GAE_4 = δ_4 + γ · λ · (1 − done_4) · 0
GAE_4 = 0.493 + 0 = **0.493**

**t = 3** (terminal step! done=true, so the propagation term is zeroed):
GAE_3 = δ_3 + γ · λ · (1 − 1) · GAE_4
GAE_3 = −3.900 + 0 = **−3.900**

This is critical: a terminal step **cuts the GAE chain**. The crash at t=3 is its own self-contained signal; it does not bleed the advantage from t=4 backwards.

**t = 2** (not terminal):
GAE_2 = δ_2 + γ · λ · 1.0 · GAE_3
GAE_2 = 0.091 + 0.99 · 0.95 · (−3.900)
GAE_2 = 0.091 + 0.9405 · (−3.900)
GAE_2 = 0.091 − 3.668 = **−3.577**

Even though t=2 itself had a positive TD residual (+0.091), the large negative advantage from the upcoming crash propagates backward, making this state look bad overall.

**t = 1** (not terminal):
GAE_1 = δ_1 + γ · λ · 1.0 · GAE_2
GAE_1 = 0.985 + 0.9405 · (−3.577)
GAE_1 = 0.985 − 3.364 = **−2.379**

**t = 0** (not terminal):
GAE_0 = δ_0 + γ · λ · 1.0 · GAE_1
GAE_0 = 0.092 + 0.9405 · (−2.379)
GAE_0 = 0.092 − 2.238 = **−2.146**

Summary:

| t | δ_t | GAE_t | Return (GAE_t + V(s_t)) |
|---|-----|-------|------------------------|
| 0 | 0.092 | −2.146 | −0.946 |
| 1 | 0.985 | −2.379 | −1.579 |
| 2 | 0.091 | −3.577 | −2.077 |
| 3 | −3.900 | −3.900 | −3.000 |
| 4 | 0.493 | 0.493 | 1.093 |

The returns (GAE_t + V(s_t)) are the targets used to train the critic's value function.

#### Phase 3: Advantage Normalisation

Raw advantages: [−2.146, −2.379, −3.577, −3.900, 0.493]

mean = (−2.146 − 2.379 − 3.577 − 3.900 + 0.493) / 5 = −11.509 / 5 = −2.302

variance = [(−2.146 + 2.302)² + (−2.379 + 2.302)² + (−3.577 + 2.302)² + (−3.900 + 2.302)² + (0.493 + 2.302)²] / 5
= [0.0243 + 0.0059 + 1.6256 + 2.5536 + 7.8175] / 5
= 12.0269 / 5 = 2.4054

std = √(2.4054 + 1e-8) = 1.5510

Normalised advantages:
A_0 = (−2.146 + 2.302) / 1.5510 = 0.156 / 1.5510 = **+0.101**
A_1 = (−2.379 + 2.302) / 1.5510 = −0.077 / 1.5510 = **−0.050**
A_2 = (−3.577 + 2.302) / 1.5510 = −1.275 / 1.5510 = **−0.822**
A_3 = (−3.900 + 2.302) / 1.5510 = −1.598 / 1.5510 = **−1.030**
A_4 = (0.493 + 2.302) / 1.5510 = 2.795 / 1.5510 = **+1.803**

After normalisation, the crashed step (t=3) has the most negative advantage (−1.030), and the post-reset step (t=4) has the most positive (+1.803). Steps 0 and 1 are near zero — they were about average.

---

## How NeuroDrive Uses This

NeuroDrive's GAE implementation lives in `src/brain/a2c/buffer.rs`, in the `compute_gae` method. The code mirrors the mathematical derivation exactly:

```rust
for t in (0..self.rewards.len()).rev() {
    let next_val = if t + 1 < self.rewards.len() {
        self.values[t + 1]
    } else {
        next_value
    };
    let mask = if self.dones[t] { 0.0 } else { 1.0 };
    let delta = self.rewards[t] + gamma * next_val * mask - self.values[t];
    gae = delta + gamma * lambda * mask * gae;
    advantages[t] = gae;
    returns[t] = gae + self.values[t];
}
```

Key implementation details:

- **Reverse iteration**: `(0..len).rev()` iterates from T−1 down to 0, matching the backward recursion
- **`mask`**: Converts `done` booleans to 0.0/1.0, implementing (1 − done_t) arithmetically
- **`next_value`**: The critic's prediction at the state *after* the last collected step, used for bootstrapping when the rollout ends mid-episode
- **Returns**: Computed as `gae + self.values[t]`, i.e. GAE_t + V(s_t). These are the regression targets for the critic
- **Normalisation**: Applied immediately after the loop, using the batch mean and standard deviation

The `next_value` bootstrap is computed in `update.rs` before calling `compute_gae`:

```rust
let mut next_value = 0.0;
if !brain.buffer.dones.last().unwrap_or(&false) {
    if let Some(last_state) = brain.buffer.states.last() {
        let (_, val) = brain.model.forward(last_state);
        next_value = val;
    }
}
```

If the last step was terminal (done=true), the bootstrap value is 0 — there is no future. Otherwise, the critic estimates the value of the state we would have transitioned into.

---

## Common Misconceptions

1. **"λ = 0 means we ignore the future."** No. λ = 0 gives GAE_t = δ_t = r_t + γ V(s_{t+1}) − V(s_t). The term γ V(s_{t+1}) already encodes the critic's estimate of *all* future rewards. What λ controls is how much we *trust* the critic's estimate vs using actual observed rewards. Low λ trusts the critic more; high λ trusts the observed trajectory more.

2. **"Done flags only matter at episode boundaries."** They matter at every step of the GAE computation. A done flag at step t zeros out both the bootstrap term in δ_t (so we do not add γ V(s_{t+1}) across episode boundaries) *and* the GAE propagation term (so advantages from after the reset do not leak backward into the terminal episode). NeuroDrive uses `current_tick_end_reason` to ensure only the actual terminal tick gets marked done.

3. **"Advantage normalisation is optional."** Technically, the algorithm converges without it. Practically, unnormalised advantages can have wildly different scales at different points in training (early on, the agent crashes constantly and advantages are dominated by large negative penalties; later, they are smaller and more nuanced). Normalisation stabilises the effective learning rate of the policy gradient and is considered essential for reliable A2C training.

---

## Glossary

| Term | Definition |
|------|-----------|
| **GAE (Generalised Advantage Estimation)** | Exponentially-weighted average of n-step advantages parameterised by λ — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **TD residual (δ_t)** | One-step advantage: r_t + γ(1−done)V(s_{t+1}) − V(s_t) — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **λ (GAE lambda)** | Bias-variance trade-off parameter ∈ [0,1]; higher = less bias, more variance |
| **n-step return** | A return estimate using n actual rewards before bootstrapping from V |
| **Advantage normalisation** | Standardising advantages to zero mean and unit variance across the batch |
| **Bootstrapping** | Using a learned estimate (e.g. V(s′)) as a stand-in for the true future return |
| **done flag / mask** | A boolean indicating episode termination, used to cut bootstrap and GAE chains at episode boundaries |

---

## Recommended Materials

1. **Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)** — The original GAE paper (arXiv: 1506.02438). Sections 1–3 derive GAE from first principles. Section 4 presents the bias-variance analysis. The paper is 10 pages and highly readable.

2. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — Section 12.1–12.5 (Eligibility Traces). The TD(λ) framework is the predecessor of GAE. Understanding eligibility traces here builds intuition for the λ parameter.

3. **OpenAI Spinning Up — "Part 3: Intro to Policy Optimization"** — The GAE section provides a clean implementation in PyTorch with step-by-step commentary. ~30 minutes to read and implement.

4. **Lilian Weng, "Policy Gradient Algorithms"** — The GAE section at lilianweng.github.io derives the formula concisely and connects it to the broader actor-critic landscape.
