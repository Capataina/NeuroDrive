# Value Functions and Critics

## Prerequisites

- Markov Decision Processes — `markov_decision_processes.md` (returns, discount factor, policies)
- Policy Gradients — `policy_gradients.md` (why we need baselines and variance reduction)
- Basic calculus: derivatives, mean squared error — `concepts/foundations/calculus_and_gradients.md`

## Target Depth for This Project

**Level 3** — Can explain with technical depth. You should be able to compute V(s) and TD updates by hand, explain the Bellman equation, and describe how NeuroDrive's critic network is structured and trained.

---

## Core Concept

### Step 1: What Does a Value Function Answer?

A value function answers the question: "how good is it to be in this state (or to take this action in this state), assuming I follow my current policy from here onwards?"

This "how good" is measured as the **expected return** — the expected sum of discounted future rewards.

### Step 2: State-Value Function V(s)

The **state-value function** V^π(s) gives the expected return starting from state s and following policy π thereafter:

V^π(s) = E_π [G_t | s_t = s] = E_π [r_t + γ r_{t+1} + γ² r_{t+2} + ... | s_t = s]

V(s) answers: "on average, how much total reward will I accumulate from here?"

### Step 3: Action-Value Function Q(s, a)

The **action-value function** Q^π(s, a) gives the expected return starting from state s, taking action a, and then following policy π:

Q^π(s, a) = E_π [G_t | s_t = s, a_t = a]

Q answers: "how much total reward will I get if I take this specific action and then follow my policy?"

The relationship between them is:

V^π(s) = Σ_a π(a | s) · Q^π(s, a)

The state-value is the policy-weighted average of the action-values. The **advantage** A(s, a) = Q(s, a) − V(s) measures how much better action a is compared to the average (see `advantage_estimation_gae.md`).

### Step 4: The Bellman Equation

The Bellman equation expresses V(s) recursively — the value of the current state depends on the immediate reward plus the discounted value of the next state:

V^π(s) = E_π [r_t + γ · V^π(s_{t+1}) | s_t = s]

This says: the value of being in state s equals the expected immediate reward plus γ times the expected value of the next state.

The Bellman equation is the foundation of nearly all value-based and actor-critic RL methods. It allows us to estimate V(s) incrementally, without waiting for the full trajectory to finish.

### Step 5: Temporal Difference Learning — TD(0)

**Monte Carlo** estimation waits until the episode ends, computes the full return G_t, and uses it as the target for V(s_t). This is unbiased but has high variance (a single trajectory can be wildly unrepresentative).

**Temporal difference** (TD) learning instead uses the Bellman equation to bootstrap — it estimates V(s_t) using the immediate reward plus the current estimate of V(s_{t+1}):

V(s_t) ← V(s_t) + α · [r_t + γ · V(s_{t+1}) − V(s_t)]

The quantity in brackets is the **TD error** (or TD residual):

δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

If δ > 0, the outcome was *better than expected* — V(s_t) was too low and should increase.
If δ < 0, the outcome was *worse than expected* — V(s_t) was too high and should decrease.

The TD error is the computational analogue of the biological **dopamine reward prediction error** described in the NeuroDrive README: δ = r + γ V(s′) − V(s). This connection is not accidental — it is exactly the teaching signal NeuroDrive plans to use for brain-inspired plasticity (see [GLOSSARY.md → Neuromodulation](../../../GLOSSARY.md)).

### Step 6: Why Critics Reduce Variance

In REINFORCE (see `policy_gradients.md`), the policy gradient is weighted by the full return G_t. This return includes all future randomness — every random action, every stochastic transition — making the gradient extremely noisy.

A **critic** learns V(s) and provides it as a baseline. The policy gradient then uses the advantage A_t = G_t − V(s_t) instead of raw G_t. Since V(s_t) captures the "average" return from state s_t, the advantage isolates the effect of the specific action taken, filtering out the noise common to all actions from that state.

Even better: with TD-based advantage estimation (GAE), we do not need the full return at all. We use a weighted combination of short-horizon TD errors, which introduces some bias but dramatically reduces variance. This is the approach NeuroDrive uses.

---

## Mathematical Foundation

### Worked Example: Computing V(s) by Hand

Consider a 4-step trajectory segment with γ = 0.99:

| t | State | Reward | Done |
|---|-------|--------|------|
| 0 | s_0 | +0.8 | false |
| 1 | s_1 | +0.3 | false |
| 2 | s_2 | −0.1 | false |
| 3 | s_3 | +1.5 | true |

**Monte Carlo return computation (backwards):**

G_3 = r_3 = 1.5 (terminal step, no future rewards)

G_2 = r_2 + γ · (1 − done_2) · G_3
G_2 = −0.1 + 0.99 · 1.0 · 1.5
G_2 = −0.1 + 1.485 = 1.385

G_1 = r_1 + γ · (1 − done_1) · G_2
G_1 = 0.3 + 0.99 · 1.0 · 1.385
G_1 = 0.3 + 1.3712 = 1.6712

G_0 = r_0 + γ · (1 − done_0) · G_1
G_0 = 0.8 + 0.99 · 1.0 · 1.6712
G_0 = 0.8 + 1.6545 = 2.4545

So V(s_0) should be approximately 2.4545 (for this single trajectory — the true V is the expectation over many trajectories).

### How a TD(0) Update Works

Suppose the critic currently estimates V(s_0) = 2.0 and V(s_1) = 1.5, and we observe:

s_0 → r_0 = 0.8 → s_1

The TD error is:

δ_0 = r_0 + γ · V(s_1) − V(s_0) = 0.8 + 0.99 · 1.5 − 2.0 = 0.8 + 1.485 − 2.0 = 0.285

With learning rate α = 0.1, the update is:

V(s_0) ← 2.0 + 0.1 · 0.285 = 2.0285

The value of s_0 nudged upward because the outcome (reaching s_1 with reward 0.8) was slightly better than expected.

**Now suppose instead r_0 = −2.0:**

δ_0 = −2.0 + 0.99 · 1.5 − 2.0 = −2.0 + 1.485 − 2.0 = −2.515

V(s_0) ← 2.0 + 0.1 · (−2.515) = 1.7485

The value dropped sharply — the outcome was much worse than expected.

### Explained Variance: Measuring Critic Quality

**Explained variance** (EV) measures how well the critic's predictions V(s_t) match the actual returns G_t:

EV = 1 − Var(G − V) / Var(G)

| EV Value | Interpretation |
|----------|---------------|
| 1.0 | Critic perfectly predicts returns |
| 0.0 | Critic is no better than predicting the mean of G |
| < 0 | Critic is actively harmful — worse than a constant |

NeuroDrive computes this after each A2C update in `update.rs` and reports it through `A2cTrainingStats`. A healthy learning run should show EV gradually rising from near 0 towards 0.5–0.9 as the critic improves.

---

## How NeuroDrive Uses This

### The Critic Architecture

NeuroDrive's critic is a **separate MLP** from the actor (no shared trunk):

```
Input (14) → Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → scalar value
```

This mirrors the actor architecture but outputs a single scalar V(s) rather than a 2-dimensional action mean. The separate architecture means the actor and critic can learn at different rates and do not interfere with each other's representations.

### Training the Critic

The critic is trained to minimise the **mean squared error** between its predictions V(s_t) and the GAE-computed returns G_t = advantage_t + V(s_t):

L_critic = (1 / 2N) Σ_t (V(s_t) − G_t)²

The gradient for each sample is:

∂L / ∂V(s_t) = (V(s_t) − G_t) / N

This gradient is backpropagated through the critic MLP using the same cache-based manual backpropagation as the actor. Each Linear layer's `.backward()` method computes weight/bias gradients and passes the input gradient to the preceding layer.

### Separate Optimiser

The critic uses its own Adam optimiser with learning rate **1e-3** (three times the actor's 3e-4). The higher learning rate is deliberate: the critic's task (regression on returns) is simpler and more stable than the actor's task (policy optimisation), so it can afford to learn faster. A fast critic means the actor always has a reasonably accurate baseline, which keeps the advantage estimates useful.

### Integration with GAE

The critic's V(s_t) predictions serve two purposes:

1. **Baseline for advantage**: A_t = G_t − V(s_t), reducing variance in the policy gradient
2. **Bootstrap target**: The TD residuals δ_t = r_t + γ V(s_{t+1}) − V(s_t) are the building blocks of GAE (see `advantage_estimation_gae.md`)

Both uses depend on the critic being reasonably accurate. If the critic is wildly wrong early in training, the advantages will be noisy (but still correct in expectation). As the critic improves, the advantages become more precise and learning accelerates.

---

## Common Misconceptions

1. **"TD learning is just a faster version of Monte Carlo."** TD and Monte Carlo make fundamentally different trade-offs. Monte Carlo has zero bias (it uses the true return) but high variance. TD(0) has lower variance (it bootstraps from a single-step estimate) but introduces bias because V(s_{t+1}) is only an approximation. GAE-λ (see `advantage_estimation_gae.md`) interpolates between these two extremes.

2. **"A critic that always predicts zero is useless."** Surprisingly, even a constant baseline reduces variance compared to no baseline at all. The optimal baseline is close to V(s), but any reasonable constant (like the mean return) helps. The critic's job is to be *more helpful* than a constant — explaining variance, not just existing.

3. **"The critic and actor should share weights to be efficient."** Weight sharing can work but introduces coupling: poor critic gradients can corrupt the actor's learned features, and vice versa. NeuroDrive deliberately uses separate networks so that the critic's faster learning rate and different loss function do not destabilise the actor. This is a standard practice in A2C/PPO implementations.

---

## Glossary

| Term | Definition |
|------|-----------|
| **State-value V(s)** | Expected return from state s under policy π — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Action-value Q(s, a)** | Expected return from state s after taking action a, then following π |
| **Bellman equation** | Recursive decomposition: V(s) = E[r + γ V(s′)] |
| **TD error (δ)** | One-step prediction error: δ = r + γ V(s′) − V(s) — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **TD(0)** | Temporal difference learning using a one-step bootstrap target |
| **Monte Carlo return** | The full discounted sum of rewards from an actual trajectory (no bootstrapping) |
| **Explained variance** | Diagnostic metric: 1 − Var(G−V)/Var(G) — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Critic** | A neural network that learns V(s) to provide baselines and bootstrap targets — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Bootstrap** | Using a current estimate (V(s′)) in place of the true quantity (G) to form learning targets |

---

## Recommended Materials

1. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — Chapter 6 (Temporal-Difference Learning). Sections 6.1–6.3 cover TD(0) with excellent worked examples. Chapter 9 covers function approximation (neural network critics).

2. **David Silver's RL Course** — Lecture 4: Model-Free Prediction. Covers MC vs TD in detail with clear diagrams of the bias-variance trade-off. ~90 minutes.

3. **OpenAI Spinning Up — "Intro to Policy Optimization"** — spinningup.openai.com. The section on value functions and baselines provides practical intuition with code. ~30-minute read.

4. **Lilian Weng, "A (Long) Peek into Reinforcement Learning"** — The value functions section at lilianweng.github.io provides a compact mathematical treatment with consistent notation matching Sutton & Barto.
