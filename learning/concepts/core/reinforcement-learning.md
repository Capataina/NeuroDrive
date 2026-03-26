# Reinforcement Learning

## Why This Matters Here

NeuroDrive's A2C baseline is a reinforcement learning (RL) system. Understanding RL from first principles is not optional for understanding the learning code — every design choice in the A2C implementation (the rollout buffer, the advantage estimates, the Bellman bootstrap) exists because of specific RL concepts.

This file builds RL from the ground up: what the problem is, how it is formalised, and what the core mathematical objects (value functions, policies, the Bellman equation) mean. Subsequent concept files (`policy-gradient-methods.md`, `advantage-estimation.md`, `actor-critic-architecture.md`) build on this foundation.

## Prerequisites

- Basic probability (expectation, conditional probability)
- No prior RL knowledge required

## Notation

| Symbol | Meaning |
|---|---|
| `s` | State |
| `a` | Action |
| `r` | Reward |
| `s'` | Next state (after taking action `a` in state `s`) |
| `π(a | s)` | Policy: probability of taking action `a` in state `s` |
| `V^π(s)` | State value function under policy π |
| `Q^π(s, a)` | Action-value (Q) function under policy π |
| `A^π(s, a)` | Advantage function |
| `γ` | Discount factor ∈ [0, 1] |
| `G_t` | Return from time t: discounted sum of future rewards |
| `T` | Episode horizon |

---

## The Reinforcement Learning Problem

Reinforcement learning is the problem of learning a behaviour policy from interaction with an environment through rewards and penalties.

The agent:
1. Observes the state `s_t`
2. Takes an action `a_t` according to policy `π(a_t | s_t)`
3. Receives reward `r_t` from the environment
4. Transitions to new state `s_{t+1}`
5. Repeats until the episode ends

The goal: find a policy `π` that maximises the expected cumulative reward.

**In NeuroDrive:** The car observes sensor data (its state `s_t`), the A2C policy produces a steering and throttle command (action `a_t`), the environment gives a reward based on progress and penalties (reward `r_t`), and the car moves to a new position (state `s_{t+1}`).

---

## Markov Decision Processes

The standard formal framework for RL is the **Markov Decision Process (MDP)**:

```
(S, A, P, R, γ)
```

Where:
- `S` — state space (all possible states)
- `A` — action space (all possible actions)
- `P(s' | s, a)` — transition dynamics: probability of reaching `s'` from `s` with action `a`
- `R(s, a, s')` — reward function
- `γ` — discount factor

### The Markov Property

An MDP assumes the **Markov property**: the next state `s'` depends only on the current state `s` and action `a`, not on the history. Formally:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

**In NeuroDrive:** The car's observation vector captures the full instantaneous driving state (raycast distances, heading error, speed, etc.) well enough that history is not strictly needed for basic driving behaviour. The Markov assumption is approximate — a human driver uses contextual memory — but it is sufficient for the A2C baseline.

### NeuroDrive as an MDP

| MDP element | NeuroDrive realisation |
|---|---|
| `s` | 23-dimensional observation vector |
| `a` | (steering, throttle) ∈ [-1,1] × [0,1] |
| `P` | Deterministic car physics (given seed) |
| `R` | Progress reward + penalties + crash/lap bonuses |
| `γ` | 0.99 (near-horizon undiscounting) |
| `T` | 30-second timeout, or crash, or lap completion |

---

## Returns and Discounting

The **return** from time `t` is the discounted sum of future rewards:

```
G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ... + γ^(T-t) * r_T
    = Σ_{k=0}^{T-t}  γ^k * r_{t+k}
```

The discount factor `γ ∈ [0, 1]` serves two purposes:
1. **Mathematical:** makes the infinite-horizon sum finite (for non-episodic tasks)
2. **Semantic:** immediate rewards are worth more than delayed rewards (a driving analogy: finishing the lap soon is better than finishing it eventually)

### NeuroDrive uses γ = 0.99

At γ = 0.99:
- A reward 10 steps in the future is worth `0.99^10 ≈ 0.905` times a current reward
- A reward 100 steps in the future is worth `0.99^100 ≈ 0.366` times a current reward

This is close enough to "undiscounted" that the agent cares about completing the lap, but discounted enough to prefer efficient early progress over eventually completing it after a long delay.

---

## Value Functions

Value functions measure "how good is it to be in state `s` (or to take action `a` in state `s`) under policy `π`?"

### State Value Function

```
V^π(s) = E_π [ G_t | s_t = s ]
       = E_π [ Σ_{k=0}^∞ γ^k * r_{t+k} | s_t = s ]
```

The expected return when starting in state `s` and following policy `π` forever.

**In NeuroDrive:** `V(s)` is what the critic estimates. A high `V(s)` near the start of a straight means the agent expects good progress ahead. A low `V(s)` at a tight corner means the agent expects trouble.

### Action-Value Function (Q-function)

```
Q^π(s, a) = E_π [ G_t | s_t = s, a_t = a ]
```

The expected return when taking action `a` in state `s`, then following policy `π`.

### Advantage Function

```
A^π(s, a) = Q^π(s, a) - V^π(s)
```

The advantage measures how much better action `a` is compared to the average action in state `s`. A positive advantage means the action was better than average; negative means it was worse.

**Why the advantage matters:** Policy gradient methods update the policy to make good actions (positive advantage) more likely and bad actions (negative advantage) less likely. Using the raw return `G_t` has high variance because it includes baseline performance noise; the advantage subtracts that baseline.

---

## The Bellman Equations

The Bellman equations express value functions as **recursive relationships**. They are the central tool for computing and estimating value functions.

### Bellman Expectation Equation for V^π

```
V^π(s) = E_π [ r_t + γ * V^π(s') | s_t = s ]
        = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V^π(s')]
```

This says: "the value of state `s` is the expected immediate reward plus discounted value of the next state."

### Practical Implication: TD Learning

The Bellman equation implies we can update an estimate `V̂(s_t)` without waiting for the full episode return:

```
TD target = r_t + γ * V̂(s_{t+1})
TD error   = TD target - V̂(s_t)
```

The TD error is the basis for the advantage estimate in A2C (see `concepts/core/advantage-estimation.md`).

### Bootstrapping

When the episode is not over, we can **bootstrap** the value estimate:

```
G_t ≈ r_t + γ * V̂(s_{t+1})   (one-step TD target)
```

Instead of rolling out the episode to completion, we use the critic's estimate for the future. This trades unbiasedness (you could be wrong) for variance reduction (the full Monte Carlo return has high variance). NeuroDrive bootstraps at the end of every rollout horizon.

---

## Policies

A **policy** `π(a | s)` tells the agent what to do. It maps states to action probabilities (stochastic) or to specific actions (deterministic).

### Deterministic vs Stochastic Policies

- **Deterministic:** always takes the same action in a given state. No exploration.
- **Stochastic:** samples an action from a distribution. Can explore.

NeuroDrive uses a **stochastic Gaussian policy** — the network outputs action means and the policy samples around them. This enables exploration: the same observation can produce different actions on different steps.

### Parameterised Policies

In deep RL, policies are represented by neural networks with parameters `θ`:

```
π_θ(a | s) = the probability of action a given state s under network θ
```

The goal is to find `θ*` that maximises expected return.

---

## Episodes and Episodic vs Continuing Tasks

NeuroDrive is **episodic**: each episode starts at the spawn point and ends at crash, timeout, or lap completion. The brain persists across episodes (weights are not reset), but environment state is reset.

This is the "one brain, one lifetime" design: the same network learns continuously across many episodes, just as a biological brain continues to learn from each new experience without wiping its weights.

---

## Exploration vs Exploitation

A classic RL tension:

- **Exploitation:** take the action you currently believe is best (maximise expected reward now)
- **Exploration:** try new actions to discover if there are better strategies

With a Gaussian policy, this tension is managed by the standard deviation:
- Large `σ` → more exploration (trying diverse actions)
- Small `σ` → more exploitation (sticking to the learned best action)

NeuroDrive regularises entropy to prevent `σ` from collapsing too early (premature exploitation).

---

## What Makes NeuroDrive Distinctive

Most RL benchmarks treat the agent as a module that interacts with a fixed external simulator (like OpenAI Gym). NeuroDrive's long-term vision is different:

1. **The learning mechanism is the subject of study**, not a tool for achieving high score on a fixed task.
2. **A2C is a baseline validation layer**, not the intended final learning architecture.
3. **The transition to local plasticity** (Milestones 2–4) is motivated by the differences between gradient-based RL and biological learning: no global gradients, no backpropagation, synapse-local credit assignment.

Understanding standard RL is required to understand what the project is moving away from.

---

## Common Misunderstandings

❌ "RL is about maximising a reward function"
✅ RL maximises *expected cumulative discounted return*. Single-step rewards are not the objective; long-term consequences are.

❌ "The policy and value function are the same network"
✅ In NeuroDrive they are explicitly separate. The actor outputs action distributions; the critic outputs scalar value estimates. They serve different purposes.

❌ "Bootstrapping is less accurate than Monte Carlo"
✅ Bootstrapping introduces bias (using an imperfect estimate of future value) but reduces variance. The trade-off is formalised in GAE (see `concepts/core/advantage-estimation.md`).

---

## Related Files

- `concepts/core/policy-gradient-methods.md` — how to actually improve the policy
- `concepts/core/advantage-estimation.md` — how to estimate A(s, a) with GAE
- `concepts/core/actor-critic-architecture.md` — the specific architecture used
- `concepts/domain-patterns/reward-shaping.md` — how NeuroDrive's reward is designed
