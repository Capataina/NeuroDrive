# Markov Decision Processes

## Prerequisites

- Basic probability (random variables, conditional probability, expectation) — see `concepts/foundations/probability_and_distributions.md`
- Familiarity with summation notation and geometric series

## Target Depth for This Project

**Level 3** — Can explain with technical depth. You should be able to formalise NeuroDrive's racing environment as an MDP, write down the components explicitly, and compute returns by hand.

---

## Core Concept

### Step 1: What Problem Are We Solving?

An agent sits inside an environment. At each moment it perceives a *state*, chooses an *action*, receives a *reward*, and transitions to a new state. The agent's objective is to choose actions that maximise the total reward it accumulates over time.

A **Markov Decision Process** (MDP) is the mathematical framework that makes this precise. It gives us a formal language for stating what the environment does, what the agent controls, and what "good behaviour" means.

### Step 2: The Five Components

An MDP is a tuple **(S, A, P, R, γ)** where:

| Symbol | Name | Meaning |
|--------|------|---------|
| **S** | State space | The set of all possible situations the agent can be in |
| **A** | Action space | The set of all actions the agent can take |
| **P(s′ \| s, a)** | Transition dynamics | The probability of arriving at state s′ given that the agent is in state s and takes action a |
| **R(s, a, s′)** | Reward function | The immediate scalar signal received after transitioning from s to s′ via action a |
| **γ** | Discount factor | A scalar in [0, 1] that controls how much future rewards are worth relative to immediate ones |

### Step 3: The Markov Property

The defining characteristic of an MDP is the **Markov property**: the probability of the next state depends *only* on the current state and action, not on the entire history of prior states and actions.

Formally: P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)

This is powerful because it means the current state contains all the information needed to predict the future. The agent does not need to remember its entire trajectory — just where it is right now.

**Why this matters for RL**: Without the Markov property, we would need to condition on arbitrarily long histories, making learning intractable. The Markov assumption lets us build algorithms that operate on fixed-size state representations.

### Step 4: Policies

A **policy** π is the agent's strategy — a rule for choosing actions given states. It can be:

- **Deterministic**: π(s) = a (one fixed action per state)
- **Stochastic**: π(a | s) = probability of choosing action a in state s

In NeuroDrive, the policy is stochastic: the actor network outputs Gaussian distribution parameters, and actions are *sampled* from that distribution (see [GLOSSARY.md → Policy](../../GLOSSARY.md)).

### Step 5: Return — Measuring Long-Term Success

The **return** G_t is the total discounted reward from timestep t onwards:

G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + γ³ · r_{t+3} + ...

The discount factor γ serves two purposes:
1. It makes the sum finite (for γ < 1, the geometric series converges)
2. It encodes a preference for sooner rewards over later ones

### Step 6: Value Functions (Preview)

Given a policy π, two functions measure how good states and actions are:

- **State-value**: V^π(s) = E_π[G_t | s_t = s] — the expected return starting from s and following π
- **Action-value**: Q^π(s, a) = E_π[G_t | s_t = s, a_t = a] — the expected return starting from s, taking a, then following π

These are explored in depth in `value_functions_and_critics.md`. For now, understand that the value function tells us "how good is this state under my current strategy?"

---

## Mathematical Foundation

### Concrete Example: A 3-State Driving Scenario

Consider a simplified version of NeuroDrive with three states:

| State | Description |
|-------|-------------|
| **STRAIGHT** | Car is on a straight section, aligned with the track |
| **TURN** | Car is approaching or navigating a turn |
| **CRASHED** | Car has left the track (terminal state) |

Two actions are available: **steer_left** and **go_straight**.

#### Transition Probabilities P(s′ | s, a)

| Current State | Action | → STRAIGHT | → TURN | → CRASHED |
|--------------|--------|------------|--------|-----------|
| STRAIGHT | go_straight | 0.7 | 0.3 | 0.0 |
| STRAIGHT | steer_left | 0.4 | 0.5 | 0.1 |
| TURN | go_straight | 0.1 | 0.3 | 0.6 |
| TURN | steer_left | 0.5 | 0.4 | 0.1 |
| CRASHED | (any) | 0.0 | 0.0 | 1.0 |

Reading row 3: if the car is in TURN and goes straight, there is a 60% chance of crashing — turns require steering.

#### Rewards R(s, a, s′)

| Transition | Reward |
|-----------|--------|
| Any → STRAIGHT | +1.0 (making forward progress) |
| Any → TURN | +0.5 (still on track, approaching difficulty) |
| Any → CRASHED | −5.0 (terminal penalty) |

#### Computing the Return for a Sample Trajectory

Suppose the agent experiences this trajectory:

| t | State | Action | Next State | Reward |
|---|-------|--------|-----------|--------|
| 0 | STRAIGHT | go_straight | STRAIGHT | +1.0 |
| 1 | STRAIGHT | go_straight | TURN | +0.5 |
| 2 | TURN | steer_left | STRAIGHT | +1.0 |
| 3 | STRAIGHT | go_straight | TURN | +0.5 |
| 4 | TURN | go_straight | CRASHED | −5.0 |

**With γ = 0.99:**

G_0 = 1.0 + 0.99(0.5) + 0.99²(1.0) + 0.99³(0.5) + 0.99⁴(−5.0)
G_0 = 1.0 + 0.495 + 0.9801 + 0.4851 + (−4.8510)
G_0 = −1.8908

**With γ = 0.5 (much more short-sighted):**

G_0 = 1.0 + 0.5(0.5) + 0.25(1.0) + 0.125(0.5) + 0.0625(−5.0)
G_0 = 1.0 + 0.25 + 0.25 + 0.0625 + (−0.3125)
G_0 = 1.25

Notice how the discount factor changes the story. With γ = 0.99, the distant crash penalty dominates and the return is negative. With γ = 0.5, the crash is so heavily discounted that the return is positive — the agent is "short-sighted" and mostly sees the early rewards.

**With γ = 0.0 (completely myopic):**

G_0 = 1.0

The agent only sees the immediate reward. It has no concept of future consequences.

This illustrates why γ matters: it determines how far into the future the agent's planning horizon extends. NeuroDrive uses γ = 0.99, giving the agent a long planning horizon where a reward 100 steps away is still worth ~37% of its face value.

### Computing the Return Recursively

In practice, we compute returns backwards using the recursive relationship:

G_t = r_t + γ · (1 − done_t) · G_{t+1}

The (1 − done) factor is critical: when an episode terminates (done = true), the future return is zero — there are no rewards beyond a terminal state. This is exactly how NeuroDrive's `buffer.rs` computes returns in `compute_gae()`.

---

## How NeuroDrive Uses This

NeuroDrive's racing environment is an MDP where:

- **States** are the 14-dimensional observation vectors (11 normalised ray distances, speed, heading error, angular velocity — see [GLOSSARY.md → Observation Vector](../../GLOSSARY.md))
- **Actions** are 2-dimensional continuous vectors: steering ∈ [−1, 1] and throttle ∈ [0, 1]
- **Transitions** are deterministic (given the same state and action, the physics engine always produces the same next state — see [GLOSSARY.md → Deterministic Simulation](../../GLOSSARY.md))
- **Rewards** are decomposed into progress reward, time penalty, and terminal bonuses/penalties
- **γ = 0.99**, giving a long planning horizon suitable for track completion

The Markov property holds approximately: the 14-dimensional observation vector is designed to capture enough information (distances to walls, current speed, heading alignment) that the agent does not need to remember previous observations. This is a design choice — if the observation were too sparse (e.g. only speed), the Markov property would not hold and the agent would need memory.

The MDP formalism is not just theoretical scaffolding. Every RL algorithm NeuroDrive uses — policy gradients, value functions, GAE, A2C — is derived from manipulating the MDP's components mathematically.

---

## Common Misconceptions

1. **"The Markov property means the environment is random."** No. Deterministic environments satisfy the Markov property trivially: P(s′ | s, a) is either 0 or 1 for every transition. NeuroDrive's physics are deterministic, and it is still a perfectly valid MDP. Stochasticity in the *policy* (sampling from a Gaussian) does not violate determinism in the *transitions*.

2. **"A higher discount factor is always better."** A very high γ (close to 1.0) gives a long planning horizon but also increases variance in return estimates, because small errors in reward predictions compound over many timesteps. A very low γ makes learning stable but myopic. The choice of γ = 0.99 is a deliberate trade-off.

3. **"States must be discrete for MDPs."** The original MDP formulation uses discrete states, but continuous-state MDPs are well-defined and are exactly what NeuroDrive uses. The 14-dimensional observation vector is a continuous state representation. The theory extends cleanly; sums become integrals.

---

## Glossary

| Term | Definition |
|------|-----------|
| **MDP** | A tuple (S, A, P, R, γ) formalising sequential decision-making — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Markov property** | The next state depends only on the current state and action, not on history |
| **Policy (π)** | A mapping from states to actions or action distributions — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Return (G)** | The discounted sum of future rewards from a given timestep — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Discount factor (γ)** | Scalar in [0,1] controlling the weight of future vs immediate rewards — see [GLOSSARY.md](../../../GLOSSARY.md) |
| **Transition dynamics** | The function P(s′ \| s, a) governing state evolution |
| **Terminal state** | A state from which no further transitions occur (e.g. CRASHED) |

---

## Recommended Materials

1. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — Chapter 3 (Finite Markov Decision Processes). The canonical treatment. Read sections 3.1–3.5 for the core formalism, then 3.6–3.7 for value functions.

2. **David Silver's RL Course (UCL, 2015)** — Lecture 2: Markov Decision Processes (available on YouTube). Covers the same material with excellent visual intuitions. Approximately 90 minutes.

3. **Lilian Weng, "A (Long) Peek into Reinforcement Learning"** — Blog post at lilianweng.github.io. Provides a concise walkthrough of MDPs with diagrams, then connects directly to policy gradients.

4. **OpenAI Spinning Up — Key Concepts in RL** — spinningup.openai.com, "Part 1: Key Concepts". A 20-minute read that covers MDPs, returns, and policies with clean notation.
