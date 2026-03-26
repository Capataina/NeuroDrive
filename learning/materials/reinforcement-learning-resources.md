# Reinforcement Learning Resources

A curated set of resources for studying the reinforcement learning theory that underpins NeuroDrive. Organised by topic and depth.

---

## Foundational Textbooks

### Sutton & Barto — Reinforcement Learning: An Introduction (2nd ed., 2018)

The standard textbook for RL. The freely available draft covers:
- Finite MDPs, Bellman equations, dynamic programming (Chapters 3–4)
- Monte Carlo and TD methods (Chapters 5–6)
- Policy gradient methods (Chapter 13) — directly relevant to NeuroDrive's A2C
- Eligibility traces (Chapter 12) — directly relevant to Milestone 2's biological architecture

**Which chapters to read for NeuroDrive:**
- Chapter 3 (MDP formalism) — prerequisite for everything
- Chapter 6 (TD learning, δ signal) — underpins both A2C and the biological learning rule
- Chapter 12 (eligibility traces) — directly relevant to Milestone 2
- Chapter 13 (policy gradient) — the theoretical basis for A2C

---

## Key Papers

### Policy Gradient Methods

**Mnih et al. (2016) — Asynchronous Methods for Deep Reinforcement Learning (A3C paper)**

Introduces A3C and, implicitly, A2C as its synchronous variant. Shows actor-critic with parallel workers can learn continuous-control tasks. The paper that established the algorithm family NeuroDrive uses.

**Schulman et al. (2015) — High-Dimensional Continuous Control Using Generalised Advantage Estimation (GAE paper)**

Introduces GAE. Provides the formal justification for the λ-weighted advantage estimator. Directly relevant: NeuroDrive uses GAE with γ=0.99, λ=0.95, which are the paper's recommended defaults.

**Williams (1992) — Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE)**

The foundational policy gradient paper. Introduces the log-derivative trick and the REINFORCE estimator. Read this to understand where the policy gradient formula comes from.

---

### Implementation and Practical RL

**Engstrom et al. (2020) — Implementation Matters in Deep RL: A Case Study on PPO and TRPO**

Shows that seemingly minor implementation choices (gradient clipping, value loss clipping, observation normalisation) dominate performance differences between on-policy methods. Directly relevant to NeuroDrive's handwritten A2C: this paper explains why implementation correctness matters more than algorithm choice.

**Andrychowicz et al. (2021) — What Matters for On-Policy Deep Actor-Critic Methods?**

Large-scale empirical study of on-policy actor-critic hyperparameter sensitivity. Key finding: separate actor/critic networks, tanh activations, and careful initialisation matter significantly. NeuroDrive uses separate networks (good) and ReLU (a known deviation from the tanh finding).

---

### Continuous Control

**Haarnoja et al. (2018) — Soft Actor-Critic (SAC)**

The main off-policy alternative to A2C for continuous control. SAC uses maximum-entropy RL (entropy in the objective, not just as a bonus) and a replay buffer. Understanding SAC clarifies why NeuroDrive chose A2C (simpler, on-policy, no replay buffer needed).

---

## Online Resources

### Spinning Up in Deep RL (OpenAI)

A practical RL education resource with:
- Clean algorithm implementations (A2C, PPO, SAC, TD3)
- Implementation tips and common bugs
- Mathematical notation consistent with most modern papers

Most directly useful: the Actor-Critic Methods and Policy Gradient Methods sections.

### David Silver's UCL Lectures (DeepMind)

Video lecture series covering:
- MDPs and dynamic programming
- Model-free prediction and control (TD, Monte Carlo)
- Policy gradient methods
- Actor-critic architectures

Lecture 7 (Policy Gradient Methods) is the most directly relevant to NeuroDrive.

---

## Topics Directly Relevant to NeuroDrive

### Continuous Action Spaces

For understanding NeuroDrive's Gaussian policy with tanh squashing:
- SAC paper (Haarnoja 2018) — Section 2 covers bounded action spaces and the squashing correction
- Spinning Up Actor-Critic page — implementation notes on squashed Gaussian log-probability

### GAE in Practice

For understanding the exact recurrence implemented in NeuroDrive:
- GAE paper (Schulman 2015) — Section 3 derives the recurrence formula
- Sutton & Barto Chapter 12 — connects GAE to the eligibility trace literature (the λ parameter is the same λ as in eligibility traces)

### Reward Shaping Theory

For understanding NeuroDrive's reward decomposition:
- Ng et al. (1999) — Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping
  This paper introduces potential-based reward shaping and proves that correctly shaped rewards do not change the optimal policy. NeuroDrive's reward components are not formally potential-based, but this paper provides the theoretical context for when reward shaping is safe.

---

## What to Read First

If you are new to RL and working through the NeuroDrive learning archive:

1. **Sutton & Barto, Chapter 3** — MDP formalism, returns, Bellman equations (5–10 hours)
2. **Sutton & Barto, Chapter 6** — TD learning, the δ signal (4–6 hours)
3. **Williams 1992** — REINFORCE; policy gradient foundation (1–2 hours, dense)
4. **GAE paper** — once you understand TD and policy gradients (2–3 hours)
5. **Engstrom 2020** — once you start thinking about implementation (1–2 hours)

This sequence gives you everything needed to understand the current A2C implementation in full technical detail.
