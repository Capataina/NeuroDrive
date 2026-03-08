# Materials: Reinforcement Learning

Topic-organised resources for learning reinforcement learning, from foundational theory through to the actor-critic methods used in NeuroDrive. Work through these alongside the corresponding concept files and exercises.

---

## MDPs and Foundations

- [ ] **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. — Chapter 3 (pp. 47–70)**
  Link: http://incompleteideas.net/book/RLbook2020.pdf
  Section: "Finite Markov Decision Processes" — covers states, actions, rewards, transitions, the Bellman equation, and the formal MDP framework.
  Why: This is the canonical treatment. Every RL paper assumes you know these definitions. Read it slowly and do the exercises at the end of the chapter.
  Difficulty: Intermediate | Time: 3–4 hours

- [ ] **David Silver — RL Course, Lecture 2: Markov Decision Processes**
  Link: https://www.youtube.com/watch?v=lfHX2hHRMVQ
  Section: Full lecture (1:27:00). Key timestamps — 0:00–12:00 (Markov property), 12:00–35:00 (MRP and returns), 35:00–58:00 (MDP definition and policies), 58:00–1:15:00 (Bellman equations and optimality).
  Why: Silver explains the mathematical structure intuitively with diagrams. Pairs well with Sutton & Barto chapter 3 — watch this first if you prefer lectures.
  Difficulty: Intermediate | Time: 1.5 hours

- [ ] **OpenAI Spinning Up — "Part 1: Key Concepts in RL"**
  Link: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
  Section: Full page. Focus on the "What Can Go Wrong" subsections — they explain failure modes early.
  Why: Concise modern reference that connects textbook definitions to practical implementation concerns. Good for a second pass after Sutton & Barto.
  Difficulty: Beginner–Intermediate | Time: 1 hour

- [ ] **Lilian Weng — "A (Long) Peek into Reinforcement Learning"**
  Link: https://lilianweng.github.io/posts/2018-02-19-rl-overview/
  Section: "What is Markov Decision Process" through "Value Function" sections.
  Why: Excellent blog-format summary that connects MDP theory to algorithms. Good for quick review before exams or interviews.
  Difficulty: Intermediate | Time: 45 minutes

---

## Policy Gradients

- [ ] **Sutton & Barto — Chapter 13 (pp. 321–340)**
  Link: http://incompleteideas.net/book/RLbook2020.pdf
  Section: "Policy Gradient Methods" — sections 13.1 (Policy Approximation) through 13.3 (REINFORCE: Monte Carlo Policy Gradient). Focus on the derivation of the policy gradient theorem (Theorem 13.1).
  Why: The primary reference. The derivation of ∇J(θ) = E[∇ log π(a|s) · Q(s,a)] is the single most important equation in policy-based RL. You must be able to reproduce it.
  Difficulty: Advanced | Time: 4–5 hours

- [ ] **David Silver — RL Course, Lecture 7: Policy Gradient Methods**
  Link: https://www.youtube.com/watch?v=KHZVXao4qXs
  Section: Full lecture (1:23:00). Key timestamps — 0:00–18:00 (motivation for policy-based methods), 18:00–42:00 (policy gradient theorem), 42:00–58:00 (REINFORCE and baselines), 58:00–end (actor-critic preview).
  Why: Silver's derivation of the policy gradient theorem is particularly clear. The whiteboard walkthrough of REINFORCE with baselines is essential.
  Difficulty: Advanced | Time: 1.5 hours

- [ ] **OpenAI Spinning Up — "Part 3: Intro to Policy Optimization"**
  Link: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
  Section: Full page. Focus on "Deriving the Simplest Policy Gradient" and "Implementing the Simplest Policy Gradient."
  Why: The cleanest modern walkthrough from theory to code. Their derivation avoids measure theory and gets to implementation fast. Read this after Sutton & Barto for the implementation perspective.
  Difficulty: Intermediate–Advanced | Time: 1.5 hours

- [ ] **Andrej Karpathy — "Deep Reinforcement Learning, Pong from Pixels"**
  Link: http://karpathy.github.io/2016/05/31/rl/
  Section: Full post. Focus on the "Policy Gradients" section and the ~130-line implementation.
  Why: A famously clear exposition. Karpathy trains a policy gradient agent to play Pong with raw pixels in ~130 lines of Python. The intuitive explanations ("when you win, increase the probability of everything you did") are excellent for building intuition.
  Difficulty: Intermediate | Time: 2 hours

---

## Value Functions

- [ ] **Sutton & Barto — Chapter 6 (pp. 119–140)**
  Link: http://incompleteideas.net/book/RLbook2020.pdf
  Section: "Temporal-Difference Learning" — sections 6.1 (TD Prediction) through 6.5 (Q-learning). Focus on the TD(0) update rule and why bootstrapping works.
  Why: TD learning is the foundation of all critic networks. Understanding how V(s) is learned via bootstrapping is essential before studying actor-critic methods or GAE.
  Difficulty: Intermediate | Time: 3–4 hours

- [ ] **David Silver — RL Course, Lecture 4: Model-Free Prediction**
  Link: https://www.youtube.com/watch?v=PnHCvfgC_ZA
  Section: Full lecture (1:28:00). Key timestamps — 0:00–25:00 (Monte Carlo vs TD), 25:00–50:00 (TD(0) algorithm), 50:00–1:10:00 (bias-variance of TD), 1:10:00–end (TD(λ) and eligibility traces).
  Why: The comparison between Monte Carlo and TD methods is particularly well-done. The bias-variance trade-off discussion maps directly to understanding GAE later.
  Difficulty: Intermediate | Time: 1.5 hours

- [ ] **Sutton & Barto — Chapter 9 (pp. 197–220)**
  Link: http://incompleteideas.net/book/RLbook2020.pdf
  Section: "On-policy Prediction with Approximation" — focus on sections 9.1–9.4 covering linear and neural network function approximation for V(s).
  Why: NeuroDrive uses a neural network as a function approximator for V(s). This chapter bridges the gap between tabular value functions and the critic network you will implement.
  Difficulty: Advanced | Time: 3 hours

- [ ] **Chris Olah — "Understanding LSTM Networks"**
  Link: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  Section: "The Core Idea Behind RNNs" section only (for understanding sequential value estimation context).
  Why: While NeuroDrive does not use LSTMs, the diagrams explaining how information flows through time steps build excellent intuition for understanding temporal difference chains and bootstrapping.
  Difficulty: Intermediate | Time: 30 minutes

---

## Actor-Critic / A2C

- [ ] **Sutton & Barto — Section 13.5: Actor-Critic Methods**
  Link: http://incompleteideas.net/book/RLbook2020.pdf
  Section: Pages 341–345. The transition from REINFORCE with baseline to a full actor-critic architecture. Focus on how the critic replaces Monte Carlo returns with TD estimates.
  Why: This is the theoretical bridge from REINFORCE to A2C. Understand why using a learned baseline (the critic) reduces variance while introducing controllable bias.
  Difficulty: Advanced | Time: 2 hours

- [ ] **Lilian Weng — "Policy Gradient Algorithms"**
  Link: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
  Section: Full post, but focus on "Actor-Critic" and "A3C / A2C" subsections. The notation is consistent with Sutton & Barto.
  Why: The best single-page reference connecting REINFORCE → actor-critic → A2C → A3C → PPO. The A2C section explains synchronous advantage estimation clearly and includes pseudocode.
  Difficulty: Intermediate–Advanced | Time: 1.5 hours

- [ ] **OpenAI Spinning Up — VPG (Vanilla Policy Gradient) Documentation**
  Link: https://spinningup.openai.com/en/latest/algorithms/vpg.html
  Section: Full page including pseudocode and "Key Equations." VPG with GAE is essentially A2C.
  Why: Clean pseudocode you can compare directly against NeuroDrive's `update.rs`. The "Quick Facts" box summarises on-policy vs off-policy distinctions.
  Difficulty: Intermediate | Time: 1 hour

- [ ] **Mnih et al. (2016) — "Asynchronous Methods for Deep Reinforcement Learning"**
  Link: https://arxiv.org/abs/1602.01783
  Section: Sections 1–4. Focus on Section 4 (A3C) and note that A2C is the synchronous, single-worker variant.
  Why: The original A3C paper. NeuroDrive's A2C is the single-threaded version of this — understanding the original helps you explain "why not A3C?" in interviews.
  Difficulty: Advanced | Time: 2 hours

---

## Generalised Advantage Estimation (GAE)

- [ ] **Schulman et al. (2016) — "High-Dimensional Continuous Control Using Generalized Advantage Estimation"**
  Link: https://arxiv.org/abs/1506.02438
  Section: Sections 1–3 (derivation from first principles), Section 4 (bias-variance analysis). The full paper is 10 pages and highly readable.
  Why: The original GAE paper. You must understand the derivation of GAE_t = Σ (γλ)^l δ_{t+l} and why λ controls the bias-variance trade-off. Work through the paper with pen and paper.
  Difficulty: Advanced | Time: 3–4 hours

- [ ] **Sutton & Barto — Sections 12.1–12.5 (Eligibility Traces)**
  Link: http://incompleteideas.net/book/RLbook2020.pdf
  Section: Pages 287–310. TD(λ) and eligibility traces — the predecessor framework to GAE.
  Why: GAE is essentially a policy-gradient adaptation of TD(λ). Understanding eligibility traces here builds direct intuition for the λ parameter in GAE and foreshadows NeuroDrive's planned biological eligibility trace system.
  Difficulty: Advanced | Time: 3 hours

- [ ] **OpenAI Spinning Up — VPG "Key Equations" section**
  Link: https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations
  Section: The GAE derivation in the Key Equations box, plus the accompanying implementation in `core.py`.
  Why: A clean, minimal implementation you can step through line by line. Compare against NeuroDrive's `buffer.rs` and the `gae_computation.py` exercise.
  Difficulty: Intermediate–Advanced | Time: 30 minutes

- [ ] **Lilian Weng — "Policy Gradient Algorithms" (GAE section)**
  Link: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#gae
  Section: The "GAE" subsection. Concise derivation connecting GAE to the broader actor-critic landscape.
  Why: If the Schulman paper is too dense on first reading, this blog post distils the key ideas in a more accessible format. Read this first, then return to the paper.
  Difficulty: Intermediate | Time: 20 minutes
