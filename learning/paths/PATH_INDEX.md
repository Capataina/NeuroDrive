# Path Index

This file describes the available learning paths and helps you choose which one to start with.

Each path has a specific audience and emphasis. Paths overlap by design — the same concept file may appear in multiple paths from different angles. That is intentional: reading a concept in the context of RL theory and then again in the context of the live code are genuinely different learning experiences.

---

## Available Paths

### `foundations-path.md`

**For:** Learners who want to build up mathematical and mechanical understanding before touching the project code. Best for those new to neural networks, automatic differentiation, or optimisation.

**Covers:** Feedforward networks, ReLU, backpropagation, gradient descent, Adam, Gaussian distributions, log-probability, entropy. Ends with the first foundation-layer exercises.

**Assumes:** Basic programming familiarity. No prior RL or neural network knowledge required.

**Time investment:** High. This path builds everything from scratch.

---

### `reinforcement-learning-path.md`

**For:** Learners who want to understand the RL theory that underpins the A2C implementation, from Markov decision processes through to the full actor-critic update step.

**Covers:** MDPs, Bellman equations, Monte Carlo vs TD learning, policy gradients, advantage estimation, GAE, continuous-action Gaussian policies, actor-critic architecture, and the project's specific A2C choices.

**Assumes:** Basic neural network familiarity (at least knows what a forward pass is).

**Time investment:** High. RL theory is not shallow.

---

### `neuroscience-path.md`

**For:** Learners who want to understand the biological learning science behind Milestones 2–9. This path covers what the project is actually trying to become, not just what it is today.

**Covers:** Hebbian learning, STDP, eligibility traces, neuromodulation, dopamine reward-prediction error, structural plasticity, continual learning, and the contrast with gradient-based methods.

**Assumes:** General scientific literacy. No prior RL or neural network knowledge strictly required, but reading `concepts/core/reinforcement-learning.md` first will make the comparisons more meaningful.

**Time investment:** High. These concepts are technically demanding in different ways from RL.

---

### `project-architecture-path.md`

**For:** Learners who want to understand NeuroDrive's runtime structure — how the code is organised, what each subsystem owns, and how the fixed-tick pipeline flows from input to analytics.

**Covers:** Module structure, Bevy ECS model, SimSet pipeline, environment system, agent interface, brain plugin, analytics, debug runtime, and the dependency graph.

**Assumes:** Some familiarity with game-engine or ECS concepts is helpful but not required.

**Time investment:** Moderate.

---

### `implementation-first-path.md`

**For:** Engineers who are already comfortable with RL and want to understand the live NeuroDrive code as quickly as possible without reading foundational theory again.

**Covers:** Observation vector, rollout buffer, GAE implementation, update path, action contract, reward structure, analytics pipeline. Goes straight to the code-grounded project files.

**Assumes:** Solid RL background. Knows what GAE, actor-critic, and rollout buffers are.

**Time investment:** Low to moderate. This is the fastest route to contribution-ready understanding.

---

### `research-directions-path.md`

**For:** Learners interested in the project's long-term scientific mission — the progression from A2C to biologically plausible local plasticity, spiking networks, and structural adaptation.

**Covers:** Full Milestone roadmap, Hebbian learning, STDP, eligibility traces, neuromodulation, structural plasticity, continual learning, the rate-based vs spiking comparison, and the planned architectural transition.

**Assumes:** Basic RL familiarity (understanding what "the A2C baseline validates the environment" means). Neuroscience background is helpful but not required.

**Time investment:** High. The science is rich.

---

## How to Choose

| Your immediate goal | Best path |
|---|---|
| Understand the current code | `implementation-first-path.md` |
| Learn RL properly | `reinforcement-learning-path.md` |
| Understand what Milestone 2 means | `research-directions-path.md` |
| Understand the runtime structure | `project-architecture-path.md` |
| Build from mathematical basics | `foundations-path.md` |
| Understand the neuroscience | `neuroscience-path.md` |

## Recommended Pairings

- [ ] `reinforcement-learning-path.md` + `neuroscience-path.md` — complete theory coverage
- [ ] `foundations-path.md` + `reinforcement-learning-path.md` — build from scratch to A2C
- [ ] `project-architecture-path.md` + `implementation-first-path.md` — full systems understanding
- [ ] `research-directions-path.md` + `exercises/project/sketch-eligibility-traces.md` — roadmap study with hands-on design practice
