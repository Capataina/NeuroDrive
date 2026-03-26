# Study Guide

This guide helps you choose a route through the archive based on your goal.
The archive is large and multi-directional. This file does not define a single correct sequence — it helps you find the one that fits your purpose.

---

## Choose a Route

### Route A — I want to understand the full project from the top down

Start with architecture, then drill into each subsystem, then connect the RL theory to the code.

- [ ] `LEARNING_MAP.md` — understand the archive's scope
- [ ] `project/architecture/runtime-overview.md` — see the system shape
- [ ] `project/architecture/fixed-tick-pipeline.md` — understand the execution order
- [ ] `project/systems/environment-system.md` — the racing world
- [ ] `project/systems/agent-interface.md` — the observation/action contract
- [ ] `project/systems/a2c-brain.md` — the current learning system
- [ ] `concepts/core/reinforcement-learning.md` — RL foundations
- [ ] `concepts/core/actor-critic-architecture.md` — what A2C is doing
- [ ] `concepts/core/advantage-estimation.md` — GAE in detail
- [ ] `project/evolution/from-baseline-to-brain.md` — where it goes next

**Recommended path file:** `paths/project-architecture-path.md`

---

### Route B — I want to learn the RL theory properly, then apply it here

Start with theory and work toward the implementation.

- [ ] `concepts/foundations/neural-networks.md` — the prerequisite mechanics
- [ ] `concepts/foundations/probability-and-distributions.md` — Gaussian policies
- [ ] `concepts/foundations/optimization-and-gradients.md` — Adam and gradient clipping
- [ ] `concepts/core/reinforcement-learning.md` — MDP, value functions, Bellman
- [ ] `concepts/core/policy-gradient-methods.md` — the policy gradient theorem
- [ ] `concepts/core/advantage-estimation.md` — GAE, variance reduction
- [ ] `concepts/core/actor-critic-architecture.md` — A2C, A3C, and variants
- [ ] `concepts/core/continuous-control.md` — Gaussian policies, tanh squashing
- [ ] `project/systems/a2c-brain.md` — the NeuroDrive implementation
- [ ] `project/decisions/a2c-as-baseline.md` — why and how to evaluate it

**Recommended path file:** `paths/reinforcement-learning-path.md`

---

### Route C — I want to understand the biological learning science behind the roadmap

Start with foundations, move into neuroscience, then study the planned architecture.

- [ ] `concepts/advanced/hebbian-plasticity.md` — co-activation rules
- [ ] `concepts/advanced/spike-timing-dependent-plasticity.md` — STDP mechanics
- [ ] `concepts/advanced/eligibility-traces.md` — delayed credit assignment
- [ ] `concepts/advanced/neuromodulation.md` — dopamine-like gating
- [ ] `concepts/advanced/structural-plasticity.md` — growth and pruning
- [ ] `concepts/advanced/continual-learning.md` — lifelong learning
- [ ] `project/comparisons/rate-based-vs-spiking.md` — how these concepts enter the project
- [ ] `project/evolution/from-baseline-to-brain.md` — the transition plan
- [ ] `project/evolution/milestone-roadmap.md` — Milestones 2–9

**Recommended path file:** `paths/neuroscience-path.md`

---

### Route D — I want the fastest path to understanding the live code

Read the code-grounded project files first, then fill in theory gaps.

- [ ] `references/observation-vector-reference.md` — the 23-dim input
- [ ] `project/systems/agent-interface.md` — how observations are built
- [ ] `project/systems/environment-system.md` — the environment contract
- [ ] `project/systems/a2c-brain.md` — rollout, GAE, update
- [ ] `concepts/core/advantage-estimation.md` — GAE maths
- [ ] `project/architecture/fixed-tick-pipeline.md` — the execution order
- [ ] `project/systems/analytics-system.md` — how runs are recorded
- [ ] `exercises/core/implement-gae.md` — verify understanding through practice

**Recommended path file:** `paths/implementation-first-path.md`

---

### Route E — I want to build foundations from scratch before anything else

Start from pure maths and mechanics, work up to the project.

- [ ] `concepts/foundations/neural-networks.md`
- [ ] `concepts/foundations/optimization-and-gradients.md`
- [ ] `concepts/foundations/probability-and-distributions.md`
- [ ] `concepts/foundations/bevy-ecs-primer.md`
- [ ] `exercises/foundations/implement-linear-layer.md`
- [ ] `exercises/foundations/implement-relu-backprop.md`
- [ ] `exercises/foundations/implement-adam-optimizer.md`
- [ ] `concepts/core/reinforcement-learning.md`
- [ ] `concepts/core/policy-gradient-methods.md`

**Recommended path file:** `paths/foundations-path.md`

---

### Route F — I want to understand the research direction (Milestones 2–9)

Focus on the science and architecture behind where the project is going.

- [ ] `project/evolution/milestone-roadmap.md` — full roadmap with learning context
- [ ] `concepts/advanced/hebbian-plasticity.md`
- [ ] `concepts/advanced/eligibility-traces.md`
- [ ] `concepts/advanced/neuromodulation.md`
- [ ] `concepts/advanced/spike-timing-dependent-plasticity.md`
- [ ] `concepts/advanced/structural-plasticity.md`
- [ ] `concepts/advanced/continual-learning.md`
- [ ] `project/comparisons/rate-based-vs-spiking.md`
- [ ] `project/evolution/from-baseline-to-brain.md`
- [ ] `exercises/project/sketch-eligibility-traces.md`
- [ ] `materials/neuroscience-resources.md`

**Recommended path file:** `paths/research-directions-path.md`

---

## Suggested Combinations

Some routes pair well together:

- [ ] **Route B + Route C** — full theory coverage from RL through to biological learning. Probably the most intellectually complete study path for this project.
- [ ] **Route D + Route B** — understand the code first, then deepen the theory. Good if you are already comfortable with RL vocabulary.
- [ ] **Route E + Route B** — build from the absolute ground up. Best for learners who are new to both neural networks and RL.
- [ ] **Any route + `exercises/EXERCISE_ORDER.md`** — exercises should accompany any study route for the concepts that have practice materials.

---

## Starting-Point Advice by Background

| Background | Suggested start |
|---|---|
| New to RL but familiar with neural networks | Route B; skip `concepts/foundations/neural-networks.md` |
| Familiar with RL but not Rust or Bevy | Route D; read `concepts/foundations/bevy-ecs-primer.md` first |
| Familiar with both RL and systems programming | Route D directly |
| Neuroscience background, new to RL | Route C, then Route B |
| Starting completely from scratch | Route E, then Route B |
| Wants to contribute to Milestone 2 | Route B + Route C + `project/evolution/from-baseline-to-brain.md` |

---

## Using Exercises

Exercises are scattered across the archive and are sequenced in `exercises/EXERCISE_ORDER.md`. You do not need to complete all exercises before continuing with theory. A good pattern is:

- Read a concept file deeply.
- Do the associated exercise while the material is fresh.
- Return to the concept file if the exercise reveals gaps.

The exercise guide (`exercises/EXERCISE_GUIDE.md`) explains the exercise types and hint strategy.
