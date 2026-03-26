# Exercise Order

Recommended sequencing for working through all NeuroDrive exercises from foundations to project-level work.

Each exercise builds on the understanding developed by the previous ones. Prerequisites are listed for each exercise; if you have prior knowledge in the area, you can skip earlier exercises and return to them if you find gaps.

---

## Foundations Tier

These exercises build the mechanical foundations that all subsequent concepts depend on. Even if you are familiar with neural networks, working through these in the NeuroDrive context is valuable.

- [ ] `foundations/implement-linear-layer.md`
  - Prerequisite: `concepts/foundations/neural-networks.md` sections on linear layers and forward pass
  - Skills: matrix multiply, weight/bias initialisation, gradient derivation

- [ ] `foundations/implement-relu-backprop.md`
  - Prerequisite: `foundations/implement-linear-layer.md`, `concepts/foundations/neural-networks.md` (backprop section)
  - Skills: ReLU gate, chain rule through nonlinearity, dead neuron problem

- [ ] `foundations/implement-adam-optimizer.md`
  - Prerequisite: `foundations/implement-relu-backprop.md`, `concepts/foundations/optimization-and-gradients.md`
  - Skills: moment estimates, bias correction, adaptive learning rates, gradient clipping

---

## Core RL Tier

These exercises require the foundations tier plus the core RL concepts. They directly address the NeuroDrive A2C implementation.

- [ ] `core/implement-gae.md`
  - Prerequisite: `concepts/core/reinforcement-learning.md`, `concepts/core/advantage-estimation.md`
  - Skills: TD error, GAE backwards recurrence, bootstrap handling, terminal masking

- [ ] `core/trace-the-policy-gradient.md`
  - Prerequisite: `concepts/core/policy-gradient-methods.md`, `concepts/core/continuous-control.md`, `concepts/foundations/probability-and-distributions.md`
  - Skills: Gaussian log-probability, tanh squashing, Jacobian correction, policy gradient with advantage

- [ ] `core/trace-observation-vector.md`
  - Prerequisite: `concepts/domain-patterns/observation-design.md`, `project/systems/agent-interface.md`
  - Skills: 23-dim vector construction, normalisation, centreline-relative features, lookahead sampling

---

## Project Tier

These exercises are grounded in the actual NeuroDrive codebase. They require understanding of the project systems.

- [ ] `project/debug-reward-shaping.md`
  - Prerequisite: `concepts/domain-patterns/reward-shaping.md`, `project/systems/environment-system.md`
  - Skills: reward decomposition, per-tick signal analysis, identifying degenerate reward behaviour

- [ ] `project/extend-observation-vector.md`
  - Prerequisite: `core/trace-observation-vector.md`, `project/systems/agent-interface.md`
  - Skills: OBSERVATION_DIM alignment, feature design, normalisation, module boundary understanding

- [ ] `project/sketch-eligibility-traces.md`
  - Prerequisite: `concepts/advanced/eligibility-traces.md`, `concepts/advanced/neuromodulation.md`, `project/evolution/from-baseline-to-brain.md`
  - Skills: designing the per-synapse trace structure, update ordering, δ broadcast integration

---

## Completion

When all exercises are checked off, revisit the concepts that felt least solid during the exercises — usually the ones where you needed the most hints. Return to the relevant concept file and re-read it with the benefit of the implementation experience.
