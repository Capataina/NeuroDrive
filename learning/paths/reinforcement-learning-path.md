# Reinforcement Learning Path

## Who This Path Is For

This path is for learners who want to understand the reinforcement learning theory that NeuroDrive's A2C baseline is built on. It takes you from the foundational MDP formalism through to the specific implementation choices in the project — GAE, Gaussian policies, tanh squashing, and the actor-critic update step.

It is the right path if you want to understand *why* the A2C code looks the way it does, not just *what* it does.

## What This Path Assumes

- You know what a neural network's forward pass does (or have completed `paths/foundations-path.md`)
- Basic familiarity with probability (Gaussian distribution, expectations)
- No prior RL knowledge required

## What You Will Understand by the End

- What a Markov Decision Process is and how it formalises the driving task
- What value functions and the Bellman equation represent
- Why the policy gradient theorem works and how it differs from supervised learning
- What the advantage function is and why it reduces variance
- How GAE interpolates between Monte Carlo returns and one-step TD
- What the actor-critic architecture does and how the two networks relate
- Why continuous-action control requires Gaussian policies instead of discrete softmax
- Why tanh squashing is used and how the log-probability correction works
- How the NeuroDrive A2C implementation maps each concept to Rust code
- What the current implementation's known gaps are and why they matter

## Recommended Sequence

- [ ] `concepts/core/reinforcement-learning.md`
  - The foundational layer: MDP, return, discount, value function, Bellman equations. Read this completely before anything else in this path.

- [ ] `concepts/core/policy-gradient-methods.md`
  - Policy parameterisation, the policy gradient theorem, REINFORCE. Includes a full derivation and worked example.

- [ ] `concepts/core/advantage-estimation.md`
  - Why raw returns produce high-variance gradients, the advantage function definition, MC vs TD, and the full GAE derivation. Read the worked example carefully.

- [ ] `concepts/core/continuous-control.md`
  - Gaussian policies, the reparameterisation trick, tanh squashing, and why the log-prob needs a Jacobian correction.

- [ ] `concepts/core/actor-critic-architecture.md`
  - How actor and critic are combined, the A3C/A2C distinction, separate vs shared networks, and what the entropy bonus does.

- [ ] `exercises/core/implement-gae.md`
  - Implement GAE from the recurrence definition before reading the source code. This is the most important single exercise for understanding the A2C update.

- [ ] `project/systems/a2c-brain.md`
  - Now read the project's A2C implementation. You should be able to map every part back to the concepts you have studied.

- [ ] `project/decisions/a2c-as-baseline.md`
  - Why A2C was chosen, what it proves, and when the project should move beyond it.

- [ ] `exercises/core/trace-the-policy-gradient.md`
  - Trace through one complete A2C policy update step using the NeuroDrive code.

- [ ] `project/comparisons/a2c-vs-ppo.md`
  - Understand what PPO adds relative to A2C and why NeuroDrive does not use it.

## After This Path

From here, proceed to:

- `paths/neuroscience-path.md` — understand the biological learning direction the project is headed
- `paths/project-architecture-path.md` — understand the full runtime that A2C operates inside
- `materials/reinforcement-learning-resources.md` — go deeper into the source literature

## Notes

- GAE is the conceptual centrepiece of this path. Spend disproportionate time on `concepts/core/advantage-estimation.md` and `exercises/core/implement-gae.md`.
- The tanh squashing log-probability correction in `concepts/core/continuous-control.md` is easy to implement incorrectly. The worked example in that file is essential reading before looking at `src/brain/a2c/update.rs`.
- The `project/systems/a2c-brain.md` file is deliberately code-grounded. Do not read it before the concept files — the theory context makes it much easier to understand.
