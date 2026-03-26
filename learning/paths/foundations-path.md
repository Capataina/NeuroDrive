# Foundations Path

## Who This Path Is For

This path is for learners who want to build up from mathematical and mechanical foundations before studying the NeuroDrive project or reinforcement learning. It is the right starting point if you are unfamiliar with neural networks, do not yet know what backpropagation is, or want to understand the handwritten Rust ML primitives at a deep level.

If you already know how feedforward networks, gradient descent, and Gaussian distributions work, you can skip this path or use it selectively to fill specific gaps.

## What This Path Assumes

- Comfortable with basic algebra and function composition
- Familiar with programming (any language)
- No prior neural network, optimisation, or probability theory knowledge required

## What You Will Understand by the End

- How a feedforward neural network computes its output (the forward pass)
- How backpropagation computes gradients through each layer
- What gradient descent does and why it needs gradient clipping
- How the Adam optimiser works and why it is preferred over vanilla SGD
- How Gaussian distributions work and how to sample from them
- What log-probability and entropy mean and why they appear in RL
- How NeuroDrive's handwritten `Linear` and `Relu` layers implement the above
- How NeuroDrive's `AdamOptimizer` implements the Adam update rule
- What the Bevy Entity Component System model is and why NeuroDrive uses it

## Recommended Sequence

- [ ] `concepts/foundations/neural-networks.md`
  - Read the full file. Pay close attention to the backpropagation derivation and the chain rule walkthrough.
- [ ] `exercises/foundations/implement-linear-layer.md`
  - Implement a forward+backward linear layer without looking at the NeuroDrive source. Verify your gradients numerically.
- [ ] `exercises/foundations/implement-relu-backprop.md`
  - Implement ReLU forward and backward passes. Check the dead-ReLU behaviour.
- [ ] `concepts/foundations/optimization-and-gradients.md`
  - Read through gradient descent, momentum, and the full Adam derivation.
- [ ] `exercises/foundations/implement-adam-optimizer.md`
  - Implement Adam step by step. Verify bias correction behaviour.
- [ ] `concepts/foundations/probability-and-distributions.md`
  - Read the Gaussian distribution, sampling, log-probability, and entropy sections carefully. These are prerequisites for understanding the Gaussian policy.
- [ ] `concepts/foundations/bevy-ecs-primer.md`
  - Read this to understand how NeuroDrive is structured as a game application. You do not need to become a Bevy expert, but you need to know what an entity, component, resource, and system mean.

## After This Path

From here, proceed to:

- `paths/reinforcement-learning-path.md` — the natural next step to understand what A2C is doing
- `concepts/core/reinforcement-learning.md` — start the RL theory immediately

## Notes

- The exercises in the foundations tier are **reconstruction exercises**: you are expected to implement the concepts from scratch without copying the NeuroDrive code. The goal is verified understanding, not code reuse.
- The `concepts/foundations/` files are longer than typical overview documents. Do not skim them. The worked examples and derivations are load-bearing.
- `bevy-ecs-primer.md` is a lighter-touch file. You do not need deep Bevy expertise to understand NeuroDrive, but you need to understand ECS well enough to follow system scheduling and resource mutation patterns.
