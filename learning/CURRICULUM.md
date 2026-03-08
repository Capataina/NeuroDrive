# Learning Curriculum: NeuroDrive

## Overview

This curriculum transforms the NeuroDrive codebase into a self-contained educational resource. NeuroDrive is a brain-inspired AI research project built in Rust/Bevy: a 2D top-down racing environment where a learning agent must acquire driving behaviour using mechanisms inspired by biological neural systems. The project implements everything from scratch — no ML frameworks, no backpropagation libraries, no evolutionary toolkits.

After completing this curriculum, you will be able to:
- Explain how neural networks, reinforcement learning, and brain-inspired plasticity work from first principles
- Implement a complete actor-critic RL agent from scratch (forward pass, backpropagation, optimiser, rollout buffer, GAE)
- Design deterministic simulation environments with proper sensor models and reward shaping
- Articulate architecture decisions and trade-offs confidently in technical interviews

**Estimated total time**: 8–12 weeks (assuming 8–10 hours per week)

**Learning path**: Mostly linear (Phases 1–3 build on each other), with Phase 4 and beyond being more modular.

---

## How To Use This Curriculum

- ✅ Check off items as you complete them
- ⭐ Star indicates critical path (must understand deeply)
- 🔵 Blue circle indicates optional (enrichment, nice-to-know)

**Different learning styles**:
- **Bottom-up learner**: Start with Phase 1 (foundations), work sequentially
- **Top-down learner**: Read `project_specific/architecture_decisions.md` first, then backfill concepts as needed
- **Hands-on learner**: Jump to exercises after each concept, read theory when you get stuck

---

## Phase 1: Mathematical and Programming Foundations (Est. 2 weeks)

### Learning Goals
After this phase, you will be able to:
- Perform matrix-vector operations and explain why they matter for neural networks
- Compute partial derivatives and apply the chain rule
- Work with probability distributions, especially the Gaussian
- Understand Rust ownership and Bevy's Entity-Component-System model

### Prerequisites
If you already have a solid undergraduate-level understanding of these topics, skip directly to Phase 2.
- [ ] Linear algebra (vectors, matrices, dot products)
- [ ] Single-variable calculus (derivatives, integrals)
- [ ] Basic probability (random variables, distributions)
- [ ] Rust basics (ownership, borrowing, traits, enums)

### Concepts to Learn
- [ ] ⭐ `concepts/foundations/linear_algebra_for_ml.md` (Est. 3–4 hours)
      Vectors, matrices, affine transforms — the language of neural networks.
- [ ] ⭐ `concepts/foundations/calculus_and_gradients.md` (Est. 3–4 hours)
      Derivatives, partial derivatives, chain rule, gradient descent — the engine of learning.
- [ ] ⭐ `concepts/foundations/probability_and_distributions.md` (Est. 3–4 hours)
      Gaussian distributions, log-probabilities, entropy — the mathematics of policy outputs.

### Materials
- [ ] `materials/neural_networks.md` — Foundations section
- [ ] `materials/rust_and_bevy.md` — Rust and ECS fundamentals

### Exercises
- [ ] `exercises/minimal_implementations/dot_product_and_matmul.py`

### Phase Milestone
**Deliverable**: Given a 2-layer neural network (weights, biases, ReLU activations), manually compute the forward pass output on paper for a 3-dimensional input vector. Verify with the exercise code.

---

## Phase 2: Neural Networks from Scratch (Est. 2–3 weeks)

### Learning Goals
After this phase, you will be able to:
- Explain how a multi-layer perceptron transforms inputs to outputs
- Derive and implement backpropagation by hand
- Explain why weight initialisation matters and how Glorot/Xavier works
- Implement the Adam optimiser and explain its advantages over SGD
- Read NeuroDrive's `src/brain/common/` code and understand every line

### Concepts to Learn
- [ ] ⭐ `concepts/core/neural_networks/forward_pass_and_layers.md` (Est. 4–5 hours)
      Linear layers, ReLU, Tanh — how data flows through a network.
- [ ] ⭐ `concepts/core/neural_networks/backpropagation.md` (Est. 6–8 hours)
      The chain rule applied to computational graphs — how networks learn.
- [ ] ⭐ `concepts/core/neural_networks/weight_initialisation_and_optimisers.md` (Est. 3–4 hours)
      Glorot init, SGD, momentum, Adam — setting up and driving the learning process.

### Materials
- [ ] `materials/neural_networks.md` — Core and advanced sections

### Exercises
- [ ] ⭐ `exercises/minimal_implementations/simple_mlp.py` — Build a 2-layer MLP from scratch
- [ ] `exercises/debugging_challenges/broken_backprop.py` — Find and fix a gradient computation bug

### Phase Milestone
**Deliverable**: Implement a complete 2-layer MLP in Python (forward pass, backpropagation, Adam optimiser) that learns to approximate sin(x) on [−π, π]. No ML libraries allowed — only NumPy for array operations. Then read NeuroDrive's `src/brain/common/mlp.rs` and `optim.rs` and explain the correspondence between your Python and their Rust.

---

## Phase 3: Reinforcement Learning — From MDPs to A2C (Est. 3–4 weeks)

### Learning Goals
After this phase, you will be able to:
- Formalise a control problem as a Markov Decision Process
- Derive the policy gradient theorem and explain REINFORCE
- Explain why baselines reduce variance and how critics work
- Implement Generalised Advantage Estimation (GAE) from scratch
- Explain the complete A2C algorithm and read NeuroDrive's implementation confidently
- Articulate the difference between on-policy and off-policy methods

### Concepts to Learn
- [ ] ⭐ `concepts/core/reinforcement_learning/markov_decision_processes.md` (Est. 3–4 hours)
      States, actions, transitions, rewards, discounting — the formal framework.
- [ ] ⭐ `concepts/core/reinforcement_learning/policy_gradients.md` (Est. 5–7 hours)
      REINFORCE, the log-probability trick, variance reduction — learning policies directly.
- [ ] ⭐ `concepts/core/reinforcement_learning/value_functions_and_critics.md` (Est. 4–5 hours)
      State-value, action-value, temporal difference, critic networks — estimating long-term value.
- [ ] ⭐ `concepts/core/reinforcement_learning/advantage_estimation_gae.md` (Est. 4–5 hours)
      TD residuals, GAE-λ, bias-variance trade-off — practical advantage computation.
- [ ] ⭐ `concepts/core/reinforcement_learning/a2c.md` (Est. 5–7 hours)
      The complete algorithm: actor-critic architecture, rollout collection, policy/value losses, entropy regularisation.

### Materials
- [ ] `materials/reinforcement_learning.md` — All sections

### Exercises
- [ ] ⭐ `exercises/minimal_implementations/reinforce.py` — REINFORCE on CartPole
- [ ] ⭐ `exercises/minimal_implementations/gae_computation.py` — GAE from a hand-crafted trajectory
- [ ] `exercises/debugging_challenges/diverging_policy_gradient.py` — Find the bug in a broken A2C

### Phase Milestone
**Deliverable**: Implement A2C from scratch in Python that solves CartPole-v1 (average reward > 195 over 100 episodes). Then read NeuroDrive's `src/brain/a2c/` directory end-to-end and write a 1-page comparison document explaining: (a) what is identical, (b) what differs, and (c) why each difference exists.

---

## Phase 4: Environment Design and Simulation Patterns (Est. 2 weeks)

### Learning Goals
After this phase, you will be able to:
- Explain how NeuroDrive's deterministic simulation works and why determinism matters
- Understand grid-based spatial systems and their O(1) point-in-area queries
- Explain raycast-based observation models and the binary-search refinement technique
- Design reward functions that provide dense learning signals without reward hacking
- Understand Bevy's ECS architecture and plugin composition pattern

### Concepts to Learn
- [ ] ⭐ `concepts/domain_patterns/simulation/deterministic_simulation.md` (Est. 2–3 hours)
      Fixed timesteps, system ordering, replay — why determinism is non-negotiable.
- [ ] ⭐ `concepts/domain_patterns/spatial_systems/grid_based_spatial_systems.md` (Est. 3–4 hours)
      Tile grids, coordinate transforms, O(1) spatial queries — the map foundation.
- [ ] ⭐ `concepts/domain_patterns/agent_perception/raycast_observation.md` (Est. 3–4 hours)
      Ray marching, binary search refinement, normalised observation vectors.
- [ ] ⭐ `concepts/domain_patterns/simulation/episode_based_rl_environments.md` (Est. 3–4 hours)
      Episode lifecycle, reward shaping, terminal conditions, moving averages.
- [ ] 🔵 `concepts/domain_patterns/ecs_architecture/entity_component_system.md` (Est. 2–3 hours)
      ECS fundamentals, Bevy plugins, system sets, resource vs component patterns.

### Materials
- [ ] `materials/rust_and_bevy.md` — ECS and simulation sections

### Exercises
- [ ] `exercises/minimal_implementations/grid_raycast.py` — Grid + raycasting from scratch

### Phase Milestone
**Deliverable**: Given a blank grid and a set of tile types, implement a minimal 2D environment with: (a) grid-based collision detection, (b) 5-ray observation model, (c) progress-based reward. Verify the observation vector looks reasonable by printing it for known car positions.

---

## Phase 5: NeuroDrive Project Deep-Dive (Est. 1–2 weeks)

### Learning Goals
After this phase, you will be able to:
- Explain every architectural decision in NeuroDrive and defend the trade-offs
- Trace data flow from sensor readings through the neural network to wheel commands
- Read any file in the repository and understand its role in the system
- Identify current limitations and propose improvements

### Concepts to Learn
- [ ] ⭐ `project_specific/architecture_decisions.md` (Est. 2–3 hours)
      Why Rust, why Bevy, why from-scratch, why A2C before plasticity.
- [ ] ⭐ `project_specific/implementations/handwritten_neural_network.md` (Est. 2–3 hours)
      NeuroDrive's MLP, cache-based autograd, manual gradient accumulation.
- [ ] ⭐ `project_specific/implementations/car_physics_model.md` (Est. 1–2 hours)
      Euler integration, drag model, pure-function extraction for replay.
- [ ] ⭐ `project_specific/implementations/observation_system.md` (Est. 2–3 hours)
      Ray angles, normalisation scales, two-phase sensor pipeline.
- [ ] 🔵 `project_specific/implementations/progress_and_lap_system.md` (Est. 1–2 hours)
      Centreline projection, arc-length parameterisation, lap-wrap detection.
- [ ] 🔵 `project_specific/implementations/analytics_pipeline.md` (Est. 1–2 hours)
      Episode tracking, chunked metrics, JSON/Markdown export, A2C health telemetry.

### Phase Milestone
**Deliverable**: Without looking at the code, draw a complete data-flow diagram of NeuroDrive: from key press / AI inference through physics, collision, progress, reward, observation, and back to the next action. Label each arrow with the Bevy resource or component that carries the data. Then verify against the actual code.

---

## Phase 6: Brain-Inspired Learning — The Road Ahead (Est. 2–3 weeks)

### Learning Goals
After this phase, you will be able to:
- Explain Hebbian plasticity, STDP, eligibility traces, and neuromodulation
- Articulate how biological learning differs from gradient-based optimisation
- Understand NeuroDrive's planned Milestones 2–5 and the scientific rationale
- Propose design decisions for implementing local plasticity in the existing codebase

### Concepts to Learn
- [ ] ⭐ `concepts/core/neuroscience/hebbian_plasticity.md` (Est. 3–4 hours)
      "Fire together, wire together" — the foundation of biological learning.
- [ ] ⭐ `concepts/core/neuroscience/stdp_and_eligibility_traces.md` (Est. 4–5 hours)
      Spike timing, eligibility windows, three-factor learning rules.
- [ ] ⭐ `concepts/core/neuroscience/neuromodulation_and_dopamine.md` (Est. 3–4 hours)
      Reward prediction error, dopamine gating, the bridge to RL.
- [ ] 🔵 `concepts/core/neuroscience/structural_plasticity.md` (Est. 2–3 hours)
      Synapse growth, pruning, capacity allocation — how brains rewire.

### Materials
- [ ] `materials/neuroscience.md` — All sections

### Phase Milestone
**Deliverable**: Write a 2-page design sketch for NeuroDrive's Milestone 2 (rate-based local plasticity + δ gating). Specify: (a) the data structures for a sparse neural graph, (b) the eligibility trace update rule, (c) the neuromodulatory signal computation, (d) how it would integrate with the existing `Brain` trait and `SimSet` pipeline. Reference the README's specifications.

---

## Interview Preparation

After completing this curriculum, you should be able to confidently answer:

1. **"How does NeuroDrive work?"**
   Start with the motivation (brain-inspired learning, not benchmark chasing). Describe the environment (2D racing, deterministic physics, raycast sensors). Explain the current A2C baseline (handwritten MLP, Gaussian policy, GAE, Adam). Then explain the planned transition to local plasticity (Hebbian/STDP + eligibility traces + neuromodulatory gating). Emphasise: everything from scratch in Rust, no ML frameworks.

2. **"Walk me through how the agent learns."**
   Observation vector (11 rays + speed + heading error + angular velocity) → forward pass through actor MLP → sample from Gaussian → clamp to valid range → physics step → reward computation (progress gain + time penalty + terminal rewards) → collect in rollout buffer → after 2048 steps, compute GAE → backpropagate policy and value losses → Adam update → repeat. Each synapse update in the future will be local, not gradient-based.

3. **"Why did you choose A2C over PPO/SAC/DQN?"**
   A2C is the simplest on-policy actor-critic that validates learnability. It requires no replay buffer (rules out off-policy), no clipping ratio (simpler than PPO), and handles continuous actions naturally (rules out vanilla DQN). It is explicitly a diagnostic baseline — the real goal is brain-inspired plasticity, not RL benchmarking.

4. **"What's the hardest technical challenge you faced?"**
   Implementing backpropagation from scratch: manual cache-based autograd for Linear/ReLU/Tanh layers, separate actor and critic gradient paths, handwritten Adam with per-layer moment vectors. Ensuring numerical stability (clamped log-std, advantage normalisation, gradient accumulation). Then integrating this into a real-time Bevy ECS simulation with deterministic fixed-timestep ordering.

5. **"What would you do differently if you rebuilt this?"**
   Consider using a computation graph / tape-based autograd rather than manual layer-by-layer backward passes. Implement a proper bounded action distribution (Beta or squashed Gaussian) rather than Gaussian + clamp. Add more comprehensive regression tests for the learning pipeline, not just physics determinism.

**Practice exercise**: Record yourself explaining NeuroDrive to a rubber duck. You should be able to give a coherent, technically accurate 5-minute overview without notes.
