# NeuroDrive

## Project Description

**NeuroDrive** is a real-time, brain-inspired AI research project built around a custom 2D top-down racing environment.  
The core objective is not to benchmark algorithms against each other, nor to maximise score using established ML libraries.  
Instead, NeuroDrive is a focused attempt to answer a specific question:

> Can we build a learning system from scratch that mimics how the human brain learns, and watch it gradually acquire driving behaviour in real time?

This project is written entirely in Rust, using Bevy for simulation and rendering.  
All learning logic, plasticity rules, and structural adaptation mechanisms are implemented from first principles.

---

## What Does the Human Brain Actually Do When It Learns?

### In Simple Terms

The human brain is a large, sparsely connected network of neurons.  
Each neuron sends signals to other neurons through connections called synapses.

Learning happens when:

- Frequently co-active neurons strengthen their connections.
- Rarely co-active neurons weaken or lose connections.
- A global “reward” signal (such as dopamine) reinforces patterns that lead to good outcomes.
- Over longer periods, connections may grow or be pruned based on usage.

The brain does **not**:

- Run backpropagation.
- Compute global gradients.
- Rely on random mutation across generations.
- Rebuild itself from scratch after every failure.

Instead, it:

- Adjusts connections locally.
- Is influenced by global reward signals.
- Gradually reorganises itself through structural plasticity.
- Learns continuously during experience.

---

### In Scientific Terms

Biological learning is believed to involve:

- **Hebbian plasticity**:  
  Synaptic weight changes based on correlated activity between presynaptic and postsynaptic neurons.

- **Spike-Timing Dependent Plasticity (STDP)**:  
  Precise timing between spikes determines strengthening or weakening.

- **Eligibility traces**:  
  Synapses temporarily store activity correlations, allowing delayed reinforcement.

- **Neuromodulation (dopamine-like signals)**:  
  A global scalar reward prediction error gates whether synaptic changes are consolidated.

- **Structural plasticity**:  
  Long-term formation and pruning of synaptic connections.

Learning is:

- Local (no global gradient transport)
- Incremental
- Dynamical (neurons have internal state)
- Continual (weights evolve during experience)

---

## Core Project Goal

NeuroDrive aims to replicate these principles in an engineered system:

- A dynamically evolving sparse neural graph
- Local synaptic update rules
- Global reward-modulated plasticity
- Optional structural growth and pruning
- Continuous online learning across episodes

We do not use:

- Genetic Algorithms
- NEAT
- Evolution Strategies
- TensorFlow / PyTorch
- Backpropagation-based training

This is not evolution across generations.

This is one persistent “brain” learning within its lifetime.

---

## Environment Overview

The environment is intentionally minimal yet non-trivial:

- Continuous 2D top-down car physics
- Steering + throttle control
- Track boundaries with collision detection
- Progress measured along a centerline spline
- Deterministic, seedable simulation loop

The car must learn to:

- Stay on track
- Maximise forward progress
- Complete laps efficiently
- Avoid catastrophic crashes

This provides a dense, interpretable reinforcement signal without trivialising the task.

---

## Brain Architecture

### High-Level Structure

The agent consists of:

- Fixed input neurons (sensor interface)
- Fixed output neurons (motor interface)
- A dynamic sparse hidden graph
- Local synapses with eligibility traces
- A global reward prediction signal

External boundary:

```
Observation → Brain → Action
```

Internal topology may change over time.

---

### Inputs

- Raycast distance sensors
- Speed
- Heading error relative to track tangent
- Optional angular velocity

These represent the “sensory cortex” of the system.

---

### Outputs

- Steering ∈ [-1, 1]
- Throttle ∈ [0, 1]

These represent motor commands.

Output nodes remain fixed even if hidden topology changes.

---

## Learning Mechanism

### Local Plasticity

Each synapse maintains:

- Weight `w_ij`
- Eligibility trace `e_ij`

Eligibility accumulates correlated activity:

```
e_ij ← λ e_ij + (pre_i × post_j)
```

---

### Global Reward Modulation

A reward prediction error signal δ is computed:

```
δ = r + γ V(s') - V(s)
```

Synaptic update:

```
Δw_ij = η × δ × e_ij
```

This mirrors biological dopamine-modulated Hebbian learning.

No gradients.
No global loss function.
No backpropagation.

---

### Structural Plasticity (Later Milestones)

- Synapses with persistently low magnitude and low eligibility are pruned.
- New synapses may form between recently co-active neurons.
- Fan-in constraints ensure bounded computation.

Topology evolves gradually during experience.

---

## Training Loop

Learning is online and episodic:

1. Reset car position.
2. Run simulation until crash, timeout, or lap completion.
3. Synapses update continuously during rollout.
4. Episode ends; brain persists.
5. Reset and repeat.

The same brain learns across episodes.

No population.
No generational replacement.

---

## Reward Structure

Primary reward:

- Positive reward for forward progress along track centerline.
- Penalty for crashing.
- Bonus for lap completion.

Reward shaping is minimal and interpretable.

Reward is not fitness.
It is a neuromodulatory teaching signal.

---

## Core Design Philosophy

- One environment, one evolving brain
- No external ML libraries
- Fully deterministic simulation
- Behaviour-first evaluation
- Watchable real-time learning
- Structural and synaptic transparency

The project is designed to make learning visible.

---

## Observability & Telemetry

NeuroDrive includes:

- Real-time episode counter
- Progress metrics (max progress, lap %)
- Moving averages across recent episodes
- Weight statistics (mean, distribution)
- Synapse count and sparsity tracking
- Dopamine (δ) visualisation
- Sensor overlays (raycasts)

Learning must be measurable, not guessed.

---

## Features & Roadmap

### 🏎️ Milestone 0 – Environment Foundation

- [x] Deterministic 2D car physics
- [x] Track representation with centerline spline
- [x] Collision detection
- [x] Raycast sensor system
- [x] Reset-on-crash episode loop
- [x] Manual / heuristic baseline controller

---

### 🧠 Milestone 1 – Brain v1 (Synaptic Plasticity Only)

- [ ] Sparse neural graph implementation
- [ ] Local eligibility traces
- [ ] Reward-modulated weight updates
- [ ] Online learning during episodes
- [ ] Real-time telemetry overlays
- [ ] Episode-level progress logging

Goal:  
Observe measurable improvement over 5–10 minutes of training.

---

### 🧪 Milestone 2 – Stability & Analysis

- [ ] Weight clamping and decay mechanisms
- [ ] Learning rate scheduling
- [ ] Moving average performance metrics
- [ ] First-half vs second-half statistical comparison
- [ ] Reward ablation experiments

Goal:  
Ensure learning signal is real and not random drift.

---

### 🌱 Milestone 3 – Structural Plasticity

- [ ] Synapse pruning rules
- [ ] Synapse growth under bounded fan-in
- [ ] Sparsity tracking over time
- [ ] Network topology visualisation
- [ ] Performance comparison: static vs adaptive topology

Goal:  
Evaluate whether structural plasticity improves adaptability.

---

### 🗺️ Milestone 4 – Generalisation

- [ ] Multiple fixed tracks
- [ ] Automatic track cycling on episode reset
- [ ] Interleaved training across tracks
- [ ] Generalisation performance metrics
- [ ] Optional procedural track generation

Goal:  
Test whether learned behaviour generalises beyond memorisation.

---

### 🧠 Milestone 5 – Biological Extensions

- [ ] Spiking neuron model (optional)
- [ ] Spike-Timing Dependent Plasticity
- [ ] Replay / sleep consolidation phase
- [ ] Catastrophic forgetting mitigation
- [ ] Lifelong learning experiments

Goal:  
Move closer to biologically grounded dynamics without abandoning tractability.

---

## What This Project Is Not

- Not a benchmark suite for mainstream RL.
- Not a competition between optimisation paradigms.
- Not a wrapper around PyTorch.
- Not an evolutionary algorithm playground.
- Not an arcade game with AI glued on top.

It is a controlled experiment in building a brain-inspired learning system from first principles.

---

## Why Racing?

A racing environment provides:

- Continuous control
- Dense reward signals
- Spatial reasoning requirements
- Stability vs aggression trade-offs
- Clear measurable progress

It is complex enough to require learning,
but simple enough to keep the focus on the brain model.

---

## Long-Term Vision

NeuroDrive is intended as a research-grade learning laboratory:

- Study synaptic plasticity in engineered systems
- Compare static vs dynamic topology learning
- Investigate reward-modulated local updates
- Explore structural adaptation under real-time constraints
- Understand learning dynamics, not just performance

The ultimate goal is not to create the fastest racing agent.

It is to build a system that visibly, measurably, and continuously learns —  
using principles inspired by how biological brains adapt to the world.
