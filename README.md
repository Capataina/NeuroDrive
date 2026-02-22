# NeuroDrive

## Project Description

**NeuroDrive** is a real-time, brain-inspired AI research project built around a custom 2D top-down racing environment.  
The goal is _not_ to benchmark standard algorithms, chase leaderboard scores, or outsource learning to external ML frameworks.

Instead, NeuroDrive is a focused attempt to answer one question:

> **Can we build a learning system from scratch that mimics how the human brain learns, and watch it gradually acquire driving behaviour in real time?**

The project is written entirely in **Rust**, using **Bevy** for simulation and rendering.  
All learning logic, plasticity rules, and structural adaptation mechanisms are implemented **from first principles**.

---

## What Does the Human Brain Actually Do When It Learns?

### In Simple Terms

The human brain is a massive, sparsely connected graph of neurons.  
Neurons communicate via synapses whose strengths change as a function of experience.

Learning happens when:

- **Co-activation strengthens connections** (useful correlations get reinforced).
- **Unhelpful connections weaken** (unused patterns fade).
- **Global reward signals modulate plasticity** (dopamine-like signals reinforce what led to better outcomes).
- **Over longer timescales, structure adapts** (connections can form, reorganise, or be pruned).

The brain does **not**:

- Run backpropagation.
- Compute global gradients.
- Train against a single static dataset.
- Reset itself after each failure.

Instead, it:

- Updates connections **locally** using only information available at the synapse.
- Uses global neuromodulation to **gate** which changes become lasting.
- Continually adapts while acting in the world.
- Slowly reshapes its structure through experience-driven plasticity.

**Hence, the brain is neither a typical reinforcement learning system nor an evolutionary algorithm; rather, it learns through ongoing, local adaptation of its own structure and connections, guided but not dictated by rewards, enabling continuous and flexible learning from experience.**

---

### In Scientific Terms (High Signal, Minimal Jargon)

Biological learning is believed to involve a few key mechanisms that compose together:

- **Hebbian plasticity**  
  Synapses strengthen when presynaptic and postsynaptic activity are correlated (“fire together, wire together”).

- **Spike-Timing Dependent Plasticity (STDP)**  
  The _timing_ of spikes matters: pre-before-post tends to strengthen; post-before-pre tends to weaken.

- **Eligibility traces**  
  Synapses maintain a short-lived “memory” of recent correlation, allowing reinforcement to arrive later.

- **Neuromodulation (dopamine-like signals)**  
  A broadcast signal (reward prediction error) gates consolidation: _which changes should stick_.

- **Structural plasticity**  
  Over longer timescales, synapses form/prune and circuits reorganise to allocate capacity where it matters.

Learning is therefore:

- **Local** (credit assignment is done using synapse-local signals, not global gradients)
- **Incremental** (continuous updates rather than rare re-training)
- **Dynamical** (neurons have internal state; behaviour depends on time)
- **Continual** (weights evolve during interaction, not just between episodes)

---

## Core Project Goal

NeuroDrive aims to replicate these principles in an engineered system:

- A **sparse neural graph** with neuron state and synapses
- **Local plasticity** rules (Hebbian / STDP-family)
- **Eligibility traces** for delayed credit assignment
- **Neuromodulation** (dopamine-like reward prediction errors)
- **Structural plasticity** (growth + pruning under constraints)
- **Continuous online learning** across episodes (“one brain, one lifetime”)

We do **not** use:

- Genetic Algorithms / NEAT
- Evolution Strategies
- TensorFlow / PyTorch / JAX
- Backpropagation-based training loops

This is not evolution across generations.  
This is **one persistent “brain”** learning within its lifetime.

---

## Environment Overview

The environment is intentionally minimal yet non-trivial:

- Continuous 2D top-down car physics
- Steering + throttle control
- Track boundaries + collision detection
- Progress measured along a centerline spline (dense signal)
- Deterministic, seedable simulation loop

The car must learn to:

- Stay on track
- Maximise forward progress
- Complete laps efficiently
- Avoid catastrophic crashes

The environment is designed to provide **dense, interpretable learning signals** without turning the task into scripted control.

---

## Brain Architecture

### High-Level Structure

The agent consists of:

- **Fixed input neurons** (sensor interface)
- **Fixed output neurons** (motor interface)
- A **dynamic sparse hidden graph**
- Local synapses with **eligibility traces**
- A global **neuromodulatory signal** (δ)

External boundary:

```

Observation → Brain → Action

```

Internal topology may change over time, but the input/output interface remains stable.

> A brain can reorganise internally while still receiving sensory input and emitting motor commands.
> NeuroDrive mirrors this: I/O is fixed; internal structure is plastic.

---

### Inputs (Sensors)

- Raycast distance sensors
- Speed
- Heading error relative to track tangent
- Optional angular velocity

These represent the engineered equivalent of sensory pathways: low-dimensional, dense, and learnable.

---

### Outputs (Actuators)

- Steering ∈ [-1, 1]
- Throttle ∈ [0, 1]

Output nodes remain fixed even if hidden topology changes.

---

## Learning Mechanism

### Local Plasticity + Eligibility

Each synapse maintains:

- Weight `w_ij`
- Eligibility trace `e_ij`

Eligibility accumulates “recent usefulness” locally:

```

e_ij ← λ e_ij + f(pre_i, post_j)

```

Where `f` is correlation-based:

- rate-based: `pre × post`
- spiking: STDP timing window

This local trace is the key ingredient that makes delayed reinforcement feasible without gradients.

---

### Neuromodulation (Dopamine-like Teaching Signal)

A reward prediction error δ is computed:

```

δ = r + γ V(s') - V(s)

```

Synaptic update:

```

Δw_ij = η × δ × e_ij

```

Interpretation:

- `e_ij` says “this synapse participated recently”
- `δ` says “that participation led to better/worse outcomes than expected”
- weight change is a gated consolidation mechanism

No gradients.
No global loss.
No backprop.

> This is the engineering analogue of “local plasticity + dopamine gating.”

---

### Structural Plasticity (Topology Updates)

Structural plasticity is not a gimmick; it is how the system reallocates capacity over time.

Rules are constrained to preserve stability and bounded compute:

- **Pruning**: remove synapses with persistently low magnitude and low eligibility contribution
- **Growth**: add synapses between recently co-active neurons when capacity is available
- **Constraints**: enforce bounded fan-in / fan-out to prevent graph blow-up

Topology evolves gradually during experience.

---

## Training Loop (Watchable, Continual Learning)

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

Reward is treated as a **neuromodulatory teaching signal**, not a fitness score.

Primary reward:

- Positive reward for **forward progress** along track centerline
- Penalty for **crashing / leaving track**
- Bonus for **lap completion**

Reward shaping is minimal, interpretable, and explicitly separated from the agent code.

> In biology, reward signals guide plasticity but do not dictate behaviour directly.
> NeuroDrive uses reward to gate learning, not to define a brittle objective function.

---

## Core Design Philosophy

- **One environment, one evolving brain**
- **No external ML libraries**
- **Deterministic simulation**
- **Behaviour-first evaluation**
- **Watchable real-time learning**
- **Structural + synaptic transparency**
- **Ablations as first-class features** (prove what causes what)

The project is designed to make learning _visible and measurable_, not just plausible.

---

## Observability & Telemetry

NeuroDrive includes real-time observability because “looks like learning” is not evidence.

Planned telemetry:

- Real-time episode counter
- Progress metrics (max progress, lap %, best-ever)
- Moving averages (e.g. last 20 episodes)
- Reward decomposition (progress vs crash penalties)
- Dopamine δ visualisation (raw + smoothed)
- Weight statistics (mean |w|, histogram bins, clamp hits)
- Graph statistics (synapse count, sparsity, churn rate)
- Sensor overlays (raycasts + hit points)
- Optional live graph view (nodes/edges)
- Optional live weight view (matrix/synapse list)

Learning must be measurable, not guessed.

---

## Features & Roadmap

### 🏎️ Milestone 0 — Environment Foundation (Deterministic Sandbox)

- [ ] Deterministic fixed-timestep 2D car physics
- [ ] Track representation with centerline spline + boundaries
- [ ] Collision detection + reset conditions
- [ ] Raycast sensor system with debug overlays
- [ ] Reset-on-crash episodic loop
- [ ] Manual / heuristic baseline controller (sanity check)
- [ ] Minimal UI controls: pause, reset episode, reset brain

**Success criteria:**
A stable environment where a controller can drive and metrics are correct.

---

### 🧠 Milestone 1 — Brain v1 (Rate-Based Neurons + Reward-Modulated Plasticity)

This milestone exists to validate the full learning pipeline before introducing spikes.

- [ ] Sparse neural graph (fixed I/O, sparse hidden connectivity)
- [ ] Neuron state dynamics (time-evolving activations)
- [ ] Eligibility traces per synapse
- [ ] Reward-modulated weight updates (δ-gated plasticity)
- [ ] Episode metrics + moving averages on-screen
- [ ] Save/load checkpoints for brain state

**Success criteria:**
Measurable improvement in progress metrics over 5–10 minutes of training.

> This is the “plumbing milestone”: it proves local learning + reward gating works end-to-end.

---

### 🧪 Milestone 2 — Scientific Control (Stability, Instrumentation, Ablations)

This milestone prevents self-deception and makes results defensible.

- [ ] Weight clamping + decay mechanisms
- [ ] Learning rate schedules and safe defaults
- [ ] Deterministic replay of episodes for debugging
- [ ] First-half vs second-half statistical comparisons
- [ ] Ablations:
  - [ ] no dopamine gating (δ fixed)
  - [ ] no eligibility traces
  - [ ] frozen weights (control baseline)
- [ ] Training-speed controls (1×, 2×, 4×) while still watchable

**Success criteria:**
Clear evidence that improvements are caused by the intended learning mechanism.

---

### ⚡ Milestone 3 — Spiking Upgrade (SNN + STDP-family Learning)

Spiking neurons are not a “nice-to-have” if the goal is biological plausibility.
This milestone upgrades the core representation.

- [ ] Spiking neuron model (membrane potential + threshold + reset)
- [ ] Spike encoding for inputs (rate or temporal encoding)
- [ ] Output decoding (spike counts → continuous steering/throttle)
- [ ] STDP-style eligibility (timing-based local trace)
- [ ] Reward-modulated STDP (δ gates consolidation)
- [ ] Side-by-side comparison vs v1 (learning curves + behaviour)

**Success criteria:**
Comparable or improved learning with a more biologically grounded mechanism.

---

### 🌱 Milestone 4 — Structural Plasticity (Constrained Growth + Pruning)

Topology change is the mechanism for capacity allocation and long-term adaptation.

- [ ] Synapse pruning rules (low weight + low contribution over time)
- [ ] Synapse growth rules (co-activity-driven, bounded)
- [ ] Bounded fan-in / fan-out constraints
- [ ] Churn metrics (edges added/removed per episode)
- [ ] Topology visualisation + snapshots
- [ ] Comparison: static topology vs plastic topology

**Success criteria:**
Structural plasticity improves stability, adaptability, or sample efficiency without graph blow-up.

---

### 🗺️ Milestone 5 — Generalisation (Multiple Tracks + Continual Learning)

Generalisation is where “learning a behaviour” diverges from memorising geometry.

- [ ] Multiple curated tracks (A/B/C)
- [ ] Track cycling on episode reset (configurable)
- [ ] Interleaved training across tracks
- [ ] Held-out evaluation track (train on A/B, test on C)
- [ ] Metrics per-track + forgetting indicators
- [ ] Difficulty progression (curriculum ordering)

**Success criteria:**
Driving skill transfers across tracks; forgetting is measurable and mitigated.

---

### 💤 Milestone 6 — Replay & Consolidation (Hippocampus-Inspired)

Replay is a strong candidate mechanism for improving continual learning.

- [ ] Trajectory buffer (state/action/reward/spikes)
- [ ] Offline replay phase between episodes (“sleep”)
- [ ] Replay scheduling (recent vs diverse)
- [ ] Consolidation rules (reduce churn, stabilise useful circuits)
- [ ] Quantify sample efficiency improvements

**Success criteria:**
Replay improves learning speed and reduces catastrophic forgetting.

---

### 🧬 Milestone 7 — Robustness (Noise, Perturbations, and Realistic Imperfections)

Brains learn under noise; robustness is part of the claim.

- [ ] Sensor noise and latency
- [ ] Domain randomisation (friction, mass, grip)
- [ ] Track perturbations (minor boundary shifts)
- [ ] Stability under long-run training (hours)
- [ ] Regression suite for learning + sim correctness

**Success criteria:**
Behaviour remains stable and learning persists under controlled perturbations.

---

### 🔬 Milestone 8 — “Explain the Brain” Mode (Interpretability & Mechanism)

If the goal is brain-inspired learning, you should be able to inspect mechanisms.

- [ ] Identify “motor primitives” emerging in circuits
- [ ] Visualise synapses that dominate steering vs throttle
- [ ] Activity clustering across situations (turns vs straights)
- [ ] Track which synapses are pruned/grown most
- [ ] Export topology + activity traces for analysis

**Success criteria:**
The system becomes explainable as a learning mechanism, not just a black box.

---

## What This Project Is Not

- Not a benchmark suite for mainstream RL.
- Not a competition between optimisation paradigms.
- Not a wrapper around PyTorch.
- Not an evolutionary algorithm playground.
- Not a racing game with AI glued on top.

It is a controlled experiment in building a brain-inspired learning system from first principles.

---

## Why Racing?

A racing environment provides:

- Continuous control (steering/throttle)
- Dense and interpretable progress signals
- Non-trivial stability constraints
- Clear measurable improvement (progress %, lap time, crash rate)
- Natural generalisation tests (new tracks)

It is complex enough to require learning,
but simple enough to keep the focus on the learning mechanism.

---

## Long-Term Vision

NeuroDrive is intended as a research-grade learning laboratory:

- Study synaptic vs structural plasticity in engineered systems
- Implement dopamine-modulated local learning without gradients
- Upgrade to spiking dynamics and STDP-family learning rules
- Evaluate generalisation and continual learning behaviour
- Build a system that _visibly learns_ and can be instrumented end-to-end

The ultimate goal is not the fastest racing agent.

It is to build a system that **visibly, measurably, and continuously learns**
using principles inspired by how biological brains adapt to the world.
