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

NeuroDrive follows a deliberate sequencing strategy:

1. Build a deterministic, observable environment.
2. Prove the task is learnable with a lightweight RL baseline (A2C).
3. Transition to brain-inspired local plasticity mechanisms.
4. Gradually increase biological fidelity and structural complexity.

This reduces debugging ambiguity and isolates representation issues from learning-rule issues.

---

## 🏎️ Milestone 0 — Environment Foundation (Deterministic Sandbox)

This milestone establishes a fully deterministic, instrumented control environment before any learning algorithm is introduced.

- [x] Deterministic fixed-timestep 2D car physics
- [x] Track representation (centerline polyline/spline + boundaries)
- [x] Collision detection + reset conditions
- [x] Progress metric via centerline projection (continuous, no jumps)
- [x] Raycast sensor system with on-screen debug overlays
- [x] Stable observation vector (normalized inputs)
- [x] Steering/throttle action interface with optional smoothing
- [x] Episode loop (crash / timeout / lap complete)
- [x] Telemetry: reward, progress %, crash count, moving averages
- [x] Deterministic replay test (same seed + same actions → identical trajectory)

- [x] Debug visual overlays:
  - [x] Raycasts + hit points
  - [x] Closest centerline projection point
  - [x] Centerline tangent vector visualisation
  - [x] Car forward vector (velocity and drag)
  - [x] Heading error readout
  - [x] Progress Percentage of the track
  - [x] F1, F2 and F3 keys to toggle the debug overlays
    - F1: Geometry overlays
    - F2: Sensor overlays
    - F3: Learning telemetry (after the ais are implemented)

**Success criteria:**
The environment is stable, deterministic, observable, and debuggable.  
A manually controlled or heuristic controller can complete laps reliably.
All geometric quantities (projection, tangent, heading error) are visually verified and stable before any learning begins.

> No learning occurs at this stage. The goal is correctness and instrumentation.

---

## 🧠 Milestone 1 — A2C Baseline (Autonomous Learnability Validation)

Before implementing biological plasticity, we validate that the task is learnable using a minimal on-policy RL algorithm.

This is not the final direction of the project.  
It is a diagnostic layer that answers:

> Is the observation space + reward structure sufficient for autonomous learning?

### Implementation Scope

- [ ] Small MLP policy network (e.g. 2×64 hidden layers)
- [ ] Value function (shared trunk or separate head)
- [ ] On-policy rollout buffer
- [ ] Advantage estimation (TD or GAE-lite)
- [ ] Policy loss + value loss + entropy regularization
- [ ] Online updates at fixed rollout intervals
- [ ] Real-time learning visualisation (watchable behaviour)
- [ ] Headless fast-training mode (optional)
- [ ] Policy snapshot + evaluation mode

### Constraints

- No replay buffer
- No target networks
- No SAC / PPO complexity
- No external ML libraries
- Fully implemented in Rust

### Observation Inputs

- Raycast distances (normalized)
- Speed
- Heading error relative to track tangent
- Optional angular velocity

No privileged geometric information (no arc-length progress, no curvature lookahead).

### Success criteria

- Measurable improvement in forward progress within minutes
- Reduced crash frequency over time
- Stable lap completion behaviour
- No reward hacking
- Learning visible in real time

If A2C fails:

- Diagnose observation scaling
- Diagnose reward magnitude
- Diagnose timestep stability
- Optionally validate representation using supervised cloning

> Milestone 1 proves that the task is learnable.  
> It isolates environment design from biological learning mechanics.

---

## 🧬 Milestone 2 — Brain v1 (Rate-Based Local Plasticity + δ Gating)

After learnability is validated, we replace gradient-based learning with biologically inspired mechanisms.

- [ ] Sparse neural graph (fixed I/O, sparse hidden connectivity)
- [ ] Neuron state dynamics (rate-based activations)
- [ ] Eligibility traces per synapse
- [ ] Reward-modulated weight updates (δ-gated plasticity)
- [ ] No backpropagation
- [ ] No global gradient computation
- [ ] Continuous online learning (single persistent brain)
- [ ] Episode metrics + moving averages
- [ ] Save/load brain state

**Success criteria:**
Observable behavioural improvement without gradients.

> This milestone transitions from optimisation to local plasticity.

---

## 🧪 Milestone 3 — Scientific Control (Stability & Ablations)

Prevent self-deception. Prove causality.

- [ ] Weight clamping + decay
- [ ] Learning rate schedules
- [ ] Deterministic episode replay
- [ ] First-half vs second-half statistics
- [ ] Ablations:
  - [ ] No dopamine gating
  - [ ] No eligibility traces
  - [ ] Frozen weights (control baseline)
- [ ] Training-speed controls (1×, 2×, 4×)

**Success criteria:**
Clear evidence that improvements arise from the intended mechanisms.

---

## ⚡ Milestone 4 — Spiking Upgrade (SNN + STDP)

Upgrade representation to spike-based dynamics.

- [ ] Spiking neuron model (membrane potential, threshold, reset)
- [ ] Spike encoding of inputs
- [ ] Spike decoding of outputs
- [ ] STDP-style eligibility traces
- [ ] Reward-modulated STDP
- [ ] Side-by-side comparison with rate-based version

**Success criteria:**
Comparable or improved learning with greater biological plausibility.

---

## 🌱 Milestone 5 — Structural Plasticity (Growth + Pruning)

Introduce constrained topology adaptation.

- [ ] Synapse pruning rules
- [ ] Synapse growth rules (co-activity driven)
- [ ] Bounded fan-in / fan-out
- [ ] Churn metrics (edges added/removed)
- [ ] Topology visualisation

**Success criteria:**
Structural adaptation improves efficiency or stability without graph explosion.

---

## 🗺️ Milestone 6 — Generalisation & Continual Learning

- [ ] Multiple curated tracks
- [ ] Interleaved training across tracks
- [ ] Held-out evaluation track
- [ ] Forgetting metrics
- [ ] Curriculum progression

**Success criteria:**
Skill transfers across tracks without catastrophic forgetting.

---

## 💤 Milestone 7 — Replay & Consolidation

- [ ] Trajectory buffer
- [ ] Offline replay (“sleep phase”)
- [ ] Consolidation rules
- [ ] Sample efficiency analysis

**Success criteria:**
Replay improves learning speed or stability.

---

## 🧬 Milestone 8 — Robustness & Perturbation Testing

- [ ] Sensor noise
- [ ] Physics randomisation
- [ ] Track perturbations
- [ ] Long-run stability testing
- [ ] Regression test suite

**Success criteria:**
Learning remains stable under controlled noise.

---

## 🔬 Milestone 9 — Interpretability & Mechanistic Analysis

- [ ] Identify emergent motor primitives
- [ ] Synapse importance visualisation
- [ ] Activity clustering (turns vs straights)
- [ ] Export topology + activity traces

**Success criteria:**
The system becomes inspectable as a learning mechanism, not just a black box.

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
