# Spike-Timing Dependent Plasticity (STDP) and Eligibility Traces

## Prerequisites

- **Hebbian plasticity** — `hebbian_plasticity.md` (co-activation strengthens synapses, the Δw = η · pre · post rule, unbounded growth problem)
- **Value functions and critics** — `concepts/core/reinforcement_learning/value_functions_and_critics.md` (temporal difference error, reward prediction)
- Basic exponential decay: y(t) = y₀ · e^(−t/τ) — understanding that signals fade over time with a characteristic time constant τ

## Target Depth for This Project

**Level 3** — Can explain intuitively and understand the maths. You should be able to describe the STDP timing window, compute weight changes for given spike timings, explain the credit assignment problem and how eligibility traces solve it, write the three-factor learning rule, and trace through a concrete numerical sequence showing trace accumulation followed by reward-modulated consolidation.

---

## Core Concept

### Step 1: Adding Timing to Hebb's Rule

Hebbian plasticity asks only "were these neurons active at the same time?" But the brain cares about *causality*. If neuron A fires and then neuron B fires shortly afterwards, A may have contributed to B's firing — this is a potentially useful causal relationship that should be reinforced. If B fires first and A fires afterwards, A could not have caused B's firing — this correlation is likely spurious.

**Spike-Timing Dependent Plasticity (STDP)** makes the direction and magnitude of the weight change depend on the relative timing of pre- and post-synaptic spikes:

- **Pre-before-post** (Δt = t_post − t_pre > 0, causal): the synapse **strengthens** (Long-Term Potentiation, LTP)
- **Post-before-pre** (Δt < 0, anti-causal): the synapse **weakens** (Long-Term Depression, LTD)

This elegantly solves one of pure Hebbian learning's critical failures: STDP provides a weakening mechanism. Connections that participate in anti-causal correlations are actively reduced, preventing unbounded growth.

### Step 2: The STDP Timing Window

The magnitude of the weight change decays exponentially with the absolute time difference. The standard model uses two exponential windows:

\[
\Delta w = \begin{cases} A_+ \cdot \exp\!\left(\dfrac{-\Delta t}{\tau_+}\right) & \text{if } \Delta t > 0 \text{ (LTP)} \\[6pt] -A_- \cdot \exp\!\left(\dfrac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \text{ (LTD)} \end{cases}
\]

where:
- A₊ and A₋ are the maximum potentiation and depression amplitudes
- τ₊ and τ₋ are the time constants (typically ~20ms for both)
- Δt = t_post − t_pre

**Numerical example** with A₊ = 0.1, A₋ = 0.12, τ₊ = τ₋ = 20ms:

| Δt (ms) | Direction | Δw |
|---------|-----------|-----|
| +5 | Pre-before-post (causal) | 0.1 × exp(−5/20) = 0.1 × 0.7788 = **+0.0779** |
| +10 | Pre-before-post (causal) | 0.1 × exp(−10/20) = 0.1 × 0.6065 = **+0.0607** |
| −5 | Post-before-pre (anti-causal) | −0.12 × exp(5/20) = −0.12 × 0.7788 = **−0.0935** |
| −10 | Post-before-pre (anti-causal) | −0.12 × exp(10/20) = −0.12 × 0.6065 = **−0.0728** |

Notice that A₋ > A₊ — the depression amplitude is slightly larger than the potentiation amplitude. This is observed experimentally and prevents runaway potentiation: on average, random spike correlations produce a net weakening, so only genuinely causal relationships survive.

### Step 3: The Credit Assignment Problem

STDP tells us *which* synapses participated in causal chains. But there is a deeper problem: **reward arrives later than the synaptic event that caused it**.

Consider NeuroDrive's agent: at timestep t, a particular synapse fires in a pattern that contributes to a good steering decision. The positive outcome (forward progress, no crash) is only measurable at timestep t+15 or later, after the car has travelled further along the track. By then, the STDP window has long closed — the spike correlation signal has vanished.

How does the brain connect "this synapse was useful" (past) with "that turned out well" (present)?

### Step 4: Eligibility Traces — The Solution

An **eligibility trace** is a decaying memory maintained at each synapse that records recent correlation activity. When a pre-post correlation occurs (Hebbian or STDP-based), the trace increases. Over time, it decays. Crucially, it persists long enough for a reward signal to arrive and act upon it.

The update rule for the eligibility trace at synapse (i, j) is:

\[
e_{ij} \leftarrow \lambda \cdot e_{ij} + f(\text{pre}_i, \text{post}_j)
\]

where:
- λ ∈ (0, 1) is the trace decay factor (controls how far into the past the trace remembers)
- f(pre_i, post_j) is the correlation function — Hebbian (pre × post) for rate-based, or the STDP kernel for spiking

The trace accumulates over multiple timesteps. Recent correlations contribute strongly; older correlations have been partially decayed away. The trace does not change any weight on its own — it merely records *which synapses are eligible* for modification when a teaching signal arrives.

### Step 5: The Three-Factor Learning Rule

The actual weight update combines three factors:

\[
\Delta w_{ij} = \eta \cdot \delta \cdot e_{ij}
\]

1. **η** — the learning rate
2. **δ** — the neuromodulatory signal (reward prediction error, covered in `neuromodulation_and_dopamine.md`)
3. **e_ij** — the eligibility trace (local synaptic memory of recent correlation)

This is called a **three-factor rule** because the weight change depends on three quantities: the learning rate, a global modulatory signal, and a local synaptic variable. The trace provides the credit assignment ("this synapse participated"), and δ provides the evaluation ("that participation led to a better/worse outcome").

---

## Mathematical Foundation — Numerical Example

We trace a single synapse through 10 timesteps with rate-based eligibility (f = pre × post), λ = 0.8, η = 0.05. A reward signal arrives at timestep 8.

| Step | pre_i | post_j | f(pre, post) | e_ij (before) | e_ij (after) = λ·e + f | δ | Δw = η·δ·e |
|------|-------|--------|-------------|---------------|------------------------|-----|-------------|
| 1 | 0.9 | 0.8 | 0.720 | 0.000 | 0.720 | 0 | 0.000 |
| 2 | 0.7 | 0.6 | 0.420 | 0.720 | 0.996 | 0 | 0.000 |
| 3 | 0.2 | 0.1 | 0.020 | 0.996 | 0.817 | 0 | 0.000 |
| 4 | 0.8 | 0.9 | 0.720 | 0.817 | 1.374 | 0 | 0.000 |
| 5 | 0.5 | 0.4 | 0.200 | 1.374 | 1.299 | 0 | 0.000 |
| 6 | 0.1 | 0.0 | 0.000 | 1.299 | 1.039 | 0 | 0.000 |
| 7 | 0.3 | 0.2 | 0.060 | 1.039 | 0.891 | 0 | 0.000 |
| 8 | 0.6 | 0.5 | 0.300 | 0.891 | 1.013 | **+2.0** | **0.101** |
| 9 | 0.4 | 0.3 | 0.120 | 1.013 | 0.930 | 0 | 0.000 |
| 10 | 0.1 | 0.1 | 0.010 | 0.930 | 0.754 | 0 | 0.000 |

Key observations:

- **Steps 1–7**: The trace accumulates correlation but no weight change occurs (δ = 0). The synapse is becoming "eligible" for modification.
- **Step 8**: A reward signal arrives (δ = +2.0). The weight changes by η × δ × e = 0.05 × 2.0 × 1.013 = **0.101**. The eligibility trace at this moment reflects not just the current correlation (0.300) but the *accumulated history* of recent co-activity.
- **Steps 9–10**: δ returns to zero. The trace continues to decay. No further weight change.

If the reward had been negative (δ = −1.5 at step 8), the weight change would have been 0.05 × (−1.5) × 1.013 = **−0.076** — weakening the synapse. The same trace memory serves both reinforcement and punishment.

---

## How NeuroDrive Will Use This

**Milestone 2** (rate-based local plasticity) uses this framework with f(pre, post) = pre × post. Each synapse in the sparse neural graph maintains an eligibility trace that accumulates Hebbian correlation and decays between timesteps. The neuromodulatory signal δ (computed from the critic's value estimates) gates which traces get consolidated into permanent weight changes. This is the three-factor rule above, implemented without backpropagation.

**Milestone 4** (spiking upgrade) replaces the rate-based correlation function with the STDP kernel. Instead of f = pre × post, the correlation function becomes the exponential timing window described above. Neurons transition from continuous rate-coded activations to discrete spikes with membrane potential dynamics. The eligibility trace and three-factor update rule remain structurally identical — only the correlation function changes.

This staged approach (rate-based first, spiking later) allows NeuroDrive to validate the trace-and-modulation architecture before adding the complexity of spike timing.

---

## Common Misconceptions

1. **"Eligibility traces are the same as RNN hidden state."** They are not. An RNN's hidden state is part of the forward computation — it affects the network's output. An eligibility trace is a *learning-side* variable — it affects how weights change but has no influence on the network's forward pass or output. The trace is invisible to the network's computation; it only matters when the optimiser (or modulatory signal) applies updates.

2. **"STDP requires exact spike times at millisecond resolution."** The rate-based approximation (pre × post) captures the essence of STDP — correlated activity drives plasticity — without requiring spike-level temporal resolution. NeuroDrive's Milestone 2 operates entirely in rate-coded space. Exact spike timing is a Milestone 4 refinement, not a prerequisite.

3. **"The three-factor rule is just REINFORCE with extra steps."** REINFORCE computes ∇ log π(a|s) · G — a gradient through the policy network. The three-factor rule computes η · δ · e — a product of a global scalar and a local trace. There is no gradient computation, no log-probability, and no chain rule. The credit assignment mechanism is fundamentally different: REINFORCE uses the structure of the policy network; the three-factor rule uses temporal correlation at each synapse.

---

## Glossary

| Term | Definition |
|------|-----------|
| **STDP** | Spike-Timing Dependent Plasticity — a learning rule where the sign and magnitude of synaptic change depend on the relative timing of pre- and post-synaptic spikes. |
| **LTP** | Long-Term Potentiation — a lasting increase in synaptic strength, triggered by causal (pre-before-post) spike timing. |
| **LTD** | Long-Term Depression — a lasting decrease in synaptic strength, triggered by anti-causal (post-before-pre) spike timing. |
| **Eligibility trace (e_ij)** | A decaying per-synapse variable that accumulates recent correlation, making the synapse "eligible" for weight change when a modulatory signal arrives. |
| **Three-factor rule** | Δw = η · δ · e — a weight update combining learning rate, global modulatory signal, and local eligibility. |
| **Trace decay (λ)** | The factor by which the eligibility trace is multiplied each timestep, controlling how far into the past the trace's memory extends. |
| **Credit assignment** | The problem of determining which past actions or synaptic events were responsible for a current outcome. |

---

## Recommended Materials

1. **Bi & Poo (1998), "Synaptic Modifications in Cultured Hippocampal Neurons"** — The landmark experimental paper demonstrating the STDP timing window. The key figures showing Δw as a function of Δt are essential reading.
2. **Izhikevich (2007), "Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling"** — Demonstrates the three-factor rule (STDP + eligibility + dopamine) in a computational model. Directly relevant to NeuroDrive's architecture.
3. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 12** — Eligibility traces in the RL context (TD(λ)). The mathematical machinery is the same, though the motivation differs.
4. **NeuroDrive README — "Learning Mechanism" section** — The eligibility trace equation e_ij ← λ·e_ij + f(pre, post) and the three-factor update Δw = η × δ × e_ij are stated explicitly. This concept file unpacks the neuroscience behind those two lines.
