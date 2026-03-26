# Comparison: Rate-Based vs Spiking Neural Networks

## Overview

NeuroDrive's biological learning roadmap passes through two fundamentally different neural computation models. Milestones 2–5 use **rate-based** networks with Hebbian-inspired local learning rules. Milestone 4 introduces **spiking neural networks (SNNs)** with spike-timing-dependent plasticity (STDP). This file explains what these models are, how they differ, what each enables, and why the project uses them in sequence rather than jumping straight to SNNs.

**Status:** Rate-based biological layer is planned (Milestone 2). SNNs are planned (Milestone 4). Both are future work. A2C is the current implementation.

## Prerequisites

- `concepts/advanced/hebbian-plasticity.md` — Hebb's rule, rate-based learning
- `concepts/advanced/spike-timing-dependent-plasticity.md` — STDP, timing windows
- `concepts/advanced/eligibility-traces.md` — the trace mechanism used in both models
- `concepts/advanced/neuromodulation.md` — the δ signal that gates learning

---

## What is a Rate-Based Neural Network?

In a rate-based (or firing-rate) model, neurons are characterised by a single scalar value: their **activation** or **firing rate** at time `t`.

A neuron's output is typically a smooth function of its inputs:

```
x_j(t) = σ( Σ_i w_ij * x_i(t) )
```

Where `σ` is a nonlinearity (ReLU, tanh, sigmoid). The activation `x_j(t)` is a continuous value representing the neuron's average firing rate over some time window.

### Properties of Rate-Based Models

- **Continuous activations:** Each neuron outputs a real number, not a discrete spike.
- **Instantaneous computation:** The network produces an output for each input without temporal dynamics within the computation.
- **Learnable with standard backprop:** The smooth nonlinearities have well-defined gradients everywhere.
- **Easier to implement:** No spike timing, no refractory periods, no membrane dynamics.

This is the model used by the standard A2C neural network — and it is also what Milestone 2's rate-based local plasticity will use, but with local Hebbian learning rules instead of backpropagation.

---

## What is a Spiking Neural Network?

In a spiking model, neurons are **integrate-and-fire** units that accumulate charge from inputs and fire a discrete spike when a threshold is crossed.

### The Leaky Integrate-and-Fire (LIF) Model

The simplest spiking neuron model:

```
τ_m * dV/dt = -(V - V_rest) + R * I(t)

if V ≥ V_threshold:
    emit spike
    V ← V_reset
    enter refractory period
```

Where:
- `V` — membrane potential
- `V_rest` — resting potential (e.g. -65 mV biologically, 0 normalised)
- `V_threshold` — spike threshold (e.g. -50 mV biologically)
- `V_reset` — post-spike reset potential
- `τ_m` — membrane time constant (e.g. 10 ms)
- `R * I(t)` — weighted input current

### Key Properties of Spiking Models

- **Discrete spike events:** Output is binary per timestep (spike or no spike).
- **Temporal dynamics:** The membrane potential integrates inputs over time. The history of recent spikes matters.
- **Timing encodes information:** *When* spikes occur relative to each other carries information beyond just how often.
- **Natural STDP:** Spike timing allows precise measurement of co-activity timing, making STDP a natural learning rule.
- **Biologically closer:** Real neurons are spiking; rate models are a simplification.

---

## How Learning Differs

### Rate-Based Hebbian Learning

In a rate-based network, a Hebbian synapse update uses the activation values:

```
Δw_ij ∝ x_i * x_j    (co-activity of firing rates)
```

The eligibility trace accumulates based on the product of activations:

```
e_ij(t) = decay * e_ij(t-1) + x_i(t) * x_j(t)
```

This is smooth and differentiable. Learning can happen on every timestep proportionally to continuous activation magnitudes.

### STDP (Spike-Timing-Dependent Plasticity)

In a spiking network, learning depends on the relative timing of pre- and post-synaptic spikes:

```
if t_post - t_pre ∈ (0, +τ):    Δw > 0  (LTP — potentiation)
if t_post - t_pre ∈ (-τ, 0):   Δw < 0  (LTD — depression)
```

The sign and magnitude of the weight change depends on whether the pre-synaptic neuron fired just before (LTP) or just after (LTD) the post-synaptic neuron. The causal relationship is key: pre before post = "neuron A contributed to neuron B firing" = strengthen; post before pre = "coincidental or anti-causal" = weaken.

The eligibility trace in STDP measures this timing signal:

```
For LTP: e_ij(t) += A_+ * exp(-|t_post - t_pre| / τ_+)  when pre fires
For LTD: e_ij(t) -= A_- * exp(-|t_post - t_pre| / τ_-)  when post fires
```

STDP is inherently temporal — it requires tracking the times of individual spike events.

---

## Computational Implications

| Property | Rate-Based | Spiking |
|---|---|---|
| Neuron state | Single scalar (activation) | Membrane potential + spike history |
| Timestep granularity | Fixed tick (e.g. 60 Hz) | Sub-millisecond (1–5 ms simulated) |
| Learning trigger | Every tick | Spike events only |
| Learning rule precision | Proportional to activation | Depends on exact spike timing |
| Information encoding | Activation magnitude | Spike rate AND timing |
| Simulation cost | O(synapses) per tick | O(spikes × synapses) per tick; spikes are sparse |
| Implementation complexity | Moderate | High |

The **sparse** nature of spiking computation is both its efficiency advantage and its implementation challenge. A network where 5% of neurons spike per tick does 95% fewer updates than a dense rate-based network — but only if the implementation efficiently handles the sparse spike lists.

---

## Why Rate-Based First (Milestones 2–3)?

The milestone sequence introduces rate-based Hebbian learning before SNNs for several reasons:

### 1. Smaller Implementation Jump

Going from A2C (backprop gradient updates) to rate-based Hebbian (local correlation updates, same activation model) is a smaller conceptual and implementation step than jumping directly to spiking neurons. The network structure, the fixed-tick integration, and the output interpretation all remain similar.

### 2. Validates Local Learning Rules First

The key scientific question at Milestone 2 is: **can local learning rules with a neuromodulatory signal produce useful behaviour?** This question can be answered in the simpler rate-based model without introducing spike timing as a confounding variable.

If rate-based local learning fails, the problem is likely in the learning rule or the reward signal — not spike timing. If it succeeds, SNNs can be introduced knowing that the core learning mechanism works.

### 3. Ablation Studies at Milestone 3

Milestone 3 plans systematic ablation studies on the biological learning components (eligibility traces, neuromodulation, structural plasticity). These ablations are much easier to interpret in the rate-based model where each component has a clear, isolated role. In SNNs, the components interact more complexly through spike timing.

### 4. Foundation for STDP

Rate-based eligibility traces are a stepping stone. STDP-based eligibility traces extend the same concept (accumulate a signal measuring co-activity over time) into the spiking timing domain. Understanding the rate-based version first makes STDP easier to understand when it is introduced.

---

## Why SNNs at Milestone 4?

Having validated the local learning principle in the rate-based model, Milestone 4 introduces spiking computation because:

### 1. Biological Realism

Real neurons spike. The rate-based model is a useful approximation but does not capture the temporal dynamics of neural computation. Moving to SNNs removes this approximation.

### 2. Temporal Coding

SNNs can encode information in the precise timing of spikes, not just in firing rate. This opens up richer representational possibilities. For driving, spike timing could potentially encode time-to-contact with walls, not just whether a wall is near.

### 3. STDP Has Stronger Biological Support

While rate-based Hebb rules are biologically motivated, STDP has much stronger experimental support for the specific learning mechanism (Bi & Poo 1998). Moving to SNNs allows the use of STDP as the primary local learning rule, closer to what real synapses do.

### 4. Energy Efficiency (Theoretical)

Biological and neuromorphic hardware benefits: sparse spike activity means most neurons are idle most of the time, making computation inherently energy-efficient. This is not relevant for NeuroDrive's CPU simulation, but it is architecturally interesting.

---

## The Transition Challenge

Moving from rate-based to spiking at Milestone 4 requires:

1. **New neuron model:** Replace scalar activation with integrate-and-fire dynamics (membrane potential, threshold, reset, refractory period).

2. **New timestep resolution:** SNNs typically require ~1 ms simulated timesteps. NeuroDrive runs at 60 Hz (16.7 ms per tick). Either multiple SNN steps per game tick, or a rate-to-spike conversion at the boundary.

3. **New observation encoding:** The observation vector (23 real numbers) must be converted to spike trains for the SNN to process. Population coding or rate-based encoding are both options.

4. **New action decoding:** The SNN output (spike trains) must be decoded into continuous steering and throttle values.

5. **STDP implementation:** The learning rule changes from rate-correlation to spike-timing events.

6. **Eligibility trace adaptation:** Trace accumulation changes from activation products to timing-based exponentials.

None of these are insurmountable, but they collectively represent a substantial implementation project. The rate-based milestones (2 and 3) exist partly to build confidence before this transition.

---

## Summary

| | Rate-Based (Milestones 2–3) | Spiking (Milestone 4+) |
|---|---|---|
| Neuron model | Continuous activation | Discrete spike events |
| Learning rule | Hebbian rate-correlation + δ | STDP timing window + δ |
| Biological realism | Moderate | High |
| Implementation complexity | Moderate | High |
| NeuroDrive status | Planned | Planned |
| Why this order | Validate local learning first | Add temporal coding after |

---

## Related Files

- `concepts/advanced/hebbian-plasticity.md` — rate-based Hebb rule details
- `concepts/advanced/spike-timing-dependent-plasticity.md` — STDP mechanism and timing windows
- `concepts/advanced/eligibility-traces.md` — how traces work in both models
- `concepts/advanced/neuromodulation.md` — the δ signal that gates both
- `project/evolution/milestone-roadmap.md` — Milestones 2, 3, and 4 in context
- `project/evolution/from-baseline-to-brain.md` — the architectural transition plan
