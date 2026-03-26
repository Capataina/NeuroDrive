# Hebbian Plasticity

## Why This Matters Here

Hebbian plasticity is the foundational principle of biological learning. Every subsequent concept in this section — STDP, eligibility traces, neuromodulation — builds on or extends Hebb's rule. Understanding it is the prerequisite for understanding what NeuroDrive's Milestone 2 is trying to build and why it looks nothing like gradient descent.

**Status:** Foundational domain knowledge. The biological learning architecture is planned for Milestone 2+, not yet implemented.

## Prerequisites

- Basic understanding of neural networks (what neurons and synapses are)
- No prior neuroscience knowledge required

## Notation

| Symbol | Meaning |
|---|---|
| `x_i` | Activity of presynaptic neuron `i` |
| `x_j` | Activity of postsynaptic neuron `j` |
| `w_ij` | Synaptic weight from `i` to `j` |
| `Δw_ij` | Change in synaptic weight |
| `η` | Learning rate |
| `τ_w` | Weight decay time constant |

---

## Core Idea: "Fire Together, Wire Together"

Donald Hebb proposed in 1949 (before anyone had a good idea of how neurons actually work in detail) that:

> "When an axon of cell A is near enough to excite cell B, and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

In modern notation: if neuron A is active while neuron B is active, the synapse from A to B strengthens.

```
Δw_ij ∝ x_i * x_j
```

This is pure **correlation-based learning**: synaptic weight change is proportional to the product of presynaptic and postsynaptic activity.

---

## The Rate-Based Hebbian Rule

The simplest formal version:

```
Δw_ij = η * x_i * x_j
```

Where `x_i` and `x_j` are firing rates (real-valued activities, typically normalised to [0, 1] or [−1, 1]).

**Intuition:**
- If both neurons are active (`x_i > 0, x_j > 0`): strengthen the synapse
- If both are inactive (`x_i ≈ 0, x_j ≈ 0`): no change
- If one is active and the other is not: the sign depends on the convention

---

## Why Hebbian Learning Alone Is Insufficient

Pure Hebbian learning has several fatal practical problems:

### 1. Unstable (Weights Explode)

Without a constraint, Hebbian learning drives weights to infinity. If A fires B, the A→B synapse strengthens. But now A fires B more easily, which fires it more, which strengthens the synapse further. Positive feedback loops cause runaway weight growth.

### 2. No Forgetting

Pure Hebbian learning only strengthens. There is no mechanism to weaken a synapse that was useful in the past but is no longer useful.

### 3. No Credit Assignment

Hebbian learning strengthens *all* co-active synapses equally. It cannot distinguish which synapses *caused* a good outcome from which happened to be active at the same time.

### 4. No Global Goal

Pure Hebbian learning has no concept of "this led to a good outcome overall." It is entirely local — it responds to correlation, not consequence.

---

## Solutions to the Instability Problem

Several mechanisms stabilise Hebbian learning in biological systems:

### Weight Decay

```
Δw_ij = η * x_i * x_j - τ * w_ij
```

The decay term `τ * w_ij` continuously reduces weights proportionally to their magnitude. This limits growth.

### BCM Rule (Bienenstock-Cooper-Munro)

Introduces a **sliding threshold** for potentiation vs depression. Neurons that fire frequently adapt their threshold upward, making them harder to potentiate. This is a homeostatic mechanism.

### Normalisation

Hard normalisation of weight vectors (Oja's rule):

```
Δw_i = η * x_j * (x_i - x_j * Σ_k w_k * x_k)
```

This projects the weight change onto the unit sphere, preventing unbounded growth while preserving the directional information.

---

## The Credit Assignment Problem

Hebbian learning is *spatially local* but *temporally immediate*. It responds to what is happening right now. But in most real tasks, the consequence of an action happens *later*:

- Steering left now → car avoids a wall → reward in 5 steps
- Which synapse changes were responsible for the good outcome?

Hebbian learning cannot answer this question alone. The synapse strengthened when it fired, but the reward arrived later. The bridge between Hebbian correlation and delayed reward is the **eligibility trace** (see `concepts/advanced/eligibility-traces.md`).

---

## Hebbian Learning vs Backpropagation

This comparison is central to understanding the NeuroDrive project's direction:

| Property | Hebbian | Backpropagation |
|---|---|---|
| Signal locality | Local (pre + post activity) | Global (loss gradient from all outputs) |
| Biological plausibility | High | Very low |
| Credit assignment | None (without eligibility traces) | Exact (via chain rule) |
| Stability | Poor without constraints | Good (with learning rate) |
| Continual learning | Natural | Catastrophic forgetting |
| Hardware | Neural tissue | GPU/TPU arrays |

Hebbian learning captures the *form* of biological learning. Backpropagation captures the *efficiency* of mathematical optimisation. NeuroDrive's research question is whether the former can be made effective enough to train a useful agent.

---

## Rate-Based vs Spike-Based Hebbian Learning

**Rate-based Hebbian** (used in Milestone 2):
- Neurons have real-valued activity levels
- Synaptic change is proportional to activity product
- Simpler to implement and analyse

**Spike-based STDP** (Milestone 4):
- Neurons emit discrete spikes
- Synaptic change depends on the precise *timing* between spikes
- More biologically detailed
- Introduces temporal complexity

Rate-based Hebbian is the natural starting point because it preserves the essential correlation structure while avoiding the complexity of spike timing.

---

## How This Enters NeuroDrive

The planned Milestone 2 brain (`src/brain/biological/`, currently an empty directory) will implement:

1. A sparse graph of neurons with real-valued activity states
2. A Hebbian-family weight update rule: `Δw_ij = η * x_i * x_j`
3. Weight decay to stabilise learning
4. Eligibility traces (see `concepts/advanced/eligibility-traces.md`) to handle delayed credit assignment
5. Neuromodulation (dopamine-like δ) to gate which Hebbian changes persist

The core architecture:
```
observation → input neurons
                ↓ (sparse connections)
            hidden neurons
                ↓ (sparse connections)
            output neurons → actions
```

With Hebbian updates flowing locally through each synapse, and a global δ signal modulating which updates are "saved."

---

## Common Misunderstandings

❌ "Hebbian learning is just correlation"
✅ Hebbian learning uses correlation, but the biological implementation includes constraints (homeostasis, weight decay) that make it more sophisticated than raw correlation-based learning.

❌ "Backpropagation is more powerful, so Hebbian learning has no value"
✅ Backpropagation requires non-local, multi-pass gradient computation. Biological systems that cannot backpropagate must use local rules. Understanding how far you can get with local rules — and how they achieve generalisation — is a genuine scientific question.

❌ "Hebbian learning produces the same result as gradient descent"
✅ They converge differently, generalise differently, and have different failure modes. Hebbian learning with eligibility traces and neuromodulation can approximate certain RL objectives, but it is not gradient descent in disguise.

---

## Related Files

- `concepts/advanced/spike-timing-dependent-plasticity.md` — the spike-based extension of Hebbian learning
- `concepts/advanced/eligibility-traces.md` — solving the temporal credit assignment problem
- `concepts/advanced/neuromodulation.md` — adding a global teaching signal to Hebbian learning
- `project/evolution/from-baseline-to-brain.md` — where Hebbian learning enters NeuroDrive
