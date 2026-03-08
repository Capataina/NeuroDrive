# Hebbian Plasticity

## Prerequisites

- **Forward pass and layers** — `concepts/core/neural_networks/forward_pass_and_layers.md` (linear layers, activations, weight matrices)
- Basic calculus: derivatives, product rule — `concepts/foundations/calculus_and_gradients.md`
- Familiarity with the NeuroDrive README's brain architecture section (sparse neural graph, local plasticity, eligibility traces)

## Target Depth for This Project

**Level 3** — Can explain intuitively and understand the maths. You should be able to state Hebb's postulate, write the weight update rule, compute weight changes for a small network by hand, explain why pure Hebbian learning is unstable, and describe how NeuroDrive plans to use Hebbian correlation as the foundation for eligibility trace accumulation.

---

## Core Concept

### Step 1: Biological Synapses

Neurons in the brain communicate through **synapses** — junctions where one neuron's output signal reaches another neuron's input. Each synapse has a *strength* (or *efficacy*) that determines how much influence the presynaptic neuron has on the postsynaptic neuron. A strong synapse means a firing presynaptic neuron is likely to contribute to the postsynaptic neuron firing. A weak synapse means the signal is largely ignored.

The brain contains roughly 100 trillion synapses. Learning, in biological terms, is the process of adjusting these synaptic strengths based on experience. The question is: what rule governs the adjustment?

### Step 2: Hebb's Postulate

In 1949, Donald Hebb proposed a deceptively simple answer:

> "When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

The modern paraphrase is: **neurons that fire together wire together**. If neuron A is active at roughly the same time as neuron B, the synapse from A to B strengthens. The correlation between their activities drives learning.

This is a **local** rule. The synapse only needs information available at its own location: the activity of its presynaptic neuron and the activity of its postsynaptic neuron. It does not need to know about any global loss function, any downstream error signal, or any other synapse in the network. This locality is what makes Hebbian learning biologically plausible — and fundamentally different from backpropagation.

### Step 3: Mathematical Formulation

Let w_ij be the weight of the synapse from presynaptic neuron i to postsynaptic neuron j. The Hebbian update rule is:

\[
\Delta w_{ij} = \eta \cdot \text{pre}_i \cdot \text{post}_j
\]

where:
- η (eta) is the learning rate, controlling the magnitude of each update
- pre_i is the activity (firing rate) of the presynaptic neuron i
- post_j is the activity (firing rate) of the postsynaptic neuron j

The weight change is proportional to the product of the two activities. When both neurons are highly active simultaneously, the weight increases substantially. When either neuron is inactive, the weight does not change.

---

## Mathematical Foundation — Numerical Example

Consider two neurons with a single synapse. We track the weight over five timesteps with a learning rate η = 0.1.

| Timestep | pre_i | post_j | Δw = η · pre · post | w (after) |
|----------|-------|--------|----------------------|-----------|
| 0        | —     | —      | —                    | 0.50      |
| 1        | 0.8   | 0.9    | 0.1 × 0.8 × 0.9 = 0.072 | 0.572 |
| 2        | 0.6   | 0.7    | 0.1 × 0.6 × 0.7 = 0.042 | 0.614 |
| 3        | 0.9   | 1.0    | 0.1 × 0.9 × 1.0 = 0.090 | 0.704 |
| 4        | 0.3   | 0.1    | 0.1 × 0.3 × 0.1 = 0.003 | 0.707 |
| 5        | 0.7   | 0.8    | 0.1 × 0.7 × 0.8 = 0.056 | 0.763 |

Observe that the weight only ever *increases*. At timestep 4, when both neurons are relatively quiet, the change is negligible (0.003) — but it is still positive. Over many timesteps, the weight will grow without bound.

### The Instability Problem

Pure Hebbian learning has a critical flaw: **unbounded weight growth**. Because the update is always non-negative (assuming non-negative activities), weights can only increase. There is no mechanism for weakening connections.

This creates a positive feedback loop: a strong synapse causes higher postsynaptic activity, which causes an even larger Δw, which makes the synapse even stronger. In the numerical example above, the weight grew from 0.50 to 0.763 in five steps. Given sustained co-activation, it would reach 10.0, 100.0, and beyond.

Biological brains solve this through several mechanisms: **synaptic normalisation** (constraining total synaptic weight per neuron), **homeostatic plasticity** (adjusting excitability to maintain target firing rates), and critically, **spike-timing dependent rules** that can produce both strengthening and weakening (see `stdp_and_eligibility_traces.md`).

---

## How NeuroDrive Will Use This

NeuroDrive does not use raw Hebbian plasticity directly. Instead, it uses the Hebbian correlation as the **accumulation function** inside the eligibility trace mechanism:

\[
e_{ij} \leftarrow \lambda \cdot e_{ij} + f(\text{pre}_i, \text{post}_j)
\]

where f(pre_i, post_j) = pre_i × post_j — the Hebbian correlation term.

In Milestone 2 (rate-based local plasticity), this correlation quantifies "how much did this synapse participate in recent computation?" The eligibility trace accumulates this participation signal over time, decaying older contributions by the factor λ. The actual weight change only occurs when the neuromodulatory signal δ arrives:

\[
\Delta w_{ij} = \eta \times \delta \times e_{ij}
\]

The Hebbian term provides the *what* (which synapses were active), while δ provides the *whether* (was the outcome good or bad). This separation is what makes the three-factor learning rule viable: credit assignment comes from local correlation, not global gradients.

### Why This Is Fundamentally Different from Backpropagation

Backpropagation requires:
1. A global scalar loss function defined over the network's output
2. The chain rule applied through every layer to compute ∂L/∂w for each weight
3. Cached activations from the forward pass at every layer
4. A reverse-order sweep through the entire network

Hebbian correlation requires:
1. The activity of the presynaptic neuron (locally available)
2. The activity of the postsynaptic neuron (locally available)
3. Nothing else

No global loss. No chain rule. No backward sweep. Each synapse computes its update independently, using only information present at its own location. This is biologically plausible in a way that backpropagation is not — real neurons cannot propagate error gradients backward through their axons.

The cost of this locality is that pure Hebbian learning has no notion of "was this correlation actually useful for the task?" That evaluative component comes from the neuromodulatory signal (see `neuromodulation_and_dopamine.md`).

### The Bridge to STDP

Hebbian plasticity treats time as irrelevant — it only cares that two neurons are active "at the same time." But in real neural circuits, timing matters enormously. A presynaptic spike arriving 5ms before the postsynaptic spike (causal timing) likely contributed to the postsynaptic firing. A presynaptic spike arriving 5ms after the postsynaptic spike (anti-causal timing) could not have caused it.

Spike-Timing Dependent Plasticity (STDP) refines the Hebbian principle by making the sign and magnitude of the weight change depend on the relative timing of pre- and post-synaptic spikes. This is covered in `stdp_and_eligibility_traces.md` and is the target for NeuroDrive's Milestone 4 spiking upgrade.

---

## Common Misconceptions

1. **"Hebbian learning is a complete learning algorithm."** It is not. Pure Hebbian plasticity is an unsupervised correlation detector with no notion of reward, loss, or task performance. On its own, it will amplify any correlation in the input, useful or not. It becomes a viable learning mechanism only when combined with a teaching signal (neuromodulation) and a temporal memory (eligibility traces).

2. **"Neurons that fire together wire together means simultaneous firing."** Hebb's original postulate says "takes part in firing it" — implying a causal, slightly pre-before-post relationship. The rate-based formulation Δw = η · pre · post loses this temporal nuance. STDP recovers it by making the sign of Δw depend on spike ordering.

3. **"Hebbian learning is the opposite of backpropagation."** They are not opposites; they answer different questions. Backpropagation asks "how should each weight change to reduce this specific loss?" Hebbian learning asks "which connections are correlated?" NeuroDrive combines Hebbian correlation (local) with a reward prediction error (global) to approximate what backpropagation achieves — but using only biologically plausible operations.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Synapse** | The junction between two neurons where signal transmission occurs; its strength (weight) is the primary substrate of learning. |
| **Hebbian plasticity** | The principle that co-active neurons should have their shared synapse strengthened; Δw ∝ pre × post. |
| **Learning rate (η)** | A scalar controlling the magnitude of each weight update. |
| **Unbounded growth** | The failure mode of pure Hebbian learning where weights increase without limit due to the absence of a weakening mechanism. |
| **Eligibility trace** | A decaying memory of recent synaptic correlation, used to bridge the gap between synapse activity and delayed reward. |
| **Local learning rule** | A weight update that uses only information available at the synapse (pre/post activity), without requiring global error signals. |

---

## Recommended Materials

1. **Dayan & Abbott, *Theoretical Neuroscience*, Chapter 8** — Covers Hebbian learning, correlation-based plasticity, and stability analysis (Oja's rule, BCM theory). The mathematical treatment is accessible and directly relevant.
2. **Hebb, *The Organization of Behaviour* (1949), Chapter 4** — The original statement of the postulate. Short and surprisingly readable. Focus on the cell assembly hypothesis.
3. **3Blue1Brown — "But what is a neural network?"** (YouTube) — While focused on artificial networks, the weight-adjustment intuition translates directly to understanding why correlated activity would strengthen connections.
4. **NeuroDrive README — "Learning Mechanism" section** — Shows the eligibility trace equation where the Hebbian term f(pre, post) appears, and how it feeds into the three-factor update rule.
