# Neuromodulation and Dopamine

## Prerequisites

- **Hebbian plasticity** — `hebbian_plasticity.md` (local correlation-based weight updates)
- **STDP and eligibility traces** — `stdp_and_eligibility_traces.md` (the three-factor rule Δw = η · δ · e_ij, the credit assignment problem)
- **Value functions and critics** — `concepts/core/reinforcement_learning/value_functions_and_critics.md` (state-value V(s), temporal difference error)

## Target Depth for This Project

**Level 3** — Can explain intuitively and understand the maths. You should be able to describe what neuromodulation is, explain the reward prediction error hypothesis, compute δ from a concrete sequence of rewards and values, and describe how NeuroDrive plans to use δ as the gating signal for eligibility-trace-based weight updates.

---

## Core Concept

### Step 1: What Is Neuromodulation?

In the brain, most communication between neurons is **point-to-point**: one presynaptic neuron connects to one postsynaptic neuron via a specific synapse. Neuromodulation is different. A small population of specialised neurons releases a chemical signal — a **neuromodulator** — that diffuses broadly and affects *many* synapses simultaneously.

Think of it as the difference between sending a letter to one person (synaptic transmission) and broadcasting an announcement over a loudspeaker (neuromodulation). The announcement does not tell any individual synapse what to do in detail. Instead, it modulates the *conditions under which learning happens*: it can amplify, suppress, or reverse the plasticity that local synaptic rules would otherwise produce.

The key neuromodulators include dopamine, serotonin, noradrenaline, and acetylcholine. Each has a different functional role. For NeuroDrive, the critical one is **dopamine**.

### Step 2: Dopamine and Reward Prediction Error

Dopamine neurons in the midbrain (specifically the ventral tegmental area and substantia nigra) do something remarkable: they do not encode *reward* directly. They encode the **difference between received reward and expected reward**.

This was demonstrated in a landmark series of experiments by Wolfram Schultz and colleagues (1997). When a monkey receives an unexpected juice reward, dopamine neurons fire vigorously. When the monkey learns to predict the reward (because a cue reliably precedes it), dopamine neurons fire at the *cue*, not at the reward itself — the reward is now expected, so it produces no prediction error. If an expected reward is *omitted*, dopamine activity dips below baseline — a negative prediction error.

This is the **Reward Prediction Error (RPE)**:

\[
\delta = r_{\text{received}} - r_{\text{expected}}
\]

- δ > 0: "better than expected" — reinforce what led here
- δ = 0: "as expected" — no change needed
- δ < 0: "worse than expected" — weaken what led here

### Step 3: The Parallel to TD Error

The RPE has a striking formal parallel to the **temporal difference (TD) error** in reinforcement learning:

\[
\delta = r + \gamma V(s') - V(s)
\]

where:
- r is the immediate reward
- γ is the discount factor
- V(s') is the estimated value of the next state
- V(s) is the estimated value of the current state

The term r + γV(s') is the "received" value (actual reward plus what we expect from here onward), and V(s) is the "expected" value (what we previously estimated this state was worth). The difference is exactly the prediction error.

This parallel is not a coincidence — it reflects a deep computational principle. Both biological dopamine and the TD error serve the same function: they signal whether an outcome was better or worse than predicted, allowing the system to adjust its expectations and behaviour accordingly.

### Step 4: How δ Gates Plasticity

The neuromodulatory signal δ does not directly tell any synapse how to change. Instead, it *gates* the consolidation of changes that were already prepared by local synaptic activity (captured in the eligibility traces). The three-factor learning rule is:

\[
\Delta w_{ij} = \eta \times \delta \times e_{ij}
\]

The three cases:

**Positive δ** (better than expected): All synapses with positive eligibility traces (recently active in correlated patterns) have their weights *increased*. The correlations that preceded the good outcome are reinforced. The larger the trace, the larger the consolidation — synapses that participated more heavily get more credit.

**Negative δ** (worse than expected): All synapses with positive eligibility traces have their weights *decreased*. The correlations that preceded the bad outcome are weakened. The system learns to avoid repeating the pattern that led to disappointment.

**Zero δ** (as expected): No weight change occurs, regardless of the eligibility traces. Expected outcomes carry no teaching signal — the system is already calibrated.

---

## Mathematical Foundation — Numerical Example

We follow a single synapse through a sequence where δ is computed from the critic's value estimates. Parameters: η = 0.05, λ = 0.8 (trace decay), γ = 0.99.

**Setup**: The synapse has been accumulating eligibility (e = 0.85) from recent pre-post correlations (see the trace accumulation example in `stdp_and_eligibility_traces.md`). Now the agent takes an action and we compute the TD error.

| Quantity | Value |
|----------|-------|
| V(s) — critic's estimate of current state | 3.20 |
| r — immediate reward received | 0.45 |
| V(s') — critic's estimate of next state | 3.50 |
| δ = r + γV(s') − V(s) | 0.45 + 0.99 × 3.50 − 3.20 = 0.45 + 3.465 − 3.20 = **0.715** |
| e_ij — eligibility trace | 0.850 |
| Δw = η × δ × e | 0.05 × 0.715 × 0.850 = **0.0304** |

The outcome was better than expected (δ = 0.715 > 0), so the synapse is strengthened. The magnitude of the change (0.0304) reflects both how eligible the synapse was (0.850) and how surprising the outcome was (0.715).

**Contrast**: Same situation, but the agent crashes instead of making progress.

| Quantity | Value |
|----------|-------|
| V(s) — critic's estimate of current state | 3.20 |
| r — crash penalty | −2.00 |
| V(s') — terminal state value | 0.00 |
| δ = r + γV(s') − V(s) | −2.00 + 0.99 × 0.00 − 3.20 = **−5.20** |
| e_ij — eligibility trace | 0.850 |
| Δw = η × δ × e | 0.05 × (−5.20) × 0.850 = **−0.221** |

The crash was much worse than expected (δ = −5.20), producing a large negative weight change. The synapse that was recently active in a correlated pattern is substantially weakened — the system is learning to avoid whatever led to the crash.

Notice the asymmetry: the crash signal (|δ| = 5.20) is much stronger than the progress signal (|δ| = 0.715). This is typical — penalties for catastrophic outcomes tend to dominate the learning signal, causing rapid avoidance learning. This mirrors biological behaviour: animals learn to avoid dangers faster than they learn to seek rewards.

---

## How NeuroDrive Will Use This

NeuroDrive's implementation plan for the neuromodulatory pathway:

1. **Compute δ from the critic**: The critic network (a separate value-estimating MLP retained from Milestone 1) produces V(s) and V(s'). The TD error δ = r + γV(s') − V(s) is computed exactly as in A2C, but its role changes: instead of driving backpropagation through the actor, it becomes a scalar broadcast signal.

2. **Broadcast δ to all synapses**: Unlike backpropagation where each weight receives a weight-specific gradient ∂L/∂w_ij, the neuromodulatory signal δ is a single scalar shared by every synapse in the network. This is the engineering equivalent of dopamine diffusing broadly across neural tissue.

3. **Modulate weight changes via eligibility**: Each synapse computes Δw_ij = η × δ × e_ij locally. The per-synapse specificity comes entirely from the eligibility trace e_ij — synapses that were recently active in correlated patterns receive larger updates, while inactive synapses are unaffected regardless of δ.

4. **Critic updates via standard gradients**: The critic itself continues to learn through conventional backpropagation of the value loss ½(V(s) − target)². Only the actor's synapses use the three-factor rule. This hybrid approach (gradient-trained critic, locally-plastic actor) is a pragmatic design choice for Milestone 2, balancing biological plausibility with stable value estimation.

### Why This Is Not Backpropagation

The crucial distinction bears repeating: in backpropagation, each weight w_ij receives a *unique* gradient ∂L/∂w_ij computed by the chain rule through all downstream layers. In the three-factor rule, each weight receives the *same* scalar δ, multiplied by its *own* eligibility trace. There is no chain rule. There is no per-weight error signal propagated backward through the network. The credit assignment comes from the temporal correlation captured in the trace, not from the structure of the computational graph.

This means the three-factor rule is less precise than backpropagation — it cannot compute exact per-weight contributions to the loss. But it is biologically plausible, computationally cheap, and sufficient for learning when the eligibility traces provide reasonable credit assignment.

---

## Common Misconceptions

1. **"Dopamine is the pleasure chemical."** Dopamine does not encode pleasure or reward directly. It encodes *surprise* — the difference between what was received and what was expected. A perfectly predicted reward produces no dopamine response. An omitted expected reward produces a *decrease* in dopamine. Pleasure and dopamine are correlated (unexpected rewards are both pleasant and dopamine-releasing) but the causal mechanism is prediction error, not hedonic value.

2. **"The global δ signal means all synapses learn the same thing."** All synapses receive the same δ, but the eligibility trace e_ij is different for each synapse. A synapse that was highly active in a correlated pattern (high e) receives a large Δw. A synapse that was quiet (e ≈ 0) receives essentially zero Δw. The per-synapse specificity comes from the local trace, not the global signal.

3. **"Three-factor learning is just REINFORCE reformulated."** REINFORCE uses ∇ log π(a|s) — a gradient through the policy network requiring the chain rule and cached activations. The three-factor rule uses e_ij — a locally-accumulated trace that requires no backward pass. They are computationally and conceptually distinct mechanisms that happen to solve the same problem (credit assignment under delayed reward).

---

## Glossary

| Term | Definition |
|------|-----------|
| **Neuromodulation** | A broadcast chemical signalling mechanism that modulates synaptic plasticity across large populations of synapses simultaneously. |
| **Dopamine** | The primary neuromodulator associated with reward prediction error in the midbrain. |
| **Reward Prediction Error (RPE / δ)** | The difference between received and expected reward: δ = r_received − r_expected. Positive δ signals "better than expected"; negative δ signals "worse than expected". |
| **TD error** | The temporal difference error δ = r + γV(s') − V(s) — the RL formalisation of reward prediction error. |
| **Gating** | The mechanism by which the modulatory signal δ controls whether eligibility-trace-prepared changes are consolidated into lasting weight modifications. |
| **Three-factor rule** | Δw = η × δ × e — weight change as a product of learning rate, global modulation, and local eligibility. |

---

## Recommended Materials

1. **Schultz, Dayan & Montague (1997), "A Neural Substrate of Prediction and Reward"** — The foundational paper linking dopamine firing to reward prediction error. The key figures showing dopamine responses to unexpected reward, predicted reward, and omitted reward are essential.
2. **Niv (2009), "Reinforcement learning in the brain"** — An accessible review connecting biological dopamine to TD learning. Excellent bridge between neuroscience and RL formalisms.
3. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 15** — "Neuroscience" chapter covering the dopamine-TD parallel, eligibility traces, and the three-factor rule from the RL perspective.
4. **NeuroDrive README — "Neuromodulation" section** — States the δ = r + γV(s') − V(s) computation and the Δw = η × δ × e update. This concept file explains the neuroscientific motivation behind those equations.
