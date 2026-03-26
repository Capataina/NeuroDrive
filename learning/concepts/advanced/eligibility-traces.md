# Eligibility Traces

## Why This Matters Here

Eligibility traces solve the **temporal credit assignment problem** — how to assign credit to a synapse that was active in the past when the reward arrives later. They are the central mechanism that makes Milestone 2's local plasticity system work without backpropagation. Understanding them is the most technically demanding conceptual step between current A2C and the planned biological brain.

**Status:** Foundational domain knowledge. The planned architecture for Milestone 2+ depends on eligibility traces as a core mechanism.

## Prerequisites

- `concepts/advanced/hebbian-plasticity.md` — why local learning needs temporal structure
- `concepts/core/reinforcement-learning.md` — returns and the temporal credit assignment problem

## Notation

| Symbol | Meaning |
|---|---|
| `e_ij(t)` | Eligibility trace of synapse (i→j) at time t |
| `λ` | Trace decay constant (not the same as GAE λ) |
| `x_i(t)` | Activity of presynaptic neuron i at time t |
| `x_j(t)` | Activity of postsynaptic neuron j at time t |
| `δ(t)` | Neuromodulatory signal (dopamine-like RPE) at time t |
| `η` | Learning rate |
| `w_ij` | Synaptic weight |

---

## The Temporal Credit Assignment Problem

Consider a synaptic connection that fired at time `t₀`, contributing to the car steering correctly into a bend. The reward from completing that bend arrives at `t₀ + Δt`, where `Δt` might be several hundred milliseconds or seconds.

How does the synapse know it was responsible for the good outcome?

Pure Hebbian learning strengthens the synapse at `t₀` based on co-activity — but it has already moved on by the time the reward arrives. The synapse has no memory of its recent activity.

The key insight:

> **The synapse must maintain a short-term memory of its recent co-activity so that it remains "eligible" to be modified when the reward eventually arrives.**

---

## The Eligibility Trace

An eligibility trace is a scalar associated with each synapse that:
1. Accumulates when the synapse participates in driving activity (pre × post)
2. Decays exponentially over time

```
e_ij(t) ← λ * e_ij(t-1) + f(x_i(t), x_j(t))
```

Where `f(x_i, x_j)` is the Hebbian correlation signal (e.g. `x_i * x_j`).

The trace `e_ij(t)` encodes "how recently and how strongly was this synapse involved in driving activity?"

---

## The Three-Factor Learning Rule

The full weight update rule uses three factors:

```
Δw_ij = η * δ(t) * e_ij(t)
```

Where:
1. `η` — learning rate
2. `δ(t)` — neuromodulatory signal (reward prediction error)
3. `e_ij(t)` — eligibility trace (recent participation)

**Interpretation:**
- `e_ij` says: "this synapse has been active recently"
- `δ` says: "recent outcomes were better (+) or worse (−) than expected"
- `Δw_ij` = "strengthen synapses that were recently active if things went well; weaken them if things went poorly"

This is the engineering analogue of dopamine-modulated Hebbian plasticity in the brain.

---

## Trace Decay and the Timing Window

The decay constant `λ ∈ (0, 1)` controls how far back in time the trace "remembers":

```
e_ij(t) = f(t) + λ * f(t-1) + λ² * f(t-2) + ...
```

- `λ` close to 1: the trace persists a long time — it can connect rewards to synaptic activity far in the past (but is noisier)
- `λ` close to 0: the trace decays quickly — only very recent activity is eligible (but reward assignment is more precise)

In practice, `λ` is a hyperparameter tuned to the time scale of reward delivery in the task. For NeuroDrive at 60 Hz, rewards (progress gains) arrive every tick, so a moderate λ that covers a few hundred milliseconds is appropriate.

---

## Comparing to GAE Lambda

Readers familiar with GAE will notice the structural similarity:

| GAE | Eligibility trace |
|---|---|
| `Â_t = δ_t + γλ * Â_{t+1}` | `e_ij(t) = λ * e_ij(t-1) + f(x_i, x_j)` |
| Exponentially weighted TD errors | Exponentially weighted Hebbian correlations |
| Bridges TD and Monte Carlo | Bridges local activity and delayed reward |
| Computed batch-wise at update | Updated continuously per tick |

Both use exponential traces to bridge a temporal gap. In A2C, the gap is between the value estimate and the actual return. In eligibility traces, the gap is between synaptic activity and the arriving reward signal.

**Key difference:** GAE is a batch computation done at update time. Eligibility traces are updated continuously, tick by tick, as the agent interacts with the environment. This is what makes them biologically plausible — the synapse does not need to store a full history, just the decaying trace.

---

## The NeuroDrive Learning Equation

From `README.md`:

```
Eligibility:   e_ij ← λ * e_ij + f(pre_i, post_j)

Weight update: Δw_ij = η × δ × e_ij
```

Where:
- `f` is correlation-based (rate: `pre × post`; spiking: STDP timing window)
- `δ = r + γ V(s') - V(s)` is the TD error (reward prediction error)

This is the same equation as the three-factor rule above, written in NeuroDrive's specific notation.

---

## Rate-Based vs STDP-Based Eligibility Traces

### Rate-Based (Milestone 2)

```
f(x_i, x_j) = x_i * x_j      (correlation between firing rates)
```

Simple, tractable, and computationally efficient. The trace is proportional to how co-active the pre and post neurons were.

### STDP-Based (Milestone 4)

```
f(x_i, x_j) = STDP_window(t_post - t_pre)
```

The trace accumulates based on the time difference between discrete pre and post spikes. This captures causal timing information that the rate-based version loses, at the cost of requiring explicit spike tracking.

---

## Implementation Requirements for Milestone 2

To implement eligibility traces in NeuroDrive, each synapse needs:

1. **A weight `w_ij`** — the current synaptic strength
2. **An eligibility trace `e_ij`** — the short-term memory of recent co-activity

Per-tick operations:
```
for each synapse (i → j):
    e_ij ← λ * e_ij + x_i * x_j      // update trace
    // (weight update is deferred until δ arrives)

// when δ is computed (at reward collection time):
for each synapse (i → j):
    w_ij ← w_ij + η * δ * e_ij        // apply gated update
    w_ij ← clamp(w_ij, w_min, w_max)  // stability
```

This is O(synapses) per tick — for a sparse graph, manageable.

---

## Why This Is Computationally Feasible Without Backpropagation

Backpropagation requires:
- A global loss
- A backward pass through every layer
- Knowledge of all weights to compute any single weight's gradient

Eligibility traces require per synapse:
- The presynaptic activity `x_i` (available locally)
- The postsynaptic activity `x_j` (available locally)
- The decaying trace `e_ij` (stored locally on the synapse)
- The global δ signal (broadcast, not computed locally)

The update is **embarrassingly local** — each synapse can update independently without knowledge of any other synapse's state. This is the key biologically plausible property.

---

## Worked Example: Three-Tick Scenario

Parameters: `λ = 0.9`, `η = 0.01`.

Suppose synapse (A→B) has pre-activity and post-activity over three ticks:

```
Tick  x_A    x_B    f = x_A*x_B   e (before update)   e (after update)
1     0.8    0.9    0.72          0.0 (initial)        0.72
2     0.1    0.2    0.02          0.9 * 0.72 = 0.648   0.648 + 0.02 = 0.668
3     0.0    0.0    0.0           0.9 * 0.668 = 0.601  0.601 + 0.0 = 0.601
```

Now suppose at tick 3, the reward arrives: `δ = +0.5` (better than expected).

```
Δw_{AB} = 0.01 * 0.5 * 0.601 = 0.003
```

The synapse strengthens slightly. Most of the credit came from the high co-activity at tick 1, which has decayed partially by tick 3 (0.72 → 0.601). The trace correctly attributes some credit to the old co-activity while down-weighting it for its distance in time.

Now suppose another synapse (C→B) was barely active:

```
Tick  x_C    x_B    f = x_C*x_B   e after 3 ticks
1     0.05   0.9    0.045         ≈ 0.09 (much smaller)
```

With the same δ:

```
Δw_{CB} = 0.01 * 0.5 * 0.09 = 0.00045
```

Synapse C→B is credited much less, correctly, because it was barely involved.

---

## Common Misunderstandings

❌ "Eligibility traces are the same as GAE's λ returns"
✅ Both use exponential weighting over time, but eligibility traces update continuously and per-synapse. GAE is a batch computation over a rollout. The mathematical structure is similar; the computational model is different.

❌ "Eligibility traces are just a way to average gradients over time"
✅ Eligibility traces do not compute gradients. They record local co-activity and remain available to be gated by a reward signal. No global backward pass is involved.

❌ "Large λ is always better because it uses more history"
✅ Large λ increases the reward assignment window but also assigns credit to synapses that may have been active long before the current outcome — including synapses from unrelated events. The optimal λ depends on the task's temporal structure.

---

## Related Files

- `concepts/advanced/hebbian-plasticity.md` — the local learning foundation
- `concepts/advanced/neuromodulation.md` — the δ signal that gates eligibility traces
- `project/evolution/from-baseline-to-brain.md` — how this replaces A2C
- `exercises/project/sketch-eligibility-traces.md` — design the trace system for Milestone 2
