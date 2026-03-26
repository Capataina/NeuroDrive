# Structural Plasticity

## Why This Matters Here

Synaptic weight changes (Hebbian, STDP, eligibility-gated) adjust *how strongly* existing connections transmit signals. But the brain also changes *which connections exist*. Synapses form, grow, and disappear. Circuits reorganise. Representational capacity is reallocated.

This is structural plasticity — and NeuroDrive's Milestone 5 plans to implement it. Understanding what structural plasticity does, why it is not just a gimmick, and what constraints it needs to remain stable is the focus of this file.

**Status:** Planned for Milestone 5. Foundational domain knowledge.

## Prerequisites

- `concepts/advanced/hebbian-plasticity.md`
- `concepts/advanced/eligibility-traces.md`

## Notation

| Symbol | Meaning |
|---|---|
| `G = (N, E)` | Neural graph: neurons N and edges E |
| `w_ij` | Synaptic weight |
| `e_ij` | Eligibility trace |
| `f_in(j)` | Fan-in of neuron j (number of incoming connections) |
| `f_out(i)` | Fan-out of neuron i (number of outgoing connections) |

---

## What Is Structural Plasticity?

Structural plasticity refers to changes in the *connectivity* of the neural network — the graph of which neurons are connected — rather than just the strengths of existing connections.

This includes:
- **Synaptogenesis**: formation of new synapses between neurons
- **Synaptic pruning**: elimination of inactive or weak synapses
- **Dendritic/axonal growth**: physical extension of neural processes that enables new connections

Structural plasticity operates on a slower timescale than synaptic weight changes. Weight changes happen within milliseconds to seconds; structural changes typically happen over hours to weeks in biology.

---

## Why Structural Plasticity Matters

### 1. Capacity Reallocation

A fixed-topology network allocates capacity (connections) upfront. If the task requires heavy processing in certain parts of the input space, fixed connectivity cannot adapt.

Structural plasticity allows the network to "invest" more connections where they are needed and prune connections that provide no value.

### 2. Efficiency

A fully connected network between N neurons has O(N²) synapses. Biological networks are **sparse** — a typical cortical neuron makes ~7,000 synapses out of a possible ~10 billion. Pruning maintains sparsity, which reduces computation and prevents overfitting.

### 3. Critical Periods and Maturation

In biological development, there are "critical periods" during which structural plasticity is highly active, followed by consolidation. NeuroDrive's Milestone 5 does not replicate this precisely, but the analogy motivates the idea that early learning might involve more structural change than later.

---

## Pruning Rules

A synapse is pruned when it is consistently uninformative:

**Criteria for pruning:**
1. **Persistently low magnitude:** `|w_ij|` remains below a threshold for an extended period
2. **Low eligibility contribution:** `e_ij` is consistently near zero (synapse has not been co-active recently)
3. **Bounded capacity:** if the total number of synapses is at a maximum, the weakest are removed when new ones are added

From the `README.md`:
> Pruning: remove synapses with persistently low magnitude and low eligibility contribution

**Practical pruning rule:**

```
if |w_ij| < w_prune_threshold AND e_ij < e_prune_threshold for T consecutive ticks:
    remove synapse (i → j)
```

The double criterion (low weight AND low eligibility) prevents pruning a synapse that is weak now but recently participated actively (it might be needed again).

---

## Growth Rules

New synapses form between neurons that are frequently co-active:

**Criteria for growth:**
1. **Co-activation:** neurons `i` and `j` fire frequently at similar times (high Hebbian correlation over time)
2. **Available capacity:** fan-in of `j` and fan-out of `i` are below their bounds
3. **No existing connection:** the synapse does not already exist

From the `README.md`:
> Growth: add synapses between recently co-active neurons when capacity is available

**Practical growth rule:**

```
if correlation(x_i, x_j) over window W > correlation_threshold:
    if f_in(j) < max_fan_in AND f_out(i) < max_fan_out:
        if synapse (i → j) does not exist:
            add synapse (i → j) with w_ij = w_init
```

The new synapse starts with a small initial weight. It will be strengthened or pruned based on subsequent activity.

---

## Bounded Fan-In and Fan-Out

The key safety constraint on structural plasticity in NeuroDrive:

```
f_in(j) ≤ max_fan_in    for all neurons j
f_out(i) ≤ max_fan_out  for all neurons i
```

Without these bounds:
- A single "hub" neuron could accumulate connections from most of the network
- The graph could grow explosively
- Compute cost per tick could become unbounded

The bounds enforce **network sparsity** and prevent graph blow-up. They are the structural analogue of weight clipping in synaptic plasticity.

---

## Metrics for Structural Plasticity (Milestone 5 Telemetry)

Planned observability from `README.md`:
- **Synapse count:** total edges at each checkpoint
- **Sparsity:** fraction of possible connections that exist
- **Churn rate:** number of synapses added + removed per N ticks

Churn rate is particularly diagnostic:
- High early churn: the network is reorganising significantly — good in early learning
- Declining churn: the network is stabilising — expected as performance plateaus
- Persistently high churn: the network may be oscillating without stabilising — a warning sign

---

## Structural Plasticity vs Weight Plasticity

| Dimension | Weight plasticity | Structural plasticity |
|---|---|---|
| What changes | Connection strengths | Connection existence |
| Timescale | Milliseconds | Hours/days (biological); configurable |
| Mechanism | Hebbian + δ gate | Co-activity + capacity constraints |
| Effect | Fine-tunes existing circuits | Reorganises which circuits exist |
| Risk | Weight explosion/collapse | Graph blow-up; instability |
| Stability tool | Weight clipping, decay | Bounded fan-in/out |

Both are needed for a truly adaptive system. Weight plasticity is fast and precise; structural plasticity is slow and architectural.

---

## Implementation Sketch for NeuroDrive

The structural plasticity tick in Milestone 5:

```
// Every T plasticity ticks:

// 1. Prune weak synapses
for each synapse (i → j):
    if should_prune(w_ij, e_ij, prune_config):
        remove (i → j) from graph

// 2. Identify growth candidates
for each (i, j) in candidate_pairs:
    if co_active_recently(i, j, growth_window) AND
       f_in(j) < max_fan_in AND
       f_out(i) < max_fan_out AND
       not exists(i → j):
        add_synapse(i, j, w_init)
        initialise e_ij = 0
```

Candidate pairs can be:
- All neuron pairs (O(N²), expensive)
- Pairs within a neighbourhood (based on spatial or functional proximity)
- Randomly sampled pairs weighted by recent co-activity metrics

---

## Why Structural Plasticity Is Not a Gimmick

The project's `README.md` makes this point explicitly:

> Structural plasticity is not a gimmick; it is how the system reallocates capacity over time.

The key argument: a fixed-topology sparse network can represent only as many distinct functions as its topology allows. If the driving task requires a specific circuit pattern that the initial random topology does not support, only structural plasticity can create it.

This is the argument for why Milestone 5 is scientifically interesting: it tests whether the network can self-organise its topology to better support the driving task, not just tune the weights of a predetermined structure.

---

## Common Misunderstandings

❌ "Structural plasticity just prunes useless connections"
✅ Structural plasticity both prunes *and* grows. The growth component is as important as the pruning — without it, the network could only lose capacity over time.

❌ "Bounded fan-in/fan-out is arbitrary"
✅ These bounds are biologically motivated (biological neurons have physically constrained connectivity) and computationally necessary (unbounded connectivity leads to O(N²) costs).

❌ "Structural plasticity requires external supervision"
✅ The rules described here are entirely local and self-organising. The only global signal is the δ neuromodulator, which already exists in the Milestone 2 architecture.

---

## Related Files

- `concepts/advanced/hebbian-plasticity.md` — the local learning that drives growth decisions
- `concepts/advanced/eligibility-traces.md` — the co-activity signal used for growth candidates
- `project/evolution/milestone-roadmap.md` — Milestone 5 details
- `concepts/advanced/continual-learning.md` — how structural plasticity interacts with forgetting
