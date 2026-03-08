# Structural Plasticity

## Prerequisites

- **Hebbian plasticity** — `hebbian_plasticity.md` (synaptic weight changes, local learning rules)
- **STDP and eligibility traces** — `stdp_and_eligibility_traces.md` (eligibility trace accumulation, three-factor rule)
- **Neuromodulation and dopamine** — `neuromodulation_and_dopamine.md` (reward prediction error, δ-gated weight updates)
- Familiarity with graph terminology: nodes, edges, fan-in (number of incoming connections), fan-out (number of outgoing connections), degree distribution

## Target Depth for This Project

**Level 3** — Can explain intuitively and understand the maths. You should be able to describe the difference between synaptic and structural plasticity, explain pruning and growth rules with concrete criteria, describe the constraints needed to prevent graph blow-up, and outline NeuroDrive's Milestone 5 plan for topology evolution.

---

## Core Concept

### Step 1: Beyond Weight Changes

Everything covered so far — Hebbian plasticity, STDP, eligibility traces, neuromodulatory gating — operates on a fixed graph topology. The set of synapses is determined at initialisation: we decide which neurons connect to which, and then learning adjusts the *strengths* of those connections. This is **synaptic plasticity**.

**Structural plasticity** goes further: it changes the graph itself. New synapses can *form* between previously unconnected neurons, and existing synapses can be *removed* entirely. The topology of the network is no longer fixed — it evolves through experience.

This distinction matters fundamentally. Synaptic plasticity can only redistribute signal flow across existing pathways. Structural plasticity can create entirely new pathways or eliminate obsolete ones. It changes what the network *can* compute, not just how strongly it computes through existing routes.

### Step 2: Biological Evidence

The brain does not maintain a fixed wiring diagram. Structural changes occur continuously, particularly in regions associated with learning:

- **Dendritic spine growth**: When a neuron receives consistently strong, correlated input from a particular source, new dendritic spines (the postsynaptic receiving structures) can sprout, creating additional synaptic contacts. This has been directly observed using two-photon microscopy in living animals during learning tasks.

- **Dendritic spine retraction**: Spines that receive little input or participate in uncorrelated activity shrink and are eventually absorbed. The synapse ceases to exist.

- **Axonal sprouting**: Over longer timescales, axons (the presynaptic transmission structures) can extend new branches to reach previously unconnected target neurons, establishing entirely new long-range connections.

These processes operate on timescales of hours to weeks — much slower than synaptic weight changes (milliseconds to minutes). In NeuroDrive, structural plasticity will similarly operate on a slower timescale than weight updates: topology changes occur periodically (e.g. every N episodes), not at every timestep.

### Step 3: Why Structural Plasticity Matters — Capacity Allocation

Consider a network initialised with random sparse connectivity. Some connections will, by chance, link neurons that process related information — these become useful and get strengthened through synaptic plasticity. Other connections will link unrelated neurons — these become weak and contribute nothing.

Without structural plasticity, those useless connections persist, occupying computational resources (memory, processing time) for zero benefit. Meanwhile, there may be pairs of neurons that *should* be connected (their activities are correlated in task-relevant ways) but are not, because the initial random wiring did not include that edge.

Structural plasticity solves this **capacity allocation** problem: remove connections that are not contributing, and add connections where they are needed. Over time, the network's topology becomes adapted to the task, concentrating connectivity where it is most useful.

### Step 4: Pruning Rules

Not every weak synapse should be pruned. A synapse might be temporarily quiet because the input pattern that activates it has not appeared recently — pruning it would destroy a useful connection that is simply dormant. NeuroDrive's pruning rule requires **both** conditions to be met:

1. **Low weight magnitude**: |w_ij| < θ_prune for a sustained period. The synapse is carrying negligible signal.
2. **Low eligibility contribution**: the recent average of |e_ij| is below a threshold. The synapse has not participated in correlated activity recently.

Both conditions together indicate a synapse that is both weak *and* unused — it is not contributing to computation and shows no sign of recent relevance. Requiring both conditions prevents premature pruning of useful-but-quiet connections.

A practical implementation evaluates pruning candidates periodically (e.g. every K episodes) rather than continuously, to avoid excessive churn and to allow synapses time to demonstrate their value under varied input conditions.

### Step 5: Growth Rules

New synapses are added between neurons that are **recently co-active** but not currently connected. The intuition: if neuron A and neuron B are frequently active at the same time (suggesting they process related information), but no direct synapse exists between them, creating one might allow useful signal flow.

Growth rules in NeuroDrive are constrained:

- **Co-activity threshold**: both neurons must have exceeded an activity threshold in recent episodes. Random co-incidental activation should not trigger growth.
- **Capacity available**: the postsynaptic neuron's fan-in must be below the maximum allowed. Growth cannot occur if the neuron is already at its connection limit.
- **Initial weight**: new synapses are initialised with a small random weight (near zero), so they do not immediately disrupt existing computation. They must "earn" their strength through subsequent synaptic plasticity.

### Step 6: Constraints — Preventing Graph Blow-Up

Without constraints, growth rules could create an ever-denser graph, eventually producing a fully-connected network that is computationally expensive and biologically implausible. NeuroDrive enforces:

- **Bounded fan-in**: each neuron has a maximum number of incoming connections (e.g. 20). This limits the computational cost of evaluating any single neuron and maintains sparse graph structure.
- **Bounded fan-out**: each neuron has a maximum number of outgoing connections, preventing any single neuron from dominating downstream computation.
- **Sparsity preservation**: the total number of synapses in the network is bounded. If growth would exceed this budget, it must be accompanied by corresponding pruning, or the growth is rejected.
- **Growth rate limit**: at most N new synapses can be added per evaluation period. This prevents sudden topological disruptions that could destabilise learned behaviour.

### Step 7: Metrics

To verify that structural plasticity is functioning correctly and beneficially, NeuroDrive will track:

- **Churn rate**: (synapses added + synapses removed) per evaluation period. High churn indicates instability — the topology is not converging. Zero churn indicates the topology has frozen — structural adaptation has stopped.
- **Sparsity**: the ratio of actual connections to possible connections. Should remain well below 1.0 (fully connected) throughout training.
- **Degree distribution**: a histogram of fan-in and fan-out across all neurons. A healthy distribution shows most neurons with moderate connectivity and a few hubs. A pathological distribution shows either extreme uniformity (random graph, no structure) or extreme concentration (star topology, fragile).
- **Pruned synapse diagnostics**: what fraction of pruned synapses had been recently created (high → growth rules are creating unhelpful connections) versus long-established (high → the task has shifted and old pathways are becoming obsolete).

---

## How NeuroDrive Will Use This

Structural plasticity is planned for **Milestone 5**, after the core learning mechanism (synaptic plasticity + eligibility traces + neuromodulation) has been validated in Milestones 2–4.

The implementation will:

1. **Run structural evaluation periodically**: every K episodes, scan all synapses for pruning candidates and all neuron pairs for growth candidates.
2. **Prune first, then grow**: removing dead connections before adding new ones ensures the synapse budget is respected and makes room for more useful connections.
3. **Log topology changes**: every structural update produces a record of which synapses were pruned (with their final weights and eligibility values) and which were created (with the co-activity evidence that triggered them).
4. **Visualise network structure**: a real-time or post-hoc graph visualisation showing the evolving topology, with edge colour/thickness encoding weight magnitude and node colour encoding activity level. This makes structural adaptation *visible*, consistent with NeuroDrive's philosophy that learning must be measurable, not guessed.

### Why Structural Plasticity Is Essential

Without structural plasticity, the network topology is frozen at initialisation. The initial random wiring determines which computational pathways *can* exist. If the initial wiring happens to lack a connection that would be useful for the task, no amount of synaptic plasticity can create it — the weight of a non-existent synapse cannot be adjusted.

Structural plasticity removes this lottery. The network can discover, through experience, which connections it needs and allocate its connectivity budget accordingly. Over long training runs, the topology should converge towards a task-adapted structure: dense connectivity in regions that process critical features (e.g. the mapping from forward-facing sensors to steering) and sparse connectivity elsewhere.

This is the engineering equivalent of how the brain dedicates more cortical area to frequently-used sensory modalities (the somatosensory homunculus devotes disproportionate area to hands and lips) — capacity flows to where it is needed.

---

## Common Misconceptions

1. **"Structural plasticity is just dropout."** Dropout randomly deactivates neurons during training as a regularisation technique, but the connections still exist and are used during inference. Structural plasticity permanently removes or creates connections. A pruned synapse is gone — its weight, its eligibility trace, and its computational contribution are all deleted. A grown synapse is genuinely new — it did not exist before and must build its weight from near-zero.

2. **"Adding more connections always helps."** More connections means more parameters, more computation, and more opportunity for noise to propagate. A fully-connected network is not better than a sparse one — it is more expensive and harder to train. The value of structural plasticity lies in *targeted* addition: connecting neurons that demonstrably benefit from direct communication, not adding connections indiscriminately.

3. **"Pruning is dangerous because you might lose important connections."** This is why NeuroDrive requires *both* low weight magnitude *and* low eligibility contribution for pruning. A connection that is important but currently quiet (low weight, high eligibility) is protected. A connection that carries signal but is not correlated with useful outcomes (high weight, low eligibility) is also protected. Only connections that are both weak and uninvolved are removed.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Structural plasticity** | Changes to the graph topology itself — formation and removal of synapses — as opposed to changes in synaptic weights. |
| **Synaptic plasticity** | Changes in the strength (weight) of existing synapses, without altering the topology. |
| **Dendritic spine** | A small protrusion on a neuron's dendrite that forms the postsynaptic side of a synapse; spines grow and retract during structural plasticity. |
| **Pruning** | The removal of a synapse from the network, deleting its weight, eligibility trace, and all associated state. |
| **Growth / sprouting** | The creation of a new synapse between previously unconnected neurons, initialised with a near-zero weight. |
| **Fan-in / fan-out** | The number of incoming / outgoing connections for a given neuron; bounded to prevent graph blow-up. |
| **Churn rate** | The sum of synapses added and removed per evaluation period; a measure of topological volatility. |
| **Sparsity** | The ratio of actual connections to possible connections; structural plasticity should maintain low sparsity. |
| **Capacity allocation** | The principle that connectivity should concentrate where it is most useful for the task, rather than being uniformly distributed. |

---

## Recommended Materials

1. **Holtmaat & Svoboda (2009), "Experience-dependent structural synaptic plasticity in the mammalian brain"** — A comprehensive review of dendritic spine dynamics observed via in vivo imaging. Covers growth, retraction, and stabilisation timescales.
2. **Lamprecht & LeDoux (2004), "Structural plasticity and memory"** — Links structural synaptic changes to memory formation, providing biological grounding for why topology changes matter for learning.
3. **Bellec et al. (2020), "A solution to the learning dilemma for recurrent networks of spiking neurons"** — Demonstrates eligibility-trace-based learning in spiking networks with structural adaptation. Relevant to NeuroDrive's Milestone 4–5 trajectory.
4. **NeuroDrive README — "Structural Plasticity" section** — States the pruning rules (low magnitude + low eligibility), growth rules (co-activity driven), and constraints (bounded fan-in/fan-out). This concept file provides the biological and computational rationale behind those design choices.
