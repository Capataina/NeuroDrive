# From Baseline to Brain: The Architectural Transition

## What This File Covers

NeuroDrive's most important architectural event has not happened yet. The project currently runs A2C — a backpropagation-based policy gradient method. The long-term goal is a biologically-inspired local plasticity system with no backpropagation, no global gradients, and no batch training.

This file traces the planned transition from the current A2C baseline to the biological brain architecture, explaining what changes, what stays the same, and what the architectural risks are along the way.

**Status:** Future direction. The A2C baseline is current. The biological architecture is planned.

## Prerequisites

- `project/systems/a2c-brain.md` — the current implementation
- `project/decisions/a2c-as-baseline.md` — why A2C exists and what it validates
- `concepts/advanced/hebbian-plasticity.md` — the local learning rule that replaces backpropagation
- `concepts/advanced/eligibility-traces.md` — the temporal credit assignment mechanism
- `concepts/advanced/neuromodulation.md` — the δ signal that gates plasticity
- `project/evolution/milestone-roadmap.md` — the milestone context

---

## The Transition in One Diagram

```
CURRENT (A2C):

Observation → [MLP with backprop] → Action
                    ↑
         Global gradient from loss
         Batch update at rollout horizon
         Separate act phase and update phase
         Weights reset on session start


TARGET (Biological Brain):

Observation → [Sparse neural graph] → Action
                    ↑
         Local weight updates at each tick
         δ = r + γV(s') - V(s)  (broadcast neuromodulator)
         Δw_ij = η * δ * e_ij   (per-synapse local update)
         No backpropagation
         One brain, one lifetime (weights persist)
```

The external interface — `ObservationVector → ActionState` — does not change. The `agent/` boundary is what makes this replacement possible without touching the environment or analytics.

---

## What Stays the Same

### The Agent/ Boundary

The observation vector (23 dimensions) and action contract (`CarAction`) are unchanged. Both A2C and the biological brain consume `ObservationVector` and write `ActionState.desired`. The environment never needs to know which controller is active.

### The δ Signal

The reward prediction error formula is identical in both systems:

```
δ = r + γ * V(s') - V(s)
```

In A2C, δ is used as the TD advantage in the policy gradient loss. In the biological architecture, the same δ is broadcast as a neuromodulatory signal to all eligible synapses.

Same signal, completely different computational mechanism.

### The Reward Structure

`EpisodeState.current_tick_reward` is produced by `episode_loop_system` in `game/`. This does not change. The reward design (progress gain, time penalty, heading-speed penalty, crash penalty, lap bonus) is environment truth — it is independent of the controller.

### The Analytics Layer

All the per-tick trace capture, episode summaries, and export machinery can remain structurally similar. The analytics layer reads from whatever the brain exposes as training stats. The schema will evolve (different stats from a biological brain: weight magnitudes, eligibility trace norms, synapse counts instead of gradient norms), but the pipeline architecture is the same.

### The Debug Runtime

The HUD and overlays will need new content — visualising δ, weight distributions, synapse graphs — but the basic structure (F1/F2/F3 toggles, world-space gizmos, fixed-position panel) remains.

---

## What Changes

### The Controller Implementation

`src/brain/biological/` replaces `src/brain/a2c/` as the active controller. The `Brain` trait in `brain/types.rs` is the interface that both must implement.

Current `Brain` trait (minimal):
```rust
pub trait Brain {
    fn act(&mut self, obs: &ObservationVector) -> CarAction;
}
```

The biological brain will likely need a richer interface — a `tick()` method called every fixed step, not just at action time, so that eligibility traces can be maintained continuously. The trait may need to be extended for this.

### The Computation Model

A2C separates **act** (run forward pass, sample action) from **update** (collect rollout, run backward pass). These happen at different times in the fixed-tick pipeline.

The biological brain collapses this distinction. Weight updates happen **every tick**, not at a separate update phase:

```
Every tick:
1. Forward pass through sparse graph → Action
2. Compute δ from current reward and value estimates
3. Update eligibility traces: e_ij ← λ * e_ij + f(x_i, x_j)
4. Update weights: w_ij ← w_ij + η * δ * e_ij
```

There is no rollout buffer. There is no horizon. Updates are continuous.

### The Network Topology

A2C uses a dense fully-connected MLP: every neuron in layer L connects to every neuron in layer L+1. The biological brain uses a **sparse graph** with bounded connectivity:

```
- Fixed input nodes (one per observation dimension)
- Fixed output nodes (one per action dimension)
- Hidden neurons with bounded fan-in and fan-out
- Initial connectivity: random sparse (some configurable sparsity fraction)
```

The sparse topology is not just biological flavour — it is computationally necessary. Structural plasticity (Milestone 5) will modify this topology dynamically. A dense topology cannot be pruned or grown meaningfully.

### The Learning Rule

**A2C:** Global backpropagation. Gradients flow backward through all layers. The update for each weight depends on the loss function value and all other weights in the network.

**Biological brain:** Local plasticity. Each weight update depends only on:
1. The pre-synaptic activation `x_i`
2. The post-synaptic activation `x_j`
3. The eligibility trace `e_ij`
4. The global δ signal

No weight needs to know about any other weight. The learning rule is local in the strictest sense.

### Persistence

A2C weights are lost when the app exits. The biological brain must persist across sessions — "one brain, one lifetime" means the brain's state is saved to disk and loaded at startup. Milestone 2 includes `save/load brain state` as an explicit requirement.

---

## The Transition Strategy

The transition from A2C to the biological brain does not require changing the environment. The strategy:

1. **Validate A2C fully** (Milestone 1): confirm the environment contract is learnable. Fix remaining gaps (persistence, headless training, reproducibility).

2. **Implement biological brain in parallel** (Milestone 2): `src/brain/biological/` is added without removing `src/brain/a2c/`. Both exist in the codebase.

3. **Use `AgentMode` to switch** (existing mechanism): A2C is mode `Ai`, biological brain becomes a new mode (or replaces `Ai` when ready).

4. **Compare on identical environment** (Milestones 2–3): run A2C and biological brain on the same environment. The comparison answers: does local plasticity produce learning comparable to backpropagation?

5. **Retire A2C eventually**: once biological learning is stable and validated, A2C can be archived or removed. The environment and analytics layers are unchanged.

---

## The Value Function Problem

The biggest unresolved question in the Milestone 2 design is: **where does the value function come from?**

The biological brain uses:
```
δ = r + γ * V(s') - V(s)
```

This requires V(s) — an estimate of the expected future return from state s. In A2C, this is the critic network, trained by backpropagation.

In a fully local architecture, there is no backpropagation. How is V(s) learned?

### Options Under Consideration

**Option A: Keep a separate small critic (hybrid)**

Maintain a simple learned value function (potentially still updated by TD error, which is a local update rule). The biological network receives δ from this critic without needing to compute gradients through it.

This is a hybrid: local plasticity in the actor, but backpropagation or TD learning in the critic. It is not fully bio-plausible but is a reasonable pragmatic starting point.

**Option B: Distributed value representation**

The basal ganglia are hypothesised to maintain a distributed state-value representation through dopaminergic circuits. A biological analog might use a small sub-network within the graph to predict value, updated by its own local learning rule.

This is more biologically motivated but substantially harder to implement correctly. Milestone 2 may start with Option A and move toward Option B in later milestones.

**Option C: No learned value function**

Use only `r_t` as the training signal (no bootstrapping). This is equivalent to `γ = 0` — no long-term credit assignment. Performance will be poor for tasks requiring multi-step planning but the learning rule is fully local.

This is a viable baseline for Milestone 2's ablation (the "no δ" ablation is essentially this). Not a long-term solution.

The final design for Milestone 2's value function is an open design question.

---

## The Timing Transition: Continuous Updates

A2C updates weights once per rollout horizon (or on terminal + min_steps). The biological brain updates weights every tick.

This changes the timing constraints in the fixed-tick pipeline. Currently:

```
SimSet::Input:
    a2c_act_system  (appends to rollout buffer)

SimSet::Measurement:
    a2c_collect_reward_system  (appends reward, maybe triggers update)
```

For the biological brain:

```
SimSet::Input:
    bio_act_system  (forward pass through sparse graph → action)
                    (update eligibility traces based on activations)

SimSet::Measurement:
    bio_update_system  (compute δ = r + γV(s') - V(s))
                       (update weights: Δw = η * δ * e)
```

The ordering requirement remains: weight updates must happen after reward is computed by `episode_loop_system`, so `bio_update_system` belongs in `SimSet::Measurement` after episode logic.

---

## What the A2C Baseline Teaches the Biological Architecture

The value of A2C is not just "proving learnability." It also informs the biological design:

1. **What observation features matter:** If A2C struggles until a particular feature is added (e.g. lookahead curvature), the biological architecture must also include that feature.

2. **What reward scale works:** The biological brain will use the same δ signal. The reward magnitudes that produce stable A2C training should also be appropriate for δ-gated plasticity.

3. **Episode length and terminal behaviour:** The interaction between crash frequency, episode length, and learning stability is environment-specific. A2C characterises this before the biological architecture has to navigate it.

4. **Failure modes:** The types of reward hacking, oscillation, and collapse that A2C exhibits tell us what the environment incentivises. The biological architecture will face the same incentive structure.

A2C is, in this sense, a diagnostic run that characterises the learning landscape before the biological brain has to navigate it.

---

## Related Files

- `project/systems/a2c-brain.md` — current implementation
- `project/decisions/a2c-as-baseline.md` — the baseline rationale
- `project/evolution/milestone-roadmap.md` — the full milestone sequence
- `concepts/advanced/hebbian-plasticity.md` — the local learning rule at Milestone 2
- `concepts/advanced/eligibility-traces.md` — the temporal credit signal
- `concepts/advanced/neuromodulation.md` — the δ gating mechanism
- `concepts/advanced/structural-plasticity.md` — Milestone 5's topology adaptation
- `project/comparisons/rate-based-vs-spiking.md` — Milestones 2 vs 4
