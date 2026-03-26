# Exercise: Sketch the Eligibility Trace Architecture

## Context

Milestone 2 will replace A2C's backpropagation-based update with local Hebbian plasticity gated by a neuromodulatory δ signal. The key component that makes delayed credit assignment possible is the eligibility trace — a per-synapse variable that remembers which synapses were recently co-active.

This exercise asks you to design the data structures and update logic for a rate-based eligibility trace system — not implement it in full, but sketch the architecture clearly enough that an implementation could follow directly from your design.

**This is a design exercise.** No code changes required.

## Prerequisites

- `concepts/advanced/eligibility-traces.md` — the three-factor rule
- `concepts/advanced/hebbian-plasticity.md` — the local Hebb rule
- `concepts/advanced/neuromodulation.md` — the δ signal and how it gates traces
- `project/evolution/from-baseline-to-brain.md` — the architectural transition plan
- `project/systems/a2c-brain.md` — the current system being replaced

---

## The Target Architecture

You are designing the biological brain's per-tick update cycle for Milestone 2.

The three-factor learning rule is:
```
Δw_ij = η * δ * e_ij

where:
  e_ij(t) ← λ * e_ij(t-1) + x_i(t) * x_j(t)
  δ(t)    = r_t + γ * V(s_{t+1}) - V(s_t)
```

This update happens every tick (unlike A2C which updates at rollout horizon).

---

## Part 1: Data Structures

Design the data structures needed for the biological brain. For each structure, specify:
- What it stores
- Whether it is per-neuron or per-synapse
- How it is initialised
- Its scope (component on the car entity? global resource?)

You need structures for:
1. The **sparse neural graph** (which synapses exist, between which neurons)
2. **Synaptic weights** `w_ij`
3. **Eligibility traces** `e_ij`
4. **Neuron activations** `x_i`
5. **Value estimate** `V(s)` (for computing δ)

---

## Part 2: Per-Tick Update Sequence

Write pseudocode for the complete per-tick update of the biological brain. The sequence should fit within the existing `SimSet` framework.

**Template to fill in:**
```
SimSet::Input:
  bio_act_system:
    1. ???  (forward pass)
    2. ???  (update eligibility traces)
    3. ???  (write to ActionState.desired)

SimSet::Measurement (after episode_loop_system):
  bio_update_system:
    1. ???  (read reward from EpisodeState)
    2. ???  (compute δ)
    3. ???  (update weights)
    4. ???  (if terminal: what happens to traces?)
```

Be specific about the ordering and what each step reads and writes.

---

## Part 3: Terminal Step Handling

In A2C, episode boundaries are handled by the rollout buffer (done flags mask the advantage recurrence). In the biological brain, updates happen continuously — what happens when the episode ends?

Answer:
1. Should eligibility traces be reset to zero when an episode ends? Why or why not?
2. Should the weight update happen on the terminal tick before or after the reset?
3. If δ is large and negative at the crash tick (bad outcome), and traces are still non-zero from recent steps, what is the effect? Is this the desired behaviour?

---

## Part 4: The Value Function Problem

The δ signal requires `V(s)` and `V(s')`. Design a minimal value function for Milestone 2:

Option A: A separate small linear network `V = W_v * obs + b_v`, updated by TD:
```
δ = r + γ * V(s') - V(s)
Δw_v = α_v * δ * obs  (TD gradient, not a backprop chain)
```

Option B: A distributed value representation in the hidden graph itself (more complex).

For each option:
1. What are the update equations?
2. Is the update truly local? (Option A's TD update is partially local — the error depends on both `V(s)` and `V(s')`, not purely per-weight information.)
3. Which option is more biologically plausible? Which is easier to implement correctly?

---

## Part 5: Concrete Example

Run through one tick manually with the following values:

```
Synapse: A → B
x_A = 0.7   (pre-synaptic activation)
x_B = 0.5   (post-synaptic activation)
e_AB(t-1) = 0.3  (previous trace)
λ = 0.9
η = 0.01
δ(t) = 0.4  (positive prediction error: better than expected)
w_AB(t-1) = 0.25
```

Compute:
1. `e_AB(t)` after trace update
2. `Δw_AB`
3. `w_AB(t)` after weight update

Then repeat with `δ(t) = -0.8` (worse than expected):
4. What is `Δw_AB` now?
5. Is the direction of the weight change correct (penalise recently active synapses that preceded a bad outcome)?

---

## Part 6: Stability Concerns

The biological learning rule has no global loss function that prevents weight explosion. Design the stabilisation mechanisms needed:

1. **Weight clamping:** Specify a clamping rule. Should it be hard (`|w_ij| ≤ w_max`) or soft (weight decay)?
2. **Trace clamping or bounding:** Should trace values be bounded?
3. **Learning rate schedule:** Should η decay over time? Why?

For each mechanism, identify which Milestone addresses it explicitly (Milestone 2 or Milestone 3).

---

## Hints

<details>
<summary>Hint 1 (where to place updates in SimSet)</summary>

The eligibility trace update should happen immediately after the forward pass (while activations are fresh). The weight update (which requires δ) must happen after `episode_loop_system` computes the reward. This suggests splitting into two systems:
- `bio_act_system` in `SimSet::Input`: forward pass + trace update
- `bio_update_system` in `SimSet::Measurement`: compute δ + update weights

</details>

<details>
<summary>Hint 2 (terminal step trace reset)</summary>

In A2C's GAE, the recurrence masks at terminal steps with `(1 - done)`. The biological analog: at a terminal step, δ is typically large (crash = large negative, lap = large positive). The weight update `η * δ * e_ij` applies this signal to all recently active synapses. After the update, resetting traces to 0 prevents the next episode's activations from being "blamed" for the current episode's terminal outcome.

</details>

<details>
<summary>Hint 3 (Part 5 numerical answers)</summary>

For positive δ:
```
e_AB(t) = 0.9 * 0.3 + 0.7 * 0.5 = 0.27 + 0.35 = 0.62
Δw_AB = 0.01 * 0.4 * 0.62 = 0.00248
w_AB(t) = 0.25 + 0.00248 = 0.25248
```

For negative δ:
```
Δw_AB = 0.01 * (-0.8) * 0.62 = -0.00496
w_AB(t) = 0.25 - 0.00496 = 0.24504
```

The direction is correct: a negative surprise weakens the synapses that were recently active (they "participated" in an action that led to a worse-than-expected outcome).

</details>

## Reflection Questions

After completing the sketch:

1. In A2C, the rollout buffer stores experiences from many ticks and processes them all at once. In the biological brain, updates happen tick by tick. What are the practical consequences for implementation? For numerical stability?

2. The rate-based trace `e_ij ← λ * e_ij + x_i * x_j` uses the activation *product*. What does it mean when `x_i > 0` and `x_j = 0`? The trace decreases toward zero but never goes negative (assuming both activations are non-negative). Is this the correct behaviour for weakening a "one-sided" synapse?

3. Compare the per-tick update cost of the biological brain versus A2C. A2C runs a full backward pass only at the rollout horizon. The biological brain updates all traces every tick. For a graph with N neurons and K synapses, what is the per-tick cost? Is this feasible?

## Related Files

- `concepts/advanced/eligibility-traces.md`
- `concepts/advanced/neuromodulation.md`
- `project/evolution/from-baseline-to-brain.md`
- `project/evolution/milestone-roadmap.md` — Milestone 2 and 3 specifications
