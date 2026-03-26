# Neuromodulation

## Why This Matters Here

Neuromodulation is the mechanism that connects the reward signal to synaptic learning in biological systems. It is the "how does the brain know that what it just did was good?" mechanism. Without neuromodulation, eligibility traces accumulate but nothing decides which synaptic changes should persist. In NeuroDrive's planned architecture, the dopamine-like δ signal is this modulatory gate.

**Status:** Foundational domain knowledge. Planned for Milestone 2+.

## Prerequisites

- `concepts/advanced/hebbian-plasticity.md`
- `concepts/advanced/eligibility-traces.md`
- `concepts/core/reinforcement-learning.md` — Bellman equations, value functions

## Notation

| Symbol | Meaning |
|---|---|
| `δ` | Reward prediction error (dopamine signal) |
| `r` | Actual reward received |
| `V(s)` | Value estimate for current state |
| `V(s')` | Value estimate for next state |
| `γ` | Discount factor |

---

## What Is Neuromodulation?

In the brain, **neuromodulators** are chemicals that are released broadly across neural circuits and modify the sensitivity of synapses to plasticity signals. The most important neuromodulator for learning is **dopamine**.

Unlike standard neurotransmission (which is point-to-point: neuron A releases a chemical that specifically affects neuron B), neuromodulation works volumetrically: dopaminergic neurons in the midbrain release dopamine that diffuses across large brain regions and affects many synapses simultaneously.

The key insight: **dopamine does not carry specific action instructions**. It carries a scalar signal that says "things are going better than expected" (+) or "things are going worse than expected" (−).

---

## Dopamine as Reward Prediction Error

The most influential computational theory of dopamine (Schultz, Dayan & Montague, 1997) proposes that dopamine neurons signal the **reward prediction error (RPE)**:

```
δ = r + γ * V(s') - V(s)
```

This is exactly the TD error from reinforcement learning:
- `r` — the reward actually received
- `γ * V(s')` — the discounted expected future value from the new state
- `V(s)` — what was expected from the current state

**Interpretation:**
- `δ > 0`: things are better than expected → dopamine release → "this was a good surprise, reinforce it"
- `δ < 0`: things are worse than expected → dopamine dip → "this was a bad surprise, weaken the active synapses"
- `δ = 0`: exactly as expected → no dopamine change → no learning signal

---

## The Three-Factor Rule with Neuromodulation

Combining eligibility traces with dopamine gives the **three-factor learning rule**:

```
Δw_ij = η * δ * e_ij
```

Where:
1. `e_ij` — eligibility trace: "synapse i→j was recently co-active"
2. `δ` — dopamine RPE: "the recent outcomes were better/worse than expected"
3. `η` — learning rate

Only synapses that are "eligible" (have a non-zero trace) are modified. The dopamine signal gates *which* synapses change — those that were recently active during the period leading up to the surprising outcome.

This is a remarkably elegant solution to the temporal credit assignment problem:
- The eligibility trace handles the **spatial** part: which synapse was involved
- The neuromodulatory signal handles the **temporal** part: was this period of activity rewarded?

---

## The Difference from A2C's δ

NeuroDrive's A2C uses the same TD error formula:

```
δ = r + γ * V(s') - V(s)
```

In A2C, this is used to compute advantages that scale the policy gradient loss — a global, batch-processed, gradient-based update.

In the biological architecture, the same δ value will be broadcast as a neuromodulatory signal to all eligible synapses simultaneously, each receiving a local update without any global gradient computation.

Same signal, completely different computational mechanism.

---

## Prediction Error Timing

A critical biological observation: dopamine neurons respond to *reward prediction error*, not to the reward itself.

**Classic experiment:**
1. A conditioned stimulus (light) is paired with a reward (food)
2. Initially, dopamine fires at the time of the food reward
3. After conditioning, dopamine fires at the time of the light (the predictor), not the food
4. If the food is omitted after the light, there is a dopamine *dip* at the expected reward time

This is exactly what a TD error signal should do:
- When the light appears unexpectedly: δ > 0 (good surprise)
- When food appears (now predicted): δ ≈ 0 (expected)
- When food is omitted (bad surprise): δ < 0

This provides strong evidence that the brain computes something like a temporal difference learning signal in dopaminergic circuits.

---

## Value Function in the Biological Context

The δ signal requires an estimate of `V(s)` — the expected future value from the current state. In A2C, this is the critic network. In the biological brain, there are theories about how the basal ganglia maintain a distributed state-value representation, but the exact mechanism is an area of active research.

For NeuroDrive's Milestone 2 implementation:
- A value function will likely be maintained as a simple learnable module
- The δ signal will be computed from this value estimate plus the actual reward
- This makes Milestone 2 a hybrid: local Hebbian plasticity with a trained value function, connected by neuromodulation

---

## What Neuromodulation Does NOT Do

Neuromodulation is often misunderstood as a teaching signal that tells the brain *what* to do. It does not:

- Neuromodulation does not carry information about *which action was correct*
- It does not directly modify weights to implement a specific behaviour
- It does not compute credit assignment — the eligibility trace does that

Neuromodulation is a *gating signal*. It decides *when* and *how much* the Hebbian changes that have been accumulating should be committed to lasting weight changes. The content of what is learned (which synapses should change) comes from the eligibility traces. The *decision to commit* comes from dopamine.

---

## Reward vs Neuromodulation in NeuroDrive

In NeuroDrive's current A2C implementation:
- `current_tick_reward` is the per-tick reward used to compute advantages
- The advantage `δ = r + γV(s') - V(s)` scales the policy gradient

In the planned biological architecture:
- The same δ computation will produce the neuromodulatory signal
- This signal will be broadcast to all active synapses (those with non-zero eligibility traces)
- Each synapse updates locally: `Δw_ij = η * δ * e_ij`

The reward computation (`src/game/episode.rs`) is unchanged between A2C and the biological brain — only how the signal is *used* changes.

---

## Worked Example: One Reward Event

Suppose at tick `t`:
- `r_t = 0.5` (good progress)
- `V(s_t) = 2.0` (expected total future return)
- `V(s_{t+1}) = 2.1` (the next state looks good too)
- `γ = 0.99`

RPE:
```
δ = 0.5 + 0.99 * 2.1 - 2.0 = 0.5 + 2.079 - 2.0 = 0.579
```

δ is positive: things are slightly better than expected.

Suppose synapse (A→B) has eligibility `e_AB = 0.6` and synapse (C→D) has `e_CD = 0.1`:

```
Δw_AB = 0.01 * 0.579 * 0.6 ≈ +0.00347
Δw_CD = 0.01 * 0.579 * 0.1 ≈ +0.00058
```

Both synapses strengthen, but A→B gets 6× more credit because it was more active recently.

---

## Common Misunderstandings

❌ "Dopamine makes you feel good"
✅ While dopamine is colloquially associated with pleasure, its computational role in learning is specifically about *prediction error*, not raw reward. Unexpected *bad* events cause dopamine dips; expected good events do not elevate dopamine much.

❌ "The biological δ signal is the same as A2C's advantage"
✅ They are computed the same way (TD error), but A2C processes them in a batch-gradient pipeline while the biological architecture broadcasts them as a continuous-time modulator.

❌ "Any reward signal can serve as the neuromodulatory signal"
✅ The modulating signal should be the *prediction error* (RPE), not the raw reward. Raw reward without baseline subtraction would cause all recent synapses to strengthen regardless of whether the outcome was surprising — eliminating the "learning from mistakes" aspect.

---

## Related Files

- `concepts/advanced/eligibility-traces.md` — what δ gates
- `concepts/advanced/hebbian-plasticity.md` — the local learning that δ modulates
- `concepts/advanced/structural-plasticity.md` — a separate form of adaptation neuromodulation influences
- `project/evolution/from-baseline-to-brain.md` — how δ fits into the planned architecture
