# Spike-Timing Dependent Plasticity (STDP)

## Why This Matters Here

STDP is the learning rule that governs how real biological synapses change in response to neural activity. It extends Hebbian learning by adding a temporal asymmetry: the *order* of presynaptic and postsynaptic spikes determines whether the synapse strengthens or weakens. This is the mechanism NeuroDrive plans to implement in Milestone 4 (the spiking network upgrade).

**Status:** Planned for Milestone 4. Foundational domain knowledge.

## Prerequisites

- `concepts/advanced/hebbian-plasticity.md` — the rate-based foundation

## Notation

| Symbol | Meaning |
|---|---|
| `Δt = t_post - t_pre` | Time difference between post and pre spike |
| `ΔW` | Change in synaptic weight |
| `A+` | Maximum potentiation magnitude |
| `A-` | Maximum depression magnitude |
| `τ+, τ-` | Time constants for potentiation / depression windows |

---

## Core Idea: Timing Determines Causality

Rate-based Hebbian learning asks: "are these two neurons active at the same time?"

STDP asks: "did the presynaptic neuron's spike *cause* the postsynaptic neuron to fire?"

This is the **causal rule**: if A fired just before B fired, A's activity may have contributed to B's firing. Strengthen that synapse. If B fired *before* A fired, A could not have caused B — weaken the synapse.

Formally:

```
If Δt > 0 (pre before post):   ΔW = +A+ * exp(-Δt / τ+)    [potentiation]
If Δt < 0 (post before pre):   ΔW = -A- * exp(+Δt / τ-)    [depression]
```

where `Δt = t_post - t_pre`.

---

## The STDP Timing Window

The weight change plotted as a function of `Δt` produces the characteristic STDP curve:

```
ΔW
 |
+|     *
 |   *   *
 |  *       *
 |----------*---------→ Δt (ms)
              *   *
               * *
                *
-|
```

- For small positive `Δt` (pre slightly before post): large potentiation
- As `Δt` increases: potentiation decays exponentially
- For small negative `Δt` (post slightly before pre): depression
- As `|Δt|` increases on the negative side: depression decays

**Biological parameter ranges:**
- `A+` ≈ 0.01–0.1 (typical potentiation magnitude)
- `A-` ≈ 0.01–0.12 (depression is often slightly larger than potentiation to keep weights stable)
- `τ+ ≈ 20 ms`, `τ- ≈ 20 ms` (order-of-magnitude estimate for cortical synapses)

---

## Why Temporal Asymmetry Is Biologically Correct

The timing window reflects a plausible mechanism for **causal attribution at the synapse level**:

1. Presynaptic terminal releases neurotransmitter (pre fires)
2. Neurotransmitter diffuses across the synaptic cleft
3. Postsynaptic receptor briefly opens
4. If the postsynaptic neuron happens to fire in this window, the synapse was "part of the cause"
5. Molecular coincidence detectors (like NMDA receptors) strengthen the synapse

If the postsynaptic neuron fires *first* (negative `Δt`), the presynaptic neuron contributed nothing — molecular mechanisms actively weaken such synapses.

---

## STDP Implements a Form of Temporal Correlation

STDP can be understood as a temporally asymmetric extension of Hebbian learning. The rate-based Hebb rule correlates activities:

```
Δw ∝ x_pre * x_post
```

STDP correlates spike *times*:

```
Δw ∝ STDP_window(t_post - t_pre)
```

Both are local rules. STDP adds causal structure by treating forward-time correlation (pre before post) as evidence of causation.

---

## Relationship to Eligibility Traces

In the rate-based model with eligibility traces, the synapse maintains a "memory" of recent pre-post co-activity that decays exponentially. This is the rate-based analogue of the STDP time window.

The key difference:
- **Rate-based eligibility traces** integrate activity over time continuously
- **STDP** responds to discrete spike-timing differences

For rate-based neurons (Milestone 2), eligibility traces capture the relevant temporal structure without requiring spike timing. For spiking neurons (Milestone 4), STDP is the direct mechanism.

---

## STDP and the Stability Problem

STDP shares the instability problem of Hebbian learning. Without constraints:
- All synapses strengthen or weaken without bound
- Networks can converge to silent (no spikes) or saturated (constant firing) states

Solutions include:
- **Synaptic scaling** (homeostatic plasticity): if a neuron fires too much, all its incoming synapses scale down
- **Hard weight bounds**: clip weights to `[w_min, w_max]`
- **Soft weight bounds**: STDP magnitude decreases near the bounds
- **Pair vs triplet STDP**: standard STDP looks at pairs of spikes; triplet rules can better reproduce biology

---

## Reward-Modulated STDP

Standard STDP responds only to spike timing — it has no concept of outcome. **Reward-modulated STDP** (R-STDP) combines STDP with a global reward signal:

```
Δw_ij = reward_signal * STDP_window(Δt)
```

Or, using eligibility traces to bridge the timing gap:

```
Δw_ij = δ * e_ij
```

where `δ` is the dopamine-like signal (reward prediction error) and `e_ij` is an eligibility trace that has accumulated the STDP-shaped synaptic activity.

This is the mechanism that will operate in Milestone 4 NeuroDrive.

---

## Connection to the NeuroDrive Plan

At Milestone 2, NeuroDrive uses **rate-based neurons** with Hebbian eligibility traces. The STDP window shape does not apply directly because there are no discrete spikes.

At Milestone 4, the upgrade to **spiking neurons** introduces:
1. Membrane potential dynamics
2. Threshold and reset mechanism
3. Spike timing records
4. STDP-shaped eligibility accumulation based on spike timing differences

The key continuity: both stages use eligibility traces and dopamine modulation. Only the form of the eligibility accumulation changes (rate-based correlation vs spike-timing-based STDP window).

---

## Worked Example: One STDP Event

Suppose:
- Pre-synaptic neuron fires at `t = 10 ms`
- Post-synaptic neuron fires at `t = 15 ms`
- `Δt = 15 - 10 = +5 ms`
- `A+ = 0.05`, `τ+ = 20 ms`

Potentiation:

```
ΔW = +0.05 * exp(-5 / 20) = +0.05 * exp(-0.25) ≈ +0.05 * 0.779 ≈ +0.039
```

Now suppose the post fires at `t = 8 ms` (before the pre):
- `Δt = 8 - 10 = -2 ms`
- `A- = 0.06`, `τ- = 20 ms`

Depression:

```
ΔW = -0.06 * exp(2 / 20) = -0.06 * exp(0.1) ≈ -0.06 * 1.105 ≈ -0.066
```

The post-before-pre event produces depression larger than the pre-before-post potentiation — which is typical of biologically observed asymmetry.

---

## Common Misunderstandings

❌ "STDP and Hebbian learning are the same thing"
✅ Hebbian learning responds to co-activity (rate correlation). STDP responds to spike timing (causal ordering). Rate-based Hebbian can be implemented without any notion of spike timing.

❌ "STDP can only strengthen synapses"
✅ STDP has both potentiation (pre before post) and depression (post before pre) windows. The ratio and shapes of these windows vary across brain regions and synapse types.

❌ "Spiking networks are strictly better than rate networks"
✅ Spiking networks are more biologically realistic. Whether they are more *useful* for the NeuroDrive task is an open question — which is why Milestone 2 uses rate-based neurons first, and Milestone 4 compares the two.

---

## Related Files

- `concepts/advanced/hebbian-plasticity.md` — the rate-based foundation
- `concepts/advanced/eligibility-traces.md` — the temporal bridge to delayed reward
- `concepts/advanced/neuromodulation.md` — the global teaching signal
- `project/comparisons/rate-based-vs-spiking.md` — when spiking matters vs when it does not
- `project/evolution/milestone-roadmap.md` — Milestone 4 (spiking upgrade)
