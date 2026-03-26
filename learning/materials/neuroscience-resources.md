# Neuroscience Resources

A curated set of resources for studying the neuroscience theory that motivates NeuroDrive's planned biological architecture. Organised by topic and depth.

---

## Foundational Texts

### Bear, Connors & Paradiso — Neuroscience: Exploring the Brain (4th ed., 2015)

A comprehensive undergraduate-level neuroscience textbook. Relevant chapters for NeuroDrive:
- Chapter 3 (Neurons): neuron structure, membrane potential, action potentials
- Chapter 6 (Synaptic Plasticity): LTP, LTD, Hebbian plasticity
- Chapter 18 (Mental Illness and Drugs): dopamine system (motivating the neuromodulation model)

**Depth:** Broad coverage; appropriate for conceptual background without extreme depth.

### Kandel et al. — Principles of Neural Science (6th ed., 2021)

The definitive advanced reference. More detailed than Bear et al. for:
- Synaptic plasticity mechanisms (Part IV)
- Basal ganglia and reward circuits (Part VII)
- Learning and memory (Part VIII)

**Depth:** Graduate/research level. Read specific chapters rather than cover-to-cover.

---

## Key Papers

### Hebbian Plasticity and STDP

**Hebb (1949) — The Organisation of Behaviour**

The original formulation of Hebb's rule: "neurons that fire together, wire together." The conceptual foundation for all Hebbian learning rules in NeuroDrive.

**Bi & Poo (1998) — Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type**

The landmark experimental paper establishing spike-timing-dependent plasticity. Shows that pre-before-post firing leads to LTP and post-before-pre leads to LTD, with a characteristic timing window. This is the biological support for NeuroDrive's planned STDP at Milestone 4.

**Abbott & Nelson (2000) — Synaptic Plasticity: Taming the Beast**

A review of STDP, its properties, and its computational consequences. Good secondary reading after Bi & Poo for understanding how STDP interacts with network dynamics.

---

### Neuromodulation and Dopamine

**Schultz, Dayan & Montague (1997) — A Neural Substrate of Prediction and Reward**

The foundational paper linking dopaminergic neuron activity to reward prediction error. Shows that dopamine neurons in the primate brain fire in a pattern exactly matching temporal difference prediction error (δ = r + γV(s') - V(s)). This is the experimental basis for NeuroDrive's δ-gated plasticity.

Reading this paper alongside NeuroDrive's `concepts/advanced/neuromodulation.md` makes the biological motivation for the three-factor learning rule concrete.

**Montague, Dayan & Sejnowski (1996) — A Framework for Mesencephalic Dopamine Systems Based on Predictive Hebbian Learning**

The companion theoretical paper to Schultz 1997. Provides the formal framework connecting dopamine, TD learning, and Hebbian plasticity into a coherent computational model.

---

### Eligibility Traces and Three-Factor Learning

**Sutton & Barto (2018) — Chapter 12 (Eligibility Traces)**

The best introduction to eligibility traces in the context of TD learning and their relationship to λ-returns. Directly connects to NeuroDrive's Milestone 2 design.

**Fremaux & Gerstner (2015) — Neuromodulated Spike-Timing-Dependent Plasticity and Theory of Three-Factor Learning Rules**

A comprehensive review of reward-modulated STDP and three-factor learning rules. Covers the theoretical conditions under which local plasticity rules can approximate policy gradient methods. Relevant to the scientific foundation of Milestones 2–4.

---

### Structural Plasticity

**Bhatt & Bhatt (2009) — Dendritic Spine Dynamics**

Review of dendritic spine (synapse) formation and elimination in the adult brain. Shows that structural plasticity is ongoing in adult neural tissue, not just during development. Motivates the Milestone 5 structural plasticity implementation.

**Butz, Wörgötter & van Ooyen (2009) — Activity-Dependent Structural Plasticity**

A computational review of structural plasticity models. Covers growth rules, pruning rules, and the role of activity in synapse formation/elimination. Directly relevant to the growth and pruning rules in Milestone 5.

---

### Memory and Consolidation

**McClelland, McNaughton & O'Reilly (1995) — Why There Are Complementary Learning Systems in the Hippocampus and Neocortex**

The foundational paper for the hippocampus-neocortex consolidation model. Argues that rapid hippocampal learning and slow neocortical consolidation are complementary systems that prevent catastrophic forgetting. The biological motivation for NeuroDrive's Milestone 7 replay.

**Rasch & Born (2013) — About Sleep's Role in Memory**

Review of sleep-dependent memory consolidation. Shows that replay during sleep transfers memories from fast-learning systems to slow-consolidation systems. The biological analogy for NeuroDrive's "sleep phase" in Milestone 7.

---

## Conceptual Reviews

### Computational Neuroscience Textbooks

**Theoretical Neuroscience — Dayan & Abbott (2001)**

The standard reference for mathematical neuroscience. Chapters directly relevant to NeuroDrive:
- Chapter 7 (Network Models): recurrent network dynamics
- Chapter 8 (Plasticity and Learning): Hebbian rules, STDP, competitive learning
- Chapter 9 (Classical Conditioning and Reinforcement Learning): dopamine, TD learning, neuromodulation

**Spikes: Exploring the Neural Code — Rieke et al. (1997)**

Focuses on spike-based information encoding and the question: what information do spike trains carry? Relevant to Milestone 4's SNN encoding design.

---

## Topics by Milestone

### Milestone 2 (Rate-Based Local Plasticity)
- Sutton & Barto, Chapter 12 (eligibility traces)
- Schultz, Dayan & Montague 1997 (dopamine RPE)
- Fremaux & Gerstner 2015 (three-factor learning rules)

### Milestone 3 (Ablations)
- Any textbook treatment of experimental design
- Engstrom et al. 2020 (RL implementation — applies methodological standards from ML to neuro-inspired systems)

### Milestone 4 (SNN + STDP)
- Bi & Poo 1998 (STDP experimental basis)
- Abbott & Nelson 2000 (STDP computational review)
- Dayan & Abbott, Chapter 8 (STDP mathematical treatment)

### Milestone 5 (Structural Plasticity)
- Butz et al. 2009 (activity-dependent structural plasticity)
- Bhatt & Bhatt 2009 (dendritic spine dynamics)

### Milestone 6 (Continual Learning)
- McClelland et al. 1995 (complementary learning systems)
- Any review of catastrophic forgetting in neural networks

### Milestone 7 (Replay)
- Rasch & Born 2013 (sleep and memory consolidation)
- McClelland et al. 1995

---

## Entry Point Recommendations

For a learner focused on the computational motivation (not pure biology):

1. **Schultz, Dayan & Montague 1997** — 10 pages; shows the dopamine-as-prediction-error result
2. **Sutton & Barto, Chapter 12** — connects eligibility traces to the ML literature you may know
3. **Bi & Poo 1998** — the STDP experiment (short, clear, figures are compelling)
4. **Fremaux & Gerstner 2015** — the theoretical bridge between STDP and policy gradient

This sequence takes a mathematically-trained reader from zero neuroscience knowledge to a solid grasp of the biological mechanisms driving NeuroDrive's planned architecture.
