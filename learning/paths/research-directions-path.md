# Research Directions Path

## Who This Path Is For

This path is for learners who want to understand what NeuroDrive is working toward — the scientific mission behind Milestones 2–9. If the A2C baseline is a scaffolding stage that validates the environment, this path teaches what the building itself will look like.

It is the right path if you want to understand the neuroscience and architecture behind local plasticity, spiking networks, structural adaptation, and continual learning — and how they connect to a concrete engineering plan.

## What This Path Assumes

- Basic understanding of what "the A2C baseline validates the environment" means (read `project/decisions/a2c-as-baseline.md` if unclear)
- Helpful but not required: some familiarity with reinforcement learning
- No prior neuroscience knowledge required

## What You Will Understand by the End

- The scientific motivation for replacing gradient descent with local plasticity
- How Hebbian learning and STDP work as local weight update rules
- How eligibility traces enable delayed credit assignment without backpropagation
- How neuromodulation (dopamine-like signals) gates learning in biological systems
- How structural plasticity allows topology reorganisation during experience
- The catastrophic forgetting problem and what memory consolidation strategies address it
- How the planned NeuroDrive architecture composes these mechanisms
- What each milestone from 2 to 9 is trying to prove and what it builds on the one before
- What the key technical challenges are between the current baseline and the intended brain

## Recommended Sequence

- [ ] `project/evolution/milestone-roadmap.md`
  - Read this first as a high-level orientation. You will not understand everything yet, but it will give you the map before you study the terrain.

- [ ] `concepts/advanced/hebbian-plasticity.md`
  - The foundation. How co-activation strengthens connections, mathematical formulations of Hebb's rule, and why Hebbian learning alone is incomplete.

- [ ] `concepts/advanced/eligibility-traces.md`
  - The core temporal mechanism. Eligibility traces are the bridge between Hebbian correlation and delayed reinforcement. This is the most technically demanding concept in the path — read it carefully.

- [ ] `concepts/advanced/neuromodulation.md`
  - How global reward signals modulate synaptic learning. Dopamine RPE, three-factor learning rules, and how the δ signal in NeuroDrive's design maps to biological neuromodulation.

- [ ] `concepts/advanced/spike-timing-dependent-plasticity.md`
  - STDP mechanics, the causal timing window, and how this compares to rate-based Hebbian learning. Connects to Milestone 4.

- [ ] `concepts/advanced/structural-plasticity.md`
  - Synapse growth and pruning, bounded fan-in/fan-out, capacity reallocation, and why this matters for long-run performance. Connects to Milestone 5.

- [ ] `concepts/advanced/continual-learning.md`
  - Why persistent, single-brain learning produces catastrophic forgetting without explicit mechanisms to prevent it. Memory consolidation, systems consolidation, and what Milestone 7 (replay/consolidation) is designed to address.

- [ ] `project/comparisons/rate-based-vs-spiking.md`
  - How the transition from Milestone 2 to Milestone 4 changes the neuron model, learning rules, and implementation approach. Why the rate-based step is valuable even though spiking is the longer-term goal.

- [ ] `project/evolution/from-baseline-to-brain.md`
  - How A2C connects to the biological architecture. What changes at each milestone and what stays the same. The architectural tensions that the transition creates.

- [ ] `exercises/project/sketch-eligibility-traces.md`
  - Design the eligibility trace system for Milestone 2. A design exercise: specify the data structures, update rules, and integration points without copying existing code.

- [ ] `materials/neuroscience-resources.md`
  - Directed reading for deeper study of the biological learning literature.

## After This Path

From here, proceed to:

- `paths/neuroscience-path.md` — if you want even deeper biological science coverage
- `paths/reinforcement-learning-path.md` — to understand A2C in full before the comparison is meaningful
- `project/systems/a2c-brain.md` — read the current baseline with the intended direction now in mind

## Notes

- This path is explicitly forward-looking. A significant portion of the content covers things that do not yet exist in the codebase. Status labels in the concept files mark what is foundational science versus what is planned project direction.
- Eligibility traces and neuromodulation are the load-bearing concepts. Every Milestone 2+ feature depends on getting those two right. Spend extra time on them.
- The contrast between gradient-based learning (A2C) and local plasticity (Brain v1+) is not just philosophical — it has concrete implications for data flow, memory access patterns, and parallelism. `project/evolution/from-baseline-to-brain.md` addresses this directly.
