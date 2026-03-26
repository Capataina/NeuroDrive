# Neuroscience Path

## Who This Path Is For

This path is for learners who want to understand the biological learning science that defines NeuroDrive's long-term mission. Milestones 2–9 are not arbitrary features — each one corresponds to a real biological mechanism. This path teaches those mechanisms from first principles and connects them to the project's intended architecture.

It is the right path if you want to understand *why* the project is trying to replace gradient-based learning with local plasticity rules, and what those rules actually do.

## What This Path Assumes

- Scientific literacy (comfortable reading mechanism descriptions)
- It helps to have a basic understanding of what a neural network's forward pass does, and to have read `concepts/core/reinforcement-learning.md` — but neither is strictly required
- No prior neuroscience knowledge required

## What You Will Understand by the End

- What Hebbian plasticity is, what "fire together, wire together" actually means mathematically
- How STDP extends Hebbian learning using spike timing
- What eligibility traces are and how they solve the temporal credit assignment problem without backpropagation
- What neuromodulation is, what dopamine reward prediction error (RPE) means, and how it gates synaptic consolidation
- How structural plasticity (synapse growth and pruning) allows the brain to reallocate representational capacity
- What continual learning means, why catastrophic forgetting happens, and what memory consolidation strategies exist
- How all these mechanisms compose into the intended NeuroDrive brain architecture
- Why NeuroDrive uses a racing task as the learning domain, and what the brain needs to solve it

## Recommended Sequence

- [ ] `concepts/advanced/hebbian-plasticity.md`
  - Start here. Hebb's rule is the foundation of biological learning. The file covers rate-based co-activation, mathematical formulations, and why Hebbian learning alone is insufficient.

- [ ] `concepts/advanced/spike-timing-dependent-plasticity.md`
  - STDP extends Hebbian learning by adding a temporal asymmetry: pre-before-post strengthens, post-before-pre weakens. This file covers the mechanism, the timing window, and its biological basis.

- [ ] `concepts/advanced/eligibility-traces.md`
  - This is the key mechanism that makes delayed reinforcement feasible without backpropagation. Read this carefully. It connects Hebbian correlation (a local signal) with reward modulation (a global signal).

- [ ] `concepts/advanced/neuromodulation.md`
  - Dopamine reward prediction error, three-factor learning rules, how the δ signal in NeuroDrive's plan corresponds to biological dopamine. This connects eligibility traces to the actual weight update mechanism.

- [ ] `concepts/advanced/structural-plasticity.md`
  - How the brain physically reorganises by forming and removing synapses. Covers growth rules, pruning rules, bounded fan-in/fan-out, and why this matters for capacity allocation.

- [ ] `concepts/advanced/continual-learning.md`
  - Catastrophic forgetting and why it is a fundamental problem in any persistent learning system. Memory consolidation, systems consolidation, and what "one brain, one lifetime" implies.

- [ ] `project/comparisons/rate-based-vs-spiking.md`
  - How the two neuron models differ, which Milestone each corresponds to, and what changes in the implementation between Milestone 2 (rate-based) and Milestone 4 (spiking).

- [ ] `project/evolution/from-baseline-to-brain.md`
  - How A2C connects to the biological architecture. This file makes the conceptual bridge between the current gradient-based baseline and the planned local-plasticity system.

- [ ] `project/evolution/milestone-roadmap.md`
  - Re-read the roadmap with the neuroscience knowledge you now have. The milestone descriptions will mean much more.

- [ ] `exercises/project/sketch-eligibility-traces.md`
  - Design the eligibility trace system that Milestone 2 will require. This is a design exercise, not a code exercise — the goal is to concretely plan what NeuroDrive needs to implement.

## After This Path

From here, proceed to:

- `materials/neuroscience-resources.md` — go deeper into the source literature
- `paths/reinforcement-learning-path.md` — understand how A2C relates to the biological direction
- `project/systems/a2c-brain.md` — read the current baseline with the planned direction in mind

## Notes

- Eligibility traces are the most technically demanding concept in this path. The file deliberately includes worked examples and multiple framings. Read it more than once if needed.
- The line between "STDP" and "eligibility traces" can be confusing. STDP is the *form* of the synaptic modification rule (timing-dependent). Eligibility traces are the *temporal mechanism* that bridges local activity with delayed reward. They are related but not identical concepts.
- The planned NeuroDrive architecture starts with rate-based neurons and eligibility traces (Milestone 2), adds STDP-family rules (Milestone 4), and then adds structural plasticity (Milestone 5). This path prepares you to understand all three stages.
