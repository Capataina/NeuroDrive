# Brain-Inspired Learning Principles

## Status

Planned project direction; not implemented yet as the main runtime learning path.

## Why This Matters Here

The README is explicit: NeuroDrive’s deeper aim is not to polish a conventional RL baseline forever. The long-term question is whether a persistent agent can learn to drive through local, online, brain-inspired adaptation rather than backpropagation-centric training machinery.

If you ignore this, you will misunderstand the repository’s identity.

## The Core Ideas

The README highlights several connected mechanisms:

- Hebbian plasticity,
- spike-timing dependent plasticity,
- eligibility traces,
- neuromodulation,
- structural plasticity,
- continual online learning.

These ideas do not all mean the same thing. They solve different parts of the learning problem.

## Local Plasticity

Local plasticity means synaptic change depends on information locally available at the connection, such as:

- presynaptic activity,
- postsynaptic activity,
- recent temporal relation between them,
- possibly a modulatory factor.

The attraction is clear:

- no global gradient graph,
- no need to backpropagate precise credit through a deep differentiable pipeline,
- learning can happen online while acting.

## Eligibility Traces

Eligibility traces are the bridge between "local event happened" and "global reward arrived later".

Intuition:

- a synapse remembers that it participated recently,
- reward arrives later,
- the system consolidates or weakens the earlier local change depending on that reward signal.

This is conceptually one of the closest links between biological intuition and delayed credit assignment.

## Neuromodulation

Neuromodulation adds a global teaching signal that says, roughly:

- what just happened was better than expected,
- or worse than expected.

Crucially, that signal does not itself specify the whole weight change. It gates local changes that were already eligible.

This is an important difference from gradient descent, where the update is computed from a global differentiable objective.

## Structural Plasticity

The README does not aim only for changing weights. It also points toward changing connectivity:

- add useful synapses,
- prune persistently weak or irrelevant ones,
- keep compute bounded,
- let the internal graph reorganise over experience.

That is much more than "train a fixed network". It is closer to life-long adaptive circuitry.

## Why This Is Harder Than A2C

The biological-learning direction raises much harder engineering questions:

- how to represent neuron state,
- how to keep updates local but useful,
- how to stabilise continual online learning,
- how to manage growth and pruning without chaos,
- how to validate whether learning is genuinely happening.

This is exactly why a simpler A2C baseline is defensible early on. The repository needs some confidence that the environment, action boundary, and observation design can support learning at all.

## Relationship To The Current Runtime

The current codebase already provides several pieces that a future biological system would still need:

- deterministic environment stepping,
- a stable observation and action contract,
- progress and reward measurement,
- analytics export,
- debug views,
- a disciplined schedule.

In that sense, the baseline is not a waste. It is part of the scaffolding.

## Misunderstanding To Avoid

Do not interpret brain-inspired learning as "copy every biological detail literally". The practical project goal is to capture useful learning principles from neuroscience, not to simulate a full biological nervous system at maximal fidelity.

## Related Files

- `concepts/advanced/a2c-vs-biological-learning.md`
- `project/decisions/why-a2c-exists-in-a-brain-inspired-project.md`
- `project/comparisons/current-baseline-vs-target-biological-system.md`
