# A2C Versus Biological Learning

## Why This Comparison Matters

This repository is easy to misread if you only look at the code or only read the README.

- If you only read the code, you may conclude this is simply an A2C racing project.
- If you only read the README, you may conclude the current implementation is still mostly aspirational.

The truth is more interesting: NeuroDrive currently uses a conventional RL baseline inside a project whose core intellectual ambition is not conventional RL.

## What A2C Gives The Project

- a learnability baseline,
- a live end-to-end autonomous control path,
- a way to validate the observation and action boundary,
- a way to stress the environment and reward design,
- a concrete training-health surface for analytics and debug tooling.

## What A2C Does Not Give The Project

- local synaptic credit assignment,
- lifelong structural adaptation,
- brain-like plasticity rules,
- a plausible model of learning from local state and global neuromodulation alone,
- the project’s eventual architectural identity.

## Comparison Table

| Dimension | Current A2C baseline | Target biological direction |
|---|---|---|
| Learning surface | policy/value networks | dynamic sparse neural graph |
| Credit assignment | gradients with advantage targets | local eligibility plus modulatory gating |
| Structure | fixed MLP topology | topology can grow or prune |
| Update cadence | periodic rollout-driven updates | continual online plasticity |
| Primary reason to exist | validate learnability | answer the project’s core research question |

## Why The Current Compromise Is Reasonable

The project would be taking an unnecessary risk if it tried to build the full biological architecture before validating:

- that the task can be learned at all,
- that the observation design exposes enough information,
- that the reward signal is usable,
- that the runtime scheduling is disciplined enough to support learning.

The A2C baseline is therefore best understood as an engineering probe rather than a betrayal of project identity.

## The Main Risk

The risk is not that A2C exists. The risk is that the repository could accidentally optimise around A2C so long that the baseline becomes the centre of gravity and the long-term architecture starts to deform around it.

This is why:

- comparisons matter,
- decisions need to be explicit,
- future-facing docs need to stay present,
- infrastructure should not overfit permanently to one baseline.

## What A Good Reader Should Conclude

A good reading of NeuroDrive is:

- the current runtime is real and technically meaningful,
- the current learner is transitional,
- the final research question is still ahead,
- today’s systems should be built so tomorrow’s biological learner can replace the baseline without wrecking the rest of the runtime.

## Related Files

- `project/decisions/why-a2c-exists-in-a-brain-inspired-project.md`
- `project/comparisons/current-baseline-vs-target-biological-system.md`
- `project/evolution/project-state-and-next-pressure-points.md`
