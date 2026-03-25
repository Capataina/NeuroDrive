# Current Baseline Versus Target Biological System

## Why This Comparison Matters

This is the single most important comparison in the archive because it keeps the repository honest about what exists now versus what it is trying to become.

## What Stayed The Same

Across both the current and target visions, several goals remain constant:

- persistent learning in a racing environment,
- online interaction with a deterministic simulation,
- stable sensory and motor boundaries,
- interpretable progress and failure analysis,
- learning that happens within one continuing agent life rather than via evolutionary populations.

## What Changed

The implemented current path is:

- a handwritten A2C baseline,
- separate actor and critic,
- periodic rollout-driven updates,
- fixed MLP topology.

The target path described by the README is:

- a sparse neural graph,
- local plasticity,
- eligibility traces,
- neuromodulatory gating,
- structural plasticity,
- continual online adaptation.

## Why The Current Project Preferred The Baseline For Now

Because the baseline answers narrower engineering questions first with lower implementation ambiguity:

- is the environment learnable,
- are observations good enough,
- is reward aligned enough,
- is the runtime ordering correct,
- do analytics and debug surfaces expose the right behaviour.

## What The Learner Should Take Away

Do not treat the current baseline as a failed version of the intended system. Treat it as scaffolding and instrumentation for the harder system.

At the same time, do not let the existence of the baseline erase the final architectural ambition. Both mistakes flatten the project.

## Related Files

- `project/decisions/why-a2c-exists-in-a-brain-inspired-project.md`
- `project/evolution/project-state-and-next-pressure-points.md`
