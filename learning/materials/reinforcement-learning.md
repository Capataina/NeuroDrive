# Materials: Reinforcement Learning

## Why This Topic Matters Here

Even though NeuroDrive does not want to remain a standard RL project forever, the current runtime still uses RL machinery seriously enough that external reading helps.

## What To Prioritise

Prioritise material that improves understanding of:

- policy gradients,
- actor-critic methods,
- GAE,
- continuous control,
- implementation details that make on-policy methods succeed or fail.

## Recommended Reading Strategy

1. first learn policy gradients and actor-critic basics,
2. then learn GAE and why variance reduction matters,
3. then read practical implementation papers or write-ups about on-policy details,
4. then return to NeuroDrive and compare the theory with the actual handwritten implementation.

## Best Resource Types

- clear policy-gradient tutorials that derive the basic update,
- actor-critic explainers with both intuition and equations,
- on-policy implementation analyses discussing seemingly small details,
- continuous-control case studies.

## How To Use This Alongside The Archive

Read these external materials after:

- `concepts/foundations/probability-value-estimation-and-return.md`
- `concepts/core/actor-critic-and-gae.md`

Then return to:

- `project/systems/a2c-baseline.md`
- `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
