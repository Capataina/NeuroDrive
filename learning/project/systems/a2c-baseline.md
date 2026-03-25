# A2C Baseline

## Status

Current in the maintained implementation. Transitional rather than final.

## What This System Does

The A2C baseline provides the current autonomous controller used to test whether NeuroDrive’s environment and representation are learnable.

It owns:

- action selection from observations,
- rollout buffering,
- reward/done collection,
- GAE-based advantage computation,
- actor and critic updates,
- training-health snapshots for downstream observability.

## Current Runtime Shape

The current brain resource contains:

- a handwritten actor-critic model,
- a rollout buffer,
- `gamma`,
- `gae_lambda`,
- rollout/update thresholds,
- a simple step counter.

The actor and critic are separate MLP stacks with two hidden layers each. The action path samples Gaussian latent values, squashes them with `tanh`, and maps throttle into `[0, 1]`.

## Schedule Placement

This subsystem only makes sense because of its placement in the simulation schedule.

- `a2c_act_system` runs in `SimSet::Input`,
- `a2c_collect_reward_system` runs in `SimSet::Measurement`,
- `a2c_flush_on_exit_system` runs in `Last`.

That means the policy acts before physics, then later receives reward after episode truth has been finalised.

## What The Baseline Already Gets Right

- continuous bounded actions,
- separate actor and critic,
- GAE,
- rollout bootstrap for non-terminal truncation,
- gradient clipping,
- training-health export,
- mode switching with rollout reset,
- flush-on-exit update for residual buffer data.

This is well beyond a toy placeholder.

## Why It Is Still Not A Trustworthy Final Baseline

Several major gaps remain:

- RNG ownership is still ad hoc,
- there is no checkpointing,
- there is no evaluation-only mode,
- there is no vectorised synchronous rollout collection,
- there is limited behavioural regression testing,
- there is limited run metadata.

These are not cosmetic omissions. They determine whether experiment results are easy or hard to misread.

## What The Training Stats Are For

`A2cTrainingStats` captures:

- policy loss,
- value loss,
- entropy,
- explained variance,
- action spread,
- clamp fraction,
- layer weight and gradient norms,
- dead-ReLU fractions.

These stats are valuable because they give the project a live training-health surface without pretending that scalar reward alone tells the whole story.

## Why This Baseline Should Stay Modular

The README’s target system is not A2C. Therefore the A2C implementation should remain a replaceable learning module, not the permanent organising principle for the whole codebase.

That means future infrastructure work should prefer abstractions that are useful beyond this specific baseline:

- experiment discipline,
- reproducibility,
- observability,
- environment scaling,
- evaluation workflows.

## Related Files

- `concepts/core/actor-critic-and-gae.md`
- `project/decisions/why-a2c-exists-in-a-brain-inspired-project.md`
- `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
