# Exercise: Derive A Reward Signal

## Goal

Design a reward decomposition for a top-down driving learner that is dense enough to guide learning but not so over-scripted that it trivialises the task.

## Starting Point

Read:

- `concepts/domain-patterns/reward-shaping-and-credit-assignment.md`
- `project/systems/environment.md`

## Tasks

- explain what behaviour each current NeuroDrive reward term is trying to encourage or suppress,
- propose one alternative reward term that might help corner anticipation,
- identify one exploit your new term could accidentally create,
- explain how analytics would need to change to validate the effect.

## Hints

- Dense reward is useful, but every extra term changes incentives.
- If a term cannot be observed later in analytics, it is harder to validate.
