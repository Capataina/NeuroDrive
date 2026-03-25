# Continuous Control And MDPs

## Why This Matters Here

NeuroDrive is a continuous-control driving task. That single phrase quietly encodes several important consequences:

- actions are not discrete menu choices,
- the environment evolves through time,
- the effect of an action depends on current velocity, heading, and track geometry,
- the learner is not trying to classify inputs but to steer a dynamical system.

If you do not internalise this, the rest of the archive will feel like an arbitrary pile of implementation details.

## The Basic Formal Picture

An MDP is usually written as:

`(S, A, P, R, gamma)`

Where:

- `S` is the state space,
- `A` is the action space,
- `P` is the transition rule,
- `R` is the reward function,
- `gamma` is the discount factor.

Plain English:

- the world is in some condition,
- you choose an action,
- the world changes,
- you receive reward,
- you care about future reward too, not just immediate reward.

## How That Maps To NeuroDrive

This repository does not expose a fully privileged simulator state directly to the learner. Instead, the practical learning loop looks more like:

`observation_t -> action_t -> physics/environment step -> reward_t, done_t, observation_(t+1)`

Important distinction:

- the **environment state** is richer than the controller sees,
- the **observation vector** is a designed interface into that state,
- the **policy** learns on observations, not on raw omniscient world truth.

That distinction matters because many design questions in NeuroDrive are really representation questions:

- what should the controller see,
- what should remain internal environment truth,
- what signals make driving easier or harder to learn.

## Continuous Control

In a discrete control problem, an action might be:

- left,
- right,
- accelerate,
- brake.

In NeuroDrive, the controller instead emits:

- steering in `[-1, 1]`,
- throttle in `[0, 1]`.

Why that matters:

- the learner must reason about magnitudes, not just categories,
- small action differences can matter,
- policy outputs are naturally modelled as continuous distributions,
- action bounding and transformation become part of the algorithmic contract.

## Sequential Dependence

Driving is a sequential problem because a single good or bad action rarely determines the whole outcome. What matters is the accumulated effect of many choices:

- turn in too late now,
- drift wide two seconds later,
- leave the road after that.

This is why the project needs:

- discounting,
- value estimation,
- credit assignment,
- dense intermediate reward,
- careful observation design.

## Partial Observability In Practice

Even if the environment were Markov in a privileged internal sense, the policy does not necessarily receive a perfectly Markov observation. NeuroDrive mitigates this by including:

- ray distances,
- speed,
- lateral offset,
- heading error,
- angular velocity,
- lookahead geometry features.

These features are designed to make the controller’s input more decision-sufficient without leaking privileged progress truth directly.

## Worked Project Framing

Imagine the car approaches a corner.

At time `t`:

- front rays begin shortening,
- lookahead curvature increases,
- heading error may still be small,
- progress remains positive,
- immediate reward may not yet look bad.

The correct action depends on anticipating the turn before the crash signal arrives. That is exactly why sequential decision-making machinery matters: the learner must treat future consequences as part of present action quality.

## Common Misunderstandings

❌ "This is just supervised regression from observations to actions."

Why wrong:
There is no dataset of correct actions. Actions are judged by future outcomes under a reward function.

❌ "Because the task is visually simple, the decision problem is simple."

Why wrong:
The task is visually simple but dynamically non-trivial. Control, delay, geometry, and recovery still matter.

❌ "If there is dense reward, long-horizon reasoning stops mattering."

Why wrong:
Dense reward helps, but it does not erase sequential dependence. The controller still must act under delayed consequences.

## How This Appears In The Project

- fixed-tick simulation creates a clean step-based interaction loop,
- the observation vector is the learner-facing state approximation,
- the action interface is continuous,
- reward is dense but still sequentially meaningful,
- A2C is a natural baseline precisely because this is a continuous-control MDP-shaped problem.

## Related Files

- `concepts/foundations/probability-value-estimation-and-return.md`
- `concepts/core/observations-actions-and-representation.md`
- `project/systems/agent-interface.md`
- `project/systems/environment.md`
