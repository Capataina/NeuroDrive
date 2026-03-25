# Actor-Critic And GAE

## Why This Matters Here

The current learner in NeuroDrive is not a generic "neural network that somehow improves". It is a specific actor-critic baseline with GAE, bounded continuous actions, and handwritten optimisation logic.

If you want to reason about the repository’s live learning path, this concept file is mandatory.

## Actor-Critic At A Glance

The actor outputs an action distribution.

The critic estimates how promising the current situation is.

The policy update uses the critic’s estimate to reduce gradient variance. Instead of learning from raw returns alone, it learns from a better-shaped signal about whether the chosen action outperformed expectation.

## Why Actor-Critic Fits NeuroDrive

This task has:

- continuous steering and throttle,
- dense but still noisy reward,
- sequential dynamics,
- a compact engineered observation vector.

Those are friendly conditions for an actor-critic baseline.

## Current Policy Shape

The live A2C path uses:

- a handwritten two-hidden-layer actor,
- a handwritten two-hidden-layer critic,
- separate stacks rather than a shared trunk,
- Gaussian latent action sampling,
- `tanh` squashing,
- a transformed throttle output from `[-1, 1]` latent space into `[0, 1]`.

That last point matters. The policy does not simply emit raw steering and throttle; it emits sampled latent values that are transformed into bounded control outputs.

## GAE

Generalised Advantage Estimation smooths the advantage target by combining TD-style information across several steps.

A common recursive view is:

`delta_t = r_t + gamma V(s_(t+1)) - V(s_t)`

`A_t = delta_t + gamma lambda A_(t+1)`

Interpretation:

- `delta_t` says whether the step was better or worse than expected,
- `lambda` decides how much longer-horizon smoothing to keep,
- smaller `lambda` means lower variance but more bias,
- larger `lambda` means more return-like behaviour.

## Why GAE Matters In Practice

Driving reward can be noisy because:

- progress arrives incrementally,
- time penalty is always present,
- crash penalty arrives only at failure,
- turn quality may matter several ticks before a terminal event.

GAE helps avoid treating each noisy tick as a standalone verdict.

## Bootstrap Logic

When the rollout ends because a horizon was reached rather than because the episode truly terminated, the critic can estimate the remaining future value of the last observation. That bootstrap value prevents the update from pretending the future suddenly became zero for no reason.

This is one of the easy places to introduce subtle training bugs, which is why the repository documents it carefully.

## Limits Of The Current Baseline

The current A2C design is competent enough to be educational and useful, but several things still keep it from being a highly trustworthy experimental baseline:

- ad hoc RNG ownership,
- no save/load path,
- no evaluation-only mode,
- no vectorised synchronous collection,
- limited behavioural testing,
- weak run metadata.

That matters because a baseline should fail only when the environment or representation is genuinely hard, not because experiment discipline is thin.

## Common Misunderstandings

❌ "A2C is here because the project changed its mind and became a normal RL project."

No. A2C is here because the repository needs a learnability baseline now.

❌ "Once A2C works, the README’s biological-learning direction stops mattering."

No. The whole educational value of this repository depends on understanding why those are different layers of the project.

## Related Files

- `project/systems/a2c-baseline.md`
- `project/decisions/why-a2c-exists-in-a-brain-inspired-project.md`
- `concepts/advanced/a2c-vs-biological-learning.md`
