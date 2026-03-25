# Probability, Value Estimation, And Return

## Why This Matters Here

NeuroDrive’s A2C baseline does not optimise lap time by direct algebra. It updates a stochastic policy from sampled interaction. That makes uncertainty, expectation, return, and value estimation central concepts.

## Return

A standard discounted return is:

`G_t = r_t + gamma r_(t+1) + gamma^2 r_(t+2) + ...`

Interpretation:

- `r_t` is immediate reward now,
- later rewards matter too,
- `gamma` controls how much future reward still counts.

In NeuroDrive, this matters because:

- a safe steering correction may sacrifice tiny immediate reward,
- but preserve future progress and avoid a crash,
- so its true quality is long-horizon, not just immediate.

## Value

The value of a state is the expected future return if you continue from there.

`V(s_t) = E[G_t | s_t]`

Plain English:
If you are in this situation, how promising is the future on average?

Project relevance:
The critic in the A2C baseline estimates this quantity for the current observation.

## Advantage

Advantage is often written as:

`A(s_t, a_t) = Q(s_t, a_t) - V(s_t)`

Plain English:
Was this action better or worse than what I usually expect from this state?

Why useful:

- raw returns can be noisy,
- subtracting a baseline reduces variance,
- the policy then learns from relative action quality rather than absolute reward alone.

## Temporal-Difference Thinking

Instead of waiting until the end of an episode to know everything, value learning uses bootstrap logic:

`delta_t = r_t + gamma V(s_(t+1)) - V(s_t)`

This is the TD error.

Interpretation:

- if reality plus next-state promise is better than expected, `delta_t` is positive,
- if worse, `delta_t` is negative.

This single quantity is a bridge between immediate reward and value learning.

## Why Probability Matters

The current policy is stochastic. It does not always emit the same steering/throttle pair for the same observation because it samples from a continuous distribution.

That means:

- actions are random variables,
- performance is measured in expectation across trajectories,
- reproducibility depends on RNG discipline,
- learning must reason about log-probabilities and entropy, not just deterministic outputs.

## Worked Example

Suppose a tick produces:

- immediate reward `r_t = 0.2`,
- current value estimate `V(s_t) = 1.1`,
- next value estimate `V(s_(t+1)) = 1.3`,
- discount `gamma = 0.99`.

Then:

`delta_t = 0.2 + 0.99 * 1.3 - 1.1`

`delta_t = 0.2 + 1.287 - 1.1 = 0.387`

Interpretation:
This step turned out better than the critic had expected. That positive surprise should increase the probability of actions that helped cause it.

## Project Connection

NeuroDrive’s reward terms are shaped enough that per-step signals exist, but not so informative that value estimation becomes unnecessary. The critic is still needed because:

- crashes often happen after an accumulating mistake,
- cornering quality has delayed effects,
- progress reward alone can be misleading,
- non-terminal rollouts need bootstrap estimates.

## Common Misunderstandings

❌ "The critic is just a second opinion about the policy."

Better view:
The critic is a learning signal generator. It is not there for decoration; it reduces policy-gradient variance and supports advantage estimation.

❌ "Dense reward means returns and values are trivial."

Better view:
Dense reward helps, but the system still needs long-horizon aggregation and estimates of future promise.

## Related Files

- `concepts/core/actor-critic-and-gae.md`
- `project/systems/a2c-baseline.md`
- `project/systems/analytics.md`
