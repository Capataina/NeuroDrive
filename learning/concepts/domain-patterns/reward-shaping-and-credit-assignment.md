# Reward Shaping And Credit Assignment

## Why This Matters Here

NeuroDrive is not a sparse-reward task where the only signal is "lap completed" or "crashed". That would make early learning almost unusably difficult. Instead, the project uses a shaped reward that tries to provide dense guidance without collapsing the task into a scripted controller.

## Current Reward Components

The implemented environment currently combines:

- positive reward for gains in best-so-far episode progress,
- a per-tick time penalty,
- a speed-weighted heading-misalignment penalty,
- a crash penalty,
- a lap-completion bonus.

This mixture tries to balance several incentives:

- move forward,
- do not stall,
- do not charge into corners while badly aligned,
- avoid leaving the track,
- complete a full lap when possible.

## Why Reward Shaping Exists

Without shaping, the learner would receive very little useful signal before crashing. With too much shaping, the learner may optimise the reward decomposition rather than the underlying driving objective.

Good shaping therefore tries to be:

- dense enough to guide learning,
- aligned enough not to reward pathological behaviour,
- interpretable enough to debug.

## Credit Assignment

Reward shaping gives a signal. Credit assignment decides who should receive it.

In the current A2C baseline:

- policy-gradient credit is assigned through returns and advantages,
- critic updates rely on TD-like value targets,
- the learner implicitly credits states and sampled actions.

In the README’s target biological direction:

- synapse-local eligibility traces would hold short-lived memory,
- neuromodulatory signals would gate consolidation,
- credit assignment would be local-plus-global rather than gradient-based.

## NeuroDrive-Specific Tension

One of the most important practical problems in this repository is avoiding "sprint then crash" behaviour:

- progress reward says move,
- time penalty says keep moving,
- but without enough turn-quality pressure the policy can learn reckless short-term gain.

The heading-speed penalty exists partly to counter that failure mode.

## What Makes This Hard

Reward design in NeuroDrive is difficult because each term influences behaviour differently:

- progress reward encourages exploration and forward motion,
- time penalty discourages idling,
- heading-speed penalty adds local caution,
- crash penalty makes failure sharp,
- lap bonus reinforces long-horizon success.

A poor balance can produce:

- timid crawling,
- aggressive wall hits,
- oscillatory steering,
- shallow local optima.

## What To Watch When Modifying Reward

If you change reward shaping, you should ask:

1. what behaviour do I expect to increase,
2. what cheap exploit have I accidentally created,
3. which analytics fields will expose the change,
4. whether observation design is the real issue instead.

## Related Files

- `project/systems/environment.md`
- `project/systems/analytics.md`
- `exercises/foundations/derive-a-reward-signal.md`
