# Exercise: Debug Reward Shaping

## Context

Reward shaping is the most underestimated source of bugs in reinforcement learning. A reward signal that looks correct can produce bizarre learned behaviours. This exercise presents several reward-shaping scenarios and asks you to reason through what behaviour each would incentivise — including some that look correct but are subtly wrong.

This is a reasoning and diagnosis exercise. You do not need to modify any code.

## Prerequisites

- `concepts/domain-patterns/reward-shaping.md` — reward decomposition theory
- `project/systems/environment-system.md` — the NeuroDrive reward components
- `concepts/core/reinforcement-learning.md` — episodes, returns

---

## Part 1: Current Reward Anatomy

The current NeuroDrive reward decomposition is:

```
r_t = progress_reward + time_reward + heading_speed_reward + crash_reward + lap_reward
```

Where:
- `progress_reward = max(0, new_fraction - best_fraction_this_episode) * progress_scale (140.0)`
- `time_reward = -0.005` (constant every tick)
- `heading_speed_reward = -0.02 * |heading_error| * speed` (scaled by actual speed)
- `crash_reward = -5.0` (one-off, when collision occurs)
- `lap_reward = +100.0` (one-off, when lap completes)

**Question 1:** The progress reward only awards for **new** best progress. If the car reaches 60% around the track in episode 1 and then reaches 65% in episode 2, what is the total progress reward earned in episode 2 for those 65 ticks (assuming uniform progress of ~1% per tick)?

**Question 2:** If the agent consistently stalls at 70% progress (crashes every episode at the same corner) but slowly improves to 72%, then 74%, then 76% over many episodes, is the progress reward design helping? Why or why not?

**Question 3:** Consider the `time_reward = -0.005` per tick. At 60 Hz, what is the total time penalty for a 60-second episode (max episode length)? Compare this to the potential lap bonus of +100.0. Is the time penalty strong enough to meaningfully discourage stalling?

---

## Part 2: Scenario Analysis

For each scenario, describe what behaviour the reward signal would incentivise. Be specific about what policy would maximise the return.

### Scenario A: Remove the time penalty

If `time_penalty = 0.0`:
- The agent still gets `progress_reward` for new best progress.
- There is still a crash penalty.
- What new behaviour might emerge? Is stalling still penalised?

### Scenario B: Replace best-progress with cumulative progress

Suppose `progress_reward = current_fraction_gain * progress_scale` — rewarding every forward tick, not just new bests:
- If the car turns around and drives backwards, what happens to `current_fraction_gain`?
- If the fraction wraps from 0.99 → 0.0 (lap complete) but backward travel creates large negative fractions, what might the policy learn?
- Is this design more or less robust than the best-progress design?

### Scenario C: Very large crash penalty

If `crash_penalty = -500.0`:
- The agent has a strong incentive to avoid crashes at all costs.
- What behaviour might emerge from this? Is "drive very slowly and conservatively" a valid policy? Would it receive high returns?
- What does this mean for the progress signal?

### Scenario D: Remove the heading-speed penalty

If `heading_speed_penalty = 0.0`:
- The agent can sprint toward a corner at full throttle without penalty, as long as it doesn't crash.
- What will the policy learn to do at corners?
- Why does the heading-speed penalty term help even if it doesn't directly prevent crashes?

---

## Part 3: Reward Hacking Analysis

A "reward-hacked" policy technically maximises the reward signal but does not exhibit the intended behaviour.

**Scenario: The stationary exploit**

Suppose the agent discovers that by staying still at the spawn point and doing nothing, it accumulates:
- `progress_reward = 0` (no progress)
- `time_reward = -0.005 * 3600 = -18.0` (60-second episode penalty)
- `crash_reward = 0` (never crashes)
- `lap_reward = 0`

**Total episode return = -18.0**

Meanwhile, an agent that drives to 50% progress and crashes after 30 seconds accumulates:
- `progress_reward = 0.5 * 140 = 70.0` (50% progress)
- `time_reward = -0.005 * 1800 = -9.0`
- `crash_reward = -5.0`

**Total episode return ≈ +56.0**

The driving agent strongly outperforms the stationary agent in expected return. But now consider: what if the agent is near a lap completion? At 99% progress, would any reasonable combination of penalties make crashing better than completing the lap?

---

## Part 4: Diagnostic Trace Analysis

You have a analytics trace file from a training run. The per-tick reward decomposition shows:

```
Episode 150, tick 200:
  progress_reward = 0.0
  time_reward = -0.005
  heading_speed_reward = -0.019
  crash_reward = 0.0
  total = -0.024

Episode 150, tick 201:
  progress_reward = 0.0
  time_reward = -0.005
  heading_speed_reward = -0.018
  total = -0.023

(continues for 200 more ticks with similar values)

Episode 150, tick 400:
  progress_reward = 0.0
  time_reward = -0.005
  heading_speed_reward = -0.020
  crash_reward = -5.0
  total = -5.025
```

**Diagnose:** What is the car doing during ticks 200–400? Why is `progress_reward` zero consistently? What does the nonzero `heading_speed_reward` tell you about the car's state? What caused the crash at tick 400?

---

## Hints

<details>
<summary>Hint 1 (best-progress semantics)</summary>

The best-progress reward uses `best_fraction_this_episode`. This resets to 0 at the start of each episode. So if the car previously reached 60% in a prior episode, this episode starts tracking from 0%. Each episode rewards from-scratch progress, not incremental improvement from the prior episode's maximum.

This means in Part 1, Question 1: the agent can earn progress reward for the first 65% of the track in every episode, not just the portion above the prior episode's 60%.

</details>

<details>
<summary>Hint 2 (Scenario B backwards driving)</summary>

In Scenario B with cumulative progress, if the fraction decreases (backwards travel), the `fraction_gain` is negative. At 60 Hz with high speed backwards, this could produce large negative rewards. But the policy might still prefer to drive forward — the incentive analysis depends on whether the positive forward progress rewards outweigh any possible backward strategy.

The main risk in Scenario B is corner cases where fraction wrapping is ambiguous.

</details>

<details>
<summary>Hint 3 (Part 4 diagnosis)</summary>

If `progress_reward = 0.0` for 200 consecutive ticks, the car is not making any new best progress. This could mean:
- The car has stalled or is circling without advancing
- The car has already reached its maximum progress for this episode and is not pushing further
- The car is going backwards

The nonzero `heading_speed_reward` (around -0.019) indicates the car is both moving (`speed > 0`) and has nonzero heading error. If the car were stationary, `heading_speed_reward` would be 0. The crash at tick 400 following this pattern suggests the car was circling or oscillating until it drifted off track.

</details>

## Reflection Questions

1. The best-progress reward creates a ratchet: the car only earns reward for reaching new territory. How does this compare to a reward that is proportional to absolute progress (distance from origin)? Which is more vulnerable to reward hacking?

2. The `progress_scale = 140.0` is much larger than the per-tick penalties. Why might this be intentional? What would happen if `progress_scale = 0.5`?

3. Design a simpler reward: `r_t = +1 if on track, -100 if crashed`. What policy would this incentivise? Would it learn to drive fast, or just to survive? How does the current reward design differ?

## Related Files

- `concepts/domain-patterns/reward-shaping.md` — potential-based shaping theory
- `project/systems/environment-system.md` — reward implementation
- `exercises/project/extend-observation-vector.md` — next project exercise
