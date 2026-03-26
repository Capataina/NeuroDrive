# Actor-Critic Architecture

## Why This Matters Here

NeuroDrive's A2C is, by name, an actor-critic method. Understanding what this architecture is — why two networks are trained simultaneously and how they interact — is required to understand why the `ActorCritic` struct in `src/brain/a2c/model.rs` looks the way it does, and what the two loss functions in `a2c_update.rs` are optimising.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — value functions
- `concepts/core/policy-gradient-methods.md` — policy gradient theorem and baseline subtraction

## Notation

| Symbol | Meaning |
|---|---|
| `π_θ(a\|s)` | Actor (policy) parameterised by θ |
| `V_φ(s)` | Critic (value function) parameterised by φ |
| `A_t` | Advantage estimate |
| `L_actor` | Actor (policy) loss |
| `L_critic` | Critic (value) loss |
| `H(π)` | Policy entropy |

---

## Core Idea

The actor-critic architecture splits the RL agent into two components that learn concurrently:

1. **The actor** (`π_θ`): the policy. It maps states to action distributions and decides what to do.
2. **The critic** (`V_φ`): the value function. It maps states to scalar estimates of expected future return and tells the actor how good each situation is.

Without the critic, the actor must estimate the baseline from raw Monte Carlo returns — high variance. Without the actor, the critic has no policy to evaluate. Together, they form a feedback loop: the actor produces experience, the critic evaluates it, the actor updates based on the critic's assessment.

---

## Why Two Separate Networks?

NeuroDrive uses **separate networks** for actor and critic, not a shared trunk with separate heads.

The empirical RL literature (Andrychowicz et al., 2020) found that separate networks performed better on most tested continuous-control tasks. The reason is a fundamental conflict of objectives:

- The actor wants to represent a distribution over actions — it needs to capture the policy geometry.
- The critic wants to represent a smooth value landscape — it needs to generalise across states.

These objectives are not the same. Forcing a shared representation often means compromising both. Separate networks allow each to optimise independently.

**In NeuroDrive:**
- Actor: `Linear(23→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2)` for action means
- Critic: `Linear(23→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1)` for value

They receive the same `observation (23-dim)` as input but have independent weights and independent optimisers.

---

## The Actor Loss

The actor loss is the negative expected advantage-weighted log-probability:

```
L_actor = - Σ_t  A_t * log π_θ(a_t | s_t)
```

Minimising this (gradient descent on the negative) is equivalent to maximising the policy gradient objective.

A positive advantage `A_t > 0` means action `a_t` was better than expected. Minimising `L_actor` increases `log π_θ(a_t | s_t)`, making that action more likely. A negative advantage decreases the log-probability.

**With entropy bonus:**

```
L_actor = - Σ_t  A_t * log π_θ(a_t | s_t) - entropy_coef * H(π)
```

Subtracting entropy (minimising negative entropy) is equivalent to maximising entropy, encouraging exploration.

---

## The Critic Loss

The critic is trained to match its predictions to the return targets:

```
L_critic = Σ_t  Loss(V_φ(s_t), R_t)
```

where `R_t = A_t + V_φ(s_t)` is the return target from GAE.

NeuroDrive uses **Huber loss** (also called smooth L1):

```
Huber(error, δ) = 0.5 * error²         if |error| ≤ δ
                = δ * (|error| - 0.5δ)  if |error| > δ
```

With `VALUE_HUBER_DELTA = 1.0`.

**Why Huber instead of MSE?**
MSE (`error²`) grows quadratically — a single large error can dominate the gradient. Huber is quadratic near zero but linear for large errors, making it more robust to outlier return estimates (e.g. a single unusually large crash penalty).

The Huber gradient is:

```
∂Huber/∂V = V - R_t      if |V - R_t| ≤ 1
           = sign(V - R_t) otherwise
```

In NeuroDrive:

```rust
let value_error = value - ret;
let value_grad = if value_error.abs() <= VALUE_HUBER_DELTA {
    value_error
} else {
    VALUE_HUBER_DELTA * value_error.signum()
};
```

---

## The A3C/A2C Distinction

**A3C** (Asynchronous Advantage Actor-Critic, Mnih et al. 2016) runs multiple actor-learners in parallel and asynchronously: each worker collects experience and applies gradient updates without waiting for the others. The asynchrony provides exploration diversity but introduces gradient staleness.

**A2C** (Synchronous) waits for all workers to complete a rollout, aggregates their gradients, and applies one synchronous update. OpenAI's empirical comparison found no benefit from the asynchrony itself — A2C matched A3C performance with simpler implementation.

**NeuroDrive's current state:** The implementation is algorithmically shaped like A2C but operationally runs with one car (one worker). It is closer to "online actor-critic" than "true synchronous multi-worker A2C." The planned vectorised trainer (`context/plans/vectorised-a2c-visual-trainer.md`) would bring it fully into A2C territory with 25 parallel environments.

---

## Update Cadence

NeuroDrive triggers an update when:
1. The rollout buffer reaches `max_steps = 512` transitions, OR
2. A terminal step occurs AND the buffer has at least `min_update_steps = 128` transitions

This means updates happen more frequently during episodes that end early (crashes), and less frequently during long episodes (lap completions or timeouts).

The actor and critic updates happen in the same call to `a2c_update()`, using the same rollout batch. This is standard for on-policy actor-critic methods — off-policy methods can update on different data.

---

## Separate Learning Rates

```
Actor learning rate:  3e-4
Critic learning rate: 5e-4
```

The critic uses a higher learning rate because the value function provides the signal the actor depends on. An outdated or poorly calibrated critic gives the actor wrong advantage estimates, directly harming policy quality. Making the critic learn somewhat faster relative to the actor is a common practical choice.

---

## What "Explained Variance" Tells You

After each update, NeuroDrive computes **explained variance**:

```
EV = 1 - Var(R - V̂) / Var(R)
```

where `R` are the return targets and `V̂` are the critic's predictions.

- `EV = 1`: the critic perfectly predicts the returns (perfect calibration)
- `EV = 0`: the critic's predictions are no better than the mean
- `EV < 0`: the critic is actively worse than using the mean

Low explained variance means the advantage estimates are noisy because the critic is wrong. Policy updates are then unreliable. Monitoring explained variance is one of the most useful early-warning signals for training health.

---

## The Interaction Loop

Each FixedUpdate tick:

```
[SimSet::Input]
actor_forward(observation) → action_distribution → sample_action → ActionState.desired

[SimSet::Physics → Collision]
physics step → collision detection

[SimSet::Measurement]
episode_loop → reward_t, done_t
observation_rebuild → new_observation
a2c_collect_reward → append(reward_t, done_t) → if horizon: a2c_update()
```

During `a2c_update()`:

```
for each (s_t, a_t, r_t, V_t, done_t) in buffer:
    rerun actor_forward(s_t)    → action_dist_t
    rerun critic_forward(s_t)   → value_t
    compute log_prob(a_t | action_dist_t)
    accumulate policy gradient, critic gradient

normalise advantages
apply gradient clip
optimiser.step(actor)
optimiser.step(critic)
clear buffer
```

---

## Common Misunderstandings

❌ "The actor and critic must share weights to be efficient"
✅ Empirical evidence favours separate networks for continuous-control tasks. Weight sharing can help in very large architectures, but for NeuroDrive's small MLP it is likely to hurt.

❌ "Explained variance close to 1 means the policy is good"
✅ Explained variance measures critic calibration, not policy quality. A critic can be well-calibrated for a bad policy.

❌ "The critic output is the action to take"
✅ The critic outputs a scalar value estimate `V(s)`. The actor (not the critic) produces the action distribution.

❌ "You need to re-run the forward pass during the update"
✅ Yes, you do. The rollout buffer stored the actions and observations, but the log-probability computation requires the current network parameters (which have been updated since the rollout was collected). NeuroDrive re-runs the forward pass for every stored transition.

---

## Related Files

- `concepts/core/policy-gradient-methods.md` — the policy gradient objective
- `concepts/core/advantage-estimation.md` — GAE and the advantage used in L_actor
- `concepts/core/continuous-control.md` — the Gaussian policy and tanh squashing
- `project/systems/a2c-brain.md` — the live NeuroDrive implementation
- `project/comparisons/a2c-vs-ppo.md` — what PPO adds to this architecture
