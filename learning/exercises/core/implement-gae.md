# Exercise: Implement Generalised Advantage Estimation (GAE)

## Context

GAE is the component of A2C that converts a rollout of rewards and value estimates into advantages used for the policy gradient. Getting GAE right is subtle: the backwards recurrence, the terminal masking, and the bootstrap handling all interact. Bugs here produce systematically wrong policy gradients that can silently degrade learning without obvious error messages.

This exercise asks you to implement GAE from scratch and verify it against a numerical example.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — returns, value functions, bootstrapping
- `concepts/core/advantage-estimation.md` — GAE recurrence formula, λ and γ roles
- `exercises/foundations/implement-adam-optimizer.md` (or equivalent background)

## The Task

Implement `compute_gae()` that takes:

- `rewards: Vec<f32>` — per-tick rewards r_0, r_1, ..., r_{N-1}
- `values: Vec<f32>` — critic estimates V(s_0), V(s_1), ..., V(s_{N-1})
- `dones: Vec<bool>` — terminal flags for each step
- `bootstrap_value: f32` — V(s_N) if the rollout ended non-terminally, else 0.0
- `gamma: f32 = 0.99`
- `lambda: f32 = 0.95`

And returns:

- `advantages: Vec<f32>` — normalised advantages for each step
- `returns: Vec<f32>` — targets for the value function (advantages + values)

### Specification

**Backwards recurrence (process from t = N-1 down to t = 0):**

```
V_{N} = bootstrap_value   (0.0 if dones[N-1] is true)

delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t

A_N = 0

A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

**After computing all advantages:**

```
return_t = A_t + V_t

mean_A = mean(A_0, ..., A_{N-1})
std_A  = std(A_0, ..., A_{N-1})
A_norm_t = (A_t - mean_A) / (std_A + 1e-8)
```

Return the normalised advantages and the raw returns.

## Constraints

- Do not look at `src/brain/a2c/buffer.rs` until you have a working implementation.
- The recurrence must run **backwards** — this is not optional.
- Terminal masking `(1 - done_t)` must be applied at the right places.

## Worked Example

Verify your implementation against this exact numerical example.

**Setup:**
```
N = 5 steps, gamma = 0.99, lambda = 0.95

rewards = [0.1,   0.2,  -5.0,   0.1,  0.3]
values  = [2.0,   2.1,   2.2,   0.05, 0.1]
dones   = [false, false, true,  false, false]
bootstrap_value = 0.5
```

**Step 1: Set bootstrap value.**

Step 4 (last step, index 4) is not terminal. So V_5 = bootstrap_value = 0.5.
But step 2 (index 2) IS terminal. This will zero the masking term at step 2.

**Step 2: Compute deltas.**

```
t=4: delta_4 = 0.3 + 0.99 * 0.5 * (1-0) - 0.1   = 0.3 + 0.495 - 0.1   = 0.695
t=3: delta_3 = 0.1 + 0.99 * 0.1 * (1-0) - 0.05  = 0.1 + 0.099 - 0.05  = 0.149
t=2: delta_2 = -5.0 + 0.99 * 0.05 * (1-1) - 2.2 = -5.0 + 0.0 - 2.2    = -7.2
t=1: delta_1 = 0.2 + 0.99 * 2.2 * (1-0) - 2.1   = 0.2 + 2.178 - 2.1   = 0.278
t=0: delta_0 = 0.1 + 0.99 * 2.1 * (1-0) - 2.0   = 0.1 + 2.079 - 2.0   = 0.179
```

**Step 3: Backwards recurrence for advantages.**

```
A_5 = 0  (initialise)

t=4: A_4 = delta_4 + 0.99 * 0.95 * (1-0) * A_5
         = 0.695 + 0.9405 * 0.0
         = 0.695

t=3: A_3 = delta_3 + 0.9405 * (1-0) * A_4
         = 0.149 + 0.9405 * 0.695
         = 0.149 + 0.6537...
         = 0.8027...

t=2: A_2 = delta_2 + 0.9405 * (1-1) * A_3
         = -7.2 + 0.9405 * 0 * A_3
         = -7.2
         (done=true at t=2 zeroes the next-step contribution)

t=1: A_1 = delta_1 + 0.9405 * (1-0) * A_2
         = 0.278 + 0.9405 * (-7.2)
         = 0.278 - 6.7716
         = -6.4936...

t=0: A_0 = delta_0 + 0.9405 * (1-0) * A_1
         = 0.179 + 0.9405 * (-6.4936...)
         = 0.179 - 6.1077...
         = -5.9287...
```

**Step 4: Returns.**

```
return_t = A_t + V_t

return_0 = -5.9287 + 2.0 = -3.9287
return_1 = -6.4936 + 2.1 = -4.3936
return_2 = -7.2    + 2.2 = -5.0
return_3 =  0.8027 + 0.05 = 0.8527
return_4 =  0.695  + 0.1  = 0.795
```

**Step 5: Normalise advantages.**

```
advantages = [-5.9287, -6.4936, -7.2, 0.8027, 0.695]

mean = (-5.9287 + -6.4936 + -7.2 + 0.8027 + 0.695) / 5
     = -18.1246 / 5
     = -3.6249

std = sqrt(mean_of_squares_minus_mean_squared)
    ≈ 3.41  (compute this precisely in your implementation)

normalised = [(A - mean) / std for A in advantages]
```

Verify your implementation produces values close to these (within floating-point precision).

## Hints

<details>
<summary>Hint 1 (where to start the backwards recurrence)</summary>

Initialise a running advantage variable `gae = 0.0`. Then loop from `t = N-1` down to `t = 0`. The "next value" V_{t+1} is either `values[t+1]` for non-final steps or `bootstrap_value` for the final step (when `t = N-1`).

</details>

<details>
<summary>Hint 2 (terminal masking position)</summary>

The masking `(1 - done_t)` appears in two places in the recurrence:
1. In the TD delta: `r_t + gamma * V_{t+1} * (1 - done_t) - V_t`
   — prevents bootstrapping from the next state if this step ended the episode
2. In the advantage recurrence: `delta_t + gamma * lambda * (1 - done_t) * A_{t+1}`
   — prevents advantage propagating across episode boundaries

Both places must be masked. Missing either one produces incorrect advantages at episode boundaries.

</details>

<details>
<summary>Hint 3 (advantage normalisation denominator)</summary>

The standard deviation in the denominator can be computed as:
```
var = mean(A^2) - mean(A)^2
std = sqrt(max(var, 0))  // max to avoid negative due to float error
normalised_A = (A - mean) / (std + 1e-8)
```

The `1e-8` prevents division by zero when all advantages are identical (e.g. in early training or when the rollout contains only zero-reward steps).

</details>

## Reflection Questions

After completing the implementation:

1. Why is the recurrence backwards? What would happen if you computed advantages forwards instead?

2. At step 2 in the worked example (done=true), the advantage A_2 equals just the TD delta (-7.2). Why does the done flag zero out the future advantage contribution? Is this correct behaviour?

3. The bootstrap value is V(s_N) from the critic. If the critic is completely wrong (e.g. always predicts 0.0), how does this affect the advantages? Is A2C still able to learn?

4. Advantage normalisation uses the mean and std of advantages within the *current batch*. Why might this be problematic if the batch contains a mix of terminal episodes and a partial non-terminal rollout at the end?

## Related Files

- `concepts/core/advantage-estimation.md` — full GAE derivation and worked examples
- `concepts/core/reinforcement-learning.md` — returns and bootstrapping
- `exercises/core/trace-the-policy-gradient.md` — the next exercise
- `src/brain/a2c/buffer.rs` — reference implementation (after completing)
