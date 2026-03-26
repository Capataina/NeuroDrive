# Exercise: Trace the Policy Gradient

## Context

The policy gradient update in NeuroDrive is not the simple REINFORCE formula. It uses a tanh-squashed Gaussian policy with a Jacobian correction in the log-probability. This correction is easy to miss and impossible to test with a unit test — the gradient will be computed, just slightly wrong. This exercise asks you to trace the full log-probability computation by hand to verify you understand it.

## Prerequisites

- `concepts/core/policy-gradient-methods.md` — policy gradient, entropy
- `concepts/core/continuous-control.md` — Gaussian policy, tanh squashing
- `concepts/foundations/probability-and-distributions.md` — log-probability, change of variables
- `exercises/core/implement-gae.md`

## The Task

Given a specific observation, policy parameters, and a sampled action, compute:

1. The policy's mean and std for each action dimension
2. The latent (pre-tanh) action sample
3. The log-probability with the Jacobian correction
4. The policy gradient loss contribution for this single step

### Setup

Assume a simplified policy with `in_features = 2`, `out_features = 2` (one output per action dimension — steering and throttle).

**Policy parameters (actor network output layer, no hidden layers for simplicity):**
```
W = [[0.3, -0.1],
     [0.5,  0.2]]  (shape: 2 × 2, out × in)
b = [0.1, -0.2]
log_std = [0.0, 0.0]  (so std = exp(0) = 1.0 for both)
```

**Observation:**
```
obs = [1.0, 0.5]
```

**Sampled latent action (the pre-tanh Gaussian sample):**
```
latent = [0.8, -0.4]
```

## Part 1: Forward Pass

Compute the policy mean:
```
mean = W * obs + b

mean[0] = W[0,0] * obs[0] + W[0,1] * obs[1] + b[0]
mean[1] = W[1,0] * obs[0] + W[1,1] * obs[1] + b[1]
```

Compute the std:
```
std = exp(log_std) = [exp(0.0), exp(0.0)] = [1.0, 1.0]
```

Compute the applied (squashed) action:
```
squashed[0] = tanh(latent[0]) = tanh(0.8)
squashed[1] = tanh(latent[1]) = tanh(-0.4)

steering  = squashed[0]
throttle  = (squashed[1] + 1.0) / 2.0
```

**Useful values:** `tanh(0.8) ≈ 0.6640`, `tanh(-0.4) ≈ -0.3799`

## Part 2: Log-Probability with Jacobian Correction

The Gaussian log-probability at the latent values:
```
log_N(latent[i] | mean[i], std[i]) = -0.5 * ((latent[i] - mean[i]) / std[i])^2
                                     - log(std[i])
                                     - 0.5 * log(2π)
```

The Jacobian correction for each dimension:
```
correction[i] = log(1 - tanh²(latent[i]))
```

Note: `1 - tanh²(x) = sech²(x)`. Since `sech(0.8) = 1/cosh(0.8)` and `cosh(0.8) ≈ 1.3374`:
```
1 - tanh²(0.8) ≈ 1 - 0.4409 ≈ 0.5591
correction[0] = log(0.5591) ≈ -0.5829
```

For `tanh(-0.4)`:
```
1 - tanh²(-0.4) ≈ 1 - 0.1443 ≈ 0.8557
correction[1] = log(0.8557) ≈ -0.1558
```

**Total log-probability:**
```
log_prob = Σ_i [ log_N(latent[i] | mean[i], std[i]) - correction[i] ]
```

Compute this numerically using your Part 1 result.

## Part 3: Policy Gradient Loss Contribution

Assume the normalised advantage for this step is `A_norm = 1.5`.

The policy gradient loss contribution for this single step:
```
loss_contrib = -log_prob * A_norm
```

This is the term that, when minimised, pushes the policy to assign higher probability to the action when the advantage is positive. Compute the numerical value.

## Part 4: What Happens Without the Jacobian Correction?

Re-compute `log_prob` without the correction term:
```
log_prob_wrong = Σ_i log_N(latent[i] | mean[i], std[i])
```

What is the percentage difference between `log_prob` and `log_prob_wrong`?

Now consider: if `latent` were `[2.5, -2.5]` (near tanh saturation), recompute both. How large is the error at saturation?

## Hints

<details>
<summary>Hint 1 (computing tanh and its derivative)</summary>

`tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})`

The correction term uses: `1 - tanh²(x) = 4 / (e^x + e^{-x})²`

For numerical stability near saturation (`|x| > 2`), the correction is important because `tanh(x)` approaches ±1 and `1 - tanh²(x)` approaches 0. The log of a value near 0 is a large negative number — this is the penalty for driving actions near the boundary.

</details>

<details>
<summary>Hint 2 (the sign of the correction)</summary>

Notice that `correction[i] = log(1 - tanh²(latent[i]))`. Since `1 - tanh²(x) ≤ 1`, the logarithm is ≤ 0. We subtract the correction:

```
log_prob = Σ_i [ log_N(latent[i]) - correction[i] ]
         = Σ_i [ log_N(latent[i]) - (something ≤ 0) ]
         = Σ_i [ log_N(latent[i]) + something ≥ 0 ]
```

Wait — does this mean the corrected log-prob is *higher* (less negative) than the uncorrected one? In what regime? Think about what this means for the gradient direction.

</details>

<details>
<summary>Hint 3 (the expected numerical values)</summary>

For Part 1, the mean should be approximately:
```
mean[0] ≈ 0.3 * 1.0 + (-0.1) * 0.5 + 0.1 = 0.35
mean[1] ≈ 0.5 * 1.0 + 0.2 * 0.5 + (-0.2) = 0.40
```

Compute Part 2 from these means.

</details>

## Reflection Questions

After completing the trace:

1. Why is the Jacobian correction always non-positive (≤ 0)? What does this mean for the effective log-probability compared to the raw Gaussian density?

2. The policy gradient minimises `-log_prob * A_norm`. If `A_norm > 0` (good action), does minimising this increase or decrease `log_prob`? Is that the correct direction?

3. In NeuroDrive, both steering and throttle contribute to `log_prob`. If the policy assigns very high probability to steering but very low probability to throttle, how does the joint `log_prob` behave? Should the two action dimensions be treated as independent?

4. At initialisation, `log_std = 0.0` (std = 1.0). The Gaussian is wide. After many training updates toward a specific driving style, `log_std` should become more negative. What would the policy look like if `log_std` became very large (e.g. 5.0)?

## Related Files

- `concepts/core/continuous-control.md` — the full squashed Gaussian theory
- `concepts/foundations/probability-and-distributions.md` — change of variables derivation
- `project/decisions/tanh-squashed-actions.md` — why this approach was chosen
- `exercises/core/trace-observation-vector.md` — the next exercise
