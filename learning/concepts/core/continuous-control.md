# Continuous Control

## Why This Matters Here

NeuroDrive requires continuous control: the car's steering must be a real number in `[-1, 1]` and throttle in `[0, 1]`. Most introductory RL material covers discrete action spaces (left/right/forward). Continuous control requires different tools: Gaussian action distributions, tanh squashing, and a log-probability correction that is easy to implement incorrectly.

Every detail in how NeuroDrive's policy produces, clips, and evaluates continuous actions follows from the concepts in this file.

## Prerequisites

- `concepts/foundations/probability-and-distributions.md` — Gaussian distributions, sampling, log-probability
- `concepts/core/policy-gradient-methods.md` — policy gradient theorem

## Notation

| Symbol | Meaning |
|---|---|
| `μ` | Action mean output by the actor network |
| `σ` | Action standard deviation (σ = exp(log_std)) |
| `z` | Latent (pre-squash) action sample |
| `a` | Final action (after squashing) |
| `log π(a\|s)` | Log-probability of action a under the Gaussian policy |

---

## Why Not Use Discrete Actions?

Discretising the action space loses precision and introduces discontinuities:

- Discretising steering to 11 positions ([-1, -0.8, ..., 0.8, 1]) means the policy cannot produce intermediate values like 0.37.
- The correct action in a mild curve is probably 0.15, not 0 or 0.2.
- With very fine discretisation, the action space explodes in dimensionality (especially for 2D joint actions).

For fine motor control like driving, a Gaussian policy that can produce any real number in its range, at any precision, is strictly superior.

---

## The Gaussian Policy for Continuous Actions

The actor outputs two quantities per action dimension:
1. **Mean** `μ` — the most likely action
2. **Standard deviation** `σ` — the spread of the distribution

An action is sampled as:

```
z ~ N(μ, σ)     →     z = μ + σ * ε,  ε ~ N(0, 1)
```

The sampled `z` is the **latent action** — it is unbounded (can be any real number).

---

## Bounding Actions with tanh

Unbounded samples `z` cannot be used directly as steering (`∈ [-1, 1]`) or throttle (`∈ [0, 1]`). We need to squash them into the correct range.

**tanh squashing:**

```
a_raw = tanh(z)    →    a_raw ∈ (-1, 1)
```

tanh is smooth, monotonic, and differentiable — all required properties for the gradient to pass through.

**Steering** is already in `[-1, 1]`:

```
a_steering = tanh(z_0)
```

**Throttle** requires an affine shift to `[0, 1]`:

```
a_throttle = 0.5 * (tanh(z_1) + 1)
```

In NeuroDrive:

```rust
let squashed = latent.tanh();
actions[i] = if i == 0 {
    squashed              // steering: [-1, 1]
} else {
    0.5 * (squashed + 1.0)  // throttle: [0, 1]
};
```

---

## The Log-Probability Correction

This is the most technically critical part of continuous control with squashing. If you compute the log-probability incorrectly, your policy gradient is biased.

The Gaussian density is:

```
p_z(z) = N(z | μ, σ)
```

After squashing `a = tanh(z)`, the density of `a` is **not** `p_z(tanh⁻¹(a))`. The change of variables formula applies:

```
p_a(a) = p_z(z) / |da/dz|
```

Taking logs:

```
log p_a(a) = log p_z(z) - log|da/dz|
```

The derivative of tanh:

```
da/dz = d/dz tanh(z) = 1 - tanh²(z) = 1 - a²
```

So:

```
log p_a(a) = log N(z | μ, σ) - log(1 - a² + ε)
```

The small epsilon `ε = 1e-6` prevents `log(0)` when `a ≈ ±1`.

For the throttle affine transform (`a = 0.5 * (tanh(z) + 1)`), there is an additional Jacobian factor:

```
da/dz = 0.5 * (1 - tanh²(z)) = 0.5 * (1 - a_raw²)
```

This adds `log(0.5) = -log(2)` to the log-probability. In NeuroDrive:

```rust
let affine_log_det = if component_idx == 1 { (2.0f32).ln() } else { 0.0 };
log_prob = gaussian_log_prob - log_det_jacobian + affine_log_det
```

Note: `+affine_log_det` with a positive `ln(2)` compensates for the `0.5` factor (dividing by `0.5` is multiplying by 2).

---

## Why the Correction Matters

Without the Jacobian correction, the policy gradient is computed as if actions were distributed under a Gaussian over the unbounded space — but they are actually distributed under a squashed Gaussian over the bounded space. The gradient points in the wrong direction.

Concretely: if `a = 0.99` (near the boundary), the squashed Gaussian assigns very low density there (because the Jacobian is small: `1 - 0.99² ≈ 0.02`). If you used the raw Gaussian log-prob (ignoring the small Jacobian), you would *overestimate* the probability of extreme actions, which would *underestimate* how much to push them (because the policy already thinks it assigns good probability there). This produces a systematic gradient error.

---

## Safety Clamping

Even after tanh squashing, floating-point edge cases can occasionally produce values slightly outside `[-1, 1]` or `[0, 1]`. NeuroDrive applies a hard clamp:

```rust
let applied_action = raw_action.clamped();
let safety_clamp_hits = [
    (applied_action.steering - raw_action.steering).abs() > 1e-6,
    (applied_action.throttle - raw_action.throttle).abs() > 1e-6,
];
```

The `safety_clamp_hits` are tracked and reported in `A2cTrainingStats.clamped_action_fraction`. If this fraction is high, the policy may be operating near the action boundaries too often — which can indicate the policy is "pushing against the wall" rather than using the full action range.

---

## Storing Both Latent and Applied Actions

The rollout buffer stores both:
- `latent_actions[i]` — the pre-squash sample `z` from `N(μ, σ)`
- `actions[i]` — the post-squash, post-clamp action `a`

During the update, the log-probability computation uses the latent action `z`:

```
log p_a(a) = log N(z | μ, σ) - log(1 - tanh(z)²)
```

We need `z` (not `a`) to evaluate the Gaussian density. If we only stored `a`, we would need to invert tanh to recover `z`, which is numerically less stable.

---

## Action Spread as a Health Signal

`A2cTrainingStats` records:
- `steering_mean`, `steering_std` — mean and std of steering actions in the batch
- `throttle_mean`, `throttle_std` — mean and std of throttle actions in the batch

A healthy policy should show:
- Non-zero steering spread (the agent is turning, not driving straight)
- Non-zero throttle spread (the agent is modulating speed)
- Neither mean stuck at an extreme value

If `steering_std ≈ 0`, the policy has collapsed to near-deterministic steering — usually a sign of premature convergence or training instability.

---

## Alternatives and Comparisons

| Approach | Pros | Cons | Used in NeuroDrive? |
|---|---|---|---|
| Discrete grid approximation | Simple | Loses precision; large action space | No |
| Gaussian + tanh squashing | Smooth; bounded; differentiable | Requires log-prob correction | **Yes** |
| Beta distribution | Naturally bounded [0,1] per dimension | Less common; harder to optimise | No |
| Deterministic policy (DDPG-style) | No sampling noise | Requires replay buffer, off-policy critic | No |
| Squashed normal (SAC-style) | Same as NeuroDrive's approach | Off-policy SAC adds replay buffer | NeuroDrive uses same squashing, on-policy |

---

## Common Misunderstandings

❌ "The mean output by the actor is the action taken"
✅ The mean is the centre of the distribution. The action taken is a sample from the distribution (which NeuroDrive stores as `latent_action`). The log-probability is computed from the sample, not the mean.

❌ "The tanh squashing correction can be skipped for simplicity"
✅ The correction is not optional. Omitting it introduces a systematic gradient bias, especially near the action boundaries where the tanh Jacobian is small.

❌ "A large clamped_action_fraction is always a bug"
✅ It usually indicates the policy is operating near action limits. Whether that is a problem depends on the driving situation. Very high fractions (> 10-20%) during normal driving may indicate a policy that is too aggressive at the boundaries.

---

## Related Files

- `concepts/foundations/probability-and-distributions.md` — Gaussian distributions and log-probability
- `concepts/core/actor-critic-architecture.md` — the full actor-critic structure
- `project/decisions/tanh-squashed-actions.md` — design rationale
- `project/systems/a2c-brain.md` — the live implementation
