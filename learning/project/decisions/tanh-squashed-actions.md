# Decision: Tanh-Squashed Bounded Actions

## The Decision

NeuroDrive's A2C policy uses a Gaussian distribution over latent (unbounded) action values, which are then passed through a `tanh` function to produce bounded actions. This is not the only way to handle bounded continuous actions. This file explains why it was done this way, what the alternatives are, and what the correctness requirements are for this approach to work properly.

**Status:** Implemented decision. The current A2C uses tanh-squashed Gaussian actions.

## Prerequisites

- `concepts/core/continuous-control.md` — Gaussian policy, squashing theory
- `concepts/foundations/probability-and-distributions.md` — log-probability, change of variables

---

## The Problem: Bounded Actions from an Unbounded Distribution

NeuroDrive's car takes:
- `steering ∈ [-1.0, 1.0]`
- `throttle ∈ [0.0, 1.0]`

A Gaussian distribution `N(μ, σ)` has support on `(-∞, +∞)`. If you sample from this distribution and use the sample directly as an action, you will sometimes get values like `steering = 3.7` or `throttle = -2.1`, which must be clamped to the valid range.

Naïve clamping causes a serious problem for the policy gradient.

---

## Why Naïve Clamping is Incorrect

Suppose the policy samples `steering = 2.5` and clamps it to `1.0`. The actual executed action is `1.0`. But the log-probability used in the policy gradient is computed as:

```
log π(1.0 | obs) = log N(2.5 | μ, σ)
```

This is wrong. The probability that is *relevant to the executed action* is not the probability of sampling `2.5` — it is the probability of sampling any value ≥ 1.0 (since all of them produce the same applied action `1.0`). The Gaussian density at `2.5` has nothing to do with the probability of taking the action `1.0` under a clamped policy.

More concretely: if the policy gradient uses an incorrect log-probability, the gradient estimate is biased. The update will push the policy in the wrong direction, creating a systematic error that accumulates across training.

**The clamp-hit diagnostics in A2cTrainingStats** exist precisely to detect this: `clamp_fraction` reports how often tanh saturates near ±1. High clamp fraction means the policy is frequently operating near the boundary, where the correction term matters most.

---

## The Tanh Squashing Approach

Instead of clamping, squash the latent sample through `tanh`:

```
latent ~ N(μ, σ)          — Gaussian sample, unbounded
squashed = tanh(latent)   — squashed to (-1, 1)
```

`tanh` maps `(-∞, +∞)` smoothly and monotonically to `(-1, 1)`. Crucially, it is differentiable everywhere, including at the boundaries.

The correct log-probability requires a **Jacobian correction** for the change of variables:

```
log π(squashed | obs) = log N(latent | μ, σ)
                        - Σ_i log(1 - tanh²(latent_i))
```

The second term is the log-determinant of the Jacobian of the `tanh` transformation. It corrects for the fact that the density of the squashed distribution is not simply the Gaussian density — the `tanh` mapping compresses probability mass near the boundaries.

### Intuition for the Correction

Near `latent = 0`, `tanh` is approximately linear: `tanh(x) ≈ x`. The Jacobian is approximately 1, so `log(1 - tanh²(0)) ≈ log(1) = 0`. The correction is small.

Near `latent = ±3`, `tanh` is nearly flat: `tanh(3) ≈ 0.995`, `d/dx tanh(3) ≈ 0.01`. The Jacobian is very small, so `log(1 - tanh²(3)) ≈ log(0.01) ≈ -4.6`. The correction is large and negative.

The correction reduces the effective log-probability for actions near the boundary, reflecting the fact that `tanh` compresses a wide range of Gaussian samples into a narrow region near ±1.

---

## Why Store Both Latent and Applied Actions

The rollout buffer stores:
- `latent_actions: Vec<[f32; 2]>` — the pre-tanh Gaussian samples
- `actions: Vec<CarAction>` — the post-tanh, post-rescaling applied actions

The applied actions go to physics (and are recorded by analytics). The latent actions are needed in the update step to recompute the log-probability correctly.

If only the applied (squashed) actions were stored, the update step could not recover the latent value and could not compute the correct Jacobian-corrected log-probability. The tanh function is not uniquely invertible in finite precision near ±1 — `atanh(0.9999...)` is numerically unstable. Storing the latent directly avoids this inversion entirely.

---

## Action Range Mapping

After tanh squashing, the action is in `(-1, 1)`. The raw squashed value is used for steering (which is already in `[-1, 1]`). For throttle (which must be in `[0, 1]`), a linear rescaling is applied:

```
throttle_applied = (squashed[1] + 1.0) / 2.0
```

This maps `(-1, 1)` to `(0, 1)`.

A symmetric mean-zero Gaussian (initial `μ = 0`) will produce a mean throttle of `0.5` at initialisation. This is a reasonable starting point — the car accelerates moderately before learning to adapt throttle to context.

---

## Alternatives Considered

### Clipping the Gaussian Mean

One approach: clip `μ` to the valid range during inference but keep the log-probability computation uncorrected. This is simple but produces the biased gradient problem described above. Rejected.

### Beta Distribution Policy

A Beta distribution has support on `(0, 1)` and can be adapted to `(-1, 1)`. It avoids the squashing correction entirely since samples are already bounded. However:
- The Beta distribution requires two positive parameters (α, β), which need separate parameterisations
- It is not symmetric by default and is harder to initialise sensibly
- It has less standard implementation in handwritten code
- The tanh approach is the industry standard (SAC uses it) and is well-documented

Rejected in favour of the tanh approach as simpler and better-studied.

### Clipped Normal Distribution (Truncated Gaussian)

A truncated Gaussian has a closed-form density within bounds. The log-probability is correct for bounded samples. However:
- The normalisation constant of the truncated Gaussian depends on the bounds and the current distribution parameters — this adds computational complexity
- Sampling from a truncated Gaussian requires rejection sampling or CDF inversion, which is more involved than the simple reparameterisation trick

Rejected as more complex without clear benefit over tanh squashing.

---

## Correctness Requirements

For the tanh approach to produce correct policy gradients:

1. **Latent values must be stored, not just applied values.** ✓ Implemented.

2. **The log-probability in the update must use the Jacobian correction.** ✓ Implemented in `a2c_update()`.

3. **The log-probability computed at act time (for the rollout buffer) must match the log-probability recomputed at update time.** Both must use the same formula. If these diverge, the importance ratio is wrong. ✓ Both use the same correction formula.

4. **Actions near tanh saturation should be monitored.** If many actions saturate, the policy distribution has collapsed to nearly-deterministic behaviour, which eliminates exploration. The `clamp_fraction` diagnostic tracks this. ✓ Implemented.

---

## Related Files

- `concepts/core/continuous-control.md` — full theory of Gaussian policy and squashing
- `concepts/foundations/probability-and-distributions.md` — change of variables, Jacobian
- `project/systems/a2c-brain.md` — the implementation
- `concepts/core/policy-gradient-methods.md` — how log-probability enters the policy gradient
