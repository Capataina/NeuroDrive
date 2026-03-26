# Probability and Distributions

## Why This Matters Here

NeuroDrive's A2C policy is a **stochastic Gaussian policy**: it does not produce deterministic actions but instead outputs a probability distribution over actions, samples from it, and then adjusts the distribution to make good actions more likely. This requires understanding Gaussian distributions, sampling, log-probability, entropy, and the chain of probability transformations that occur when a continuous action passes through tanh squashing.

Every training step computes the log-probability of the actions that were taken, multiplied by the advantage. The entropy is added as a regularisation bonus. Without understanding what these quantities mean, the update code in `src/brain/a2c/update.rs` is just arithmetic.

## Prerequisites

- Basic understanding of what a probability distribution is
- No prior formal probability theory required

## Notation

| Symbol | Meaning |
|---|---|
| `N(μ, σ²)` | Normal (Gaussian) distribution with mean μ and variance σ² |
| `p(x)` | Probability density at x |
| `log p(x)` | Log-probability (log-likelihood) |
| `H(p)` | Entropy of distribution p |
| `μ` | Mean of a Gaussian |
| `σ` | Standard deviation (σ² is variance) |
| `z` | A sample from the distribution (latent action) |
| `a` | The final action (after squashing) |

---

## The Gaussian (Normal) Distribution

The Gaussian distribution is the most important continuous distribution. A random variable `z ~ N(μ, σ²)` has probability density:

```
p(z | μ, σ) = (1 / (σ * √(2π))) * exp( -(z - μ)² / (2σ²) )
```

Properties:
- **Mean:** `E[z] = μ` — the centre of the distribution
- **Variance:** `Var[z] = σ²` — how spread out the distribution is
- **Standard deviation:** `σ = √(Var[z])`
- 68% of probability mass lies within 1σ of the mean
- 95% lies within 2σ

### Why Use a Gaussian Policy?

A stochastic policy outputs a distribution over actions. For continuous actions like steering `∈ [-1, 1]` and throttle `∈ [0, 1]`, the Gaussian is the natural choice because:

1. It is fully characterised by just two parameters (mean and std) — the network only needs to output two numbers per action dimension.
2. It is differentiable everywhere — log-prob and entropy have clean analytical forms.
3. It naturally models uncertainty — a high `σ` means "I do not know what to do"; a low `σ` means "I am confident in this action".

---

## Log-Probability

In practice, probabilities are almost always computed in log-space:

```
log p(z | μ, σ) = -0.5 * log(2π) - log(σ) - (z - μ)² / (2σ²)
```

**Why log?**
- Products of probabilities (across time steps) become sums of log-probabilities
- Avoids numerical underflow (products of many small numbers shrink to zero floating-point)
- Gradients of log-probability with respect to μ and σ are numerically stable

In the policy gradient theorem, the objective involves `∇ log π(a | s)`. The log makes this gradient tractable:

```
∂ log p / ∂μ  = (z - μ) / σ²
∂ log p / ∂σ  = (z - μ)² / σ³ - 1/σ
```

Or equivalently, the gradient with respect to `log σ` (which is what NeuroDrive uses as the learnable parameter):

```
∂ log p / ∂(log σ) = (z - μ)² / σ² - 1
```

---

## Sampling from a Gaussian

To produce an action during a forward pass, the policy samples `z ~ N(μ, σ)`. This is done using the **reparameterisation trick**:

```
z = μ + σ * ε    where ε ~ N(0, 1)
```

This separates the randomness (`ε`) from the learned parameters (`μ`, `σ`). In NeuroDrive:

```rust
let latent = sample_normal(mean, std, &mut rng);
// which internally computes: mean + std * standard_normal_sample
```

The sampled `z` (called `latent_action` in the code) is stored alongside the action for use during gradient computation.

---

## Learnable Log Standard Deviation

In NeuroDrive, the standard deviation `σ` is not directly output by the actor network. Instead, the network outputs the **mean** `μ`, and the standard deviation is parameterised as a separate **learnable parameter `log σ`** (log-std):

```
σ = exp(log σ)
```

This ensures `σ > 0` regardless of the value of `log σ` (exponential is always positive), while allowing unconstrained optimisation (the gradient of `L` w.r.t. `log σ` has no constraints).

In NeuroDrive: `a_log_std: Vec<f32>` — one value per action dimension, separately updated with their own Adam moments.

The log-std is clamped: `a_log_std[j].clamp(-2.0, 0.5)`, corresponding to standard deviations in roughly `[0.14, 1.65]`. This prevents extreme distributions.

---

## Entropy of a Gaussian

The entropy of a distribution measures its "unpredictability" — higher entropy means more spread, less certainty:

```
H(N(μ, σ²)) = 0.5 * (1 + log(2π) + 2 * log σ)
             = 0.5 * log(2πeσ²)
```

For NeuroDrive's purposes, the important thing is that entropy is a function of `σ` alone — it does not depend on `μ`. So the entropy regularisation term in the policy loss encourages the policy to keep its standard deviation large.

**Why entropy regularisation?** A policy that has collapsed to near-zero variance (near-deterministic) has stopped exploring. In early training especially, a minimum entropy level ensures the policy continues to try different actions and cannot get stuck on a locally good but globally suboptimal strategy.

The entropy coefficient in NeuroDrive is `0.01`:

```
L = -(policy_gradient) + entropy_coef * entropy
  (minimised by maximising the negative policy gradient + entropy bonus)
```

---

## The Tanh Squashing Transformation

The Gaussian policy samples actions `z ~ N(μ, σ)` which are unbounded. But NeuroDrive's action space is bounded:
- Steering: `[-1, 1]`
- Throttle: `[0, 1]`

To enforce these bounds, the sampled `z` is passed through `tanh`:

```
a_raw = tanh(z)   → in [-1, 1]
```

For throttle, a linear mapping shifts the range:

```
a_throttle = 0.5 * (tanh(z) + 1)   → in [0, 1]
```

### The Log-Probability Correction

Squashing changes the probability density. If `z = g⁻¹(a)` and `a = tanh(z)`, then by the change-of-variables formula:

```
log p(a) = log p(z) - log |da/dz|
```

The derivative of `tanh` is:

```
d/dz tanh(z) = 1 - tanh²(z) = 1 - a²
```

So the log-probability correction for the tanh transform is:

```
log p(a) = log p(z) - log(1 - a²)
```

NeuroDrive adds a small epsilon `1e-6` for numerical stability:

```rust
let log_det_jacobian = (1.0 - squashed * squashed + 1e-6).ln();
log_prob = gaussian_log_prob - log_det_jacobian
```

**Why this matters:** Without this correction, the policy gradient estimate is biased. The gradient tells the policy "how much should this action have been preferred?", and the answer depends on the true probability of that action under the squashed distribution. An incorrect log-prob gives a wrong gradient.

For the throttle action, there is an additional affine factor of ×0.5 from the scaling, adding `log(0.5)` = `-log(2)` to the log-probability. NeuroDrive compensates with:

```rust
let affine_log_det = if component_idx == 1 { (2.0f32).ln() } else { 0.0 };
log_prob = gaussian_log_prob - log_det_jacobian + affine_log_det
```

---

## Worked Example: One Complete Action Sample

Suppose the actor outputs `μ = [0.0, 0.0]` and `log_std = [0.0, 0.0]` (so `σ = [1.0, 1.0]`).

**Sampling:**
- Draw `ε₀ = 0.5` from N(0, 1), compute `z₀ = 0.0 + 1.0 * 0.5 = 0.5`
- Draw `ε₁ = -0.3` from N(0, 1), compute `z₁ = 0.0 + 1.0 * (-0.3) = -0.3`

**Squashing:**
- `a_raw₀ = tanh(0.5) ≈ 0.462` → steering
- `a_raw₁ = 0.5 * (tanh(-0.3) + 1) = 0.5 * (1 - 0.291) ≈ 0.355` → throttle

**Log-probability of steering action:**
```
gaussian_log_prob(z₀=0.5 | μ=0, σ=1) = -0.5*log(2π) - 0.5*(0.5)² = -1.043
log_det = log(1 - 0.462² + 1e-6) = log(0.787) ≈ -0.240
log_prob₀ = -1.043 - (-0.240) = -1.043 + 0.240 = -0.803
```

This is the log-probability that the Gaussian policy (after tanh squashing) assigned to the actual steering action `0.462`.

During training, this log-probability is multiplied by the advantage estimate. If the advantage is positive (the action was better than expected), the gradient increases the log-probability — making the policy more likely to take this action in this state in the future.

---

## Common Misunderstandings

❌ "The Gaussian mean is the action taken"
✅ The mean is the *expected* action. The actual action is a *sample* from the distribution. During training, the sampled action (including its latent pre-squash value) is recorded for gradient computation.

❌ "The log-probability correction is optional"
✅ It is not. Omitting it biases the gradient estimate. This is a common source of subtle bugs in actor-critic implementations with bounded action spaces.

❌ "Entropy regularisation prevents good performance"
✅ Entropy regularisation prevents *premature* convergence to a suboptimal policy. The coefficient `0.01` is small enough that it only penalises near-zero entropy — it does not stop the policy from becoming confident once it has found a good strategy.

---

## Related Files

- `concepts/foundations/neural-networks.md` — the network that produces μ
- `concepts/core/continuous-control.md` — the full continuous action story in the RL context
- `project/decisions/tanh-squashed-actions.md` — design rationale for this specific implementation
- `project/systems/a2c-brain.md` — where these concepts appear in the live code
