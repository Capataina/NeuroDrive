# Optimisation and Gradients

## Why This Matters Here

NeuroDrive computes gradients of a policy gradient loss through a manually written backpropagation pass, then applies those gradients using a hand-implemented Adam optimiser. The optimiser code lives in `src/brain/common/optim.rs`. Understanding what gradient descent does, why Adam improves upon it, and why gradient clipping is necessary are prerequisites for reading the A2C update path intelligently.

Beyond the current code: the project's biological learning direction eventually replaces gradient-based optimisation entirely. Understanding the mechanics of what is being replaced — and its limitations — is part of understanding why biologically plausible alternatives are interesting.

## Prerequisites

- `concepts/foundations/neural-networks.md` — forward pass and backpropagation

## Notation

| Symbol | Meaning |
|---|---|
| `θ` | Model parameters (all weights and biases) |
| `L(θ)` | Loss function, scalar |
| `∇_θ L` | Gradient of loss with respect to parameters |
| `η` | Learning rate |
| `t` | Timestep (update count) |
| `m_t` | First moment estimate (momentum) in Adam |
| `v_t` | Second moment estimate (RMS) in Adam |
| `β₁, β₂` | Adam decay rates |
| `ε` | Small constant for numerical stability |

---

## Core Idea: Gradient Descent

A loss function `L(θ)` measures how bad the current parameters are. The gradient `∇_θ L` points in the direction of steepest increase. To reduce the loss, move opposite to the gradient:

```
θ ← θ - η * ∇_θ L
```

This is vanilla gradient descent. With neural networks, `θ` encompasses every weight and bias in the network, and `∇_θ L` is computed by backpropagation.

**Why this works:** If `η` is small enough, the linear approximation of `L` near the current `θ` is accurate, and stepping in the negative gradient direction is guaranteed to reduce the loss locally. This is not the same as finding the global minimum — it is a local descent.

**Why this alone is insufficient:** Vanilla gradient descent has several practical problems:
1. The same `η` for every parameter is suboptimal — some parameters need larger steps, others smaller.
2. It treats all gradients equally regardless of their history.
3. It is sensitive to the scale of the loss and the geometry of the loss surface.

---

## Stochastic Gradient Descent and Mini-Batches

In practice, gradients are computed over batches of data rather than the entire dataset. This introduces noise into each gradient estimate, which is actually beneficial for escaping sharp local minima. In the on-policy RL context, the "batch" is the rollout buffer collected since the last update.

NeuroDrive's A2C accumulates gradients across the entire rollout batch before calling the optimiser once. This is a full-batch update over the collected rollout — not stochastic within a single update, but stochastic across updates (since each rollout captures different experience).

---

## Momentum

Momentum accelerates gradient descent by accumulating a velocity vector in the direction of persistent gradient components:

```
v ← β * v + (1 - β) * ∇_θ L
θ ← θ - η * v
```

Intuition: if the gradient has consistently pointed in a direction for many steps, momentum builds up speed in that direction. If the gradient oscillates (as in sharp ravines in the loss surface), the opposing components cancel out.

Adam incorporates a generalised form of this.

---

## Adam Optimiser

Adam (Adaptive Moment Estimation) is the standard optimiser for most deep learning applications. NeuroDrive uses it for both actor and critic updates.

Adam maintains two running estimates per parameter:
1. **First moment** `m_t` — exponential moving average of gradients (like momentum)
2. **Second moment** `v_t` — exponential moving average of squared gradients (like RMSProp)

### Adam Update Rule

At each update step `t`:

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t         [first moment update]
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         [second moment update]

m̂_t = m_t / (1 - β₁^t)                        [bias correction]
v̂_t = v_t / (1 - β₂^t)                        [bias correction]

θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)         [parameter update]
```

Standard hyperparameters: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.

### Why Adam Works Better than Vanilla SGD

The key insight is the division by `√v̂_t + ε`:
- If a parameter has had large gradients historically (large `v̂_t`), its effective learning rate is small.
- If a parameter has had small, sparse gradients (small `v̂_t`), its effective learning rate is larger.

This **adaptive per-parameter learning rate** lets different weights learn at different effective speeds, which is especially valuable in networks with varying gradient scales.

### Bias Correction

The exponential moving averages are initialised at zero. Without bias correction, the first few estimates are biased toward zero because the average has not "warmed up" yet. The bias correction terms `1 / (1 - β^t)` compensate for this cold-start effect. As `t` grows large, `β^t → 0`, and the bias correction factor approaches 1, having no effect.

### Worked Example

Suppose `η = 0.001`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, and at step `t=1`, a parameter has gradient `g = 0.5`:

```
m₁ = 0.9 * 0 + 0.1 * 0.5 = 0.05
v₁ = 0.999 * 0 + 0.001 * 0.25 = 0.00025

m̂₁ = 0.05 / (1 - 0.9¹) = 0.05 / 0.1 = 0.5
v̂₁ = 0.00025 / (1 - 0.999¹) = 0.00025 / 0.001 = 0.25

θ ← θ - 0.001 * 0.5 / (√0.25 + 1e-8)
     = θ - 0.001 * 0.5 / 0.5000000001
     ≈ θ - 0.001
```

At step 1, the Adam update with this gradient is approximately `Δθ = -0.001`. The bias correction rescaled both estimates to what they would be if we had been running Adam since the beginning with this gradient.

Now suppose at step `t=2`, the gradient drops to `g = 0.01`:

```
m₂ = 0.9 * 0.05 + 0.1 * 0.01 = 0.046
v₂ = 0.999 * 0.00025 + 0.001 * 0.0001 = 0.00025

m̂₂ = 0.046 / (1 - 0.81) = 0.046 / 0.19 ≈ 0.242
v̂₂ = 0.00025 / (1 - 0.998) = 0.00025 / 0.002 = 0.125

θ ← θ - 0.001 * 0.242 / (√0.125 + ε) ≈ θ - 0.000684
```

The second update is smaller — Adam "remembers" the large previous gradient in `v̂₂`, so it is more conservative now.

### In NeuroDrive

The actor and critic have separate Adam optimisers:

```rust
pub a_opt: AdamOptimizer,  // lr = 3e-4
pub c_opt: AdamOptimizer,  // lr = 5e-4
```

The `log_std` parameters (the learnable action standard deviations) have their own manually implemented Adam moment updates directly in `a2c_update.rs`.

---

## Gradient Clipping

Even with Adam, gradients can occasionally become very large — due to extreme advantage values, numerical instability, or rare trajectory events. If very large gradient steps are taken, the parameters can "explode" into a region where the loss is large and recovery is slow.

**Global gradient clipping** rescales all gradients by a common factor if their combined L2 norm exceeds a threshold:

```
if ||g|| > max_norm:
    g ← g * (max_norm / ||g||)
```

This preserves the direction of the gradient but caps its magnitude.

NeuroDrive clips both actor and critic gradients separately at a maximum norm of `0.5`:

```rust
const ACTOR_GRAD_CLIP_NORM: f32 = 0.5;
const CRITIC_GRAD_CLIP_NORM: f32 = 0.5;
```

This is done in `clip_linear_gradients` in `src/brain/a2c/update.rs`, called after gradient accumulation but before the optimiser step.

### Why Clip?

In RL especially, advantages can be large and noisy. Without clipping, a single very bad episode can produce a gradient that catastrophically updates the policy. Clipping provides a form of gradient-level trust region: "never change parameters by more than a bounded amount in a single step."

---

## Learning Rate Choices

NeuroDrive currently uses fixed learning rates:
- Actor: `3e-4`
- Critic: `5e-4`

The critic uses a slightly higher learning rate because value function learning can afford more aggressive updates than the policy — a poor critic makes the policy gradient estimates noisy, so having the critic learn faster provides better signal earlier.

Fixed learning rates are a known limitation. The empirical RL literature suggests that learning rate annealing (gradually decreasing `η` over training) can improve stability and convergence. This is a documented gap in the current implementation.

---

## Why Biology Does Not Use Gradient Descent

This connects directly to NeuroDrive's long-term mission:

**Gradient descent requires:**
1. A global loss function defined over all outputs
2. Backward propagation of gradients through every layer
3. Knowledge of every weight in the network to compute the update for any single weight

The brain has none of these. Neurons update their synapses using only local information:
- The activity of the presynaptic neuron
- The activity of the postsynaptic neuron
- Optionally, a global modulatory signal (dopamine)

This is the core motivation for Milestones 2–4: implement learning rules that are local, not global; synapse-level, not network-level; incremental, not batch.

---

## Alternatives and Comparisons

| Optimiser | Key property | Used in NeuroDrive? |
|---|---|---|
| Vanilla SGD | Simple; sensitive to scale | No |
| SGD + momentum | Faster convergence | No |
| RMSProp | Adaptive per-param LR | No |
| **Adam** | Adaptive LR + momentum | **Yes** |
| LAMB / AdamW | Weight decay variants | No |

---

## Common Misunderstandings

❌ "Adam always converges to the best solution"
✅ Adam finds a local minimum. The loss surface of neural networks is non-convex, so convergence guarantees apply only locally.

❌ "Gradient clipping is a hack to fix bad code"
✅ Gradient clipping is a principled and widely used technique in RL. It is especially important in on-policy methods where individual rollouts can produce very large advantage estimates.

❌ "Higher learning rate always means faster learning"
✅ Higher LR can cause divergence or oscillation if the step size exceeds the radius of local loss-surface curvature.

---

## Related Files

- `concepts/foundations/neural-networks.md` — backpropagation that produces the gradients
- `exercises/foundations/implement-adam-optimizer.md` — implement Adam step by step
- `project/systems/a2c-brain.md` — where the optimiser is used in NeuroDrive
