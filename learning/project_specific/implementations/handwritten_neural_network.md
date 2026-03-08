# NeuroDrive's Handwritten Neural Network Implementation

NeuroDrive implements neural networks entirely from scratch — no ML frameworks, no autograd, no GPU kernels. Every forward pass, backward pass, and parameter update is explicit Rust code operating on `Vec<f32>` buffers. This document explains how each component works and where it lives in the codebase.

> **Prerequisites:** [Neural Network Fundamentals](../../concepts/core/neural_networks/), [Backpropagation](../../concepts/core/neural_networks/backpropagation.md), [Optimisation](../../concepts/core/neural_networks/optimisation.md)

---

## Linear Layer

**Code location:** `src/brain/common/mlp.rs`

A `Linear` layer stores weights as `Vec<Vec<f32>>` in row-major layout — `weights[i][j]` is the connection from input neuron `j` to output neuron `i`. The dimensions are `[out_dim][in_dim]`. Biases are a `Vec<f32>` of length `out_dim`.

**Forward pass:** Computes `y = Wx + b`. For each output neuron `i`, the result is the dot product of `weights[i]` with the input vector, plus `bias[i]`. The input is cached in `input_cache` for use during backpropagation. This cache is essential — without it, the backward pass cannot compute weight gradients.

**Backward pass:** Given `grad_output` (the gradient of the loss with respect to this layer's output), three quantities are computed:

1. **Weight gradients:** `grad_weights[i][j] = grad_output[i] * input_cache[j]` — the outer product of the output gradient and the cached input.
2. **Bias gradients:** `grad_bias[i] = grad_output[i]` — simply the output gradient itself, because the bias contributes additively.
3. **Input gradients (passed downstream):** `grad_input[j] = Σ_i weights[i][j] * grad_output[i]` — the transpose multiplication `W^T · grad_output`, which propagates the error signal to the preceding layer.

The weight and bias gradients are stored for the optimiser; the input gradient is returned so the chain rule can continue backward through the network.

---

## ReLU Activation

**Code location:** `src/brain/common/mlp.rs`

ReLU (Rectified Linear Unit) applies `f(x) = max(0, x)` element-wise.

**Forward pass:** Each element of the input is clamped to a minimum of zero. The raw input (before clamping) is stored in `input_cache`.

**Backward pass:** The gradient is masked — wherever the cached input was less than or equal to zero, the corresponding gradient is set to zero. Where the input was positive, the gradient passes through unchanged. This is the derivative of `max(0, x)`: it is 1 for positive inputs and 0 otherwise.

---

## Tanh Activation

**Code location:** `src/brain/common/mlp.rs`

Tanh squashes values to the range `(-1, 1)`, which is useful for bounding action outputs.

**Forward pass:** Computes `y = tanh(x)` element-wise. Unlike ReLU, tanh caches the *output* (`output_cache`), not the input.

**Backward pass:** Multiplies the incoming gradient by `(1 − y²)`, where `y` is the cached output. This is the standard derivative of tanh, and using the cached output avoids recomputing the forward pass.

---

## ActorCritic Model

**Code location:** `src/brain/a2c/model.rs`

The `ActorCritic` struct contains two independent MLP stacks plus a learnable exploration parameter:

**Actor (policy network):**
- `a_fc1`: Linear(obs_dim → hidden_dim) → ReLU
- `a_fc2`: Linear(hidden_dim → hidden_dim) → ReLU
- `a_mean`: Linear(hidden_dim → action_dim)

The output is the mean of a Gaussian distribution over continuous actions. The actor does *not* output the standard deviation — that is handled separately.

**Critic (value network):**
- `c_fc1`: Linear(obs_dim → hidden_dim) → ReLU
- `c_fc2`: Linear(hidden_dim → hidden_dim) → ReLU
- `c_value`: Linear(hidden_dim → 1)

The output is a scalar estimate of the state value `V(s)`.

**Exploration parameter:** `a_log_std` is a `Vec<f32>` of length `action_dim`, representing the logarithm of the Gaussian standard deviation. It is *not* conditioned on the state — it is a global learnable vector updated alongside the actor's weights. During the forward pass, `std = exp(clamp(log_std, -2.0, 0.5))` provides bounded exploration noise. The clamp prevents the standard deviation from collapsing to near-zero (premature convergence) or growing excessively large (destructive noise).

The actor and critic are updated by separate optimisers, preventing value-function gradients from interfering with policy gradients.

---

## Adam Optimiser

**Code location:** `src/brain/common/optim.rs`

The `AdamOptimizer` maintains per-parameter first-moment (`m`) and second-moment (`v`) vectors for every trainable layer.

**Step procedure:** For each parameter `θ` with gradient `g`:

1. Update biased first moment: `m ← β₁·m + (1−β₁)·g`
2. Update biased second moment: `v ← β₂·v + (1−β₂)·g²`
3. Bias-correct: `m̂ = m / (1−β₁^t)`, `v̂ = v / (1−β₂^t)` where `t` is the step count
4. Update parameter: `θ ← θ − lr · m̂ / (√v̂ + ε)`

Default hyperparameters are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`. The step count `t` is incremented once per `step()` call, not once per parameter — this is important for correct bias correction.

The optimiser operates on the gradient buffers stored inside each `Linear` layer after the backward pass. It iterates over all registered layers and applies the Adam update to both weights and biases.

---

## Glorot (Xavier) Initialisation

**Code location:** `src/brain/common/math.rs`

Weights are initialised by sampling uniformly from `U[-limit, +limit]` where `limit = √(6 / (fan_in + fan_out))`. Here `fan_in` is the number of input neurons and `fan_out` is the number of output neurons for each layer.

This scaling keeps the variance of activations roughly constant across layers at initialisation time, preventing signals from vanishing or exploding before training begins. Biases are initialised to zero.

The random number generator used for initialisation is a simple linear congruential generator (LCG), ensuring reproducibility across runs given the same seed. This determinism extends to the entire training pipeline — the same seed produces the same initial weights, the same action samples, and the same gradient updates.

---

## Design Rationale

Every component described here could be replaced by a single `model = nn.Sequential(...)` call in PyTorch. The reason we do not is that NeuroDrive exists to make these mechanisms visible. When a gradient explodes, we can print the exact `grad_weights` matrix. When Adam oscillates, we can inspect the moment vectors. When initialisation is wrong, we can see the activation magnitudes drift across layers.

This transparency comes at a cost — manual backpropagation is error-prone and must be validated with numerical gradient checks — but it transforms neural networks from opaque API calls into understandable arithmetic.

> **See also:** [Architecture Decisions](../architecture_decisions.md) for why this trade-off was made, [A2C Algorithm](../../concepts/core/reinforcement_learning/a2c_algorithm.md) for how the model is trained.
