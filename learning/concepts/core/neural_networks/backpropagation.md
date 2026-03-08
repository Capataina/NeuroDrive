# Backpropagation

## Prerequisites

- **Forward pass and layers**: you must already understand how a linear layer computes y = Wx + b and how ReLU/Tanh apply non-linearities (see `forward_pass_and_layers.md`).
- **Chain rule of calculus**: if y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). You should be able to apply this to compositions of 3–4 functions.
- **Partial derivatives**: given f(x₁, x₂, x₃), you can compute ∂f/∂x₁ while treating x₂ and x₃ as constants.

## Target Depth for This Project

**Level 4** — This is the most important concept in the curriculum. You must be able to hand-compute every gradient in a 2-layer MLP with concrete numbers, explain why each layer caches its input, derive the gradient formulas for weights/biases/inputs of a linear layer, and trace the full backward flow through NeuroDrive's actor and critic networks line by line in `src/brain/a2c/update.rs`.

---

## Core Concept

### The Problem Backpropagation Solves

A neural network is a chain of parameterised functions. Training means adjusting the parameters (weights, biases) so that the network's output gets closer to a desired output. To adjust parameters, we need to know *how each parameter affects the loss* — that is, we need the gradient of the loss with respect to every parameter in the network.

For a single linear layer with 64 inputs and 64 outputs, there are 64×64 + 64 = 4,160 parameters. NeuroDrive's actor alone has three linear layers totalling over 5,000 parameters. Computing each gradient independently would be intractable. Backpropagation solves this by exploiting the **chain rule** to compute all gradients in a single backward sweep through the network, reusing intermediate results at every step.

### The Chain Rule on Computational Graphs

Think of the forward pass as a directed acyclic graph. Each node is an operation (matrix multiply, add bias, apply ReLU). Data flows forward through the graph. Backpropagation reverses this flow: starting from the loss at the output, it propagates the gradient backward through each node, applying the chain rule at every step.

For a composition L = f₃(f₂(f₁(x))):

\[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}
\]

Each layer receives the gradient of the loss with respect to its *output* (called `grad_output` in NeuroDrive) and must compute two things:

1. **Parameter gradients**: ∂L/∂W and ∂L/∂b — used to update this layer's weights and biases.
2. **Input gradient**: ∂L/∂x — passed to the previous layer so it can continue the chain.

### Gradient Formulas for a Linear Layer

Given the forward computation y = Wx + b, with `grad_output` = ∂L/∂y:

**Bias gradient** — How does each bias affect the loss?

\[
\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i} = \text{grad\_output}[i]
\]

The bias enters the computation as a simple addition, so its gradient is exactly the upstream gradient. No multiplication needed.

**Weight gradient** — How does each weight affect the loss?

\[
\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} \cdot x_j = \text{grad\_output}[i] \cdot \text{input}[j]
\]

This is why the layer must **cache the input** during the forward pass. Without the cached input values, we cannot compute the weight gradients.

**Input gradient** — How does each input affect the loss? (Passed to the previous layer.)

\[
\frac{\partial L}{\partial x_j} = \sum_{i=0}^{\text{out\_dim}-1} W_{ij} \cdot \frac{\partial L}{\partial y_i} = \sum_{i} W_{ij} \cdot \text{grad\_output}[i]
\]

This is a matrix-transpose multiplication: grad_input = Wᵀ · grad_output.

### Gradient Formulas for ReLU

ReLU(x) = max(0, x). Its derivative is:

\[
\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
\]

During backward, ReLU acts as a **gradient mask**: it passes through the gradient where the cached input was positive, and blocks it (sets to zero) where the input was non-positive. This is why ReLU caches its *input* — it needs to know which elements were positive.

### Gradient Formulas for Tanh

Tanh caches its *output* y, and the gradient is:

\[
\frac{\partial L}{\partial x_i} = \text{grad\_output}[i] \cdot (1 - y_i^2)
\]

Near y = 0, the gradient is close to 1 (signal flows freely). Near y = ±1 (saturation), the gradient approaches 0 (signal is killed). This is why tanh suffers from vanishing gradients for extreme inputs.

---

## Mathematical Foundation — Complete Numerical Example

We use the same 2-layer MLP from the forward pass document and compute the full backward pass.

**Architecture**: Input (3) → Linear₁ (3→2) → ReLU → Linear₂ (2→1) → output

**Cached values from the forward pass**:
- Input: x = [0.5, −0.3, 0.8]
- Layer 1 output (pre-ReLU): z₁ = [0.43, −0.15]
- ReLU output: h = [0.43, 0.0]
- Final output: ŷ = 0.308

**Suppose our target is 1.0 and we use MSE loss**:

\[
L = \frac{1}{2}(\hat{y} - \text{target})^2 = \frac{1}{2}(0.308 - 1.0)^2 = \frac{1}{2}(−0.692)^2 = 0.2394
\]

**Gradient of loss with respect to output**:

\[
\frac{\partial L}{\partial \hat{y}} = \hat{y} - \text{target} = 0.308 - 1.0 = -0.692
\]

### Step 1: Backward Through Linear₂ (2→1)

Recall: W₂ = [[0.6, −0.4]], b₂ = [0.05], cached input = h = [0.43, 0.0].

`grad_output` = [−0.692]

**Bias gradient**:

\[
\frac{\partial L}{\partial b_2} = -0.692
\]

**Weight gradients**:

\[
\frac{\partial L}{\partial W_2[0][0]} = (-0.692)(0.43) = -0.29756
\]
\[
\frac{\partial L}{\partial W_2[0][1]} = (-0.692)(0.0) = 0.0
\]

The second weight gradient is zero because the cached input (ReLU output) at that position was zero. A dead ReLU stops gradient flow for its connected weights.

**Input gradient** (to pass to ReLU):

\[
\frac{\partial L}{\partial h_0} = W_2[0][0] \cdot (-0.692) = (0.6)(-0.692) = -0.4152
\]
\[
\frac{\partial L}{\partial h_1} = W_2[0][1] \cdot (-0.692) = (-0.4)(-0.692) = 0.2768
\]

grad_input for ReLU = [−0.4152, 0.2768]

### Step 2: Backward Through ReLU

Cached input to ReLU was z₁ = [0.43, −0.15].

- z₁[0] = 0.43 > 0 → gradient passes through: −0.4152
- z₁[1] = −0.15 ≤ 0 → gradient is blocked: 0.0

grad_input for Linear₁ = [−0.4152, 0.0]

Notice: the second gradient component (0.2768) that Linear₂ carefully computed is now entirely discarded by the ReLU mask. The dead neuron blocks all gradient flow, confirming that no parameter updates will improve the path through that neuron for this sample.

### Step 3: Backward Through Linear₁ (3→2)

Recall: W₁ = [[0.2, −0.5, 0.1], [0.4, 0.3, −0.2]], b₁ = [0.1, −0.1], cached input = x = [0.5, −0.3, 0.8].

`grad_output` = [−0.4152, 0.0]

**Bias gradients**:

\[
\frac{\partial L}{\partial b_1[0]} = -0.4152, \quad \frac{\partial L}{\partial b_1[1]} = 0.0
\]

**Weight gradients**:

Row 0 (grad_output[0] = −0.4152):

\[
\frac{\partial L}{\partial W_1[0][0]} = (-0.4152)(0.5) = -0.2076
\]
\[
\frac{\partial L}{\partial W_1[0][1]} = (-0.4152)(-0.3) = 0.12456
\]
\[
\frac{\partial L}{\partial W_1[0][2]} = (-0.4152)(0.8) = -0.33216
\]

Row 1 (grad_output[1] = 0.0): all weight gradients are 0.0.

**Input gradient** (would be passed to the previous layer if one existed):

\[
\frac{\partial L}{\partial x_0} = W_1[0][0] \cdot (-0.4152) + W_1[1][0] \cdot 0.0 = (0.2)(-0.4152) = -0.08304
\]
\[
\frac{\partial L}{\partial x_1} = W_1[0][1] \cdot (-0.4152) + W_1[1][1] \cdot 0.0 = (-0.5)(-0.4152) = 0.2076
\]
\[
\frac{\partial L}{\partial x_2} = W_1[0][2] \cdot (-0.4152) + W_1[1][2] \cdot 0.0 = (0.1)(-0.4152) = -0.04152
\]

This input gradient is not used further (there is no previous layer), but in a deeper network it would continue the chain.

### Summary of All Gradients

| Parameter | Gradient |
|-----------|----------|
| b₂[0] | −0.692 |
| W₂[0][0] | −0.298 |
| W₂[0][1] | 0.0 |
| b₁[0] | −0.415 |
| b₁[1] | 0.0 |
| W₁[0][0] | −0.208 |
| W₁[0][1] | +0.125 |
| W₁[0][2] | −0.332 |
| W₁[1][0..2] | all 0.0 |

Observe the pattern: every gradient in the second neuron's path is zero. The ReLU killed z₁[1], which zeroed the gradient for that neuron's entire weight row and bias. This is the dead-ReLU effect made visible through concrete numbers.

---

## Gradient Accumulation

In NeuroDrive's A2C update, the forward-backward cycle runs **once per sample** in the rollout buffer (batch_size ≈ 2,048 steps). Each `backward()` call **adds** (+=) to `grad_weights` and `grad_biases` rather than overwriting them. The gradients accumulate across all samples. After the loop finishes, the optimiser steps once using the sum of all gradients. Then `zero_grad()` resets the accumulators back to zero before the next update cycle.

This is equivalent to computing the average gradient over the batch (NeuroDrive divides grad_output by batch_size before calling backward, so the accumulated sum is already the mean). The accumulation pattern is visible in `Linear::backward()`:

```rust
self.grad_biases[i] += grad_output[i];          // += not =
self.grad_weights[i][j] += grad_output[i] * input[j];  // += not =
```

The `+=` operator is the single most important character in the entire backward pass. It enables batch gradient computation without storing all intermediate gradients simultaneously.

---

## How NeuroDrive Uses This

### The Full Backward Flow

In `src/brain/a2c/update.rs`, for each sample in the rollout buffer:

**Critic backward** (value loss: ½(V − return)²):

```
d_value = [(value - return) / batch_size]     ← gradient of MSE loss, pre-averaged
    → c_value.backward(d_value)               ← Linear(64→1) backward
    → c_relu2.backward(...)                   ← ReLU mask
    → c_fc2.backward(...)                     ← Linear(64→64) backward
    → c_relu1.backward(...)                   ← ReLU mask
    → c_fc1.backward(...)                     ← Linear(14→64) backward
```

**Actor backward** (policy gradient loss: −advantage · log_prob):

```
d_mean = [−advantage · d_logprob_d_mean / batch_size]    ← per-action-dim
    → a_mean.backward(d_mean)                             ← Linear(64→2) backward
    → a_relu2.backward(...)                               ← ReLU mask
    → a_fc2.backward(...)                                 ← Linear(64→64) backward
    → a_relu1.backward(...)                               ← ReLU mask
    → a_fc1.backward(...)                                 ← Linear(14→64) backward
```

The actor also computes gradients for `a_log_std` — the learnable log-standard-deviation of the Gaussian policy. These gradients are accumulated into `a_log_std_grad` and updated with a standalone Adam step after the main loop.

### Cache-Based Autograd

NeuroDrive implements what could be called a *cache-based manual autograd*. There is no tape, no computation graph object, no automatic differentiation library. Instead, each layer is responsible for:

1. **forward()**: compute the output and cache what backward() will need.
   - `Linear` caches the **input** (needed for ∂L/∂W = grad_output · inputᵀ).
   - `Relu` caches the **input** (needed to know which elements to mask).
   - `Tanh` caches the **output** (needed because the gradient is 1 − y²).

2. **backward(grad_output)**: compute and accumulate parameter gradients (∂L/∂W, ∂L/∂b), then return grad_input (∂L/∂x) to the caller.

3. **zero_grad()**: reset all gradient accumulators to zero.

The caller (the update function) is responsible for calling these in reverse order and threading the grad_input from one layer's backward into the next layer's backward. This is manual but transparent — every gradient computation is explicit and inspectable.

### The `zero_grad()` Ceremony

Before the batch loop begins, `brain.model.zero_grad()` is called. This resets `grad_weights` and `grad_biases` in all six linear layers and `a_log_std_grad` to zero. Without this, gradients from the previous update would leak into the current one, producing incorrect parameter updates. The pattern is: **zero → accumulate across batch → step → zero**.

### L2 Norms for Health Monitoring

After the update, NeuroDrive computes the L2 norm of both weights and gradients for every layer:

\[
\|W\|_2 = \sqrt{\sum_{i,j} W_{ij}^2 + \sum_i b_i^2}
\]

\[
\|\nabla W\|_2 = \sqrt{\sum_{i,j} (\nabla W_{ij})^2 + \sum_i (\nabla b_i)^2}
\]

These norms serve as health indicators:
- **Weight L2 norm growing unboundedly**: weights are exploding — learning rate may be too high.
- **Gradient L2 norm near zero**: vanishing gradients — the network is not learning.
- **Gradient L2 norm spiking**: unstable updates — may need gradient clipping.

NeuroDrive reports these per-layer in `A2cLayerHealth` and exports them through the analytics pipeline.

---

## Common Misconceptions

1. **"Backpropagation is a learning algorithm."** Backpropagation is *only* a gradient computation algorithm. It tells you the direction and magnitude of the gradient. The actual parameter update is performed by an optimiser (SGD, Adam, etc.) that consumes those gradients. Conflating the two leads to confusion about what can be changed and what is fixed.

2. **"The backward pass needs the forward pass's output."** For a linear layer, the backward pass needs the *input*, not the output. For Tanh, it needs the *output*. For ReLU, it needs the *input*. Each layer caches exactly what its specific gradient formula requires — there is no universal rule. Getting this wrong is a common source of bugs in hand-written backpropagation.

3. **"Gradients flow equally through all paths."** ReLU can completely block gradient flow (dead neurons). Tanh saturation can nearly block it. Even in a linear layer, weights with cached inputs near zero receive near-zero gradients. The actual gradient flow is highly non-uniform and depends entirely on the specific input values seen during the forward pass.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Backpropagation** | Algorithm for efficiently computing ∂L/∂θ for all parameters θ using the chain rule in reverse order through the computational graph. |
| **Chain rule** | d/dx f(g(x)) = f'(g(x)) · g'(x). The fundamental identity enabling gradient computation through composed functions. |
| **grad_output** | ∂L/∂y — the gradient of the loss with respect to a layer's output, received from the layer above. |
| **grad_input** | ∂L/∂x — the gradient of the loss with respect to a layer's input, passed to the layer below. |
| **Gradient accumulation** | Summing (+=) gradients across multiple samples before performing a single parameter update. |
| **zero_grad()** | Resetting all gradient accumulators to zero before starting a new accumulation cycle. |
| **Gradient masking** | ReLU's backward behaviour: passing gradients where input > 0, blocking where input ≤ 0. |
| **Vanishing gradients** | Gradients shrinking toward zero as they propagate through many layers, preventing learning in early layers. |
| **L2 norm** | √(Σ xᵢ²) — a scalar summary of a vector's magnitude, used to monitor weight and gradient health. |
| **Autograd** | Automatic differentiation — computing gradients automatically. NeuroDrive uses a manual cache-based variant rather than a framework-provided tape. |

---

## Recommended Materials

1. **Andrej Karpathy — "Yes you should understand backprop"** (blog post). Explains why hand-deriving gradients matters, with examples of subtle bugs that only manual understanding can catch.
2. **3Blue1Brown — "Backpropagation calculus"** (YouTube, Chapter 4 of the *Neural Networks* series). Builds visual intuition for how gradients flow backward through a network, one layer at a time.
3. **CS231n (Stanford) — Backpropagation lecture notes**. The definitive reference for deriving linear layer, ReLU, and loss gradients with computational graph diagrams. Works through the same formulas used in this file.
4. **NeuroDrive source — `src/brain/common/mlp.rs` and `src/brain/a2c/update.rs`**. Read `Linear::backward()` alongside the formulas above. Then read the update loop to see how the backward calls chain together for actor and critic. Every line corresponds to a specific gradient formula from this document.
