# Neural Networks

## Why This Matters Here

NeuroDrive's A2C brain is implemented as a handwritten multi-layer perceptron (MLP) in Rust with no external ML frameworks. Every forward pass, every gradient, and every parameter update is computed manually. Understanding neural networks at this level — not just as a black box that "learns" but as a specific computational graph with specific arithmetic — is required to reason about what the code in `src/brain/common/` and `src/brain/a2c/` is actually doing.

Beyond the current implementation, the project's long-term direction involves replacing gradient-based learning with biologically plausible local rules. To understand what is being replaced and why, you first need to understand what gradient descent and backpropagation actually do.

## Prerequisites

- Basic algebra (functions, derivatives, chain rule)
- No prior neural network knowledge required

## Notation

| Symbol | Meaning |
|---|---|
| `x` | Input vector to a layer |
| `W` | Weight matrix of a layer, shape `[out, in]` |
| `b` | Bias vector, shape `[out]` |
| `z` | Pre-activation output: `z = Wx + b` |
| `a` | Post-activation output: `a = f(z)` |
| `f` | Activation function (e.g. ReLU) |
| `L` | Scalar loss value |
| `∂L/∂w` | Partial derivative of loss with respect to weight `w` |
| `η` | Learning rate |

---

## Core Idea

A neural network is a function approximator built from stacked linear transformations punctuated by non-linear activations.

In plain terms: take a vector of numbers (an observation), repeatedly multiply it by matrices and apply a simple non-linear function, and eventually produce an output vector. The matrices (the weights) are the learnable parameters. The goal is to adjust them so that the output is useful.

Without non-linear activations, the composition of multiple linear layers would simply be another linear layer. The activation functions (ReLU, tanh, sigmoid) are what give networks the capacity to represent complex non-linear mappings.

---

## The Linear Layer

The fundamental building block is the linear layer:

```
z = Wx + b
```

Where:
- `x` is the input vector of dimension `n`
- `W` is the weight matrix of shape `[m, n]` (m output neurons, n inputs each)
- `b` is the bias vector of dimension `m`
- `z` is the pre-activation output of dimension `m`

Each output neuron `i` computes:

```
z_i = Σ_j  W[i][j] * x[j]  +  b[i]
```

This is a dot product of the i-th row of `W` with `x`, plus the bias.

### In NeuroDrive

The `Linear` struct in `src/brain/common/mlp.rs` implements exactly this. Its `forward` method:

```rust
pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
    self.input_cache = Some(input.to_vec());  // cache for backward pass
    let mut output = vec![0.0; self.biases.len()];
    for i in 0..self.biases.len() {
        let mut sum = self.biases[i];
        for j in 0..input.len() {
            sum += self.weights[i][j] * input[j];
        }
        output[i] = sum;
    }
    output
}
```

Note the `input_cache`: the input is saved during the forward pass because the backward pass (gradient computation) needs it.

---

## The ReLU Activation

ReLU (Rectified Linear Unit) is the activation function used in NeuroDrive's A2C:

```
ReLU(z) = max(0, z)
```

For a vector input, it applies element-wise:

```
a_i = max(0, z_i)
```

ReLU is the most common activation in modern deep learning for several practical reasons:
- Computationally cheap (just a comparison and clamp)
- Does not saturate for positive inputs (no vanishing gradient on that side)
- Creates sparse activations (neurons can be "dead" or "alive")

**The dead ReLU problem:** A neuron is "dead" if its pre-activation `z_i` is always ≤ 0. Then `ReLU(z_i) = 0`, and its gradient is also 0, so it never recovers. This is tracked in NeuroDrive's `A2cLayerHealth.dead_relu_fraction`.

### Derivative

The derivative of ReLU is:

```
d/dz  ReLU(z) = 1  if z > 0
                0  if z ≤ 0
```

This is used in backpropagation.

### In NeuroDrive

```rust
pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
    self.input_cache = Some(input.to_vec());
    input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
}

pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
    let input = self.input_cache.as_ref().unwrap();
    input.iter().zip(grad_output.iter())
        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
        .collect()
}
```

The backward pass multiplies the incoming gradient by the derivative of ReLU at each position.

---

## A Complete Feedforward Network

NeuroDrive's actor uses this structure:

```
observation (23-dim)
    ↓
Linear(23 → 64)  [a_fc1]
    ↓
ReLU             [a_relu1]
    ↓
Linear(64 → 64)  [a_fc2]
    ↓
ReLU             [a_relu2]
    ↓
Linear(64 → 2)   [a_mean]  → action means (steering, throttle)
```

The critic uses an identical structure:

```
observation (23-dim)
    ↓
Linear(23 → 64)  [c_fc1]
    ↓ ReLU
Linear(64 → 64)  [c_fc2]
    ↓ ReLU
Linear(64 → 1)   [c_value]  → scalar value estimate
```

These are separate networks — the actor and critic do not share weights. This is a deliberate design choice (see `project/decisions/a2c-as-baseline.md` and `concepts/core/actor-critic-architecture.md`).

---

## Backpropagation

Backpropagation is the algorithm for computing gradients of the loss with respect to all parameters. It applies the **chain rule** of calculus backwards through the computational graph.

### The Chain Rule

If `L` is the loss, `z = f(x)` is any intermediate computation, and we know `∂L/∂z`, then:

```
∂L/∂x = (∂L/∂z) * (∂z/∂x)
```

This telescopes through every layer in the network.

### Backward Through a Linear Layer

Given a linear layer `z = Wx + b` and upstream gradient `∂L/∂z`, we compute:

```
∂L/∂W[i][j]  =  (∂L/∂z_i) * x[j]    (outer product)
∂L/∂b[i]     =  ∂L/∂z_i
∂L/∂x[j]     =  Σ_i  W[i][j] * (∂L/∂z_i)  (W transposed times grad)
```

In plain terms:
- The gradient of the weight connecting input `j` to output `i` is the upstream gradient for output `i` multiplied by the input `j` that passed through that weight.
- The gradient of the bias for output `i` is just the upstream gradient for that output.
- The gradient flowing back to the input `j` is the sum of all upstream gradients weighted by the corresponding weights.

### In NeuroDrive

```rust
pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
    let input = self.input_cache.as_ref().expect("Must call forward first");
    let out_dim = self.biases.len();
    let in_dim = input.len();
    let mut grad_input = vec![0.0; in_dim];

    for i in 0..out_dim {
        self.grad_biases[i] += grad_output[i];
        for j in 0..in_dim {
            self.grad_weights[i][j] += grad_output[i] * input[j];  // outer product
            grad_input[j] += self.weights[i][j] * grad_output[i];  // W^T * grad
        }
    }
    grad_input
}
```

Note `+=` for gradient accumulation. The A2C update accumulates gradients across the whole batch before calling the optimizer step.

### Backward Through ReLU

ReLU's gradient is easy: pass the upstream gradient through if the input was positive, zero it if the input was non-positive.

```
∂L/∂z_i = (∂L/∂a_i) * (1 if z_i > 0 else 0)
```

This is the "mask" that NeuroDrive's `Relu::backward` applies.

---

## Worked Example: One Forward Pass

Suppose `x = [1.0, 2.0]`, `W = [[0.5, -0.3], [0.1, 0.8]]`, `b = [0.0, 0.0]`.

**Forward through Linear:**

```
z[0] = 0.5 * 1.0 + (-0.3) * 2.0 + 0.0 = 0.5 - 0.6 = -0.1
z[1] = 0.1 * 1.0 +  0.8  * 2.0 + 0.0 = 0.1 + 1.6  =  1.7
```

**Forward through ReLU:**

```
a[0] = max(0, -0.1) = 0.0   (this neuron is "dead" for this input)
a[1] = max(0,  1.7) = 1.7
```

**Backward (suppose upstream gradient is `[0.0, -0.5]` at the ReLU output):**

Through ReLU:
```
dz[0] = 0.0 * (z[0] ≤ 0 → 0)  = 0.0
dz[1] = -0.5 * (z[1] > 0 → 1) = -0.5
```

Through Linear:
```
dW[0][0] = dz[0] * x[0] = 0.0 * 1.0 = 0.0
dW[0][1] = dz[0] * x[1] = 0.0 * 2.0 = 0.0
dW[1][0] = dz[1] * x[0] = -0.5 * 1.0 = -0.5
dW[1][1] = dz[1] * x[1] = -0.5 * 2.0 = -1.0

dx[0] = W[0][0]*dz[0] + W[1][0]*dz[1] = 0.5*0.0 + 0.1*(-0.5) = -0.05
dx[1] = W[0][1]*dz[0] + W[1][1]*dz[1] = -0.3*0.0 + 0.8*(-0.5) = -0.4
```

This is exactly the computation NeuroDrive performs in `Linear::backward` and `Relu::backward`.

---

## Weight Initialisation

The quality of training depends heavily on how weights are initialised.

NeuroDrive uses **Glorot uniform** (also called Xavier uniform) initialisation:

```
W[i][j] ~ Uniform(-limit, +limit)
where limit = sqrt(6 / (fan_in + fan_out))
```

This initialises weights to a scale that keeps the variance of activations roughly stable across layers, which helps gradients flow without exploding or vanishing in the early stages of training.

Zero bias initialisation is used (`b = 0`), which is standard. Initialising biases to zero is safe because the non-zero weight randomness breaks the symmetry between neurons.

---

## Glorot Uniform in NeuroDrive

From `src/brain/common/math.rs` (the `glorot_uniform` helper):

```rust
pub fn glorot_uniform(out_dim: usize, in_dim: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    let limit = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
    (0..out_dim).map(|_| {
        (0..in_dim).map(|_| rng.gen_range(-limit..=limit)).collect()
    }).collect()
}
```

---

## Why Not Use tanh as Activation in A2C?

The large-scale empirical study by Andrychowicz et al. (see `materials/reinforcement-learning-resources.md`) found tanh outperforms ReLU in tested on-policy continuous-control settings. NeuroDrive currently uses ReLU, which is a credible upgrade candidate if the baseline remains unstable.

The practical difference:
- ReLU neurons can die (permanently output zero)
- tanh neurons saturate symmetrically but never fully die
- tanh gradients vanish at extreme values; ReLU gradients are binary (pass or block)

For a small MLP like NeuroDrive's 2×64 network, this distinction is unlikely to be decisive, but it is a real effect in longer or wider networks.

---

## Alternatives and Comparisons

| Feature | This project | Common alternative |
|---|---|---|
| Activation | ReLU | tanh (lit. recommendation for on-policy RL) |
| Initialisation | Glorot uniform | Orthogonal init (sometimes used in RL) |
| Gradient strategy | Full batch over rollout | Mini-batches (PPO-style) |
| Architecture | Separate actor + critic | Shared trunk (discouraged by literature) |

---

## Common Misunderstandings

❌ "Backpropagation is the learning algorithm"
✅ Backpropagation is the *gradient computation* algorithm. The learning algorithm is gradient descent (or Adam). Backpropagation computes what to change; the optimiser decides how.

❌ "More layers always helps"
✅ NeuroDrive uses 2 hidden layers, which empirical work confirms is sufficient for most continuous-control tasks of this scale.

❌ "ReLU is always better than tanh"
✅ In on-policy actor-critic settings, tanh is empirically competitive or better. ReLU is chosen for simplicity and speed in NeuroDrive.

❌ "Gradient accumulation with += is a bug"
✅ NeuroDrive accumulates gradients across the entire batch before calling the optimiser. This is standard practice. The optimiser step divides by batch size implicitly through the normalised loss.

---

## Related Files

- `concepts/foundations/optimization-and-gradients.md` — what happens after backpropagation
- `concepts/foundations/probability-and-distributions.md` — the Gaussian policy layer
- `project/systems/a2c-brain.md` — the full MLP used in the live A2C implementation
- `exercises/foundations/implement-linear-layer.md` — rebuild the forward/backward layer
- `exercises/foundations/implement-relu-backprop.md` — rebuild ReLU and verify the gradient
