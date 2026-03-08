# Forward Pass and Neural Network Layers

## Prerequisites

- Vectors and matrices: what a dot product is, how matrix-vector multiplication works.
- Basic function composition: f(g(x)).
- Comfort reading Rust struct definitions (you will see NeuroDrive's `Linear`, `Relu`, `Tanh` types).

## Target Depth for This Project

**Level 4** — You must be able to hand-compute a full forward pass through a multi-layer perceptron with specific numerical values, explain why each non-linearity exists, and map every step to the corresponding line in NeuroDrive's `src/brain/common/mlp.rs`.

---

## Core Concept

### What Is a Neural Network?

A neural network is a **composition of parameterised functions**. Each function (called a *layer*) takes a vector of numbers, transforms it using learnable parameters, and produces a new vector. Chain several layers together and you get a function that can, in principle, approximate any continuous mapping from inputs to outputs — provided it contains at least one non-linear layer (the universal approximation theorem).

The critical word is *parameterised*. A neural network is not a fixed formula. Its behaviour is determined by numerical parameters — weights and biases — that are adjusted during training. Before training, the network outputs nonsense. After training, those same parameters encode a useful input-output relationship.

### The Linear Layer (Affine Transformation)

The simplest and most fundamental layer is the **linear layer**, which computes an affine transformation:

\[
\mathbf{y} = W\mathbf{x} + \mathbf{b}
\]

where:
- **x** is the input vector of dimension `in_dim`
- **W** is the weight matrix of shape `[out_dim × in_dim]`
- **b** is the bias vector of dimension `out_dim`
- **y** is the output vector of dimension `out_dim`

Each output element is a weighted sum of all inputs, plus a bias term:

\[
y_i = \sum_{j=0}^{\text{in\_dim}-1} W_{ij} \cdot x_j + b_i
\]

The weight matrix controls *how much each input contributes to each output*. The bias shifts the result. Together they define an affine (linear + translation) mapping from one vector space to another.

### Non-Linear Activation Functions

#### ReLU — Rectified Linear Unit

\[
\text{ReLU}(x) = \max(0, x)
\]

ReLU is applied element-wise. If the input is positive, it passes through unchanged. If negative, it is clamped to zero.

**Gradient**: The derivative is 1 for positive inputs and 0 for negative inputs. This makes backpropagation trivially cheap — the gradient is either passed through or blocked entirely.

**Dead ReLU problem**: If a neuron's pre-activation is consistently negative for all training inputs (perhaps due to a large negative bias or unfortunate initialisation), its gradient is permanently zero and it can never recover. NeuroDrive monitors this with a `dead_relu_fraction` metric per layer.

#### Tanh — Hyperbolic Tangent

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

Output range is \([-1, 1]\). This is useful when outputs should be centred around zero (e.g. steering commands).

**Gradient**: \(\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - y^2\). Notice that NeuroDrive's `Tanh` caches the *output* (not the input) because the gradient formula uses the output value directly.

**Saturation**: For large |x|, tanh approaches ±1 and the gradient approaches 0. This can slow learning in deep networks (the "vanishing gradient" problem). NeuroDrive uses ReLU in hidden layers to avoid this, reserving Tanh only where bounded outputs are needed.

#### Why Non-Linearities Are Essential

Without non-linearities, stacking N linear layers collapses into a single linear layer:

\[
W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'
\]

No matter how many layers you stack, the composite function is still linear. The network could never learn curved decision boundaries, thresholds, or any relationship that is not a straight line. Non-linearities between linear layers break this collapse and give the network its expressive power.

---

## Mathematical Foundation — Complete Numerical Example

We will compute the forward pass of a 2-layer MLP from raw numbers.

**Architecture**: Input (3) → Linear (3→2) → ReLU → Linear (2→1) → output

**Input vector**:

\[
\mathbf{x} = \begin{bmatrix} 0.5 \\ -0.3 \\ 0.8 \end{bmatrix}
\]

### Layer 1: Linear (3→2)

**Weights** (shape [2×3]):

\[
W_1 = \begin{bmatrix} 0.2 & -0.5 & 0.1 \\ 0.4 & 0.3 & -0.2 \end{bmatrix}
\]

**Biases**:

\[
\mathbf{b}_1 = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
\]

**Computation** — each output neuron computes a dot product with the input, plus bias:

\[
z_1[0] = (0.2)(0.5) + (-0.5)(-0.3) + (0.1)(0.8) + 0.1
\]
\[
= 0.10 + 0.15 + 0.08 + 0.1 = 0.43
\]

\[
z_1[1] = (0.4)(0.5) + (0.3)(-0.3) + (-0.2)(0.8) + (-0.1)
\]
\[
= 0.20 - 0.09 - 0.16 - 0.1 = -0.15
\]

**Result**: \(\mathbf{z}_1 = [0.43, -0.15]\)

### ReLU Activation

\[
\text{ReLU}([0.43, -0.15]) = [\max(0, 0.43), \max(0, -0.15)] = [0.43, 0.0]
\]

The first neuron passes through. The second is clamped to zero — it is "dead" for this input. During backpropagation, no gradient will flow through the second neuron for this sample.

**Result**: \(\mathbf{h} = [0.43, 0.0]\)

### Layer 2: Linear (2→1)

**Weights** (shape [1×2]):

\[
W_2 = \begin{bmatrix} 0.6 & -0.4 \end{bmatrix}
\]

**Bias**:

\[
b_2 = 0.05
\]

**Computation**:

\[
\text{output} = (0.6)(0.43) + (-0.4)(0.0) + 0.05 = 0.258 + 0.0 + 0.05 = 0.308
\]

**Final output**: **0.308**

Notice how the second hidden neuron's contribution is exactly zero because ReLU killed it. The network effectively used only one of its two hidden neurons for this particular input.

---

## How NeuroDrive Uses This

### Architecture

NeuroDrive uses two **separate** multi-layer perceptrons — they share no parameters:

**Actor network** (selects actions):
```
Input [14] → Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2) → action means
```

**Critic network** (estimates value):
```
Input [14] → Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → state value
```

The 14-dimensional input is the observation vector: 11 normalised raycast distances + speed + heading error + angular velocity.

The actor outputs 2 values — the *mean* of a Gaussian distribution for steering and throttle. The standard deviation is a separate learnable parameter (`a_log_std`), not an output of the network.

The critic outputs 1 scalar — the estimated discounted return from the current state.

### The `Linear` Struct

In `src/brain/common/mlp.rs`, each `Linear` layer stores:
- `weights: Vec<Vec<f32>>` — shape `[out_dim][in_dim]`
- `biases: Vec<f32>` — shape `[out_dim]`
- `grad_weights` and `grad_biases` — same shapes, accumulate gradients during backward passes
- `input_cache: Option<Vec<f32>>` — stores the input seen during `forward()`, needed later by `backward()`

The `forward()` method computes `y = Wx + b` and caches the input. The cache is the bridge between the forward pass and the backward pass: without it, `backward()` would not know what input values to use when computing weight gradients.

### The `Relu` Struct

The `Relu` layer caches its *input* during forward (not the output). During backward, it checks each cached input: if positive, the gradient passes through; if non-positive, the gradient is zeroed. This is the gradient masking behaviour.

### The `ActorCritic` Struct

In `src/brain/a2c/model.rs`, the `ActorCritic` struct holds all six linear layers (`a_fc1`, `a_fc2`, `a_mean`, `c_fc1`, `c_fc2`, `c_value`), four ReLU activations (`a_relu1`, `a_relu2`, `c_relu1`, `c_relu2`), and separate Adam optimisers for actor and critic. The `forward()` method chains them in sequence and returns both the action distribution and the value estimate.

### Why Separate Networks?

The actor and critic have different jobs and different loss functions. Sharing parameters would create conflicting gradient signals — the actor wants features that distinguish good from bad actions, while the critic wants features that predict cumulative reward. Separate networks ensure each can specialise without interference. NeuroDrive reinforces this separation with different learning rates: 3×10⁻⁴ for the actor and 1×10⁻³ for the critic.

---

## Common Misconceptions

1. **"More layers always means better."** Additional layers add parameters and computational cost. Without sufficient training data and careful initialisation, deeper networks suffer from vanishing gradients, overfitting, or simply being harder to optimise. NeuroDrive uses only two hidden layers — this is enough for the 14-dimensional continuous control task and keeps gradient flow manageable with hand-written backpropagation.

2. **"Biases are optional."** Removing biases means every linear layer's output must pass through the origin. This constrains the network's representational capacity. A bias vector shifts the activation function's "sweet spot" to wherever the data needs it. NeuroDrive initialises all biases to zero — they learn their required offsets during training.

3. **"ReLU(x) = x for all x."** ReLU passes through positive values unchanged, but it *destroys* information for negative values. This asymmetry is precisely what gives ReLU its non-linear power, but it also means that neurons can "die" if they consistently receive negative pre-activations. NeuroDrive tracks the dead-ReLU fraction per layer to catch this failure mode early.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Affine transformation** | A linear map followed by a translation: y = Wx + b. |
| **Activation function** | A non-linear function applied element-wise after a linear layer. |
| **ReLU** | Rectified Linear Unit: max(0, x). |
| **Tanh** | Hyperbolic tangent: maps inputs to (−1, 1). |
| **Forward pass** | Computing the network's output from an input by passing data through each layer sequentially. |
| **Input cache** | A stored copy of a layer's input during the forward pass, retained for use in backpropagation. |
| **Dead neuron** | A ReLU neuron that always outputs zero because its pre-activation is consistently non-positive. |
| **MLP** | Multi-Layer Perceptron: a feedforward network of stacked linear layers and activations. |
| **Actor** | The network that outputs action parameters (policy). |
| **Critic** | The network that outputs a scalar value estimate. |

---

## Recommended Materials

1. **3Blue1Brown — "But what is a neural network?"** (YouTube, Chapter 1 of the *Neural Networks* series). Visual, intuition-first introduction to what neurons and layers actually compute.
2. **Michael Nielsen — *Neural Networks and Deep Learning*, Chapter 1**. Free online textbook that builds from single neurons to multi-layer networks with worked examples.
3. **Andrej Karpathy — "micrograd"** (GitHub + YouTube lecture). A tiny autograd engine that implements exactly the same forward/backward pattern NeuroDrive uses, but in Python. Watching this after reading this file will cement the connection between theory and code.
4. **NeuroDrive source — `src/brain/common/mlp.rs`**. Read the `Linear::forward()`, `Relu::forward()`, and `Tanh::forward()` implementations. They are direct translations of the mathematics above into Rust.
