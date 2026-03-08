# Calculus and Gradients for Neural Networks

## Prerequisites

- Comfortable with functions (y = f(x), input → output).
- Basic algebra (exponents, substitution).
- Vectors and dot products (see `linear_algebra_for_ml.md`).

## Target Depth for This Project

**Level 2–3.** You need to understand what a derivative and gradient are, how gradient descent works mechanically, and why the chain rule enables backpropagation. You do not need to derive gradients by hand for complex networks — automatic differentiation handles that — but you must understand the process conceptually.

---

## Core Concept

### Derivatives: rates of change

A **derivative** measures how quickly a function's output changes as its input changes. For a function f(x), the derivative at a point x is the slope of the tangent line to the curve at that point.

Notation: f'(x) or df/dx.

**Simple example.** Let f(x) = x². The derivative is f'(x) = 2x.

- At x = 3: f'(3) = 6. The function is increasing at a rate of 6 units of output per unit of input.
- At x = 0: f'(0) = 0. The function is momentarily flat — this is the minimum.
- At x = −2: f'(−2) = −4. The function is decreasing.

The sign tells you the direction; the magnitude tells you how steeply.

### Partial derivatives

When a function has **multiple inputs**, a **partial derivative** measures the rate of change with respect to one input while holding all others constant.

For f(x, y) = x² + 2y²:

```
∂f/∂x = 2x      (treat y as a constant, differentiate with respect to x)
∂f/∂y = 4y      (treat x as a constant, differentiate with respect to y)
```

At the point (3, 2):

```
∂f/∂x = 2(3) = 6       — "if we nudge x slightly, f increases at rate 6"
∂f/∂y = 4(2) = 8       — "if we nudge y slightly, f increases at rate 8"
```

---

## Mathematical Foundation

### The gradient vector

The **gradient** collects all partial derivatives into a single vector:

```
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
```

The gradient has a critical geometric property: **it points in the direction of steepest increase** of the function. Its magnitude tells you how steep that increase is.

For our function f(x, y) = x² + 2y² at (3, 2):

```
∇f = (6, 8)
```

This vector points "uphill". To decrease f, we move in the **opposite** direction: (−6, −8).

### Gradient descent

**Gradient descent** is the optimisation algorithm at the heart of neural network training. The idea is simple: to minimise a loss function L, repeatedly take small steps in the direction opposite to the gradient.

**Update rule:**

```
θ ← θ − η · ∇L(θ)
```

where θ is the parameter vector, η (eta) is the **learning rate**, and ∇L(θ) is the gradient of the loss with respect to the parameters.

**Worked example: 3 steps of gradient descent.**

Minimise f(x, y) = x² + 2y², starting at (3, 2), with learning rate η = 0.1.

**Step 0** — Starting point: (x, y) = (3, 2), f = 9 + 8 = 17.

Gradient: ∇f = (2x, 4y) = (6, 8).

Update:

```
x ← 3 − 0.1 × 6 = 3 − 0.6 = 2.4
y ← 2 − 0.1 × 8 = 2 − 0.8 = 1.2
```

**Step 1** — (x, y) = (2.4, 1.2), f = 5.76 + 2.88 = 8.64.

Gradient: ∇f = (4.8, 4.8).

Update:

```
x ← 2.4 − 0.1 × 4.8 = 2.4 − 0.48 = 1.92
y ← 1.2 − 0.1 × 4.8 = 1.2 − 0.48 = 0.72
```

**Step 2** — (x, y) = (1.92, 0.72), f = 3.6864 + 1.0368 = 4.7232.

Gradient: ∇f = (3.84, 2.88).

Update:

```
x ← 1.92 − 0.1 × 3.84 = 1.92 − 0.384 = 1.536
y ← 0.72 − 0.1 × 2.88 = 0.72 − 0.288 = 0.432
```

**After step 2** — (x, y) = (1.536, 0.432), f = 2.3593 + 0.3732 = 2.7325.

The loss dropped from 17 → 8.64 → 4.72 → 2.73. Each step moves closer to the minimum at (0, 0). The steps get smaller because the gradient shrinks as we approach the minimum.

### The chain rule

The **chain rule** tells us how to differentiate a **composition** of functions — a function inside a function.

**Rule:**

```
d/dx f(g(x)) = f'(g(x)) · g'(x)
```

Read it as: "the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function."

**Worked example.** Let h(x) = (2x + 1)³. This is a composition: the outer function is f(u) = u³ and the inner function is g(x) = 2x + 1.

```
f'(u) = 3u²
g'(x) = 2
```

Applying the chain rule:

```
h'(x) = f'(g(x)) · g'(x)
      = 3(2x + 1)² · 2
      = 6(2x + 1)²
```

At x = 1: h'(1) = 6(3)² = 6 × 9 = 54.

**Why does this matter?** A neural network is a chain of function compositions:

```
output = f₃(f₂(f₁(input)))
```

Each fᵢ is a layer (affine transformation + activation). To train the network, we need to know how the loss changes when we adjust parameters in f₁ — the earliest layer. The chain rule lets us "unwind" the composition layer by layer:

```
∂L/∂θ₁ = (∂L/∂f₃) · (∂f₃/∂f₂) · (∂f₂/∂f₁) · (∂f₁/∂θ₁)
```

This is **backpropagation**: applying the chain rule repeatedly, working backwards from the loss to each parameter. Without the chain rule, training deep networks would be intractable.

---

## How NeuroDrive Uses This

**Gradient descent in A2C.** NeuroDrive's advantage actor-critic algorithm computes three loss components (actor loss, critic loss, entropy bonus). These are combined into a total loss, and the gradient of this total loss with respect to all network parameters is computed via backpropagation. The optimiser (Adam) then applies a gradient descent step — conceptually θ ← θ − η · ∇L(θ), though Adam adds momentum and adaptive scaling.

**Learning rate.** NeuroDrive uses a learning rate of 3 × 10⁻⁴. This small value ensures each gradient step is conservative — large steps risk overshooting the minimum and destabilising training.

**Gradient norms.** NeuroDrive clips gradient norms (max_grad_norm = 0.5) before applying updates. This prevents a single large gradient from causing a catastrophic parameter jump — a safety rail enabled by understanding what gradient magnitude means.

**Backpropagation through the network.** The actor network takes a 12-dimensional observation vector, passes it through two hidden layers with ReLU activations, and outputs Gaussian parameters (μ, σ). The chain rule propagates the policy gradient loss backward through every layer, adjusting each weight matrix and bias vector.

---

## Common Misconceptions

1. **"Gradient descent always finds the global minimum."** It finds a **local** minimum. For non-convex loss landscapes (which neural networks have), there may be many local minima. In practice, most local minima in high-dimensional networks are "good enough."

2. **"A zero gradient means we have found the best answer."** A zero gradient indicates a **critical point** — which could be a minimum, a maximum, or a saddle point. Saddle points are common in high dimensions.

3. **"The learning rate does not matter much."** The learning rate is one of the most sensitive hyperparameters. Too large and training diverges; too small and it stalls. Getting it right is essential.

---

## Glossary

| Term | Definition |
|---|---|
| **Derivative** | The rate of change of a function at a point; the slope of the tangent line. |
| **Partial derivative** | The derivative with respect to one variable, holding others constant. |
| **Gradient (∇f)** | A vector of all partial derivatives; points in the direction of steepest increase. |
| **Gradient descent** | Iterative optimisation: θ ← θ − η · ∇L(θ). |
| **Learning rate (η)** | Step size for gradient descent; controls how far each update moves. |
| **Chain rule** | Rule for differentiating compositions: d/dx f(g(x)) = f'(g(x)) · g'(x). |
| **Backpropagation** | Applying the chain rule layer-by-layer to compute gradients in a neural network. |

---

## Recommended Materials

1. **3Blue1Brown — "Essence of Calculus"** (YouTube series). Builds visual intuition for derivatives, the chain rule, and gradients — the best starting point for anyone new to calculus.
2. **Andrej Karpathy — "micrograd"** (YouTube lecture + GitHub). Builds a tiny autograd engine from scratch, showing exactly how the chain rule becomes backpropagation in code.
3. **Mathematics for Machine Learning** — Deisenroth, Faisal & Ong, Chapter 5. Covers vector calculus and gradients with an ML focus, including worked gradient descent examples.
