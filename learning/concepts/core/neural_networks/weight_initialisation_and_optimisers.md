# Weight Initialisation and Optimisers

## Prerequisites

- **Forward pass**: understand how data flows through linear layers and activations (see `forward_pass_and_layers.md`).
- **Backpropagation**: understand how gradients are computed and accumulated (see `backpropagation.md`).
- **Basic probability**: the concept of sampling from a uniform distribution U(−a, a).

## Target Depth for This Project

**Level 3–4** — You must be able to derive the Glorot/Xavier initialisation bound for a given layer shape, walk through 3 steps of the Adam optimiser with concrete numbers, and explain why NeuroDrive uses separate optimisers with different learning rates for actor and critic.

---

## Core Concept

### Why Initialisation Matters

Before training begins, every weight in the network must be assigned an initial value. This seemingly trivial step has an outsized effect on whether training succeeds or fails.

**If weights are too large**: the output of each layer grows exponentially as data passes through the network. After a few layers, activations overflow to extreme values, gradients explode, and the optimiser produces enormous, destructive updates. The network diverges on the first step.

**If weights are too small**: the output of each layer shrinks toward zero. After a few layers, activations and gradients are so small that they underflow to zero. No meaningful learning signal reaches the early layers. The network appears to train but never improves.

**If all weights are identical**: every neuron in a layer computes exactly the same function. Gradients are identical for every neuron. Updates are identical. The symmetry never breaks — the network effectively has one neuron per layer, regardless of width. This is called the *symmetry problem*.

Good initialisation keeps signals flowing at a stable magnitude through both the forward and backward passes. It does not need to be perfect — the optimiser will adjust everything during training — but it must avoid the catastrophic regimes above.

### Glorot/Xavier Uniform Initialisation

The Glorot initialisation (also called Xavier initialisation, after Xavier Glorot who proposed it in 2010) is designed to preserve the variance of activations across layers. The core idea: if the input to a layer has variance σ², the output should also have approximately variance σ². This prevents signals from growing or shrinking as they pass through.

For a linear layer with `fan_in` inputs and `fan_out` outputs, weights are sampled uniformly from:

\[
W_{ij} \sim U\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}},\; +\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)
\]

**Why this specific formula?**

For a uniform distribution U(−a, a), the variance is a²/3. Setting the variance of the output equal to the variance of the input (under assumptions of zero-mean inputs and linear activation) leads to:

\[
\text{Var}(y) \approx \text{fan\_in} \cdot \text{Var}(W) \cdot \text{Var}(x)
\]

For Var(y) = Var(x), we need Var(W) = 1/fan_in. From the output side, we need Var(W) = 1/fan_out. Glorot compromises by using the harmonic mean: Var(W) = 2/(fan_in + fan_out). Setting a²/3 = 2/(fan_in + fan_out) gives a = √(6/(fan_in + fan_out)).

**Worked example — NeuroDrive's first actor layer (14→64)**:

\[
a = \sqrt{\frac{6}{14 + 64}} = \sqrt{\frac{6}{78}} = \sqrt{0.07692} \approx 0.2774
\]

Every weight in this layer is sampled from U(−0.2774, +0.2774). For the second hidden layer (64→64):

\[
a = \sqrt{\frac{6}{64 + 64}} = \sqrt{\frac{6}{128}} = \sqrt{0.04688} \approx 0.2165
\]

Larger layers get smaller initial weights — this is the self-adjusting property. A 14→64 layer gets weights in [−0.28, +0.28]; a 64→64 layer gets [−0.22, +0.22]; and the mean head (64→2):

\[
a = \sqrt{\frac{6}{64 + 2}} = \sqrt{\frac{6}{66}} = \sqrt{0.09091} \approx 0.3015
\]

This final layer gets slightly larger weights because it has few outputs. The formula automatically accounts for the asymmetry.

**Biases**: NeuroDrive initialises all biases to **zero**. This is standard practice — the bias is just an offset, and there is no symmetry-breaking issue for biases (each output neuron already has a unique weight vector).

---

## Mathematical Foundation — Optimisers

### Gradient Descent: The Core Idea

Once backpropagation gives us the gradient ∂L/∂θ for every parameter θ, we update parameters to reduce the loss:

\[
\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}
\]

where η (eta) is the **learning rate** — a scalar that controls the step size.

This is **stochastic gradient descent** (SGD) when the gradient is computed on a mini-batch rather than the full dataset. It works, but it has problems: it oscillates in narrow valleys, it is slow on flat plateaux, and it treats all parameters equally regardless of their gradient history.

### SGD with Momentum

Momentum adds a "velocity" that accumulates past gradients:

\[
v \leftarrow \beta v + \frac{\partial L}{\partial \theta}
\]
\[
\theta \leftarrow \theta - \eta \cdot v
\]

where β (typically 0.9) controls how much history is retained. Momentum smooths oscillations and accelerates movement along consistent gradient directions. Think of it as a heavy ball rolling downhill — it builds speed on slopes and coasts through minor bumps.

### Adam: Adaptive Moment Estimation

Adam (Kingma & Ba, 2014) combines momentum with per-parameter adaptive learning rates. It maintains two running averages for each parameter:

**First moment** m — the exponential moving average of gradients (like momentum):

\[
m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g
\]

**Second moment** v — the exponential moving average of *squared* gradients:

\[
v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2
\]

**Bias correction** — because m and v are initialised to zero, they are biased toward zero in early steps. Corrected estimates:

\[
\hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t}
\]

where t is the step count.

**Update rule**:

\[
\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\]

The key insight: dividing by √v̂ gives each parameter its own effective learning rate. Parameters with consistently large gradients get a smaller effective step (preventing overshooting). Parameters with small gradients get a larger effective step (preventing stalling). The ε term (typically 10⁻⁸) prevents division by zero.

### Worked Example — 3 Steps of Adam for One Parameter

Let us track a single weight parameter through 3 Adam updates with these hyperparameters:

- η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸
- Initial: θ = 0.5, m = 0, v = 0, t = 0

**Step 1**: gradient g = 0.2

t = 1

m = 0.9(0) + 0.1(0.2) = 0.02

v = 0.999(0) + 0.001(0.04) = 0.00004

m̂ = 0.02 / (1 − 0.9¹) = 0.02 / 0.1 = 0.2

v̂ = 0.00004 / (1 − 0.999¹) = 0.00004 / 0.001 = 0.04

θ = 0.5 − 0.001 × 0.2 / (√0.04 + 10⁻⁸) = 0.5 − 0.001 × 0.2 / 0.2 = 0.5 − 0.001 = **0.499**

Notice: the bias correction in step 1 makes the effective update approximately equal to the learning rate (0.001), regardless of the gradient magnitude. This is a remarkable property of Adam in early training — it auto-calibrates.

**Step 2**: gradient g = 0.15

t = 2

m = 0.9(0.02) + 0.1(0.15) = 0.018 + 0.015 = 0.033

v = 0.999(0.00004) + 0.001(0.0225) = 0.00003996 + 0.0000225 = 0.00006246

m̂ = 0.033 / (1 − 0.9²) = 0.033 / 0.19 = 0.17368

v̂ = 0.00006246 / (1 − 0.999²) = 0.00006246 / 0.001999 = 0.031246

θ = 0.499 − 0.001 × 0.17368 / (√0.031246 + 10⁻⁸) = 0.499 − 0.001 × 0.17368 / 0.17677 = 0.499 − 0.000983 ≈ **0.498017**

**Step 3**: gradient g = −0.3 (sign reversed — perhaps the loss landscape curved)

t = 3

m = 0.9(0.033) + 0.1(−0.3) = 0.0297 − 0.03 = −0.0003

v = 0.999(0.00006246) + 0.001(0.09) = 0.00006240 + 0.00009 = 0.00015240

m̂ = −0.0003 / (1 − 0.9³) = −0.0003 / 0.271 = −0.001107

v̂ = 0.00015240 / (1 − 0.999³) = 0.00015240 / 0.002997 = 0.050851

θ = 0.498017 − 0.001 × (−0.001107) / (√0.050851 + 10⁻⁸) = 0.498017 + 0.001 × 0.001107 / 0.22550

= 0.498017 + 0.0000049 ≈ **0.498022**

Key observations:
- When the gradient reversed sign in step 3, the momentum m nearly cancelled to zero (−0.0003). Adam's response was a very small update, showing its natural resistance to noisy sign-flipping gradients.
- The second moment v grows slowly because β₂ = 0.999, providing a long memory of gradient magnitudes.
- Bias correction was strongest at t = 1 (dividing by 0.1) and weakened by t = 3 (dividing by 0.271). By t ≈ 50, the correction is negligible.

---

## How NeuroDrive Uses This

### Glorot Initialisation in Code

In `src/brain/common/math.rs`, the `glorot_uniform` function computes:

```rust
let limit = (6.0 / (rows as f32 + cols as f32)).sqrt();
```

Then samples each weight uniformly from `−limit..limit`. Note that `rows` corresponds to `out_dim` (fan_out) and `cols` to `in_dim` (fan_in), matching the formula exactly.

### AdamOptimizer Struct

In `src/brain/common/optim.rs`, the `AdamOptimizer` stores per-layer first-moment vectors (`m_weights`, `m_biases`) and second-moment vectors (`v_weights`, `v_biases`), a step counter `t`, and the hyperparameters β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸.

The `step()` method iterates over all layers, computes bias-corrected moments, and updates weights and biases in place. Every line maps directly to the Adam formulas above.

### Separate Optimisers for Actor and Critic

NeuroDrive creates **two** `AdamOptimizer` instances:

```rust
let a_opt = AdamOptimizer::new(&[&a_fc1, &a_fc2, &a_mean], 3e-4);
let c_opt = AdamOptimizer::new(&[&c_fc1, &c_fc2, &c_value], 1e-3);
```

The actor uses a learning rate of **3×10⁻⁴** and the critic uses **1×10⁻³** — the critic learns roughly 3× faster.

**Why different rates?** The critic's job is simpler and more stable: it is fitting a regression target (the discounted return). A higher learning rate lets it converge quickly, providing a useful baseline for advantage estimation early in training. The actor's job is harder and more sensitive: it is adjusting a stochastic policy. Too-fast policy updates can collapse exploration prematurely or oscillate between competing behaviours. A lower learning rate keeps the actor's updates conservative, stabilising the learning process.

This asymmetry is standard practice in actor-critic methods. If the critic is inaccurate, the advantages it computes are noisy, which corrupts the actor's gradient signal. Letting the critic learn faster ensures it provides reliable value estimates by the time the actor starts making meaningful policy changes.

### Standalone Adam for Log-Std

The log-standard-deviation parameters (`a_log_std`) are not part of any `Linear` layer — they are free-standing learnable scalars. NeuroDrive maintains separate first and second moment vectors (`log_std_opt_m`, `log_std_opt_v`) and a shared step counter (`opt_t`) for these parameters. The update uses the same Adam formula with the actor's learning rate (3×10⁻⁴) and clamps the result to [−2.0, 0.5] to prevent the standard deviation from collapsing to zero or growing excessively.

### Three Optimiser Instances, One Algorithm

To summarise NeuroDrive's optimiser topology:

| Optimiser | Parameters | Learning Rate | Purpose |
|-----------|-----------|---------------|---------|
| `a_opt` | Actor Linear layers (a_fc1, a_fc2, a_mean) | 3×10⁻⁴ | Conservative policy updates |
| `c_opt` | Critic Linear layers (c_fc1, c_fc2, c_value) | 1×10⁻³ | Fast value function fitting |
| Standalone Adam state | a_log_std (2 scalars) | 3×10⁻⁴ | Exploration control |

All three use the same β₁, β₂, ε hyperparameters. The only difference is the learning rate. Each maintains independent moment vectors, so the momentum of the actor never interferes with the momentum of the critic.

---

## Common Misconceptions

1. **"Initialisation does not matter if you train long enough."** In theory, any non-degenerate initialisation can eventually converge. In practice, bad initialisation causes numerical overflow/underflow within the first few updates, and training never recovers. Glorot initialisation is cheap insurance — it costs nothing at runtime and prevents a class of failures that no amount of training can fix.

2. **"Adam adapts the learning rate, so the initial learning rate does not matter."** Adam adapts *relative* scaling between parameters, but the base learning rate η still controls the overall step size. Setting η = 1.0 with Adam will still diverge on most problems. The learning rate remains the single most important hyperparameter even when using adaptive methods.

3. **"Momentum and Adam's first moment are the same thing."** Classical momentum accumulates the gradient itself: v = βv + g. Adam's first moment accumulates an exponential moving average: m = β₁m + (1−β₁)g. The (1−β₁) scaling factor means Adam's first moment is a *weighted average* rather than a running sum. This is why Adam does not accelerate unboundedly — its effective step size is naturally bounded.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Fan-in** | The number of input connections to a neuron (in_dim for a linear layer). |
| **Fan-out** | The number of output connections from a neuron (out_dim for a linear layer). |
| **Glorot/Xavier init** | Weight initialisation sampling from U(−√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))). |
| **Learning rate (η)** | Scalar controlling the step size of parameter updates. |
| **SGD** | Stochastic Gradient Descent: θ ← θ − η·∇L. |
| **Momentum** | Accumulating past gradients to smooth and accelerate parameter updates. |
| **Adam** | Adaptive Moment Estimation: combines momentum (first moment) with per-parameter scaling (second moment). |
| **First moment (m)** | Exponential moving average of gradients — tracks the direction of recent updates. |
| **Second moment (v)** | Exponential moving average of squared gradients — tracks the magnitude of recent updates. |
| **Bias correction** | Dividing moments by (1 − βᵗ) to compensate for zero-initialisation bias in early steps. |
| **ε (epsilon)** | Small constant (10⁻⁸) added to √v̂ to prevent division by zero in the Adam update. |

---

## Recommended Materials

1. **Xavier Glorot & Yoshua Bengio — "Understanding the difficulty of training deep feedforward neural networks" (2010)**. The original paper deriving the Glorot initialisation. Section 4 contains the variance analysis that leads to the √(6/(fan_in + fan_out)) formula.
2. **Diederik Kingma & Jimmy Ba — "Adam: A Method for Stochastic Optimization" (2014)**. The original Adam paper. Section 2 derives the algorithm in full. The bias correction proof in Section 3 is worth reading carefully.
3. **Andrej Karpathy — CS231n Lecture 6: Training Neural Networks I** (slides + video). Covers initialisation, SGD, momentum, and Adam in the context of training practical networks. The visualisations of loss landscape trajectories under different optimisers are particularly instructive.
4. **NeuroDrive source — `src/brain/common/math.rs` and `src/brain/common/optim.rs`**. Read `glorot_uniform()` (6 lines, direct translation of the formula) and `AdamOptimizer::step()` (the full Adam algorithm in ~30 lines of Rust). Compare every line to the formulas in this document.
