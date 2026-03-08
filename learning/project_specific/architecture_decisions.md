# NeuroDrive Architecture Decisions

This document records the major architectural decisions made in NeuroDrive, explaining what was chosen, what alternatives were considered, why we went in this direction, and what trade-offs we accepted. These decisions form the structural foundation of the project.

> **See also:** [Neural Network Concepts](../concepts/core/neural_networks/), [Reinforcement Learning Concepts](../concepts/core/reinforcement_learning/)

---

## 1. Why Rust + Bevy Instead of Python + PyTorch/Gymnasium?

**What we chose:** Pure Rust with the Bevy ECS framework for both the simulation environment and the learning system.

**What we considered:** Python with PyTorch and Gymnasium/Farama — the industry-standard reinforcement learning stack used by virtually every RL research lab and tutorial.

**Why this approach:** The entire purpose of NeuroDrive is implementing learning mechanisms from first principles. Using PyTorch would hide the very thing we are trying to understand — gradient computation, parameter updates, network architecture wiring. By writing everything in Rust, every gradient is a line of code we authored, every cache is a `Vec<f32>` we allocated, and every update rule is arithmetic we can step through in a debugger.

Rust also provides deterministic performance with no garbage collector pauses, which matters for reproducible training runs. Having the environment and the brain in the same language eliminates serialisation boundaries and Python↔C++ interop complexity. Bevy's Entity Component System gives us structured concurrency, explicit system ordering, and clean separation between simulation systems and learning systems.

**Trade-offs we accepted:** There is no ecosystem of pre-built RL tools — no vectorised environments, no pre-made replay buffers, no TensorBoard integration out of the box. Everything must be implemented manually. Iteration speed is slower than Python prototyping; a one-line PyTorch change might be twenty lines of manual backpropagation. Compilation times add friction to the experiment loop.

---

## 2. Why Handwritten Neural Networks Instead of a Rust ML Crate?

**What we chose:** From-scratch implementations of Linear layers, ReLU, Tanh, and Adam optimisation using plain `Vec<f32>` storage.

**What we considered:** Rust ML crates such as `burn`, `candle`, and `tch-rs` (Rust bindings to libtorch).

**Why this approach:** Educational transparency. Every weight matrix is a `Vec<Vec<f32>>` we can print. Every backward pass computes explicit gradients we can inspect. Every optimiser step performs the full Adam update with bias correction in visible arithmetic. There is no autograd graph hiding the chain rule — the chain rule is the code.

This makes it possible to reason about what happens when gradients explode, why Glorot initialisation matters, and exactly how Adam's momentum interacts with sparse gradients. These are the concepts the project exists to teach.

**Trade-offs we accepted:** No GPU acceleration (all computation is single-threaded CPU). No automatic differentiation — if we get a backward pass wrong, gradients silently corrupt. Manual backpropagation is error-prone and requires careful unit testing of each layer's gradient.

> **Reference:** [Backpropagation](../concepts/core/neural_networks/backpropagation.md), [Optimisation](../concepts/core/neural_networks/optimisation.md)

---

## 3. Why A2C Before Brain-Inspired Plasticity?

**What we chose:** Validate that the environment is learnable by implementing a standard Advantage Actor-Critic (A2C) baseline before attempting any biologically-inspired learning rules.

**What we considered:** Jumping directly to Hebbian plasticity, STDP, or neuromodulated learning.

**Why this approach:** If A2C cannot learn to drive on our track, the problem lies in the environment design, the reward function, or the observation space — not in the learning algorithm. A2C is well-understood and has known convergence properties. By establishing it first, we isolate environment bugs from learning algorithm bugs. Once A2C drives successfully, we know the observation→action mapping is feasible and can confidently experiment with novel learning rules.

**Trade-offs we accepted:** This delays the core research goal (brain-inspired plasticity) by several development cycles. However, it dramatically reduces debugging ambiguity — without a working baseline, every failure of a novel algorithm would raise the question "is it the algorithm or the environment?"

> **Reference:** [Actor-Critic Methods](../concepts/core/reinforcement_learning/actor_critic.md), [A2C Algorithm](../concepts/core/reinforcement_learning/a2c_algorithm.md)

---

## 4. Why Separate Actor and Critic MLPs?

**What we chose:** Independent multi-layer perceptrons for actor (policy) and critic (value function), each with its own optimiser.

**What we considered:** A shared-trunk architecture with two output heads — one for action means and one for the state value.

**Why this approach:** Gradient interference. When actor and critic share a feature extractor, value-function gradients can destabilise the policy's internal representations, and vice versa. The policy wants features that distinguish good actions; the critic wants features that predict cumulative reward. These objectives are related but not identical, and shared gradients create implicit coupling that is difficult to debug.

Separate networks are also simpler to reason about: each has its own learning rate, its own gradient norms, and its own Adam state. When something goes wrong, we know exactly which network is misbehaving.

**Trade-offs we accepted:** More total parameters (no feature sharing), slightly higher memory usage, and no representation transfer between actor and critic. For our 14-dimensional observation space and small hidden layers, this overhead is negligible.

> **Reference:** [Neural Network Architecture](../concepts/core/neural_networks/architecture.md)

---

## 5. Why Gaussian Policy with Learnable Log-Std Instead of State-Dependent Std?

**What we chose:** A global learnable `log_std` parameter vector, clamped to `[-2.0, 0.5]`, shared across all states.

**What we considered:** A state-dependent standard deviation output from the actor network (a second head producing per-state exploration noise).

**Why this approach:** Simplicity and stability. A global log-std has fewer failure modes — it cannot collapse to zero for some states while exploding for others. The clamp bounds provide hard safety rails: `exp(-2.0) ≈ 0.135` prevents premature convergence, and `exp(0.5) ≈ 1.65` prevents excessively noisy actions. For a baseline whose purpose is validating environment learnability, this is sufficient.

**Trade-offs we accepted:** The agent cannot adapt its exploration strategy per state. It explores with the same noise magnitude whether it is on a straight (where low noise suffices) or entering a corner (where more exploration might help). This limits asymptotic performance but does not prevent learning.

> **Reference:** [Policy Gradient Methods](../concepts/core/reinforcement_learning/policy_gradients.md)

---

## 6. Why Tile-Grid Track Instead of Spline/Polygon Geometry?

**What we chose:** A discrete tile grid where each cell is a `TilePart` enum variant (straight, inner/outer corner arcs, grass).

**What we considered:** Continuous spline curves with polygon collision geometry, or procedurally generated Bézier track boundaries.

**Why this approach:** O(1) spatial queries — given a world position, integer division yields the tile index and `is_road_at()` is a lookup. Collision detection reduces to point-in-tile tests. Building new tracks is straightforward: place tiles in a grid, and the centreline construction algorithm traverses connectivity automatically.

**Trade-offs we accepted:** Lower geometric resolution — curves are approximated by quarter-circle arcs discretised into 8 line segments per tile. Track edges have visible stair-stepping at tile boundaries. The tile grid constrains track design to 90° turns on a fixed grid spacing, making complex geometry (hairpins, chicanes, varying-radius bends) difficult to represent.

> **Reference:** [Progress and Lap System](implementations/progress_and_lap_system.md), [Car Physics Model](implementations/car_physics_model.md)
