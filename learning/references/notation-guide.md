# Notation Guide

A consolidated reference for the mathematical notation used across the NeuroDrive learning archive. Use this when a symbol appears in a concept file without an immediate definition.

---

## Reinforcement Learning Notation

| Symbol | Meaning | First introduced |
|---|---|---|
| `s` | State | `concepts/core/reinforcement-learning.md` |
| `a` | Action | `concepts/core/reinforcement-learning.md` |
| `r` or `r_t` | Reward at time t | `concepts/core/reinforcement-learning.md` |
| `π(a|s)` | Policy: probability of action a in state s | `concepts/core/reinforcement-learning.md` |
| `V^π(s)` | State-value function under policy π | `concepts/core/reinforcement-learning.md` |
| `Q^π(s,a)` | Action-value function under policy π | `concepts/core/reinforcement-learning.md` |
| `γ` | Discount factor (0.99 in NeuroDrive) | `concepts/core/reinforcement-learning.md` |
| `G_t` | Return from time t: `Σ_{k=0}^∞ γ^k r_{t+k}` | `concepts/core/reinforcement-learning.md` |
| `δ` | TD error / reward prediction error: `r + γV(s') - V(s)` | `concepts/core/reinforcement-learning.md` |
| `A(s,a)` | Advantage: `Q(s,a) - V(s)` | `concepts/core/advantage-estimation.md` |
| `λ` | GAE decay parameter (0.95 in NeuroDrive) | `concepts/core/advantage-estimation.md` |
| `Â_t` | Normalised advantage estimate at step t | `concepts/core/advantage-estimation.md` |
| `θ` | Policy parameters | `concepts/core/policy-gradient-methods.md` |
| `J(θ)` | Policy objective (expected return) | `concepts/core/policy-gradient-methods.md` |
| `∇_θ J(θ)` | Policy gradient | `concepts/core/policy-gradient-methods.md` |
| `H(π)` | Policy entropy | `concepts/core/policy-gradient-methods.md` |

---

## Neural Network Notation

| Symbol | Meaning | First introduced |
|---|---|---|
| `x` or `x_i` | Neuron activation / input vector | `concepts/foundations/neural-networks.md` |
| `W` or `W_{ji}` | Weight matrix / weight from neuron i to j | `concepts/foundations/neural-networks.md` |
| `b` or `b_j` | Bias vector / bias of neuron j | `concepts/foundations/neural-networks.md` |
| `z` or `z_j` | Pre-activation: `z = Wx + b` | `concepts/foundations/neural-networks.md` |
| `σ(·)` | Activation function (generic) | `concepts/foundations/neural-networks.md` |
| `L` | Loss function | `concepts/foundations/neural-networks.md` |
| `∂L/∂W` or `dW` | Gradient of L with respect to W | `concepts/foundations/neural-networks.md` |
| `η` | Learning rate | `concepts/foundations/optimization-and-gradients.md` |
| `m_t` | First moment estimate (Adam) | `concepts/foundations/optimization-and-gradients.md` |
| `v_t` | Second moment estimate (Adam) | `concepts/foundations/optimization-and-gradients.md` |
| `β1, β2` | Adam decay rates (0.9 and 0.999) | `concepts/foundations/optimization-and-gradients.md` |

---

## Probability and Distributions Notation

| Symbol | Meaning | First introduced |
|---|---|---|
| `N(μ, σ²)` | Gaussian distribution with mean μ and variance σ² | `concepts/foundations/probability-and-distributions.md` |
| `μ` | Distribution mean | `concepts/foundations/probability-and-distributions.md` |
| `σ` | Standard deviation | `concepts/foundations/probability-and-distributions.md` |
| `log_std` | Logarithm of standard deviation (learnable parameter) | `concepts/foundations/probability-and-distributions.md` |
| `p(x)` | Probability density at x | `concepts/foundations/probability-and-distributions.md` |
| `log p(x)` | Log-probability (log-density) | `concepts/foundations/probability-and-distributions.md` |

---

## Biological Learning Notation

| Symbol | Meaning | First introduced |
|---|---|---|
| `w_ij` | Synaptic weight from neuron i to j | `concepts/advanced/hebbian-plasticity.md` |
| `x_i` | Pre-synaptic activation (neuron i) | `concepts/advanced/hebbian-plasticity.md` |
| `x_j` | Post-synaptic activation (neuron j) | `concepts/advanced/hebbian-plasticity.md` |
| `Δw_ij` | Synaptic weight change | `concepts/advanced/hebbian-plasticity.md` |
| `e_ij` | Eligibility trace for synapse i→j | `concepts/advanced/eligibility-traces.md` |
| `λ` (trace) | Eligibility trace decay constant | `concepts/advanced/eligibility-traces.md` |
| `t_pre` | Time of pre-synaptic spike | `concepts/advanced/spike-timing-dependent-plasticity.md` |
| `t_post` | Time of post-synaptic spike | `concepts/advanced/spike-timing-dependent-plasticity.md` |
| `A_+, A_-` | STDP learning rate constants (LTP and LTD) | `concepts/advanced/spike-timing-dependent-plasticity.md` |
| `τ_+, τ_-` | STDP timing window constants | `concepts/advanced/spike-timing-dependent-plasticity.md` |
| `V` (membrane) | Membrane potential (spiking model) | `project/comparisons/rate-based-vs-spiking.md` |
| `τ_m` | Membrane time constant | `project/comparisons/rate-based-vs-spiking.md` |
| `G = (N, E)` | Neural graph: neuron set N, synapse set E | `concepts/advanced/structural-plasticity.md` |
| `f_in(j)` | Fan-in of neuron j | `concepts/advanced/structural-plasticity.md` |
| `f_out(i)` | Fan-out of neuron i | `concepts/advanced/structural-plasticity.md` |

---

## NeuroDrive-Specific Notation

| Symbol | Meaning |
|---|---|
| `OBSERVATION_DIM = 23` | Fixed dimension of `ObservationVector` |
| `obs_t` | Observation vector at tick t |
| `a_t` | Applied action at tick t |
| `latent_t` | Pre-tanh Gaussian sample at tick t |
| `squashed_t` | Post-tanh action at tick t |
| `V(s_t)` | Critic value estimate at tick t |
| `r_t` | Per-tick reward at tick t |
| `done_t` | Terminal flag at tick t |
| `s` | Arc-length along centreline (centreline progress) |
| `fraction` | `s / total_centreline_length` — lap completion fraction |

---

## Common Subscript/Superscript Conventions

| Convention | Meaning |
|---|---|
| `_t` | At time step t |
| `_{t+1}` | At the next time step |
| `_i` | For neuron/feature/parameter index i |
| `_ij` | For synapse from neuron i to neuron j |
| `^π` | Under policy π |
| `^*` | Optimal (e.g. `V^*` = optimal value function) |
| `hat (^)` | Estimated / bias-corrected value |
| `bar (ˉ)` | Mean / normalised value |
