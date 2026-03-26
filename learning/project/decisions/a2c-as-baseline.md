# Decision: A2C as the Baseline Controller

## The Decision

NeuroDrive uses a handwritten Advantage Actor-Critic (A2C) implementation as its first autonomous controller. This was not the only option. This file explains what problem A2C solves, why it was chosen over alternatives, what trade-offs it brings, and what it means for the project's long-term direction.

**Status:** Implemented decision. A2C is live in the current runtime.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — MDP, episodes, value functions
- `concepts/core/policy-gradient-methods.md` — policy gradient, entropy, baseline subtraction
- `concepts/core/actor-critic-architecture.md` — the A2C model structure

---

## The Problem A2C Was Chosen to Solve

The problem is not "how do we make the car drive well?" — it is a narrower, earlier question:

> Is the NeuroDrive environment and observation contract learnable at all?

Before investing in biologically-inspired learning algorithms (the actual project goal), the environment, reward shaping, observation vector, and episode lifecycle all need validation. If a standard, well-understood algorithm cannot learn to drive on this track with this observation design, one of these components is broken.

A2C serves as the validation baseline. If A2C cannot learn, the problem is the environment. If A2C can learn, the environment contract is sound and the project can proceed to the biologically-motivated work.

---

## Why Actor-Critic?

The first choice was the algorithm *family*, not the specific variant.

**Why not Q-learning / value-only methods (DQN)?**

NeuroDrive has a continuous action space: steering ∈ [-1, 1], throttle ∈ [0, 1]. Value-only methods like DQN are designed for discrete action spaces. Continuous DQN variants (DDPG, TD3, SAC) require a separate action maximisation step and are substantially more complex. An actor-critic method — which directly represents and optimises a stochastic continuous policy — is a more natural fit.

**Why not a pure policy gradient (REINFORCE)?**

REINFORCE estimates gradients using full Monte Carlo returns. The variance of this estimator is very high, especially in longer episodes. The car might drive well for 30 seconds and then crash; the REINFORCE gradient would attribute the crash equally to all 1800 actions taken before it. This makes learning slow and unstable.

Actor-critic methods replace the Monte Carlo return with a learned value-function baseline, which substantially reduces variance. A2C uses GAE (Generalised Advantage Estimation) to further control the variance-bias trade-off. This is the right choice for a dense-reward, continuous-control environment.

---

## Why A2C Specifically?

Within the actor-critic family, the choices were A2C, PPO, SAC, and others. A2C was chosen for several reasons:

### 1. Simplicity of Implementation

NeuroDrive implements everything from scratch in Rust — no ML frameworks, no RL libraries. This is a core project constraint (the handwritten implementation is itself an educational artefact).

A2C is the simplest actor-critic variant:
- No clipping or trust-region constraints (PPO)
- No separate replay buffer or entropy-tuned temperature (SAC)
- No delayed target network updates or noise injection (TD3)
- Just: collect rollout → compute GAE → gradient update → repeat

The implementation ladder from A2C to a working policy is much shorter than any of the alternatives.

### 2. On-Policy Design Matches the Project

A2C is an on-policy algorithm: it uses data collected by the current policy to update the current policy, then discards it. This matches NeuroDrive's interaction model:
- The car drives for a rollout horizon of steps
- Those steps are used for one update
- The buffer is cleared and the next rollout begins

Off-policy algorithms (SAC, DDPG) require a replay buffer that stores experience from many past policy versions. This adds complexity and changes the timing model significantly. On-policy A2C fits the fixed-tick, single-car runtime more cleanly.

### 3. Dense Reward Compatibility

A2C benefits from dense, informative rewards. NeuroDrive's reward design (progress gain, time penalty, heading-speed penalty, crash penalty, lap bonus) provides a reward signal nearly every tick. Dense rewards are where A2C tends to perform well compared to sparse-reward settings.

### 4. Handwritten Code Validation

Writing the A2C update from scratch (GAE, policy gradient, Huber loss, entropy, gradient clipping, Adam) is itself a learning objective for the project. Understanding each mathematical piece deepens understanding of what the biological architecture must eventually replace.

---

## The Trade-offs A2C Brings

A2C is not the best possible algorithm for this environment. It was chosen as the *validation baseline*, not the *optimal controller*.

### Sample Inefficiency

On-policy methods discard all experience after each update. Every sample from the environment is used exactly once. This is wasteful compared to off-policy methods that can replay old experience many times.

For a single car at 60 Hz, sample inefficiency means long training times. If the agent needs 1 million steps to learn, that is approximately 4.6 hours of real-time simulation (at 60 Hz). Off-policy methods can often learn comparable policies with fewer environment steps.

**Mitigation in NeuroDrive context:** The project plans a vectorised trainer (25 simultaneous cars) as an A2C-specific extension. This does not improve sample efficiency per environment step, but it improves wall-clock efficiency by running 25 environments in parallel.

### Single-Track Overfitting Risk

With one hard-coded circuit and one car, A2C can learn brittle circuit-specific behaviours. A policy that has memorised "at this visual pattern, turn left" is not learning to drive — it is learning to recognise the track. This cannot be detected until generalisation is tested on a new track.

This is why Milestone 6 (multi-track generalisation) exists. The baseline is intentionally validated on one track first before generalisation is demanded.

### High Sensitivity to Implementation Details

The research literature shows that on-policy actor-critic performance is extremely sensitive to implementation choices: learning rates, gradient clipping, initialisation, advantage normalisation, entropy coefficient. A handwritten implementation has more surface area for subtle bugs than a battle-tested framework baseline.

NeuroDrive's A2C is still better described as a validation harness than a trusted high-performance policy learner.

### Not the Final Architecture

The most important trade-off: A2C is explicitly a temporary system. The project's eventual goal is to replace it with a biologically-inspired local plasticity architecture. A2C must therefore remain modular — it should not become so deeply embedded in the runtime that removing it later requires major surgery.

The `Brain` trait in `brain/types.rs` is the architectural insurance: any controller that implements `Brain` can replace A2C without changing the environment or agent layers.

---

## What A2C Validates

When A2C successfully learns to drive on NeuroDrive's track, it validates:

1. **Reward design:** The reward components collectively provide enough signal for gradient-based learning.
2. **Observation contract:** The 23-dimensional observation vector contains enough information for a policy to learn good driving behaviour.
3. **Episode lifecycle:** Reset, progress measurement, lap detection, and terminal conditions behave correctly over thousands of episodes.
4. **Fixed-tick pipeline:** The SimSet ordering is correct; the A2C learning loop operates at the right temporal position in the pipeline.
5. **Handwritten ML primitives:** Linear, ReLU, Adam, and backpropagation are correctly implemented.

If A2C demonstrates clear learning (increasing returns, decreasing crash rate, improving best-progress), all of these components can be considered validated. The project can then invest in the more experimental biological learning architecture with confidence.

---

## Related Files

- `project/systems/a2c-brain.md` — the implementation
- `project/comparisons/a2c-vs-ppo.md` — why PPO was not chosen
- `project/evolution/from-baseline-to-brain.md` — how A2C transitions to the biological architecture
- `concepts/core/actor-critic-architecture.md` — the algorithm family
- `concepts/core/advantage-estimation.md` — the GAE component
