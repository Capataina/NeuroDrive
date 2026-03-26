# Glossary

Definitions for all technical terms used across the NeuroDrive learning archive. Entries are listed alphabetically. Each definition includes a cross-reference to the file where the term is treated in depth.

---

## A

**A2C (Advantage Actor-Critic)**
A synchronous on-policy actor-critic reinforcement learning algorithm. Collects a rollout of experience, computes advantages using GAE, then updates both actor and critic via gradient descent. NeuroDrive's current autonomous controller is a handwritten A2C implementation.
→ `concepts/core/actor-critic-architecture.md`, `project/systems/a2c-brain.md`

**Action space**
The set of all possible actions an agent can take. NeuroDrive has a 2D continuous action space: steering ∈ [-1, 1] and throttle ∈ [0, 1].
→ `concepts/core/continuous-control.md`, `project/systems/agent-interface.md`

**ActionState**
A Bevy resource in `src/agent/action.rs` that stores the `desired` action (what the controller outputs) and the `applied` action (what physics uses). The separation allows optional smoothing without changing either interface.
→ `project/systems/agent-interface.md`

**Advantage function A(s,a)**
The advantage measures how much better or worse action a is compared to the average action in state s: `A(s,a) = Q(s,a) - V(s)`. Used in the policy gradient to reduce variance. NeuroDrive uses GAE to estimate advantages.
→ `concepts/core/advantage-estimation.md`

**Advantage normalisation**
After computing all advantages in a rollout batch, subtract the batch mean and divide by the batch standard deviation. Ensures advantages are zero-centred and unit-variance within each update, stabilising the policy gradient magnitude.
→ `concepts/core/advantage-estimation.md`

**AgentMode**
A Bevy resource in `src/brain/types.rs` that switches between `Keyboard` and `Ai` control. F4 toggles the mode at runtime. Mode toggle clears the A2C rollout buffer to prevent mixed-controller trajectories.
→ `project/systems/a2c-brain.md`

**Analytics**
NeuroDrive's data capture and export subsystem. Captures per-tick traces, episode summaries, and A2C training stats during runtime, then exports to JSON and Markdown reports on exit.
→ `project/systems/analytics-system.md`

**Angular velocity**
The car's yaw rate — how quickly it is rotating. Included in the observation vector (index 14) as it helps the policy distinguish between controlled cornering and spinning.
→ `references/observation-vector-reference.md`

---

## B

**Backpropagation**
An algorithm for computing the gradient of a loss function with respect to network parameters by applying the chain rule backwards through the computation graph. Used by A2C to update the actor and critic. Notably absent from the planned biological architecture.
→ `concepts/foundations/neural-networks.md`

**Bellman equation**
A recursive relationship expressing the value of a state in terms of the immediate reward and the value of successor states: `V(s) = E[r + γV(s')]`. The foundation of temporal difference learning.
→ `concepts/core/reinforcement-learning.md`

**Bevy ECS**
The Entity Component System framework NeuroDrive is built on. Entities are IDs, components are data structs attached to entities, systems are functions over queries, resources are global singletons. Scheduling in Bevy's `FixedUpdate` determines the order systems execute.
→ `concepts/foundations/bevy-ecs-primer.md`

**Bootstrap value**
When a rollout ends at a non-terminal step (horizon reached), the value of the final state V(s_N) is estimated by the critic and used to initialise the GAE recurrence. If the rollout ends at a terminal step, the bootstrap value is 0.
→ `concepts/core/advantage-estimation.md`

**Brain trait**
A minimal Rust trait in `src/brain/types.rs` defining the controller interface: `fn act(&mut self, obs: &ObservationVector) -> CarAction`. Both A2C and future biological controllers implement this trait.
→ `project/systems/a2c-brain.md`, `project/evolution/from-baseline-to-brain.md`

---

## C

**CarAction**
A struct containing `steering: f32` and `throttle: f32`. The stable motor output format written by all controllers and read by the physics system.
→ `project/systems/agent-interface.md`

**Catastrophic forgetting**
A failure mode in neural networks where training on a new task overwrites the representations learned for previous tasks. Critical concern for Milestone 6 (multi-track generalisation) under the one-brain-one-lifetime design principle.
→ `concepts/advanced/continual-learning.md`

**Centreline**
A closed polyline tracing the centre of the driveable track path. Used for progress measurement (arc-length projection), lookahead sampling, spawn pose, and centreline-relative observation features.
→ `project/systems/environment-system.md`

**Chain rule**
A calculus rule for differentiating compositions of functions: `d/dx f(g(x)) = f'(g(x)) * g'(x)`. The mathematical basis for backpropagation.
→ `concepts/foundations/neural-networks.md`

**Churn rate**
In structural plasticity: the number of synapses added plus synapses removed per N ticks. High early churn indicates reorganisation; declining churn indicates stabilisation.
→ `concepts/advanced/structural-plasticity.md`

**Collision detection**
The system in `src/game/collision.rs` that checks the four corners of the rotated car rectangle against `TrackGrid`. Emits `CollisionEvent` when any corner is off-road.
→ `project/systems/environment-system.md`

**CollisionEvent**
A zero-payload Bevy event emitted when any car corner leaves the road. Consumed by `episode_loop_system` to trigger the crash penalty and episode termination.
→ `project/systems/environment-system.md`

**Critic**
The value-function network in an actor-critic architecture. In NeuroDrive: `23 → 64 → 64 → 1`, producing a scalar V(s) estimate. Trained to minimise Huber loss against GAE return targets.
→ `concepts/core/actor-critic-architecture.md`, `project/systems/a2c-brain.md`

**Curriculum learning**
A training strategy where easier tasks are presented before harder ones. Planned for Milestone 6: simpler tracks before more complex ones.
→ `project/evolution/milestone-roadmap.md`

---

## D

**Dead neuron**
A ReLU unit that outputs zero for all inputs in a training batch. Its gradient is zero, so no learning signal flows through it. Tracked in `A2cTrainingStats.dead_relu_fraction`.
→ `concepts/foundations/neural-networks.md`

**Discount factor (γ)**
A value between 0 and 1 that reduces the weight of future rewards in the return. NeuroDrive uses γ = 0.99, meaning rewards 100 steps in the future are worth `0.99^100 ≈ 0.37` of immediate rewards.
→ `concepts/core/reinforcement-learning.md`

**Dopamine**
A neuromodulator in the biological brain whose release pattern tracks reward prediction error (δ). The inspiration for NeuroDrive's neuromodulatory δ signal that gates synaptic plasticity in the planned biological architecture.
→ `concepts/advanced/neuromodulation.md`

---

## E

**Eligibility trace (e_ij)**
A per-synapse variable that accumulates recent co-activity. Decays over time: `e_ij ← λ * e_ij + f(x_i, x_j)`. Allows a reward signal arriving after a delay to update the synapses that were recently active, solving temporal credit assignment without backpropagation.
→ `concepts/advanced/eligibility-traces.md`

**Entropy (policy entropy H)**
A measure of how spread out a probability distribution is. High entropy = diverse action selection (more exploration). Low entropy = near-deterministic policy. NeuroDrive adds an entropy bonus to the A2C loss to prevent premature policy collapse.
→ `concepts/core/policy-gradient-methods.md`

**EpisodeConfig**
A Bevy resource storing tunable episode parameters: `max_steps`, `progress_scale`, `time_penalty`, `heading_speed_penalty`, `crash_penalty`, `lap_bonus`, `moving_average_window`.
→ `project/systems/environment-system.md`

**EpisodeState**
A global singleton resource that stores the complete state of the current episode and the most-recently-completed episode. The authoritative source of reward truth and terminal truth.
→ `project/systems/environment-system.md`

**Explained variance**
A diagnostic metric for the critic: `1 - Var(returns - V(s)) / Var(returns)`. Near 1.0 means the critic explains most return variation (good). Near 0 means the critic is uninformative. Negative means the critic is actively wrong.
→ `project/systems/a2c-brain.md`

---

## F

**Fan-in / fan-out**
The number of incoming connections (fan-in) or outgoing connections (fan-out) of a neuron. Bounded in structural plasticity to prevent graph blow-up.
→ `concepts/advanced/structural-plasticity.md`

**Fixed-tick pipeline**
NeuroDrive's deterministic 60 Hz simulation loop. Systems are ordered within four `SimSet` stages: `Input → Physics → Collision → Measurement`.
→ `project/architecture/fixed-tick-pipeline.md`

**Forgetting metric**
A quantitative measure of catastrophic forgetting: `F = Performance(Track A after Track B training) / Performance(Track A before Track B training)`. A value near 1.0 means minimal forgetting.
→ `concepts/advanced/continual-learning.md`

---

## G

**GAE (Generalised Advantage Estimation)**
An advantage estimator that interpolates between pure TD error (low variance, high bias) and Monte Carlo return (high variance, zero bias) using a decay parameter λ. Computes advantages via backwards recurrence with terminal masking.
→ `concepts/core/advantage-estimation.md`

**Gaussian policy**
A policy that outputs a mean and standard deviation for each action dimension, then samples actions from the resulting Gaussian distribution. Used by NeuroDrive's A2C for continuous steering and throttle control.
→ `concepts/core/continuous-control.md`

**Glorot uniform initialisation**
Weight initialisation that samples from `Uniform(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`. Designed to keep activation variance approximately constant across layers.
→ `concepts/foundations/neural-networks.md`

**Gradient clipping**
A technique that rescales all gradients if their global norm exceeds a threshold before the optimiser step. NeuroDrive clips at 0.5 to prevent explosive updates.
→ `concepts/foundations/optimization-and-gradients.md`

---

## H

**Heading error**
The angle between the car's forward direction and the centreline tangent at the nearest projection point. Observation vector index 13. Positive = car is pointing left of tangent.
→ `references/observation-vector-reference.md`

**Hebbian plasticity**
A local learning rule: "neurons that fire together, wire together." Synapses strengthen when pre- and post-synaptic activities are correlated: `Δw_ij ∝ x_i * x_j`. The foundation of NeuroDrive's planned biological learning architecture.
→ `concepts/advanced/hebbian-plasticity.md`

**HUD**
Heads-Up Display. NeuroDrive's runtime diagnostics panel (F3 toggle) showing current state, moving averages, live A2C training stats, and recent-quarter run assessment.
→ `project/systems/debug-runtime.md`

**Huber loss**
A loss function that is quadratic (like MSE) for small errors and linear (like L1) for large errors. Used for the A2C critic to reduce sensitivity to outlier value targets.
→ `concepts/core/actor-critic-architecture.md`

---

## I

**Importance ratio**
In PPO (not in NeuroDrive's current A2C): `r_t = π_new(a|s) / π_old(a|s)`. Measures how much the current policy differs from the policy that collected the data. PPO clips this ratio to constrain policy updates.
→ `project/comparisons/a2c-vs-ppo.md`

**Integrate-and-fire (LIF) neuron**
A spiking neuron model that accumulates input current as membrane potential and fires a spike when the threshold is crossed. The planned neuron model for Milestone 4's SNN.
→ `project/comparisons/rate-based-vs-spiking.md`

---

## J

**Jacobian correction**
The change-of-variables correction applied to the log-probability of a tanh-squashed Gaussian policy: `log π(a|s) = log N(latent|μ,σ) - Σ_i log(1 - tanh²(latent_i))`. Without this correction, the policy gradient is biased for actions near the action boundaries.
→ `concepts/core/continuous-control.md`, `project/decisions/tanh-squashed-actions.md`

---

## K

**Kinematic car model**
NeuroDrive's physics model. Simulates car motion using steering sensitivity, acceleration, drag, and maximum speed without full rigid-body physics. Deterministic and reproducible.
→ `project/systems/environment-system.md`

---

## L

**Lap completion**
An episode termination condition. Detected when `TrackProgress.fraction` wraps around from near 1.0 to near 0.0. Triggers a +100.0 lap bonus.
→ `project/systems/environment-system.md`

**Lateral offset**
The perpendicular distance from the car's position to the nearest centreline point. Observation vector index 12. Positive = car is to the left of centreline direction.
→ `references/observation-vector-reference.md`

**Learning rate (η)**
A hyperparameter controlling the step size of parameter updates. NeuroDrive's A2C uses `actor_lr = 3e-4` and `critic_lr = 5e-4` for Adam.
→ `concepts/foundations/optimization-and-gradients.md`

**Log-probability**
The natural logarithm of a probability density. Used in the policy gradient: `∇_θ J = E[∇_θ log π(a|s) * A(s,a)]`. More numerically stable than working with raw probabilities.
→ `concepts/foundations/probability-and-distributions.md`

**Lookahead samples**
Four centreline points sampled ahead of the car's current position. Each contributes `heading_delta` and `curvature` features (indices 15–22) to the observation vector.
→ `references/observation-vector-reference.md`, `concepts/domain-patterns/observation-design.md`

**LTD (Long-Term Depression)**
A weakening of synaptic strength. In STDP: occurs when post-synaptic firing precedes pre-synaptic firing (post-before-pre timing).
→ `concepts/advanced/spike-timing-dependent-plasticity.md`

**LTP (Long-Term Potentiation)**
A strengthening of synaptic strength. In STDP: occurs when pre-synaptic firing precedes post-synaptic firing (causal timing).
→ `concepts/advanced/spike-timing-dependent-plasticity.md`

---

## M

**MDP (Markov Decision Process)**
The mathematical framework for sequential decision-making. Defined by states S, actions A, transition probabilities P(s'|s,a), rewards R(s,a,s'), and discount factor γ. NeuroDrive's environment is an MDP.
→ `concepts/core/reinforcement-learning.md`

**Memory consolidation**
The process by which recently learned information is transferred from fast-learning, high-forgetting-risk memory systems (hippocampus) to stable long-term storage (neocortex). The biological motivation for Milestone 7's replay mechanism.
→ `concepts/advanced/continual-learning.md`

**MLP (Multi-Layer Perceptron)**
A feedforward neural network with fully connected layers. NeuroDrive's A2C uses two MLPs: actor (`23 → 64 → 64 → 2`) and critic (`23 → 64 → 64 → 1`).
→ `concepts/foundations/neural-networks.md`

**Module boundaries**
The ownership rules between NeuroDrive's seven subsystems. Key rule: `game/` does not depend on `brain/`. The environment must be controller-agnostic.
→ `project/architecture/module-boundaries.md`

---

## N

**Neuromodulation**
A biological mechanism where a chemical signal (dopamine) is broadcast broadly to modulate synaptic plasticity. The dopamine-like δ signal gates which Hebbian trace changes become lasting weight updates.
→ `concepts/advanced/neuromodulation.md`

**Normalisation**
Scaling raw feature values to a bounded range (typically [-1, 1] or [0, 1]) before using them as neural network inputs. Prevents large-magnitude features from dominating gradient updates.
→ `concepts/domain-patterns/observation-design.md`

---

## O

**ObservationVector**
A Bevy component (`[f32; 23]`) attached to the car entity containing the 23-dimensional normalised policy input. Rebuilt every tick in `SimSet::Measurement` after episode resets.
→ `project/systems/agent-interface.md`, `references/observation-vector-reference.md`

**OBSERVATION_DIM**
The constant `23` defined in `src/agent/observation.rs` and used by the A2C model input layer. Changing this requires coordinated changes in both files.
→ `references/observation-vector-reference.md`

**On-policy learning**
A learning paradigm where the agent uses only data collected by the current policy to update the policy. A2C is on-policy. Data is discarded after each update. Contrasted with off-policy methods (SAC, DDPG) that use a replay buffer.
→ `concepts/core/reinforcement-learning.md`, `project/comparisons/a2c-vs-ppo.md`

**One brain, one lifetime**
NeuroDrive's core design principle. The same neural network (or, in the future, sparse biological graph) learns continuously across all episodes without resetting weights. Weights persist between sessions at Milestone 2+.
→ `README.md`, `project/evolution/from-baseline-to-brain.md`

---

## P

**Policy (π)**
A function mapping states to action probabilities: `π(a|s)`. NeuroDrive's A2C uses a Gaussian policy with tanh squashing.
→ `concepts/core/reinforcement-learning.md`

**Policy gradient**
The gradient of expected return with respect to policy parameters: `∇_θ J = E[∇_θ log π(a|s) * A(s,a)]`. Used to update the actor in A2C.
→ `concepts/core/policy-gradient-methods.md`

**PPO (Proximal Policy Optimisation)**
An on-policy actor-critic algorithm that adds a clipped probability ratio to constrain policy updates. The main alternative to A2C for continuous control. Not currently used in NeuroDrive.
→ `project/comparisons/a2c-vs-ppo.md`

**Progress reward**
The primary reward signal in NeuroDrive. Rewards only new best-episode progress: `max(0, current_fraction - best_fraction_this_episode) * 140.0`.
→ `project/systems/environment-system.md`, `concepts/domain-patterns/reward-shaping.md`

---

## R

**Rate-based network**
A neural network where neurons are characterised by continuous activation values (firing rates) rather than discrete spikes. NeuroDrive's current A2C uses a rate-based MLP. The planned Milestone 2 biological architecture also uses rate-based neurons, with Hebbian rather than backpropagation-based learning.
→ `project/comparisons/rate-based-vs-spiking.md`

**Raycast**
A ray-casting operation that shoots a ray from the car's position in a given direction and returns the distance to the first non-road cell. Used to compute the 11-dimensional ray sensor array.
→ `project/systems/agent-interface.md`, `references/observation-vector-reference.md`

**ReLU (Rectified Linear Unit)**
An activation function: `max(0, x)`. Used in all A2C hidden layers. Gradient is 1 for positive inputs, 0 for negative inputs (dead neuron problem).
→ `concepts/foundations/neural-networks.md`

**Return (G_t)**
The discounted sum of future rewards from time t: `G_t = Σ_{k=0}^∞ γ^k r_{t+k}`. The quantity that RL algorithms seek to maximise.
→ `concepts/core/reinforcement-learning.md`

**Reward prediction error (δ)**
`δ = r + γV(s') - V(s)`. Positive when outcomes are better than expected; negative when worse. The signal used to update both the A2C critic and the planned biological brain's synaptic plasticity.
→ `concepts/core/reinforcement-learning.md`, `concepts/advanced/neuromodulation.md`

**Reward shaping**
The practice of designing reward functions to guide learning toward desired behaviour. NeuroDrive's reward combines progress gain, time penalty, heading-speed penalty, crash penalty, and lap bonus.
→ `concepts/domain-patterns/reward-shaping.md`

**Rollout buffer**
A data structure that stores one rollout's worth of experience: observations, actions, rewards, dones, values, and log-probs. Processed at the end of a rollout to compute GAE and run a training update.
→ `project/systems/a2c-brain.md`

---

## S

**Sample efficiency**
A measure of how much environment experience is needed to achieve a given performance level. On-policy methods like A2C are relatively sample-inefficient because they discard experience after one use.
→ `project/decisions/a2c-as-baseline.md`

**SensorReadings**
A Bevy component containing raw (un-normalised) sensor measurements: ray distances, lateral offset, heading error, angular velocity, lookahead points. Used by debug overlays (raw display) and observation builder (normalised conversion).
→ `project/systems/agent-interface.md`

**SimSet**
An enum defining four ordered system sets in NeuroDrive's `FixedUpdate` schedule: `Input → Physics → Collision → Measurement`. Every fixed-tick system belongs to exactly one set.
→ `project/architecture/fixed-tick-pipeline.md`

**Spiking neural network (SNN)**
A network where neurons communicate via discrete spike events rather than continuous activations. Planned for Milestone 4. Enables STDP and more biologically faithful computation.
→ `project/comparisons/rate-based-vs-spiking.md`

**STDP (Spike-Timing Dependent Plasticity)**
A synaptic learning rule where the sign and magnitude of weight change depend on the relative timing of pre- and post-synaptic spikes. Pre-before-post → LTP (strengthen); post-before-pre → LTD (weaken). Planned for Milestone 4.
→ `concepts/advanced/spike-timing-dependent-plasticity.md`

**Stability-plasticity dilemma**
The fundamental tension in continual learning: high plasticity enables rapid learning but causes forgetting; low plasticity retains old knowledge but prevents learning new things. Biological brains navigate this through multiple memory systems.
→ `concepts/advanced/continual-learning.md`

**Structural plasticity**
Adaptation of neural network topology through formation and elimination of synapses. Contrasted with synaptic weight plasticity (which changes the strength of existing connections). Planned for Milestone 5.
→ `concepts/advanced/structural-plasticity.md`

---

## T

**Tanh squashing**
Applying `tanh` to a latent (unbounded) action sample to constrain it to `(-1, 1)`. Requires a Jacobian correction in the log-probability to avoid biased policy gradients. Used by NeuroDrive's A2C for bounded steering and throttle outputs.
→ `concepts/core/continuous-control.md`, `project/decisions/tanh-squashed-actions.md`

**TD error (temporal difference error)**
See "Reward prediction error (δ)".

**Three-factor learning rule**
The biological synaptic update rule: `Δw_ij = η * δ * e_ij`. Three factors: the learning rate η, the neuromodulatory signal δ, and the eligibility trace e_ij. Each provides different information; none alone is sufficient for credit assignment.
→ `concepts/advanced/eligibility-traces.md`, `concepts/advanced/neuromodulation.md`

**Track**
The static racing environment. Built from tile parts in `src/maps/monaco.rs`. Comprises `TrackGrid` (driveable-area queries), `TrackCenterline` (progress and lookahead), and spawn pose.
→ `project/systems/environment-system.md`

**TrackProgress**
A Bevy component attached to the car storing centreline projection state: arc length `s`, lap fraction, closest point, tangent, and signed distance. Updated every tick by `update_track_progress_system`. Intentionally excluded from the observation vector.
→ `project/systems/environment-system.md`

**Trust region**
A constraint on policy updates that prevents the policy from changing too drastically in a single step. PPO implements this via clipping. A2C does not have a trust region constraint.
→ `project/comparisons/a2c-vs-ppo.md`

---

## V

**Value function V(s)**
The expected cumulative discounted return from state s under a given policy: `V(s) = E[G_t | s_t = s]`. Estimated by the critic network in A2C. Used to compute δ and to compute returns as targets for the critic.
→ `concepts/core/reinforcement-learning.md`

**Vectorised training**
A planned extension where 25 cars run simultaneously in 25 environments, collecting experience in parallel. Would improve wall-clock training efficiency without improving sample efficiency.
→ `project/evolution/milestone-roadmap.md`

---

## W

**Weight decay**
A regularisation technique that adds a penalty proportional to the squared magnitude of weights to the loss function. Equivalent to L2 regularisation. Keeps weights small and prevents explosion.
→ `concepts/advanced/hebbian-plasticity.md`
