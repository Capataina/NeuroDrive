# Glossary

A comprehensive, alphabetically-ordered glossary of every technical term used across the NeuroDrive learning curriculum. Each entry gives a clear definition, a concrete example, and references to where the term is explored in depth.

---

### Activation Function
A non-linear function applied element-wise after a linear transformation in a neural network. Without activation functions, stacking linear layers would collapse into a single linear transformation (because the composition of two linear functions is still linear). Common examples: ReLU (max(0, x)), Tanh, Sigmoid. In NeuroDrive, the actor and critic MLPs use ReLU activations between hidden layers.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`

### Actor
The component of an actor-critic architecture that learns the *policy* — a mapping from states to action probabilities (or distribution parameters). In NeuroDrive, the actor is a 2-hidden-layer MLP (14→64→64→2) that outputs the mean of a Gaussian distribution over steering and throttle.
See: `concepts/core/reinforcement_learning/a2c.md`

### Actor-Critic
A reinforcement learning architecture that combines two learners: an *actor* (learns the policy) and a *critic* (learns the value function). The critic's value estimates are used to compute advantages, which reduce variance in the policy gradient. NeuroDrive uses separate actor and critic MLPs with independent optimisers.
See: `concepts/core/reinforcement_learning/a2c.md`

### Adam (Adaptive Moment Estimation)
An optimisation algorithm that maintains per-parameter exponential moving averages of both the gradient (first moment, m) and the squared gradient (second moment, v). Update rule: θ ← θ − lr · m̂ / (√v̂ + ε), where m̂ and v̂ are bias-corrected estimates. Adam adapts the learning rate for each parameter individually, making it more robust than vanilla SGD. NeuroDrive implements Adam from scratch in `src/brain/common/optim.rs`.
See: `concepts/core/neural_networks/weight_initialisation_and_optimisers.md`

### Advantage
A measure of how much better an action is compared to the average action in a given state. Defined as A(s, a) = Q(s, a) − V(s), where Q is the action-value and V is the state-value. If the advantage is positive, the action was better than expected; if negative, worse. Used to weight policy gradient updates — good actions get reinforced, bad actions get suppressed.
See: `concepts/core/reinforcement_learning/advantage_estimation_gae.md`

### Affine Transformation
A linear transformation followed by a translation: y = Wx + b, where W is a weight matrix, x is the input vector, and b is a bias vector. Each layer in an MLP performs an affine transformation before applying an activation function.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`

### Arc-Length Parameterisation
A way of describing a curve by the distance travelled along it from a reference point, rather than by coordinates. NeuroDrive's track centreline uses cumulative arc-length so that progress can be measured as a fraction of total track length. If the total centreline is 500 units long and the car is at arc-length 250, its progress fraction is 0.5.
See: `project_specific/implementations/progress_and_lap_system.md`

### Backpropagation
The algorithm for computing gradients of a loss function with respect to all parameters in a neural network. It applies the chain rule of calculus layer-by-layer in reverse order (from output to input). Each layer stores its input during the forward pass ("cache"), then uses that cache during the backward pass to compute local gradients. NeuroDrive implements this manually — there is no autograd library.
See: `concepts/core/neural_networks/backpropagation.md`

### Bevy
A data-driven game engine written in Rust, built around an Entity-Component-System (ECS) architecture. NeuroDrive uses Bevy for its simulation loop, rendering, UI, and system scheduling. Version 0.18.0 is used.
See: `concepts/domain_patterns/ecs_architecture/entity_component_system.md`

### Bias (Neural Network)
A learnable parameter added to the output of a linear transformation: y = Wx + b. The bias allows the network to shift its activation function horizontally, enabling it to fit functions that don't pass through the origin. Each layer in NeuroDrive's MLP has a bias vector.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`

### Bias Correction (Adam)
In the Adam optimiser, the moment estimates m and v are initialised to zero, which biases them towards zero early in training. Bias correction compensates by dividing by (1 − β^t), where t is the timestep. This ensures accurate estimates during the first few updates.
See: `concepts/core/neural_networks/weight_initialisation_and_optimisers.md`

### Binary Search Refinement
A technique used in NeuroDrive's raycast system to find precise road boundaries. After coarse ray marching identifies the approximate boundary (the last on-road point and the first off-road point), 8 iterations of bisection narrow the boundary position to approximately 0.012-unit precision (3 / 2^8).
See: `concepts/domain_patterns/agent_perception/raycast_observation.md`

### Chain Rule
A calculus rule for computing derivatives of composed functions: d/dx f(g(x)) = f'(g(x)) · g'(x). In neural networks, the chain rule is applied repeatedly through layers to compute how each parameter affects the final loss. This is the mathematical foundation of backpropagation.
See: `concepts/core/neural_networks/backpropagation.md`, `concepts/foundations/calculus_and_gradients.md`

### Centreline
The reference path running through the middle of the track. NeuroDrive constructs this as a closed polyline from tile-grid connectivity. It is used for progress measurement, heading error computation, reward shaping, and debug visualisation — but it is explicitly excluded from the agent's observation vector to prevent privileged information leakage.
See: `project_specific/implementations/progress_and_lap_system.md`

### Component (ECS)
A data-only struct attached to an entity. Components have no behaviour — they are pure data. In Bevy/NeuroDrive, `Car`, `TrackProgress`, `SensorReadings`, and `ObservationVector` are all components. Systems query for entities with specific component combinations.
See: `concepts/domain_patterns/ecs_architecture/entity_component_system.md`

### Cosine Similarity
A measure of similarity between two vectors based on the angle between them: cos(θ) = (A · B) / (||A|| × ||B||). Returns 1.0 for identical directions, 0.0 for perpendicular, −1.0 for opposite. Not directly used in NeuroDrive but a foundational concept for understanding vector-space representations.
See: `concepts/foundations/linear_algebra_for_ml.md`

### Critic
The component of an actor-critic architecture that learns the *value function* — an estimate of expected future return from a given state. In NeuroDrive, the critic is a separate 2-hidden-layer MLP (14→64→64→1) that outputs a single scalar value estimate. Its predictions are used to compute TD residuals and advantages for the actor's policy gradient.
See: `concepts/core/reinforcement_learning/value_functions_and_critics.md`

### Dead ReLU
A neuron whose ReLU activation always outputs zero because its pre-activation input is consistently negative. Dead neurons contribute nothing to the network's output and receive zero gradients, so they can never recover. NeuroDrive tracks the dead-ReLU fraction per hidden layer as a network health diagnostic.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`

### Deterministic Simulation
A simulation where identical inputs always produce identical outputs, regardless of when or where the simulation runs. NeuroDrive achieves this through fixed 60 Hz timesteps, explicit system ordering via `SimSet`, and pure-function physics stepping. Determinism enables replay, debugging, and reproducible experiments.
See: `concepts/domain_patterns/simulation/deterministic_simulation.md`

### Discount Factor (γ)
A scalar in [0, 1] that determines how much future rewards are valued relative to immediate rewards. The discounted return is G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + .... A discount factor of 0.99 means a reward 100 steps in the future is worth 0.99^100 ≈ 0.366 of its face value. NeuroDrive uses γ = 0.99.
See: `concepts/core/reinforcement_learning/markov_decision_processes.md`

### Dot Product
An algebraic operation on two vectors that produces a scalar: a · b = Σ aᵢbᵢ. Geometrically, it equals ||a|| × ||b|| × cos(θ), where θ is the angle between the vectors. The dot product is the fundamental operation in neural network linear layers (each output neuron computes a dot product of weights and inputs).
See: `concepts/foundations/linear_algebra_for_ml.md`

### Eligibility Trace
A synapse-level memory that records "this synapse was recently active." In biological learning, eligibility traces bridge the gap between a synaptic event and a later reward signal. The trace decays over time: e ← λe + f(pre, post). When a reward signal arrives, the weight change is proportional to the trace: Δw = η · δ · e. This is the mechanism NeuroDrive plans to use for local plasticity (Milestones 2+).
See: `concepts/core/neuroscience/stdp_and_eligibility_traces.md`

### Entity (ECS)
A unique identifier (essentially an integer ID) that groups components together. An entity is not an object — it is an address. In NeuroDrive, the car is a single entity with components `Car`, `Transform`, `Sprite`, `TrackProgress`, `SensorReadings`, and `ObservationVector`.
See: `concepts/domain_patterns/ecs_architecture/entity_component_system.md`

### Entropy (Policy)
A measure of the randomness/uncertainty in a probability distribution. For a Gaussian policy with standard deviation σ, the differential entropy is H = 0.5 + 0.5·ln(2π) + ln(σ). Higher entropy means more exploration (wider action distribution). NeuroDrive adds an entropy bonus to the policy loss to prevent premature convergence: L_entropy = −entropy_coef × H.
See: `concepts/core/reinforcement_learning/a2c.md`

### Episode
A complete sequence of interactions from an initial state to a terminal condition. In NeuroDrive, an episode starts when the car is placed at the spawn point and ends when the car crashes, times out, or completes a lap. The agent's brain persists across episodes — it is "one brain, one lifetime."
See: `concepts/domain_patterns/simulation/episode_based_rl_environments.md`

### Epoch
A complete pass through the entire training dataset. In on-policy RL like A2C, the term is less relevant — data is collected and used once (on-policy), then discarded. NeuroDrive's A2C collects a 2048-step rollout, performs one update, and discards the data.
See: `concepts/core/reinforcement_learning/a2c.md`

### Euler Integration
The simplest numerical integration method: x_{t+1} = x_t + v_t · Δt. It approximates the continuous dynamics by taking a single linear step per timestep. NeuroDrive uses Euler integration for both position (from velocity) and velocity (from acceleration). At 60 Hz with small forces, Euler is sufficiently accurate and perfectly deterministic.
See: `project_specific/implementations/car_physics_model.md`

### Explained Variance
A diagnostic metric for the critic's quality: EV = 1 − Var(G − V) / Var(G), where G is the true return and V is the critic's prediction. EV = 1.0 means the critic perfectly predicts returns; EV ≤ 0 means the critic is no better than predicting the mean. NeuroDrive computes this after each A2C update.
See: `concepts/core/reinforcement_learning/value_functions_and_critics.md`

### Fixed Timestep
A simulation design where the physics update runs at a constant frequency regardless of rendering frame rate. NeuroDrive uses 60 Hz (Δt = 1/60 s). This ensures determinism: the same actions always produce the same trajectory, unlike variable-timestep designs where frame rate fluctuations cause physics drift.
See: `concepts/domain_patterns/simulation/deterministic_simulation.md`

### GAE (Generalised Advantage Estimation)
An algorithm for computing advantages that interpolates between high-bias/low-variance (TD(0)) and low-bias/high-variance (Monte Carlo) estimates. It uses a parameter λ ∈ [0, 1]: GAE_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}, where δ_t is the TD residual. NeuroDrive uses λ = 0.95. The recursive computation is: GAE_t = δ_t + γ·λ·(1−done)·GAE_{t+1}.
See: `concepts/core/reinforcement_learning/advantage_estimation_gae.md`

### Gaussian Distribution (Normal Distribution)
A probability distribution defined by mean μ and standard deviation σ: p(x) = (1/√(2πσ²)) exp(−(x−μ)²/(2σ²)). NeuroDrive's actor outputs the mean of a Gaussian for each action dimension; the standard deviation is a separate learnable parameter (log-space). Actions are sampled from this Gaussian during training.
See: `concepts/foundations/probability_and_distributions.md`

### Glorot Initialisation (Xavier Initialisation)
A weight initialisation strategy that samples weights from U[−√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))], where fan_in is the number of input units and fan_out is the number of output units. This keeps the variance of activations roughly constant across layers, preventing vanishing or exploding signals at initialisation. NeuroDrive uses Glorot uniform initialisation for all MLP layers.
See: `concepts/core/neural_networks/weight_initialisation_and_optimisers.md`

### Gradient
The vector of partial derivatives of a function with respect to its parameters. The gradient points in the direction of steepest increase. In optimisation, we move parameters in the *negative* gradient direction to minimise a loss function. For a scalar function f(w₁, w₂), the gradient is ∇f = (∂f/∂w₁, ∂f/∂w₂).
See: `concepts/foundations/calculus_and_gradients.md`

### Gradient Descent
An iterative optimisation algorithm: θ_{t+1} = θ_t − η·∇L(θ_t), where η is the learning rate and ∇L is the gradient of the loss. Each step moves parameters slightly in the direction that reduces the loss. NeuroDrive uses Adam (a variant of gradient descent with adaptive learning rates) for the A2C baseline.
See: `concepts/foundations/calculus_and_gradients.md`

### Heading Error
The angular difference between the car's forward direction and the track centreline tangent at the nearest point. Measured as a signed angle in [−π, π]. A heading error of 0 means the car is perfectly aligned with the track. NeuroDrive includes heading error in the observation vector (normalised to [−1, 1] by dividing by π).
See: `project_specific/implementations/observation_system.md`

### Hebbian Plasticity
A synaptic learning rule where connections strengthen when pre- and post-synaptic neurons are co-active: "neurons that fire together wire together." Formalised as Δw_ij = η · pre_i · post_j. This is the foundation of biological learning and the starting point for NeuroDrive's planned brain-inspired mechanisms (Milestone 2+).
See: `concepts/core/neuroscience/hebbian_plasticity.md`

### Log-Probability
The natural logarithm of a probability: log p(x). For a Gaussian with mean μ and std σ: log p(x) = −0.5·[(x−μ)²/σ² + ln(2π) + 2·ln(σ)]. Log-probabilities are used instead of raw probabilities because: (1) they are numerically stable for very small probabilities, (2) products become sums (log(ab) = log a + log b), and (3) the policy gradient theorem uses log-probabilities directly.
See: `concepts/core/reinforcement_learning/policy_gradients.md`

### Loss Function
A scalar function that measures how wrong a model's predictions are. The optimiser minimises this function. In A2C, there are two losses: the *policy loss* (−advantage × log_prob, encouraging good actions) and the *value loss* (0.5 × (V(s) − G_t)², encouraging accurate value predictions).
See: `concepts/core/reinforcement_learning/a2c.md`

### Markov Decision Process (MDP)
A mathematical framework for sequential decision-making defined by (S, A, P, R, γ): states S, actions A, transition probabilities P(s'|s,a), reward function R(s,a,s'), and discount factor γ. The Markov property states that the future depends only on the current state, not on the history of how we got there. NeuroDrive's racing environment can be formalised as an MDP.
See: `concepts/core/reinforcement_learning/markov_decision_processes.md`

### MLP (Multi-Layer Perceptron)
A neural network consisting of fully connected (linear) layers with non-linear activations between them. Data flows in one direction: input → hidden layers → output. NeuroDrive's actor MLP is: Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2). The critic MLP has the same structure but outputs a single scalar.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`

### Neuromodulation
A biological mechanism where a diffuse chemical signal (e.g., dopamine) modulates synaptic plasticity across many synapses simultaneously. In computational terms, it acts as a global "gate" that determines which local synaptic changes should be consolidated. NeuroDrive models this as the TD error δ = r + γV(s') − V(s), analogous to dopamine-based reward prediction error.
See: `concepts/core/neuroscience/neuromodulation_and_dopamine.md`

### Observation Vector
A fixed-size numerical array that represents everything the agent can perceive about the environment. In NeuroDrive, this is a 14-dimensional vector: 11 normalised ray distances, speed, heading error, and angular velocity. Critically, it excludes privileged information like track progress, centreline distance, or curvature lookahead.
See: `project_specific/implementations/observation_system.md`

### On-Policy
A property of RL algorithms that use only data collected by the current policy for learning. After each update, old data is discarded because it was generated by a different (now outdated) policy. A2C is on-policy. This contrasts with off-policy methods (like SAC or DQN) that can reuse old data from a replay buffer.
See: `concepts/core/reinforcement_learning/a2c.md`

### Plugin (Bevy)
A modular unit that adds resources, systems, and configuration to a Bevy application. Plugins encapsulate domain-specific concerns. NeuroDrive has six plugins: MonacoPlugin (track), AgentPlugin (sensors/actions), BrainPlugin (A2C), AnalyticsPlugin (tracking/export), GamePlugin (physics/episodes), DebugPlugin (overlays/HUD).
See: `concepts/domain_patterns/ecs_architecture/entity_component_system.md`

### Policy
A function (or distribution) that maps states to actions: π(a|s). A deterministic policy outputs a single action; a stochastic policy outputs a probability distribution over actions. NeuroDrive's policy is a stochastic Gaussian parameterised by the actor MLP's output (mean) and learnable log-std parameters.
See: `concepts/core/reinforcement_learning/policy_gradients.md`

### Policy Gradient
A family of RL algorithms that optimise the policy directly by estimating the gradient of expected return with respect to policy parameters: ∇J(θ) = E[∇log π(a|s; θ) · R]. Intuitively: increase the probability of actions that led to high returns, decrease the probability of actions that led to low returns.
See: `concepts/core/reinforcement_learning/policy_gradients.md`

### Progress Fraction
The car's position along the track expressed as a value in [0, 1], computed by projecting the car's world position onto the centreline and dividing the arc-length by the total centreline length. A progress fraction of 0.5 means the car is halfway around the track.
See: `project_specific/implementations/progress_and_lap_system.md`

### Ray Marching
A technique for finding intersections by stepping along a ray at fixed intervals and checking whether each sample point satisfies a condition (e.g., is on road). NeuroDrive marches rays at 3-unit intervals from the car, checking `is_road_at()` at each step. When a step transitions from on-road to off-road, binary search refinement pinpoints the exact boundary.
See: `concepts/domain_patterns/agent_perception/raycast_observation.md`

### ReLU (Rectified Linear Unit)
An activation function: f(x) = max(0, x). Its gradient is 1 for x > 0 and 0 for x ≤ 0. ReLU is popular because it is computationally cheap, does not saturate for positive inputs, and produces sparse activations. Its main risk is "dying ReLU" — neurons that get stuck outputting zero. NeuroDrive uses ReLU between hidden layers.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`

### Resource (Bevy)
A globally-accessible, singleton piece of state in the ECS world. Unlike components (which are per-entity), resources are shared across all systems. In NeuroDrive, `A2cBrain`, `EpisodeState`, `ActionState`, `EpisodeTracker`, and `DebugOverlayState` are all resources.
See: `concepts/domain_patterns/ecs_architecture/entity_component_system.md`

### Return (G)
The total discounted reward from time t onwards: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + .... In practice, computed recursively as G_t = r_t + γ·(1−done)·G_{t+1}. The return is what the value function tries to predict and what the policy tries to maximise.
See: `concepts/core/reinforcement_learning/markov_decision_processes.md`

### Reward Shaping
The practice of designing the reward function to provide dense, informative learning signals. NeuroDrive's reward has three components: (1) progress reward (positive for new best-progress within the episode), (2) time penalty (small negative per tick to discourage stalling), and (3) terminal rewards (crash penalty, lap bonus). This decomposition ensures the agent always receives gradient-rich signals.
See: `concepts/domain_patterns/simulation/episode_based_rl_environments.md`

### Rollout Buffer
A data structure that stores a sequence of (state, action, reward, done, value, log_prob) tuples collected during on-policy interaction. Once enough transitions are collected (2048 in NeuroDrive), GAE is computed and the model is updated. The buffer is then cleared because the data is from the old policy.
See: `concepts/core/reinforcement_learning/a2c.md`

### STDP (Spike-Timing Dependent Plasticity)
A biological synaptic learning rule where the relative timing of pre- and post-synaptic spikes determines the direction of weight change. Pre-before-post (causal) strengthens the synapse; post-before-pre (anti-causal) weakens it. This timing sensitivity makes STDP more selective than basic Hebbian learning. Planned for NeuroDrive Milestone 4.
See: `concepts/core/neuroscience/stdp_and_eligibility_traces.md`

### Structural Plasticity
The biological mechanism of forming new synapses and pruning existing ones based on activity patterns. Unlike synaptic plasticity (which changes weights), structural plasticity changes the topology of the neural graph. Planned for NeuroDrive Milestone 5.
See: `concepts/core/neuroscience/structural_plasticity.md`

### System (ECS)
A function that operates on entities with specific component combinations. Systems contain all the behaviour; components and entities contain none. In Bevy, systems are regular Rust functions that declare their data dependencies through parameter types. Example: `car_physics_system` queries for entities with `Car` and `Transform` components.
See: `concepts/domain_patterns/ecs_architecture/entity_component_system.md`

### SystemSet (Bevy)
A named group that defines execution ordering for systems within a schedule. NeuroDrive's `SimSet` defines four ordered phases: Input → Physics → Collision → Measurement. Systems declare which set they belong to, and Bevy ensures they run in the correct order.
See: `concepts/domain_patterns/simulation/deterministic_simulation.md`

### TD Residual (Temporal Difference Error)
The one-step prediction error of the value function: δ_t = r_t + γ·V(s_{t+1}) − V(s_t). If δ > 0, the outcome was better than expected; if δ < 0, worse. The TD residual is the building block of GAE and is analogous to the biological dopamine reward prediction error signal.
See: `concepts/core/reinforcement_learning/advantage_estimation_gae.md`

### Value Function
A function that estimates the expected return from a state: V(s) = E[G_t | s_t = s]. The value function answers: "how good is it to be in this state?" In NeuroDrive, the critic MLP learns V(s) from experience. The value function is used to compute advantages (A = Q − V) and to bootstrap returns at the end of a rollout.
See: `concepts/core/reinforcement_learning/value_functions_and_critics.md`

### Weight (Neural Network)
A learnable parameter that determines the strength of the connection between neurons in adjacent layers. In a linear layer y = Wx + b, the weight matrix W has shape [out_dim × in_dim]. Learning consists of adjusting weights to minimise the loss function. NeuroDrive stores weights as `Vec<Vec<f32>>` in row-major order.
See: `concepts/core/neural_networks/forward_pass_and_layers.md`
