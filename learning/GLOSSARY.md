# Glossary

## A

**A2C**
Advantage Actor-Critic. A synchronous actor-critic family method in which a policy and value estimator are trained together.

Plain English:
The policy decides what to do; the critic estimates how good states are; the advantage term tells the policy whether a sampled action was better or worse than expected.

Project relevance:
NeuroDrive currently uses a handwritten A2C-style baseline as its live autonomous controller.

See:
`concepts/core/actor-critic-and-gae.md`, `project/systems/a2c-baseline.md`

**Action smoothing**
A low-pass filter between desired controller output and applied control sent to the physics system.

Plain English:
The controller says what it wants now; smoothing can soften sudden changes before the car actually uses them.

Project relevance:
The interface supports smoothing, but it is disabled by default.

See:
`project/systems/agent-interface.md`

**Advantage**
The relative usefulness of an action compared with the state’s expected value.

Plain English:
It answers: "Was this action better or worse than what I generally expected from here?"

Project relevance:
NeuroDrive’s A2C update uses GAE-derived advantages.

See:
`concepts/core/actor-critic-and-gae.md`

## B

**Baseline**
A simpler reference system used to validate whether a problem setup is learnable before investing in more ambitious approaches.

Project relevance:
The current A2C learner is a baseline, not the intended final architecture.

See:
`project/decisions/why-a2c-exists-in-a-brain-inspired-project.md`

**Bootstrap value**
The critic estimate used to continue return estimation when a rollout ends because of truncation rather than a true terminal state.

Project relevance:
NeuroDrive bootstraps non-terminal rollouts when the update horizon is reached.

See:
`project/systems/a2c-baseline.md`

## C

**Centreline**
The track’s canonical path used for measuring progress, tangent direction, and geometry-relative features.

Plain English:
It is the track’s "middle ribbon" that tells the project what counts as forward movement and where the car sits relative to the ideal path.

Project relevance:
Progress measurement, heading error, lateral offset, and lookahead features all depend on the centreline.

See:
`project/systems/maps-and-centreline.md`

**Credit assignment**
The problem of determining which past states, actions, or synapses deserve blame or credit for later outcomes.

Project relevance:
A2C handles this with returns and advantages; the README’s target biological system would handle it with local eligibility traces gated by neuromodulation.

See:
`concepts/domain-patterns/reward-shaping-and-credit-assignment.md`, `concepts/domain-patterns/brain-inspired-learning-principles.md`

## D

**Determinism**
The property that the same initial conditions and inputs lead to the same behaviour.

Project relevance:
NeuroDrive’s environment core is fairly deterministic, but the current A2C path is not yet meaningfully reproducible because RNG ownership is still ad hoc.

See:
`concepts/core/determinism-and-fixed-timestep-simulation.md`, `project/architecture/data-flow-and-schedule.md`

## E

**Eligibility trace**
A short-lived memory of recent local activity that allows later reinforcement to affect earlier participating connections.

Project relevance:
This is central to the README’s intended biological-learning path, but not yet implemented in code.

See:
`concepts/domain-patterns/brain-inspired-learning-principles.md`

**Episode**
A bounded interaction segment ending in crash, timeout, or lap completion.

Project relevance:
Reward accumulation, resets, moving averages, analytics snapshots, and A2C rollout completion all depend on episode structure.

See:
`project/systems/environment.md`, `project/systems/analytics.md`

## F

**Fixed timestep**
A simulation loop that advances by a constant delta rather than variable frame time.

Project relevance:
NeuroDrive runs at `60 Hz`, which matters for physics stability, reward alignment, observation timing, and replay-friendly reasoning.

See:
`concepts/core/determinism-and-fixed-timestep-simulation.md`

## G

**GAE**
Generalised Advantage Estimation. A method for computing smoother policy-gradient targets using a trade-off between variance and bias.

Project relevance:
NeuroDrive’s A2C baseline uses GAE with `gamma` and `lambda` parameters.

See:
`concepts/core/actor-critic-and-gae.md`

## H

**Heading error**
The angular mismatch between the car’s forward vector and the local track tangent.

Project relevance:
It appears both in observations and in reward shaping via a speed-weighted penalty.

See:
`project/systems/environment.md`, `project/systems/agent-interface.md`

## L

**Local plasticity**
Weight change rules based only on information locally available at a connection or nearby neurons.

Project relevance:
This is part of the README’s end-state ambition and a major reason the project does not intend to remain an A2C repository forever.

See:
`concepts/domain-patterns/brain-inspired-learning-principles.md`

## M

**MDP**
Markov Decision Process. A standard formal model for sequential decision making with states, actions, transitions, rewards, and returns.

Project relevance:
The current A2C baseline fits naturally into this framing even though the README’s future direction aims to move beyond conventional RL abstractions.

See:
`concepts/foundations/continuous-control-and-mdps.md`

## N

**Neuromodulation**
A broadcast signal, often compared with dopamine-like reinforcement, that gates which local changes should consolidate.

Project relevance:
Central to the README’s desired biological-learning story; not yet implemented.

See:
`concepts/domain-patterns/brain-inspired-learning-principles.md`

## O

**Observation vector**
The fixed-size normalised feature vector consumed by controllers.

Project relevance:
Current dimension is `23`, combining ray distances, kinematic state, and centreline lookahead features.

See:
`project/systems/agent-interface.md`

## P

**Policy**
A mapping from observations or states to actions or action distributions.

Project relevance:
The current policy is a handwritten Gaussian actor with tanh-squashed outputs.

See:
`project/systems/a2c-baseline.md`

**Progress fraction**
Normalised position along the track centreline in `[0, 1]`.

Project relevance:
Used by episode reward shaping, lap wrap detection, analytics, and debug displays, but intentionally not leaked directly into the observation vector.

See:
`project/systems/environment.md`, `project/systems/agent-interface.md`

## R

**Reward shaping**
The deliberate design of intermediate reward terms so learning receives denser guidance than a sparse success/failure signal alone.

Project relevance:
NeuroDrive currently mixes progress reward, time penalty, heading-speed penalty, crash penalty, and lap bonus.

See:
`concepts/domain-patterns/reward-shaping-and-credit-assignment.md`

**Rollout buffer**
A transient store of state, action, reward, done, and critic information collected before an update.

Project relevance:
The A2C baseline appends transitions online and updates on horizon or terminal conditions.

See:
`project/systems/a2c-baseline.md`

## S

**Structural plasticity**
Longer-timescale formation, reorganisation, or pruning of network structure.

Project relevance:
Important to the README’s target architecture; not part of the current baseline runtime.

See:
`concepts/domain-patterns/brain-inspired-learning-principles.md`

## T

**Temporal-difference learning**
Updating value estimates using bootstrapped comparisons between current and next predictions.

Project relevance:
The critic component of A2C depends on this style of reasoning.

See:
`concepts/foundations/probability-value-estimation-and-return.md`, `concepts/core/actor-critic-and-gae.md`

## V

**Value function**
An estimate of expected future return from a state.

Project relevance:
The critic in NeuroDrive’s current actor-critic baseline produces scalar value estimates used for GAE and diagnostics.

See:
`concepts/core/actor-critic-and-gae.md`
