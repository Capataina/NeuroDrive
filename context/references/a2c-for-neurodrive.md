# Reference — A2C for NeuroDrive

## Scope / Purpose

- Capture external A2C research and implementation lessons in one durable reference file.
- Translate those lessons into project-specific guidance for NeuroDrive rather than leaving them as generic RL advice.
- Distinguish clearly between source-backed findings and repository-specific inference.

## Current Relevance

- NeuroDrive currently uses a handwritten A2C baseline as its only autonomous learning implementation.
- The project intent in `README.md` is eventually brain-inspired local plasticity, but the near-term engineering need is still a trustworthy baseline that can answer a narrower question:
  - is the current environment and observation contract learnable at all?
- This file is therefore not a decision that “A2C is the final direction”.
- It is a research-backed reference for how far to take A2C, how to evaluate it properly here, and which implementation upgrades are worth the effort in this repository.

## Content

### 1. What A2C actually is

| Topic | Research-backed summary | Why it matters for NeuroDrive |
|---|---|---|
| Actor-critic foundation | Policy-gradient methods improve a parameterised policy directly, while a learned value or advantage estimate reduces variance in the gradient estimate. Actor-critic methods sit in that family rather than in value-only RL. | NeuroDrive’s current A2C is the right class of baseline if the aim is continuous steering/throttle control from learned state features instead of discrete action lookup. |
| A3C origin | The A3C paper established that parallel actor-learners can stabilise training and that actor-critic works in continuous-control settings as well as Atari. | The main conceptual justification for using actor-critic in this repo is sound; the open question is implementation quality, not whether actor-critic is fundamentally mismatched. |
| A2C variant | OpenAI’s Baselines write-up describes A2C as the synchronous version of A3C: multiple actors collect experience, the implementation waits for them, averages updates, and benefits from larger batched computation. OpenAI explicitly reported no evidence that asynchronous noise itself helped. | NeuroDrive’s current implementation is algorithmically “A2C-shaped”, but operationally it is closer to a single-environment online actor-critic. The biggest missing A2C property is synchronous batched rollout collection across multiple actors. |
| GAE role | GAE exists to reduce policy-gradient variance while controlling bias through the `gamma` / `lambda` trade-off. | NeuroDrive already uses GAE, which is the right default for a continuous-control baseline. The remaining issue is not whether to use GAE, but whether the rest of the training stack is strong enough to benefit from it. |

### 2. Core research takeaways

#### 2.1 Foundational lessons

- Sutton et al. establish the core policy-gradient framing in which the policy is represented directly and can be improved using an approximate action-value or advantage function. That is the conceptual base A2C stands on, not an implementation trick.
- Mnih et al. show that actor-critic with parallel actor-learners can train successfully without replay, including on continuous motor-control tasks.
- Schulman et al. show why advantage estimation matters in practice: lower-variance gradients are not optional polish in continuous control, they are one of the main enablers of stable learning.

#### 2.2 Implementation lessons from later on-policy work

- Engstrom et al. show that supposedly “minor” implementation choices can dramatically alter deep policy-gradient performance. Their work is PPO/TRPO-specific, but the lesson generalises directly to A2C-style code: if the baseline is brittle, the cause is often in optimiser, scaling, initialisation, and batching details rather than in the headline algorithm.
- Andrychowicz et al. show the same issue at larger scale for on-policy RL generally: practical performance depends heavily on choices such as architecture separation, activations, initial policy scale, batch construction, and normalisation.

#### 2.3 What consistently matters in practice

The following are source-backed general lessons from the on-policy literature and are highly relevant to A2C in NeuroDrive:

| Implementation choice | Research signal | NeuroDrive implication |
|---|---|---|
| Separate policy and value networks | Large-scale on-policy study found separate value and policy networks performed better on most tested environments. | NeuroDrive already does this. Keep it. |
| Two hidden layers | Large-scale on-policy study found two hidden layers worked well across tested tasks. | NeuroDrive already uses a two-layer actor and two-layer critic. This is adequate for the baseline stage. |
| Tanh activations | Large-scale on-policy study found `tanh` better than `ReLU` in the tested on-policy continuous-control settings. | NeuroDrive currently uses `ReLU`. This is a credible upgrade candidate if the baseline remains unstable after more urgent fixes. |
| Careful initial policy scale | Large-scale on-policy study found initial policy centring and small standard deviation materially affect training. | NeuroDrive currently uses Glorot init plus zero log-std. That is reasonable, but not obviously tuned for “safe driving first” behaviour. |
| Observation and reward normalisation | Engstrom et al. identify normalisation and clipping choices as highly consequential implementation details. | NeuroDrive currently uses fixed feature scaling and clipping, but not running normalisation of observations or returns/advantages beyond standardised advantages. |
| Gradient clipping | Engstrom et al. highlight global gradient clipping as an important practical optimisation. | NeuroDrive already clips actor and critic gradients. Keep it. |
| Learning-rate schedules / annealing | Engstrom et al. report strong sensitivity to optimiser details including Adam LR annealing. | NeuroDrive currently uses fixed learning rates. That is acceptable for a minimal baseline but weak for a stronger one. |

### 3. What A2C is good at, and what it is bad at, in this project

#### 3.1 Where A2C fits NeuroDrive well

- Continuous action space:
  - steering and throttle are naturally continuous, so a Gaussian policy is a better baseline fit than forcing a discrete action grid.
- Dense per-tick feedback:
  - NeuroDrive already has progress, time, heading-risk, crash, and lap terms, which is much friendlier to on-policy actor-critic than a sparse-success-only environment.
- Interpretable observations:
  - the current observation vector is compact, engineered, and geometry-rich, which lowers the sample burden relative to pixels.
- Online fixed-tick interaction:
  - A2C fits a fixed-timestep environment naturally and does not require offline datasets or replay infrastructure to become useful.

#### 3.2 Where A2C fits NeuroDrive badly

- Sample efficiency:
  - on-policy methods throw away data after use, which is expensive in a single-environment setup.
- Single-track overfitting risk:
  - with one hand-built circuit and one car, A2C can learn brittle circuit-specific habits while still looking superficially improved.
- High sensitivity to implementation details:
  - because the repo uses handwritten ML code, there is more room for subtle numerical or optimisation mistakes than in a mature framework baseline.
- Final project mismatch:
  - A2C is not biologically inspired and should not be allowed to become the architectural centre of gravity.

### 4. Minimal, mediocre, and expert A2C implementations for NeuroDrive

This section is an inference from the literature plus direct repository inspection, not a claim copied from one paper.

#### 4.1 Minimal implementation

A minimal A2C for NeuroDrive is good enough to answer only one question:

> can a stable actor-critic improve above manual-randomish behaviour on this environment at all?

Required properties:

- fixed-tick action/reward alignment
- correct terminal handling and bootstrap logic
- GAE
- bounded continuous actions
- separate actor and critic
- gradient clipping
- enough telemetry to detect divergence

Current NeuroDrive status:

- already meets most of this threshold
- strongest remaining minimal-stage gaps:
  - no controlled RNG ownership
  - no multi-environment batching
  - no observation/reward normalisation
  - no explicit behavioural regression tests

Verdict:

- NeuroDrive is already beyond “toy A2C”, but not yet at “trustworthy baseline”.

#### 4.2 Mediocre implementation

A mediocre A2C is the stage where the code is not wrong, but the experiments are still weak and easy to misread.

Typical traits:

- single environment only
- fixed learning rates
- no run metadata
- weak reproducibility
- weak evaluation separation from training
- enough metrics to see training curves, but not enough to explain failures cleanly

Current NeuroDrive status:

- this is very close to the current state
- strengths:
  - coherent schedule placement
  - useful A2C health metrics
  - decent post-run analytics
  - sensible squashed continuous-action contract
- weaknesses:
  - one environment stream
  - ad hoc RNG
  - no save/load
  - no evaluation mode
  - no periodic export/checkpoint
  - no broad behavioural test harness

Verdict:

- “mediocre but promising” is the fairest classification of the current baseline.

#### 4.3 Expert implementation

For NeuroDrive, an expert A2C would not mean “most complicated possible code”. It would mean the baseline is good enough that, if it still fails, the failure likely belongs to the environment/reward/observation problem rather than obvious training-stack flaws.

Expected properties:

| Area | Expert bar in NeuroDrive |
|---|---|
| Reproducibility | single owned RNG/seed path, seed recorded in reports, deterministic-enough repeated runs |
| Collection | true synchronous vectorised rollout collection across multiple car environments or parallel world instances |
| Scaling | running normalisation for observations and value targets or returns, plus controlled clipping |
| Evaluation | distinct training and evaluation modes, no-learning evaluation rollouts, checkpointed model snapshots |
| Optimisation | tuned LR schedules, tested init, sensible entropy handling, explicit NaN/non-finite guards |
| Diagnostics | run metadata, policy/value drift, TD-error distributions, per-seed comparisons, behavioural success criteria |
| Safety | tests for buffer alignment, GAE correctness, action-transform correctness, and seed reproducibility |

Verdict:

- NeuroDrive is not near this level yet, and it should not attempt to jump there all at once.

### 5. What should go well with A2C in NeuroDrive

#### 5.1 High-value companions

- Vectorised environment rollout:
  - this is the single most “actually A2C” upgrade missing from the repo.
  - If the project keeps A2C for a while, multiple synchronous car instances should be the first substantial systems improvement.
- Stronger experiment metadata:
  - seed, config, git revision, observation version, active mode, and track identity should be exported with reports.
- Training/evaluation split:
  - one mode updates the policy; the other runs checkpoints without learning and writes comparable reports.
- Running normalisation:
  - current static clipping is better than nothing, but running mean/variance for observations and either returns or value targets is the more robust on-policy practice.
- Checkpointing:
  - not because A2C is the final system, but because without saved states you cannot compare improvements rigorously.

#### 5.2 NeuroDrive-specific companions

- Track variation or curriculum:
  - not necessarily many maps, but enough variation to tell genuine driving skill from track memorisation.
- Episode-start variation:
  - spawn offsets, headings, or localised start sectors would make the controller less brittle and produce better diagnostics.
- Geometry-first observation ablations:
  - because the project already has strong centreline-derived features, A2C is a good consumer for controlled observation experiments.
- Clear baseline scorecard:
  - examples:
    - median best progress over N eval episodes
    - crash rate
    - lap completion rate
    - centreline distance trend
    - turn-entry speed and understeer metrics

### 6. What should not be overbuilt around A2C

- Do not turn A2C into the repository’s long-term abstraction centre just because it is the first working learner.
- Do not build large replay/persistence infrastructure that assumes an on-policy Gaussian actor-critic is permanent.
- Do not confuse “better training metrics” with “better driving”; in this project the turn-execution analytics matter more than scalar reward alone.

### 7. Project-specific gap analysis against current code

| Topic | Current NeuroDrive state | Assessment |
|---|---|---|
| Separate actor / critic | yes | good; matches empirical best practice |
| GAE | yes | good; keep |
| Squashed bounded actions | yes | good fit for steering/throttle |
| Gradient clipping | yes | good |
| Value loss robustness | Huber | reasonable |
| Running observation normalisation | no; only fixed scaling/clipping | meaningful gap |
| Running return/value normalisation | no | meaningful gap |
| LR annealing / schedules | no | moderate gap |
| Seed-controlled RNG | no | major experimental gap |
| Multi-env synchronous rollout | no | major A2C-identity gap |
| Save/load checkpoints | no | major practical gap |
| Evaluation-only mode | no | major practical gap |
| Behavioural test harness | very limited | major validation gap |
| Run metadata export | no seed/config/git metadata | major analytics gap |
| Activation choice | `ReLU` | plausible improvement candidate, not first priority |

### 8. Recommended priority order for NeuroDrive

This priority order is an inference for this repository.

#### 8.1 Recommended next step

Implement controlled reproducibility and experiment discipline before adding more policy complexity.

Why now:

- Without seed control, checkpoints, and evaluation separation, A2C improvements are too easy to misread.
- These upgrades also improve every later brain baseline, not just A2C.

What it unlocks:

- apples-to-apples comparisons
- behavioural regression testing
- honest judgement on whether the current observation/reward design is working

Main risks:

- it does not make the agent immediately stronger, so it can feel slower than algorithmic tweaking

#### 8.2 Credible alternative: true synchronous multi-environment A2C

Why now:

- it is the most algorithm-faithful improvement and the biggest likely sample-efficiency gain within A2C itself

What it unlocks:

- stabler gradients
- less per-episode brittleness
- better GPU/batch utilisation if the project later leans into larger models

Main risks:

- it is structurally invasive in an ECS game codebase because environment instancing and analytics/debug assumptions are currently single-car oriented

#### 8.3 Credible alternative: observation and reward normalisation first

Why now:

- literature suggests scaling and normalisation choices strongly affect on-policy performance

What it unlocks:

- less brittle optimisation
- fewer learning-rate cliffs

Main risks:

- if added before stronger experiment discipline, it improves training stability without proving behavioural correctness

### 9. Practical interpretation for whether A2C should remain the baseline

- Keep A2C if the aim is still “learnability validation with continuous controls and dense engineered observations”.
- Replace or deprioritise A2C if:
  - the repo needs much higher sample efficiency from a single environment,
  - the engineering cost of vectorised on-policy training becomes too high,
  - or the project is ready to move from validation toward biologically motivated learning rules.
- A2C should be treated as a falsification tool:
  - if a competently implemented A2C still cannot learn stable driving here, the problem may be in observation design, reward semantics, environment variability, or task framing rather than only in algorithm choice.

### 10. Source list

Primary sources used:

- Sutton et al., “Policy Gradient Methods for Reinforcement Learning with Function Approximation” (NeurIPS 1999): https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation
- Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning” (arXiv / ICML 2016): https://arxiv.org/abs/1602.01783
- Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation” (arXiv 2015): https://arxiv.org/abs/1506.02438
- OpenAI, “OpenAI Baselines: ACKTR & A2C” (official OpenAI write-up): https://openai.com/index/openai-baselines-acktr-a2c/
- Engstrom et al., “Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO” (arXiv 2020): https://arxiv.org/abs/2005.12729
- Andrychowicz et al., “What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study” (arXiv 2020): https://arxiv.org/abs/2006.05990

## Implications for the Repository

- NeuroDrive should keep A2C modular and baseline-scoped.
- The highest-leverage A2C work is not “make the network bigger”; it is:
  - reproducibility,
  - vectorised rollout collection,
  - normalisation,
  - evaluation discipline,
  - stronger analytics metadata.
- If the current A2C remains weak after those upgrades, the project should be willing to conclude that the limitation is in the environment/task formulation or that A2C has served its purpose and should stop absorbing engineering effort.

## Open Constraints / Follow-Up Questions

- How much structural change is acceptable to support multi-environment synchronous rollout in the current Bevy/ECS setup?
- Does the project want A2C only as a baseline validator, or as a stronger medium-term training harness while biological-learning work matures?
- Which variations of the environment are acceptable for better generalisation testing without violating the project’s “minimal but interpretable” design goal?
