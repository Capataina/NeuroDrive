# Directory Tree

This file shows the full structure of the `learning/` archive with a short description of each folder and key file.

```text
learning/
├── LEARNING_MAP.md              — what this archive is for and how to use it
├── DIRECTORY_TREE.md            — this file; full structural index
├── GLOSSARY.md                  — comprehensive alphabetical glossary of all major terms
├── STUDY_GUIDE.md               — high-level route selector; start here to pick a path
│
├── paths/
│   ├── PATH_INDEX.md            — overview of all paths and how to choose one
│   ├── foundations-path.md      — bottom-up study for learners who want maths and mechanics first
│   ├── reinforcement-learning-path.md — focused RL theory through to the A2C implementation
│   ├── neuroscience-path.md     — biological learning from Hebbian rules to structural plasticity
│   ├── project-architecture-path.md — top-down study of the runtime architecture and systems
│   ├── implementation-first-path.md — fast route through the live code for experienced engineers
│   └── research-directions-path.md — roadmap-facing study of Milestones 2–9 and the science behind them
│
├── concepts/
│   ├── foundations/
│   │   ├── neural-networks.md           — feedforward networks, linear layers, ReLU, backpropagation
│   │   ├── optimization-and-gradients.md — SGD, Adam, gradient clipping, learning-rate intuition
│   │   ├── probability-and-distributions.md — Gaussian distributions, sampling, log-prob, entropy
│   │   └── bevy-ecs-primer.md           — Entity Component System model used throughout NeuroDrive
│   │
│   ├── core/
│   │   ├── reinforcement-learning.md    — Markov decision processes, value functions, Bellman equations
│   │   ├── policy-gradient-methods.md   — REINFORCE, policy gradient theorem, variance reduction
│   │   ├── advantage-estimation.md      — GAE, lambda returns, bias-variance trade-off
│   │   ├── actor-critic-architecture.md — combining policy and value networks; A2C and variants
│   │   └── continuous-control.md        — action spaces, Gaussian policies, tanh squashing
│   │
│   ├── domain-patterns/
│   │   ├── reward-shaping.md            — potential-based shaping, dense vs sparse, pitfalls
│   │   ├── observation-design.md        — feature engineering, raycasts, centreline geometry
│   │   └── determinism-and-reproducibility.md — why determinism matters; seeds, fixed timesteps
│   │
│   └── advanced/
│       ├── hebbian-plasticity.md        — Hebb's rule, co-activation, local learning
│       ├── spike-timing-dependent-plasticity.md — STDP mechanism, timing windows, biological basis
│       ├── eligibility-traces.md        — synapse-local credit assignment, λ-returns vs traces
│       ├── neuromodulation.md           — dopamine-like reward gating, RPE, three-factor rules
│       ├── structural-plasticity.md     — synapse growth and pruning, topology adaptation
│       └── continual-learning.md        — catastrophic forgetting, lifelong learning, memory consolidation
│
├── project/
│   ├── architecture/
│   │   ├── runtime-overview.md          — high-level map of all subsystems and their relationships
│   │   ├── fixed-tick-pipeline.md       — the SimSet execution chain and why order matters
│   │   └── module-boundaries.md         — ownership, data flow, and dependency direction
│   │
│   ├── systems/
│   │   ├── environment-system.md        — track, physics, collision, progress, reward, episodes
│   │   ├── agent-interface.md           — observation vector, action contract, sensor pipeline
│   │   ├── a2c-brain.md                 — model, rollout buffer, GAE, update path, training stats
│   │   ├── analytics-system.md          — trackers, derived metrics, JSON/Markdown export
│   │   └── debug-runtime.md             — F1/F2/F3 overlays, HUD, live learning display
│   │
│   ├── decisions/
│   │   ├── a2c-as-baseline.md           — why A2C was chosen, what it validates, when to retire it
│   │   └── tanh-squashed-actions.md     — why tanh is used for continuous actions; log-prob correction
│   │
│   ├── comparisons/
│   │   ├── a2c-vs-ppo.md                — A2C versus PPO for this project; what PPO adds
│   │   └── rate-based-vs-spiking.md     — rate-coding neurons versus spiking neurons; transition plan
│   │
│   └── evolution/
│       ├── milestone-roadmap.md         — Milestones 0–9 explained with learning context
│       └── from-baseline-to-brain.md    — the planned transition from A2C to local plasticity
│
├── exercises/
│   ├── EXERCISE_GUIDE.md                — exercise types, how to use hints, where to start
│   ├── EXERCISE_ORDER.md                — recommended practice sequence with checkboxes
│   │
│   ├── foundations/
│   │   ├── implement-linear-layer.md    — build a forward+backward linear layer from scratch
│   │   ├── implement-relu-backprop.md   — implement ReLU and verify its gradient
│   │   └── implement-adam-optimizer.md  — implement Adam update rule step by step
│   │
│   ├── core/
│   │   ├── implement-gae.md             — implement GAE from the recurrence definition
│   │   ├── trace-the-policy-gradient.md — manually trace through one A2C policy update
│   │   └── trace-observation-vector.md  — trace a single tick's observation construction
│   │
│   └── project/
│       ├── debug-reward-shaping.md      — diagnose a reward-shaping scenario and fix it
│       ├── extend-observation-vector.md — design and add a new observation feature
│       └── sketch-eligibility-traces.md — design the eligibility trace system for Milestone 2
│
├── materials/
│   ├── reinforcement-learning-resources.md — papers, books, and references for RL study
│   ├── neuroscience-resources.md           — neuroscience resources for biological learning
│   └── rust-and-systems-resources.md       — Rust, Bevy, and game-engine systems resources
│
└── references/
    ├── notation-guide.md                   — symbols and conventions used across the archive
    ├── status-conventions.md               — meanings of Current / Planned / Foundational labels
    └── observation-vector-reference.md     — the 23-dim observation vector, feature by feature
```

## Key Locations by Purpose

### If you want to understand the current implementation

- `project/architecture/runtime-overview.md`
- `project/systems/environment-system.md`
- `project/systems/a2c-brain.md`
- `project/systems/agent-interface.md`
- `references/observation-vector-reference.md`

### If you want to understand the RL theory behind A2C

- `concepts/core/reinforcement-learning.md`
- `concepts/core/policy-gradient-methods.md`
- `concepts/core/advantage-estimation.md`
- `concepts/core/actor-critic-architecture.md`
- `project/decisions/a2c-as-baseline.md`

### If you want to understand where the project is heading

- `project/evolution/from-baseline-to-brain.md`
- `project/evolution/milestone-roadmap.md`
- `concepts/advanced/hebbian-plasticity.md`
- `concepts/advanced/eligibility-traces.md`
- `concepts/advanced/neuromodulation.md`

### If you want to practise

- `exercises/EXERCISE_GUIDE.md`
- `exercises/EXERCISE_ORDER.md`

### If you want a quick reference

- `GLOSSARY.md`
- `references/notation-guide.md`
- `references/observation-vector-reference.md`
