# Directory Tree

```text
learning/
├── LEARNING_MAP.md
├── DIRECTORY_TREE.md
├── GLOSSARY.md
├── STUDY_GUIDE.md
├── paths/
│   ├── PATH_INDEX.md
│   ├── foundations-path.md
│   ├── project-architecture-path.md
│   ├── implementation-first-path.md
│   ├── reinforcement-learning-path.md
│   ├── neuroscience-path.md
│   └── debugging-and-observability-path.md
├── concepts/
│   ├── foundations/
│   │   ├── continuous-control-and-mdps.md
│   │   └── probability-value-estimation-and-return.md
│   ├── core/
│   │   ├── observations-actions-and-representation.md
│   │   ├── determinism-and-fixed-timestep-simulation.md
│   │   └── actor-critic-and-gae.md
│   ├── domain-patterns/
│   │   ├── reward-shaping-and-credit-assignment.md
│   │   └── brain-inspired-learning-principles.md
│   └── advanced/
│       └── a2c-vs-biological-learning.md
├── project/
│   ├── architecture/
│   │   ├── runtime-overview.md
│   │   └── data-flow-and-schedule.md
│   ├── systems/
│   │   ├── maps-and-centreline.md
│   │   ├── environment.md
│   │   ├── agent-interface.md
│   │   ├── a2c-baseline.md
│   │   ├── analytics.md
│   │   └── debug-runtime.md
│   ├── decisions/
│   │   └── why-a2c-exists-in-a-brain-inspired-project.md
│   ├── comparisons/
│   │   ├── current-baseline-vs-target-biological-system.md
│   │   └── singleton-runtime-vs-vectorised-trainer.md
│   └── evolution/
│       └── project-state-and-next-pressure-points.md
├── exercises/
│   ├── EXERCISE_GUIDE.md
│   ├── EXERCISE_ORDER.md
│   ├── foundations/
│   │   ├── derive-a-reward-signal.md
│   │   └── reason-about-returns-and-advantages.md
│   └── project/
│       ├── inspect-the-observation-pipeline.md
│       ├── debug-a2c-rollout-alignment.md
│       ├── extend-the-analytics-schema.md
│       └── design-the-vectorised-trainer-boundaries.md
├── materials/
│   ├── reinforcement-learning.md
│   ├── computational-neuroscience.md
│   └── rust-bevy-and-game-loop-engineering.md
└── references/
    ├── notation-guide.md
    ├── status-conventions.md
    └── system-cheat-sheet.md
```

## Key Locations

- `paths/`
  Route files for different learner goals. Use this when you want progression rather than a raw file list.
- `concepts/`
  Theory and reusable ideas. Read these when you need the domain scaffolding behind the project systems.
- `project/`
  Project-grounded explanations of the actual NeuroDrive runtime, architecture, decisions, and evolution.
- `exercises/`
  Practice tasks. These are intentionally project-specific where that teaches more than generic toy tasks would.
- `materials/`
  Curated external-study guidance by topic rather than by medium.
- `references/`
  Quick lookup files that support the rest of the archive.

## File Highlights

- `LEARNING_MAP.md`
  Explains what this archive is for and how to use it.
- `STUDY_GUIDE.md`
  The best high-level route selector if you do not yet know where to begin.
- `GLOSSARY.md`
  Shared vocabulary across RL, simulation, architecture, and brain-inspired learning.
- `project/architecture/runtime-overview.md`
  The best first project-specific file for most readers.
- `project/comparisons/current-baseline-vs-target-biological-system.md`
  The file that makes the repository’s transitional state easiest to understand.
- `project/evolution/project-state-and-next-pressure-points.md`
  A compact explanation of what is missing next and why it matters.

## How The Tree Is Intended To Feel

This is not a small archive with a decorative structure. The tree exists so that a large amount of educational material stays navigable.

The guiding split is:

- `concepts/` explains the ideas,
- `project/` explains this repository,
- `paths/` tells you how to move through both,
- `exercises/` forces you to prove understanding,
- `materials/` sends you outward when the archive should not try to become a full textbook,
- `references/` reduces lookup friction while reading.
