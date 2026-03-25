# Directory Tree

```text
learning/
├── LEARNING_MAP.md                          # What this learning system is for and how to use it
├── DIRECTORY_TREE.md                        # Structural index of the learning tree
├── GLOSSARY.md                              # Shared terminology for the whole project
├── STUDY_GUIDE.md                           # High-level route selector
├── paths/
│   ├── PATH_INDEX.md                        # Index of focused study paths
│   ├── project-architecture-path.md         # Top-down route through the current runtime
│   ├── reinforcement-learning-path.md       # RL-first route to the A2C baseline
│   ├── systems-and-simulation-path.md       # Determinism, physics, scheduling, and environment route
│   └── implementation-first-path.md         # Fast route for rebuilding key project systems
├── concepts/
│   ├── foundations/
│   │   └── fixed-timestep-simulation.md     # Why deterministic fixed-step simulation matters here
│   ├── core/
│   │   ├── actor-critic-and-gae.md          # A2C and GAE explained from first principles
│   │   └── observation-design.md            # Observation vectors, scaling, and leakage risks
│   └── domain-patterns/
│       ├── ecs-plugin-scheduling.md         # Bevy plugin/set ordering as a design pattern
│       └── deterministic-racing-environment.md
│                                              # Dense-signal continuous-control environment design
├── project/
│   ├── architecture/
│   │   └── runtime-architecture.md          # NeuroDrive runtime map in learner-friendly form
│   ├── systems/
│   │   ├── environment.md                   # Track, physics, reward, reset, and episode lifecycle
│   │   ├── agent-interface.md               # Actions, sensors, observations, and controller boundary
│   │   ├── a2c-baseline.md                  # Current baseline learner and its limitations
│   │   └── analytics-and-debugging.md       # Runtime and post-run observability surfaces
│   ├── comparisons/
│   │   └── a2c-baseline-vs-biological-target.md
│   │                                          # Why the current learner exists and why it is temporary
│   └── evolution/
│       └── current-state-and-next-gaps.md   # What is implemented now and what is still missing
├── exercises/
│   ├── EXERCISE_GUIDE.md                    # How to use the exercise layer
│   ├── EXERCISE_ORDER.md                    # Recommended practice sequence with checkboxes
│   └── project/
│       ├── reason-about-schedule-order.md   # Reconstruct the fixed-tick ordering contract
│       ├── extend-observation-vector.md     # Think through a safe observation change
│       └── debug-a2c-reproducibility.md     # Analyse current determinism gaps in the learner
├── materials/
│   ├── rust-and-bevy.md                     # Suggested background study for the implementation stack
│   ├── reinforcement-learning.md            # Suggested background study for the baseline learner
│   └── neuroscience-and-local-learning.md   # Suggested background study for the long-term target
└── references/
    └── status-conventions.md                # Short guide to current vs superseded wording
```

## Key Locations

- `paths/` — checklist-driven study routes for different goals
- `concepts/` — first-principles explanations that the project relies on
- `project/` — NeuroDrive-specific architecture, systems, evolution, and comparisons
- `exercises/` — practice tasks that make you reason about the actual repository
- `materials/` — topic-grouped external study suggestions
- `references/` — small support files that help the learner interpret the rest
