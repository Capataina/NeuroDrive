# Learning Map

`learning/` is the repository's teaching archive. It is not the same thing as `context/`.

`context/` is the maintained implementation memory used to keep the project grounded in current runtime reality. `learning/` is the slower, larger, more explanatory layer whose job is to help a motivated engineer actually master NeuroDrive: what exists now, what the repository is trying to become, which theory sits underneath both, and where the main trade-offs live.

This archive therefore covers two truths at once:

- **Current implementation reality**: a deterministic Bevy racing environment, a stable controller boundary, a handwritten A2C baseline, analytics export, and a live debug HUD.
- **Project direction and intellectual territory**: local plasticity, neuromodulation, eligibility traces, structural plasticity, and the larger question of whether a brain-like learning system can acquire driving behaviour online from first principles.

You should use the archive differently depending on what you need.

## Start Here

- If you want to understand the archive shape first, read [DIRECTORY_TREE.md](./DIRECTORY_TREE.md).
- If you want route guidance rather than a tree, read [STUDY_GUIDE.md](./STUDY_GUIDE.md).
- If you already know what kind of learner you are, jump to [paths/PATH_INDEX.md](./paths/PATH_INDEX.md).
- If you want quick term lookup while reading, keep [GLOSSARY.md](./GLOSSARY.md) open in parallel.
- If you learn best by rebuilding or debugging, use [exercises/EXERCISE_GUIDE.md](./exercises/EXERCISE_GUIDE.md) and [exercises/EXERCISE_ORDER.md](./exercises/EXERCISE_ORDER.md).

## What This Archive Covers

The material is organised around six broad surfaces:

1. **Foundations**
   The minimum reinforcement-learning, simulation, control, and maths concepts you need before NeuroDrive’s systems make full sense.
2. **Core concepts**
   The ideas that appear repeatedly in the project: observations, action spaces, fixed-timestep simulation, actor-critic learning, advantage estimation, reward shaping, and credit assignment.
3. **Project architecture and systems**
   How the actual Rust/Bevy repository is wired together today.
4. **Comparisons and decisions**
   Why the project currently uses A2C even though the README’s end goal is not A2C.
5. **Evolution and future direction**
   Where the current implementation is incomplete and what tensions matter next.
6. **Practice**
   Exercises that force reconstruction, extension, debugging, and design reasoning rather than passive reading only.

## How To Read Status Labels

Some files explicitly label their status so you do not confuse current runtime truth with roadmap-facing or foundational material.

- **Current in the project runtime**: implemented in code now.
- **Current in the maintained implementation**: reflected in code and `context/`, even if still immature.
- **Foundational domain knowledge**: required theory, whether or not implemented directly.
- **Planned project direction**: central to the README mission, but not yet built.
- **Historical or transitional**: no longer the final direction, but still important for understanding how the repository got here.

## How To Navigate Efficiently

If you are unsure where to start, choose one of these entry points:

- **Top-down project understanding**
  Read `STUDY_GUIDE.md`, then `paths/project-architecture-path.md`.
- **Fastest route to safe implementation work**
  Use `paths/implementation-first-path.md`.
- **Algorithm and learning-focus**
  Use `paths/reinforcement-learning-path.md`.
- **Brain-inspired and future-direction focus**
  Use `paths/neuroscience-path.md`.
- **Observability, debugging, and validation**
  Use `paths/debugging-and-observability-path.md`.

## Progress Tracking

Progress is tracked only in the files where checkboxes actually help:

- [STUDY_GUIDE.md](./STUDY_GUIDE.md)
- files in [paths/](./paths/)
- [exercises/EXERCISE_ORDER.md](./exercises/EXERCISE_ORDER.md)

Concept files, project files, and glossary entries intentionally do not contain learner checkboxes; those files are reference and teaching material, not progress state.

## Recommended Reading Style

Do not read the archive as one giant linear book.

Use this rhythm instead:

1. pick a route,
2. keep the glossary nearby,
3. alternate theory files with project files,
4. pause regularly for exercises,
5. use the materials guides when you want external depth.

## Important Constraint To Keep In Mind

NeuroDrive is in a transitional state.

The current codebase is already substantial enough that you should treat the A2C baseline, analytics, and debug systems as real implementation. At the same time, you should not mistake them for the final intended architecture. Much of the educational value of this repository comes from understanding that gap clearly rather than pretending it does not exist.
