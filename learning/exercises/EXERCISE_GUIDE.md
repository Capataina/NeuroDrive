# Exercise Guide

## What Exercises Are For

The exercises in this archive are not tests of whether you read the material. They are practice routes for building genuine understanding through doing.

Reading about backpropagation and implementing backpropagation are different things. A learner who has computed a GAE by hand is in a fundamentally different position than one who has only read the formula. These exercises close that gap.

## How Exercises Are Structured

Each exercise file contains:

- **Context:** why this exercise matters and what it connects to
- **Prerequisites:** concepts and files you need to have read first
- **The task:** what you need to build, trace, compute, or explain
- **Constraints:** what help is available (code you can look at, code you cannot)
- **Checkpoints:** intermediate milestones so you know you are on the right track
- **Hints:** one or more progressive hints if you are stuck, each progressively more revealing
- **Reflection questions:** conceptual questions to answer after completing the task

**Solutions are never included in exercise files.** If you are stuck, work through the hints in order before looking at the source code. The source code is always available — but using it as a first resort eliminates the learning value.

---

## Exercise Tiers

### Foundations Tier

Reconstruction exercises for the core ML primitives. These exercises require writing code from scratch given the mathematical specification.

**When to do these:** Before or alongside reading `concepts/core/`. These exercises build the mechanical understanding that makes higher-level concepts legible.

| Exercise | Skill Built |
|---|---|
| `foundations/implement-linear-layer.md` | Matrix multiply, forward pass, gradient derivation |
| `foundations/implement-relu-backprop.md` | Activation functions, backprop through nonlinearities |
| `foundations/implement-adam-optimizer.md` | Moment estimation, bias correction, step |

### Core RL Tier

Tracing and derivation exercises on the A2C algorithm and its components.

**When to do these:** After reading `concepts/core/`. These exercises require working with the actual NeuroDrive source or deriving computations on paper.

| Exercise | Skill Built |
|---|---|
| `core/implement-gae.md` | GAE recurrence, bootstrap handling, advantage normalisation |
| `core/trace-the-policy-gradient.md` | Log-prob computation, squashed Gaussian, tanh Jacobian |
| `core/trace-observation-vector.md` | Full 23-dim vector construction, normalisation, feature meaning |

### Project Tier

Debugging, extension, and sketch exercises grounded in the actual NeuroDrive codebase.

**When to do these:** After reading `project/systems/`. These exercises assume familiarity with the codebase.

| Exercise | Skill Built |
|---|---|
| `project/debug-reward-shaping.md` | Reward decomposition, per-tick signal analysis |
| `project/extend-observation-vector.md` | Interface extension, dimension alignment, feature design |
| `project/sketch-eligibility-traces.md` | Biological learning rule design for the planned Milestone 2 architecture |

---

## Recommended Exercise Sequence

For a learner following the **foundations path:**

```
implement-linear-layer
    → implement-relu-backprop
    → implement-adam-optimizer
    → (read concepts/core/ first, then)
    → implement-gae
    → trace-the-policy-gradient
    → trace-observation-vector
```

For a learner following the **implementation-first path:**

```
trace-observation-vector
    → trace-the-policy-gradient
    → debug-reward-shaping
    → implement-gae
    → extend-observation-vector
    → sketch-eligibility-traces
```

For a learner following the **research-directions path:**

```
implement-gae  (understand the bridge between A2C and biological learning)
    → sketch-eligibility-traces
    → (read concepts/advanced/ then revisit)
    → extend-observation-vector
```

See `exercises/EXERCISE_ORDER.md` for a single recommended linear sequence.

---

## How to Use Hints

Each exercise provides hints in order of increasing specificity. Work in this sequence:

1. Read the full task description carefully.
2. Attempt the task without any hints.
3. If stuck after a genuine effort (more than 15 minutes for a focused attempt), read **Hint 1** only.
4. Attempt again.
5. If still stuck, read **Hint 2** only.
6. Continue this pattern until you make progress.
7. Only look at source code after you have exhausted the hints.

The hints are calibrated so that Hint 1 narrows the search space, Hint 2 points to the specific mechanism, and later hints provide the mathematical ingredient without giving the code.

---

## After Each Exercise

Answer the reflection questions in the exercise file. You do not need to write them anywhere permanent — just think through them. They are designed to surface conceptual gaps that the implementation alone might not reveal.

---

## Related Files

- `exercises/EXERCISE_ORDER.md` — single recommended sequencing
- `paths/foundations-path.md` — curriculum route including these exercises
- `STUDY_GUIDE.md` — route selection based on background and goals
