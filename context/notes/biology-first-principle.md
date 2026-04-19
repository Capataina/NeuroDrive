# The Biology-First Principle

## Current Understanding

NeuroDrive's single most important guiding principle is: **when we hit a problem, the answer comes from biology, not from the machine-learning toolkit.**

Standard ML has a well-worn playbook for every failure mode:

| Problem | Standard ML response |
|---------|----------------------|
| Overfitting | Dropout, weight decay, early stopping, data augmentation |
| Slow learning | Higher LR, warmup schedules, adaptive optimisers |
| Poor generalisation | Batch normalisation, label smoothing, more data |
| Exploding gradients | Gradient clipping |
| Vanishing gradients | Skip connections, better initialisation |
| Catastrophic forgetting | Elastic Weight Consolidation, replay buffers, task-specific heads |

That playbook is a specific cultural response to the specific failure modes of backprop-trained networks. It is **not the only way to solve these problems** — it is the way the ML community converged on given its foundational choice of gradient descent.

**NeuroDrive rejects that playbook entirely.** When we hit a problem:

| Problem NeuroDrive might see | Biological answer to consult |
|------------------------------|------------------------------|
| Overfitting / over-commitment | Homeostatic plasticity, sparse coding, neuromodulated consolidation, sleep replay |
| Slow learning | Multi-timescale plasticity, attention, salience modulation, neurogenesis in specific regions |
| Network collapse / dead neurons | Excitatory/inhibitory balance, synaptic homeostasis, intrinsic plasticity |
| Catastrophic forgetting | Complementary learning systems, hippocampal replay, structural consolidation |
| Poor generalisation | Structural plasticity, lateral inhibition, efference copy |
| Exploration collapse | Noradrenergic arousal, novelty-driven dopamine, curiosity circuits |

**And crucially — when biology does not have a clear answer, we stop and research the biology further. We do not reach for the ML toolkit as a shortcut.**

## Rationale

This is not an aesthetic preference. It is the thesis of the project.

NeuroDrive's stated goal in `README.md` is:

> Can we build a learning system from scratch that mimics how the human brain learns, and watch it gradually acquire driving behaviour in real time?

That goal is only meaningful if we are actually building biology-inspired machinery. If our response to every problem is to import an ML-toolkit solution, the project becomes "a layered MLP with Hebbian updates bolted on" — which is exactly what the README explicitly says NeuroDrive is not.

The biology-first discipline gives the project a clear identity:

- Every design choice has a biological justification or an explicit "biology doesn't have this yet" acknowledgement.
- Every problem becomes a prompt to learn more neuroscience.
- Every milestone is tied to a specific biological feature that addresses a specific measured pathology.

## How to Apply This

When a future session encounters a problem with the brain-inspired learner, the decision flow is:

1. **Observe the pathology.** What is actually happening? Measure it. Do not guess.
2. **Consult biology first.** What does the brain do in an analogous situation? Research neuroscience, not ML.
3. **If biology has an answer**, translate it into a computational mechanism that fits NeuroDrive's Rust-from-scratch constraint.
4. **If biology has multiple candidate answers**, pick the simplest / most tractable one; note the others in `context/references/` for later.
5. **If biology does not have a clear answer**, pause and do a proper research pass (as a background agent). Do not fall back on the ML toolkit.
6. **Only if the biological answer is fundamentally out of reach** (too slow, architecturally incompatible, purely speculative research-frontier territory) should we consider an ML-shaped workaround — and if we do, we flag it as a compromise, not a default.

## Why It Matters

Three concrete reasons this discipline is load-bearing:

1. **It prevents scope collapse.** Without this discipline, "brain-inspired" erodes one convenient shortcut at a time until what remains is RL with cosmetic biological vocabulary. The ML toolkit has 50 years of momentum behind it; biology does not. Each time we take an ML shortcut, we owe biology more to catch up.
2. **It makes problems into research opportunities.** When backprop-era RL hits a wall, the tools get more elaborate. When NeuroDrive hits a wall, we learn more about brains. That compounds over time — the project gets smarter about neuroscience with every problem solved.
3. **It makes claims defensible.** If someone asks "why do you say this is brain-inspired?", the answer is "every design decision is traceable to a specific biological mechanism, and we can show you the reference." That is a stronger claim than "we used Hebbian updates."

## What Was Tried

Prior to this principle being articulated explicitly (2026-04-19), the project drifted toward engineering-pragmatic answers a few times — e.g., the research recommended "reuse PPO's GAE δ as the neuromodulator signal" as an engineering shortcut, which would have made the brain-inspired learner dependent on a backprop-trained component. That recommendation was **rejected** in favour of Option C (raw per-tick reward as the modulator) specifically because the biology-first principle said "build your own reward predictor later, don't borrow one from the ML toolkit."

This is the kind of call the principle makes easy: pragmatic shortcut vs faithful-to-the-thesis answer, and we pick the second.

## Guiding Principles (Concrete Rules)

- **No backpropagation** anywhere in the brain-inspired learner, ever. PPO can keep using it because PPO is the diagnostic baseline.
- **No genetic algorithms, no evolution strategies, no NEAT population evolution.** "One brain, one lifetime" is explicit in the README.
- **No imported ML libraries** (PyTorch, TensorFlow, JAX, ndarray-with-autograd). Everything from Rust primitives.
- **No ML-toolkit defaults when hitting a pathology.** Consult biology first. If biology doesn't help, research biology more.
- **Standard ML techniques are allowed only if they have a direct biological analogue** (e.g., weight clipping → synaptic homeostasis is fine; dropout → no clear biological analogue, avoid).
- **Every new milestone names a biological mechanism and a pathology it addresses.** "Add feature X because biology has X" is valid; "add feature X because DeepMind found it helps" is not.

## References

- `README.md` §"Core Project Goal" — explicit list of what we don't use (no GA, no ES, no TF/PyTorch/JAX, no backprop).
- `context/references/brain-inspired-learning/overview.md` — synthesis of the 7 research papers, all of which respect this principle.
- `context/references/brain-inspired-learning/biological-learning-foundations.md` — the biology baseline this principle draws on.
- `context/notes/baseline-to-brain-inspired.md` — the transition framing, updated 2026-04-19.
- `context/notes/brain-v1-design.md` — the concrete v1 design decisions that apply this principle in practice.
