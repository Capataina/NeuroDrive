# Continual Learning

## Why This Matters Here

NeuroDrive's core design principle is "one brain, one lifetime" — the same neural network learns continuously across all episodes, never resetting its weights. This is in direct contrast with most machine learning practice, where training happens on a fixed dataset and the model is then deployed.

Continual learning studies how a learning system can acquire new skills over time without catastrophically forgetting the ones it already has. Understanding this challenge and the mechanisms that address it is essential for understanding Milestones 6 and 7.

**Status:** Foundational domain knowledge. Milestones 6 (generalisation) and 7 (replay/consolidation) address continual learning directly.

## Prerequisites

- `concepts/advanced/hebbian-plasticity.md`
- `concepts/advanced/eligibility-traces.md`

---

## The Catastrophic Forgetting Problem

When a neural network is trained on a new task, gradient updates to the weights tend to overwrite the representations learned for old tasks. This is called **catastrophic forgetting** (also called catastrophic interference).

**Example in NeuroDrive terms:**
1. The brain learns to navigate the current track's left turn at position X
2. The track is extended with a right turn at position Y
3. Training on the right turn updates the same weights that encode the left turn behaviour
4. The agent now knows how to turn right at Y but has "forgotten" how to turn left at X

For the current single-track, single-episode NeuroDrive, forgetting is less critical — the agent is always training on the same task. But:
- Milestone 6 introduces multiple tracks
- The "one brain, one lifetime" principle means the same weights serve all tracks
- Forgetting becomes a genuine failure mode

---

## Why Forgetting Happens

In a standard neural network:
- Weights are shared across all inputs
- Updating weights for input pattern A affects the network's response to input pattern B (unless the patterns are orthogonal)
- There is no mechanism to "protect" old knowledge from new updates

Gradient descent actively optimises the current loss. It does not know or care about previous tasks. If the gradient for the new task points in a direction that increases the old task's loss, that is not a constraint — it is just the gradient.

---

## The Memory Hierarchy

Biological memory is not a single system. Neuroscience identifies at least:

| System | Timescale | Function |
|---|---|---|
| Synaptic plasticity | Milliseconds–seconds | Rapid encoding of experiences |
| Short-term memory | Seconds–minutes | Working memory, temporary buffering |
| Long-term potentiation | Hours–days | Consolidation of important memories |
| Systems consolidation | Days–years | Transfer from hippocampus to neocortex |

This hierarchy provides a natural protection mechanism: important memories are consolidated into slow-changing representations while rapid learning continues in fast-changing systems.

---

## Memory Consolidation

**Synaptic consolidation** occurs at the synapse level: recently potentiated synapses are "marked" through molecular mechanisms (e.g. protein synthesis) that make them more resistant to subsequent modification. This takes minutes to hours.

**Systems consolidation** occurs across brain systems: the hippocampus rapidly encodes new experiences (high plasticity, high forgetting risk), and then "replays" them to the neocortex during sleep/rest. The neocortex integrates these patterns slowly over many replays, building stable, compressed representations.

This is the biological motivation for NeuroDrive's **Milestone 7 (Replay & Consolidation)**:

> Trajectory buffer → offline replay ("sleep phase") → consolidation rules

---

## Continual RL Challenges

In reinforcement learning, the non-stationarity problem is even more acute than in supervised learning:

1. **The data distribution changes with the policy.** As the agent learns, it visits different states. Representations learned from early-training distributions may no longer be appropriate for late-training distributions.

2. **Rewards are not i.i.d.** Unlike supervised labels, rewards are temporally correlated and depend on the current policy.

3. **There is no separate train/test split.** The agent learns and acts simultaneously. The weights that are being updated are the same weights producing the behaviour being evaluated.

---

## Strategies for Continual Learning

### 1. Elastic Weight Consolidation (EWC)

After learning task A, compute the **Fisher information** of each weight (a measure of how important it is for task A's performance). When learning task B, add a regularisation penalty that prevents large changes to important weights:

```
L_total = L_task_B + λ * Σ_i F_i * (θ_i - θ*_i)²
```

Where `F_i` is the Fisher importance of weight `i` and `θ*_i` is its value after task A.

This explicitly protects task A's critical weights while allowing task B to update non-critical ones.

### 2. Progressive Neural Networks

Maintain separate network columns for each task. When learning a new task, freeze all previous columns and only train a new column (with lateral connections from previous columns). Zero forgetting by construction, but linear growth in parameters.

### 3. Experience Replay

Maintain a buffer of experiences from previous tasks. When learning a new task, periodically include samples from old tasks in the training batch. This prevents the catastrophic overwriting by keeping old tasks "alive" in the loss signal.

Biological correlate: memory reactivation during sleep. NeuroDrive's Milestone 7 implements this approach.

### 4. Meta-Continual Learning

Learn representations that are specifically suited to rapid adaptation without forgetting — essentially learning a prior over tasks that can be quickly specialised without disturbing general knowledge.

---

## "One Brain, One Lifetime" and Continual Learning

The NeuroDrive design principle implies:

1. The network must handle skill retention across episodes (current: same track, no forgetting issue)
2. The network must generalise across tracks without forgetting (Milestone 6)
3. The network may benefit from sleep-phase consolidation (Milestone 7)
4. The network must maintain stable performance under sensor noise and physics variation (Milestone 8)

Each of these is a continual learning challenge. The biological plasticity mechanisms (eligibility traces, structural plasticity, neuromodulation) provide some natural forgetting resistance because:
- Local learning rules do not propagate catastrophic updates globally
- Slow structural plasticity provides stable representational scaffolding
- Weight decay naturally limits the magnitude of any single update

But these properties do not *eliminate* forgetting — they just reduce the rate. For multi-track generalisation (Milestone 6), explicit consolidation mechanisms will likely be needed.

---

## The Forgetting-Plasticity Trade-off

Any continual learning system must navigate a fundamental tension:

- **High plasticity:** learns new things quickly, but forgets old things quickly
- **Low plasticity (high stability):** retains old knowledge, but adapts slowly to new situations

This is the **stability-plasticity dilemma**. No universal solution exists. Biological systems navigate it through:
- Different plasticity rates in different brain systems (hippocampus: fast; neocortex: slow)
- Memory consolidation as a bridge
- Selective protection of important memories through molecular tagging

---

## Measuring Forgetting in NeuroDrive

When Milestone 6 introduces multiple tracks, measuring forgetting requires:

1. Train on Track A until performance plateau
2. Introduce Track B and train
3. Re-evaluate on Track A (without additional training)
4. Measure the performance drop

**Forgetting metric:**

```
F = Performance_after_B_training(Track A) / Performance_before_B_training(Track A)
```

A value close to 1.0 means minimal forgetting. A value close to 0 means catastrophic forgetting.

This is one of the planned diagnostics in Milestone 6: `context/systems/README.md` mentions "forgetting metrics."

---

## Common Misunderstandings

❌ "Continual learning is only relevant for multi-task settings"
✅ Even within a single task, the non-stationarity of RL (changing policy → changing data distribution) creates continual learning challenges.

❌ "Catastrophic forgetting only happens in neural networks"
✅ Any parameterised function approximator with shared parameters can exhibit catastrophic forgetting. The problem is not specific to neural networks.

❌ "Replay completely solves the forgetting problem"
✅ Replay is effective but requires sufficient buffer capacity and an appropriate mixing ratio between old and new experiences. Insufficient replay of old tasks still leads to gradual forgetting.

---

## Related Files

- `concepts/advanced/structural-plasticity.md` — how structural changes contribute to stability
- `project/evolution/milestone-roadmap.md` — Milestones 6 and 7
- `materials/neuroscience-resources.md` — biological memory research resources
