# Brain-Inspired Learner v1 — Design Decisions

Captures the concrete v1 design agreed on 2026-04-19 after the seven-paper research pass in `context/references/brain-inspired-learning/`. This note complements `biology-first-principle.md` (the guiding discipline) and `baseline-to-brain-inspired.md` (the transition framing).

## Current Understanding

### Topology: graph, not layered

The v1 brain is a **sparse directed graph of neurons and synapses**, not a layered feed-forward MLP. This is a deliberate choice rooted in biological realism — real brains are graph-structured, not layered — and it enables structural plasticity and visualisation in a way layers cannot.

- **Neurons** are nodes. Each has its own state (activation, firing-rate history, intrinsic excitability).
- **Synapses** are directed edges. Each has weight, eligibility trace, and possibly (later) synaptic delay.
- **Cyclic** is allowed. Real neurons have recurrent connections everywhere.
- **Topological sort not required.** Forward pass uses one-step propagation per tick: each tick, every neuron computes its new activation from the *previous* tick's activations of its inputs. No settling iteration.

### Reserved I/O neurons

- **Input neurons** (43): one per observation dimension. Bound to the stable 43-dim observation contract. These neurons' activations are set directly from `ObservationVector` each tick.
- **Output neurons** (2): steering and throttle. Their activations after forward pass are read and written to `ActionState.desired`.
- **Hidden neurons** (initial ~15–20, grows over time): the plastic interior. No predetermined structure.

This preserves the stable agent-interface contract (`ObservationVector` → `ActionState.desired`) that PPO also consumes. Both controllers remain drop-in interchangeable via `AgentMode`.

### Learning rule: three-factor plasticity with eligibility traces

For each synapse `(i, j)`:

```text
e_ij ← λ·e_ij + pre_i · post_j        (eligibility trace update)
δw_ij = η · M · e_ij                  (weight update, gated by modulator M)
```

Where:

- `λ` controls eligibility decay. Target time constant `τ_e ≈ 2s` (120 ticks at 60 Hz) per `reward-design.md`.
- `η` is the learning rate.
- `M` is the global neuromodulator signal — in v1, **raw per-tick reward** (Option C from the discussion).
- `pre_i` and `post_j` are pre- and post-synaptic activations (rate-coded).

This is the Frémaux–Gerstner three-factor rule, widely referenced in the neuroscience literature and implementable from the existing Rust primitives.

### Neuromodulator: Option C — raw reward, no critic (v1)

The v1 modulator is the per-tick reward directly from `EpisodeState.current_tick_reward`. No value predictor. No TD error calculation.

**Why Option C and not Option B (plastic value predictor) for v1:**

- Option C is the smallest, purest, most biology-first starting point. The entertainment-first reward is already dense per-tick, so the eligibility trace does the credit-assignment work — we do not need a value function to bridge sparse-reward gaps.
- Option B (building a plasticity-trained value predictor) is a meaningful architectural addition and deserves its own milestone (M8) where it can be designed carefully.
- Option A (reusing PPO's GAE δ) was **rejected** — it would make the brain-inspired learner depend on a backprop-trained component, violating the biology-first principle.

If v1 learns acceptably with raw reward, a critic may never be needed. If it does not, Option B becomes the natural next step.

### Homeostasis

Two homeostatic mechanisms run alongside plasticity, both biologically settled:

1. **Synaptic scaling** — per-neuron, slow. If the total incoming synaptic weight to a neuron drifts, scale all incoming weights multiplicatively to return to a target sum. Prevents weight explosion and weight death.
2. **Intrinsic excitability homeostat** — per-neuron, slow. Each neuron tracks its mean firing rate. If it drifts from a target band (too silent or too active), adjust the neuron's intrinsic bias/threshold.

Both are slow compared to the per-tick plasticity updates — they kick in over seconds to minutes of in-game time, not per tick.

### Structural plasticity: continual-backprop style, adapted to graph

Per `structural-plasticity-neuroevolution.md` research, the chosen technique is **continual backprop** (Dohare et al. 2024) — the only published structural-plasticity technique with demonstrated PPO + continuous-control + continual-training results. Adapted to our graph topology:

- **Per-neuron utility metric:** track each neuron's contribution (mean absolute output × mean absolute sum of outgoing weights, or similar). Low-utility neurons get recycled.
- **Apoptosis + neurogenesis (slot-based):** low-utility neurons effectively "die" — outgoing synapses zeroed, incoming synapses resampled — and a fresh neuron takes the slot. Mechanically this is one operation; biologically it is two (cell death + new-neuron formation). Behaviour-preserving at the moment of replacement. See the "Known Biological Simplifications" section in `README.md` for rationale — real biology does not have slot-reuse, but at our scale the bookkeeping simplification is reasonable.
- **Plateau-triggered neurogenesis:** when the running reward stops improving for N episodes, add a new neuron with random connections. Net2Wider-style width growth, adapted to graph form. Location-unrestricted (see Known Simplifications — real neurogenesis is localised).
- **Synapse pruning:** below-threshold magnitude synapses get removed entirely from the graph.
- **Synapse sprouting:** occasionally, two highly-correlated but unconnected neurons get a new random edge (biological "sprouting" analog). No spatial-proximity constraint (see Known Simplifications — real synaptogenesis requires axonal growth to physical neighbours).

**Depth is fixed, width is variable.** Net2DeeperNet's identity-preserving depth growth requires ReLU and fails with tanh. We are staying on tanh per the PPO baseline's hard-learned lesson, so we grow the graph's *width* (neuron count + synaptic density) but not its *depth*. In graph terms, "depth" is the typical path length from input to output — we constrain it by limiting per-hop distances rather than explicit layer count.

### Activation function

**Tanh**, matching the PPO baseline. The structural-plasticity paper flagged that Net2DeeperNet requires ReLU, but since we are not growing depth, tanh works. Tanh also avoids the dead-neuron failure mode the PPO baseline hit hard with ReLU (34–57% dead neurons) — critical given our plasticity rules need every neuron to contribute something over time.

### Initialisation

Small random seed graph:

- 43 input neurons (bound to observation dims).
- ~15 hidden neurons.
- 2 output neurons (bound to steering/throttle).
- Initial synapses: random connectivity at ~10% density. Roughly 120–150 edges total.
- Weight initialisation: small Gaussian (σ ≈ 0.1).

No predetermined structure. The brain grows its shape from this seed.

### Storage

Sparse, not dense. We do **not** use dense weight matrices. Each synapse is a struct:

```rust
struct Synapse {
    source: NeuronId,
    target: NeuronId,
    weight: f32,
    eligibility: f32,
    // Future fields: delay, facilitation, depression, dale_sign, etc.
}
```

Neurons live in a separate Vec with their own state. Adding/removing neurons and synapses is a Vec push/remove — no matrix reshaping, no layer rearrangement.

### Forward pass

One pass per tick:

1. Input neurons' activations set from `ObservationVector`.
2. For every neuron (iterating synapses), compute `pre-activation = sum over incoming synapses of source_activation_at_previous_tick × weight`.
3. Apply `tanh` to get new activation.
4. Output neurons' activations read and written to `ActionState.desired`.

Cyclic connections work because we read *previous tick's* activations. No settling loop. Biologically plausible (neurons have integration time constants; activations propagate with delay). Cheap computationally.

### Performance expectations

At v1 scale (500 neurons, 5000 synapses, 8 cars):

- Forward pass: ~5000 multiply-adds + 500 tanh per car per tick → ~15 µs per car → ~120 µs per tick for 8 cars.
- Eligibility updates: ~5000 per car per tick → similar.
- Weight updates: bounded by synapse count.
- Structural plasticity: infrequent (per-episode or per-plateau, not per-tick).

**Well under 1 ms per tick total.** The frame budget headroom from the 2026-04-18 performance overhaul absorbs this trivially. No GEMM / AMX needed; sparse graph traversal is faster than dense matrix ops at this scale.

### Integration with the existing runtime

- **New `AgentMode::BrainInspired` variant.** F4 becomes three-way: Keyboard / PPO / Brain-Inspired.
- **New module** `src/brain/inspired/` parallel to `src/brain/ppo/`.
- **Consumes** `ObservationVector` identically to PPO.
- **Writes** `ActionState.desired` identically to PPO.
- **Consumes** `EpisodeState.current_tick_reward` as the modulator signal.
- **PPO stays permanently live** as the diagnostic baseline. Any environment change must work for both.

## Rationale

### Why graph, not layers

Three reasons, in order of importance:

1. **Biological faithfulness.** Real brains are graph-structured. Layers are a machine-learning convention, not a biological feature.
2. **Structural plasticity is natural.** Adding/removing neurons and synapses on a graph is trivial (Vec push/remove). Doing the same on dense matrices requires reshaping, zero-insertion, and matrix-size management per layer.
3. **Visualisation payoff.** A graph brain is natively renderable — neurons as dots, synapses as lines, growth and pruning animated. The emotional core of "watch a brain grow" is impossible without graph structure.

### Why Option C (raw reward, no critic)

1. **Purest starting point.** No imported machinery from the ML toolkit.
2. **Biology-first discipline.** Building a borrowed reward predictor is a pragmatic shortcut; building our own plasticity-native one is the correct biological answer and deserves its own milestone (M8).
3. **Dense reward makes it feasible.** Velocity projection + centreline proximity is per-tick — the eligibility trace handles credit-assignment without a value function.

### Why tanh

Inherited from the PPO baseline's hard-learned lesson: ReLU produced 34–57% dead neurons that starved the actor. Tanh also fits the biology better (bounded activations, symmetric around zero).

### Why continual-backprop for structural plasticity (not NEAT)

1. **NEAT is population-based.** Our paradigm is "one brain, one lifetime". NEAT's add-node primitive is reusable but its population framing is not.
2. **Continual backprop has published PPO + continuous-control results.** The only candidate in the survey with direct task analogy.
3. **Graph adaptation is straightforward.** "Replace a neuron" maps to "clear outgoing edges, resample incoming edges" on a graph.

## What Was Tried (or would have been)

- **Reusing PPO's critic** as the dopamine signal was recommended by the research as an engineering shortcut. **Rejected** under the biology-first principle.
- **NEAT's full population evolution** was considered and **rejected** as incompatible with the "one brain, one lifetime" framing.
- **Spiking neurons with STDP** was considered for v1 and **deferred** to the Long-Term Plan — requires sub-tick scheduling, architectural rework beyond v1's scope.

## Guiding Principles

- **Every future brain-inspired addition must have a biological justification and a named pathology it addresses.** "Adding feature X because DeepMind paper found it helps" is not a valid justification.
- **Fixed depth, variable width.** Tanh constrains depth growth; width and synaptic density grow freely.
- **PPO coexistence is a permanent constraint.** Any environment, reward, or observation change must work for both controllers.
- **Visualisation is a first-class feature**, not an afterthought. The M7 brain inspector is the emotional core of the project.

## References

- `context/notes/biology-first-principle.md` — the discipline this design obeys.
- `context/notes/baseline-to-brain-inspired.md` — the transition framing.
- `context/references/brain-inspired-learning/overview.md` — seven-paper synthesis.
- `context/references/brain-inspired-learning/local-learning-rules.md` — three-factor plasticity derivation.
- `context/references/brain-inspired-learning/biological-learning-foundations.md` — the biology baseline.
- `context/references/brain-inspired-learning/structural-plasticity-neuroevolution.md` — continual-backprop technique.
- `context/references/brain-inspired-learning/reward-design.md` — modulator + eligibility trace specifics.
- `context/references/brain-inspired-learning/training-paradigms.md` — why single-agent, not population.
- `README.md` — project intent, milestone structure.
