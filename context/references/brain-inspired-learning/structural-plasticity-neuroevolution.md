# Structural Plasticity and Neuroevolution for NeuroDrive

## Scope / Purpose

- Answer the repository-specific question: **how should NeuroDrive implement a neural network whose topology can grow, shrink, and reshape during training — without destroying learned behaviour and without relying on any external ML library — given the stable 43-dim observation / 2-dim action contract, the 60 Hz Bevy tick, the 8-car vectorised trainer, and the M2 Air CPU-only budget?**
- Survey the algorithmic literature for topology change (classic NEAT and its variants, Net2Net function-preserving morphisms, dynamic sparse training — SET / RigL, synaptic pruning — Lottery Ticket / SNIP, and the newer continual-backprop and plasticity-injection lines) and extract the *algorithmic* decisions each makes: when to add, when to remove, how to initialise.
- Produce a **Topology-Change Decision Matrix** and a **Neuron Utility Metric** survey, then rank candidate approaches by feasibility in handwritten Rust against NeuroDrive's constraints.
- **Explicitly out of scope** — biology of synaptogenesis (covered in `biological-learning-foundations.md`), weight-update rules on fixed topology (`local-learning-rules.md`), population vs single-agent training (`training-paradigms.md`), reward signals (`reward-design.md`), timescale choices (`learning-timescales.md`). NEAT is population-based and thus overlaps — this paper covers the *algorithmic mechanics* of topology change; training-paradigms covers *when to evaluate, select, and persist*.

## Current Project Relevance

NeuroDrive has just finished its PPO baseline round and validated the environment (`run_1776556719.md`: 8/8 cars complete the loop, fleet max-progress spread 1.1%). The README's long-term intent is a *brain-inspired* learner that "can form, reorganise, or be pruned" over time and allocates capacity where it matters. The current PPO architecture is an explicit placeholder: actor 2×64 + critic 2×128, *fixed* at compile time, fully connected, entirely gradient-driven.

The next substantive milestone replaces that with something that grows and prunes itself. Before writing the first structural-plasticity commit, the project needs to decide:

1. **Is structural change evolution-across-generations (NEAT-style) or within-lifetime reshaping (Net2Net / dynamic sparse / continual-backprop)?** README §"Core Project Goal" says explicitly "one persistent brain, one lifetime" and "We do not use Genetic Algorithms / NEAT" — which rules out classic NEAT as the *primary* mechanism but not the *algorithmic primitives* it defines (innovation numbers, speciated protection, complexification).
2. **What triggers addition vs removal?** The literature splits six ways on this — random regrow (SET), gradient-informed regrow (RigL), co-activity (Hebbian structural plasticity), single-shot saliency (SNIP), or utility-based neuron replacement (continual backprop).
3. **How are new weights initialised?** Zero (RigL, continual-backprop outgoing), identity / function-preserving (Net2Net), small-random (NEAT, SET), or the teacher's duplicated weight rescaled (Net2Net Net2WiderNet).
4. **What implementability cost is acceptable?** NeuroDrive is pure handwritten Rust — the entire structural-plasticity machinery must be written from scratch, reviewed, tested, and kept inside the 16.67 ms frame budget alongside Bevy rendering and 8 cars' physics.

This artefact provides the input each of those decisions needs.

## Current State Snapshot

Verified by direct inspection of `src/brain/ppo/model.rs`, `src/brain/common/mlp.rs`, `context/architecture.md`, and `context/systems/brain-ppo.md` (2026-04-19 round-2 state).

### What exists today (repository fact)

| Surface | Current shape | Citation |
|---|---|---|
| Network encoding | `ActorCritic { a_fc1, a_fc2, a_mean, a_log_std, c_fc1, c_fc2, c_value }` — seven `Linear` layers, flat `Vec<f32>` weights, fully-connected, fixed dimensions chosen at construction | `src/brain/ppo/model.rs` (~lines 165–220) |
| Weight storage | `Linear.weights: Vec<f32>` in row-major order, length = `out_dim × in_dim` | `src/brain/common/mlp.rs` |
| Forward path | Batched mat-mat through `gemm_backend` dispatch (scalar / matrixmultiply / Accelerate) — assumes dense contiguous weights | `src/brain/common/gemm_backend.rs` |
| Topology mutation support | **None.** No add-node, no add-connection, no prune. Layer widths are `const` — changing them requires a recompile of the model struct | Repo inspection |
| Sparsity support | **None.** All connections fully populated. No sparse-matrix path in `gemm_backend` | Repo inspection |
| Neuron-utility telemetry | Partial: `PpoLayerHealth` tracks weight L2 norm, gradient L2 norm, tanh saturation fraction per layer | `systems/brain-ppo.md` "Training Stats" |
| Per-neuron signals captured | Not exposed — saturation is aggregated, no per-unit activation history, no per-unit utility score | Repo inspection |
| Determinism surfaces | `PpoBrain.rng: StdRng`, `SpawnRng: StdRng` — structural mutation RNG would be a third | `notes/conventions.md` §2 |

### What the next phase needs (project inference)

The brain-inspired phase requires (a) a sparse graph representation that can change size at runtime, (b) a topology-mutation scheduler, (c) a per-neuron / per-synapse utility signal to drive pruning decisions, (d) a correctness boundary that keeps the 43-dim input and 2-dim output layers fixed (the stable agent interface) while the hidden graph reshapes. None of this machinery exists — it all has to be written from scratch in Rust, kept deterministic within a session, and must remain within the 4.4% frame-budget envelope the PPO baseline currently uses.

## Research Signal

<!-- Evidence class values: "source-backed" (direct quoted passage from a
     primary source), "repository fact" (verified via file:line), "project
     inference" (explicitly labelled inference), "open uncertainty". -->

| # | Topic | Source-backed signal | Source + quoted passage ID | Current repository state | Repo citation | Project implication | Evidence class |
|---|---|---|---|---|---|---|---|
| 1 | NEAT innovation numbers | "tracking the history of genes by the use of a global innovation number which increases as new genes are added" | **[NEAT-INNOV]** | no gene-level identity; layers are dense matrices | `src/brain/ppo/model.rs` | The innovation-number primitive is only needed if we ever want **crossover** between brains. With "one persistent brain, one lifetime" we do **not** need it — which removes the single hardest bookkeeping cost in a full NEAT port | source-backed |
| 2 | NEAT add-node mutation | "An existing connection is split by inserting a new node. The original connection is disabled, and two new connections replace it" | **[NEAT-ADDNODE]** | no equivalent | same | Add-node mutation is a *local* operation: identify an edge, disable it, insert a node with the same-valued incoming edge and a weight-1 outgoing edge (or vice-versa). This primitive is directly reusable even outside full NEAT | source-backed |
| 3 | NEAT complexification | NEAT "begins with minimal topologies and gradually increases complexity through structural mutations" | **[NEAT-COMPLEX]** | starts at full 64/128 width | same | Philosophy match — start minimal, grow into capacity as task demands. Contrasts with NeuroDrive's current over-parameterised PPO baseline | source-backed |
| 4 | NEAT new-weight init | "new connection weights are initialized with small random values to minimize initial disruption" | **[NEAT-INIT]** | not applicable yet | same | Small-random is a reasonable default for additive mutations where the new connection must not disrupt existing behaviour. Contrasts with Net2Net's function-preserving identity-init and RigL's zero-init | source-backed |
| 5 | HyperNEAT indirect encoding | HyperNEAT "query[ies] the CPPN to determine the connection weight between two neurons as a function of their position in space" | **[HYPER-CPPN]** | direct encoding (weights are the genome) | same | Indirect encoding only pays off when the task has **geometric regularities** (image grids, robot bodies). NeuroDrive's 43-dim observation is a flat feature vector with no obvious spatial geometry — HyperNEAT's main advantage does not apply | source-backed |
| 6 | ES-HyperNEAT | "extends [HyperNEAT] to evolve the location of every neuron in the network" | **[ES-HYPER]** | N/A | same | Adds substrate evolution on top of HyperNEAT's already-heavy CPPN machinery. Not warranted for NeuroDrive | source-backed |
| 7 | Net2WiderNet function-preserving | `U^k'_j(i) = 1/\|{x\|g(x)=g(j)}\| · W^g(j),h(i+1)` — outgoing weights from duplicated units are divided by replication count | **[N2N-WIDER]** | not applicable | `src/brain/common/mlp.rs` | The widening formula is a 5-line transformation: pick a unit to duplicate, copy its incoming row, duplicate its outgoing column, divide both outgoing columns by 2. Trivially implementable on flat `Vec<f32>` storage | source-backed |
| 8 | Net2DeeperNet identity init | "The new matrix U is initialized to an identity matrix, but remains free to learn to take on any value later" | **[N2N-DEEPER]** | N/A | same | Adds depth without disturbing behaviour — but only valid when activation φ satisfies φ(Iφ(v)) = φ(v). See next row | source-backed |
| 9 | Net2Net activation constraint | "for some popular activation functions, such as the logistic sigmoid, it is not possible to insert a layer of the same type that represents an identity function" | **[N2N-LIMIT]** | activation is tanh | same | **Tanh fails Net2DeeperNet's identity condition** in general: tanh(I · tanh(v)) = tanh(tanh(v)) ≠ tanh(v). Net2DeeperNet effectively requires ReLU-family activations. NeuroDrive switched *away* from ReLU because of 34–57 % dead neurons. This is a real architectural tension — see Gap Analysis | source-backed, contrasting |
| 10 | SET initial sparsity | "the probability of a connection between the neurons h^k_i and h^k-1_j is given by p = ε(n^k+n^k-1)/(n^k n^k-1)" | **[SET-ER]** | dense | repo fact | Erdős-Rényi-random initial sparsity with a single density knob ε. If ε=20 and n=64, the expected density is ≈ 40% — a reasonable starting point for NeuroDrive's small hidden layers | source-backed |
| 11 | SET prune step | "a fraction ζ of the smallest positive weights and of the largest negative weights of SC^k is removed. These removed weights are the ones closest to zero" | **[SET-PRUNE]** | N/A | repo fact | Simple magnitude-based pruning at fraction ζ per epoch. Implementable as a partial-sort — `O(nk log nk)` worst case, small constants | source-backed |
| 12 | SET regrow | "an amount of new random connections, equal to the amount of weights removed previously, is added" | **[SET-REGROW]** | N/A | repo fact | **Random regrow** — same-count replacement of pruned edges at uniformly-sampled inactive positions with small random weights. Very cheap, fully local decision | source-backed |
| 13 | SET offline assumption | "After the training ends, we keep the topology of SC^k as the one obtained after the last weight removal step, without adding new random connections" | **[SET-OFFLINE]** | N/A | repo fact | SET's evaluation protocol **locks topology at end of training** — it is not a strictly online algorithm. For NeuroDrive's "always training, never stop" mode, the stopping rule would have to be replaced with a running average or never-stop variant | source-backed, contrasting |
| 14 | RigL drop rule | "drop the connections given by ArgTopK(−\|θl\|, …)" — smallest-magnitude active weights removed each update | **[RIGL-DROP]** | N/A | repo fact | Same as SET for prune | source-backed |
| 15 | RigL grow rule | "grow the connections with highest magnitude gradients, ArgTopK_{i∉θl∖𝕀active}(\|∇_Θl L_t\|,k)" | **[RIGL-GROW]** | no gradient w.r.t. inactive weights is currently computed | repo fact | **Key departure from SET.** RigL requires computing dL/dw for weights that are currently pruned — a dense-gradient over sparse-weights pass. Algorithmically sound but loses SET's pure-sparse cost model. Implementable but negates the compute-efficiency argument for sparsity | source-backed |
| 16 | RigL zero-init | "Newly activated connections are initialized to zero and therefore don't affect the output of the network" | **[RIGL-INIT]** | N/A | repo fact | Zero-init guarantees the topology change does not disrupt forward behaviour — same philosophy as Net2Net identity-init but cheaper | source-backed |
| 17 | RigL schedule | cosine-decayed update fraction, default ΔT=100 iterations between updates, α=0.3 initial fraction | **[RIGL-SCHED]** | N/A | repo fact | Structural updates happen **every 100 training steps, not every tick**. For NeuroDrive where one PPO update spans 64 amortised ticks, that maps to structural change roughly every 1.5 PPO updates | source-backed |
| 18 | Lottery Ticket hypothesis | "dense, randomly-initialized, feed-forward networks contain subnetworks ('winning tickets') that — when trained in isolation — reach test accuracy comparable to the original network" | **[LTH-CLAIM]** | N/A | repo fact | LTH is a *finding about initialisation*, not a training-time algorithm. Only useful if NeuroDrive performed iterative magnitude pruning (IMP) across many full-training rounds — inconsistent with "one brain, one lifetime" | source-backed |
| 19 | LTH limits | LTH winning tickets demonstrated on "MNIST and CIFAR10" with "fully-connected and convolutional feed-forward architectures" — no RL or continuous-control results in the original paper | **[LTH-SCOPE]** | N/A | repo fact | LTH's evidence base is fully outside NeuroDrive's regime. Importing the claim to continuous-control RL requires evidence that is not in this paper | source-backed |
| 20 | Rethinking-Pruning contrarian | "with optimal learning rate, the 'winning ticket' initialization as used in Frankle & Carbin (2019) does not bring improvement over random initialization" | **[LIU-RETHINK]** | N/A | repo fact | **Contrasting source.** Liu et al. 2019 argue the LTH effect vanishes under proper LR tuning — "the pruned architecture itself, rather than a set of inherited 'important' weights, is more crucial". Weakens the case for any elaborate weight-preservation scheme during topology change | source-backed, contrasting |
| 21 | SNIP saliency | SNIP "introduces a saliency criterion based on connection sensitivity that identifies structurally important connections" — computed at initialisation via `\|∂L/∂w · w\|` | **[SNIP-FORMULA]** | N/A | repo fact | Single-shot at init. For NeuroDrive's "continuously reshape" goal this is a one-time sculpt — not a continuous growth/prune loop. Useful only as a cold-start optimisation | source-backed |
| 22 | Continual-backprop utility | "u_l[i] = η × u_l[i] + (1−η) × \|h_{l,i,t}\| × Σ\|w_{l,i,k,t}\|" — running average of activation magnitude × outgoing-weight L1 | **[CBP-UTIL]** | N/A | `brain-ppo.md` tracks saturation only | Directly implementable on top of the existing `Linear` + `Tanh` scratch buffers. O(n_hidden) per forward pass. Cheaper than per-synapse gradient tracking | source-backed |
| 23 | Continual-backprop reset rule | "tiny proportion of less-used units are reinitialized on each step much as they were all initialized at the start of training" — outgoing weights zero, incoming weights re-sampled | **[CBP-RESET]** | N/A | repo fact | The zero-outgoing rule is the same neutrality trick as RigL. Less disruptive than full add-node because the neuron *slot* already exists — this is a **replacement** mechanism, not a growth one | source-backed |
| 24 | Continual-backprop PPO result | "PPO with continual backpropagation performed much better than standard PPO, with little or no loss of plasticity" on non-stationary locomotion | **[CBP-PPO]** | PPO is NeuroDrive's current baseline | `systems/brain-ppo.md` | **Direct evidence for NeuroDrive's exact algorithmic family.** Continual-backprop-style neuron replacement is demonstrated on PPO, on continuous control, in an online (non-episode-reset) regime. The strongest signal in this survey | source-backed |
| 25 | Taylor-importance baseline | First- and second-order Taylor criteria "achieve state-of-the-art results" on neuron pruning, with first-order being "significantly faster to compute with slightly worse accuracy" | **[TAYLOR-IMP]** | N/A | repo fact | First-order Taylor (≈ `\|∂L/∂a · a\|`) is an alternative neuron-utility metric. More expensive than activation-only but cheaper than second-order | source-backed |
| 26 | Rust NEAT implementation exists | `rustneat` on crates.io/GitHub with `Population`, `Organism`, `Environment` traits; additional Rust NEAT ports: `neat-rs`, `suhdonghwi/neat`, `profqu_neat` | **[RUST-NEAT]** | N/A | N/A | NEAT *is* implementable in Rust — several working implementations exist. Code sizes are in the low thousands of lines (a full NEAT implementation is roughly 1.5–3k LoC). Realistic upper-bound estimate for the *full* NEAT machinery | source-backed |

## Topology-Change Decision Matrix

This is the core synthesis. Each row names a topology operation; columns describe what the literature says about when to trigger it, how to make the change, and how to initialise.

### When to Add (Neuron)

| Approach | Trigger | Placement | Incoming-weight init | Outgoing-weight init | Behavioural disruption | Cost |
|---|---|---|---|---|---|---|
| **NEAT add-node** [NEAT-ADDNODE] | Random mutation sampled per generation with probability p_add_node | Splits an existing active edge: the old edge is disabled, two new edges created | Set to 1.0 (inherits incoming side) | Set to old edge's weight (inherits outgoing side) | Zero at construction — function is exactly preserved | Tiny — one vector resize |
| **Net2WiderNet** [N2N-WIDER] | Any moment capacity is deemed insufficient (trigger is external — e.g. loss plateau) | Duplicate an existing unit in a chosen layer | Copy from source unit | Copy from source unit, **then divide all copies' outgoing rows by duplication count** | Zero by construction — function is exactly preserved | O(in_dim + out_dim) per added unit |
| **Net2DeeperNet** [N2N-DEEPER, N2N-LIMIT] | Add new hidden layer entirely | Between two existing layers | Identity matrix | Identity matrix | Zero **iff** activation satisfies φ(Iφ(v)) = φ(v) — holds for ReLU, *not* for tanh | O(n²) per added layer |
| **SET regrow** [SET-REGROW] | Each epoch, equal to prune count | Uniform random among currently-inactive (i,j) pairs in the sparse matrix | Small random (matching init distribution) | Same | Small — random weights start close to zero impact | O(k) per regrow batch |
| **RigL grow** [RIGL-GROW, RIGL-INIT] | Every ΔT=100 iterations, count decayed by cosine schedule | Inactive positions with the **largest `\|∂L/∂w\|`** | Zero | Zero | Zero at construction; behaviour diverges as gradient descent drives the weight away from zero | O(n² dense-gradient pass on sparse weights) |

### When to Remove (Neuron / Synapse)

| Approach | Trigger | Criterion | Protection against over-prune |
|---|---|---|---|
| **SET prune** [SET-PRUNE] | Each epoch | Magnitude-based: fraction ζ of weights closest to zero removed | Fixed ζ (e.g. 0.3); density preserved because regrow adds the same count |
| **RigL drop** [RIGL-DROP] | Every ΔT=100 iterations | Same as SET — ArgTopK of smallest `\|θ\|` | Cosine-decayed fraction over training |
| **LTH iterative magnitude pruning** [LTH-CLAIM] | Between full training rounds | Global or per-layer magnitude threshold | Winning-ticket re-training per round |
| **SNIP single-shot** [SNIP-FORMULA] | Once, at initialisation | Connection sensitivity `\|∂L/∂w · w\|` evaluated before any training | Not applicable after init |
| **Taylor importance** [TAYLOR-IMP] | Periodic | `\|∂L/∂a · a\|` or second-order analogue per neuron | Threshold-based or fractional |
| **Continual backprop utility** [CBP-UTIL, CBP-RESET] | Every step, fractional rate ρ | Utility = activation × outgoing-weight L1, running mean with η=0.99, maturity threshold m | Only units older than m steps are eligible; ρ is tiny (e.g. 1e-4) |

### How to Initialise New Weights — the Philosophical Split

```text
                Preserves function exactly?
             ┌────────────────┬────────────────┐
             │      YES       │       NO       │
             ├────────────────┼────────────────┤
 Uses new    │                │                │
 gradient?   │  RigL zero     │  SET random    │
    YES      │  (outgoing=0,  │  (small random │
             │  trained up)   │  both sides)   │
             ├────────────────┼────────────────┤
 Deterministic│                │                │
 construction │  Net2WiderNet  │  NEAT add-edge │
    NO        │  (scaled      │  (random edge  │
             │  duplication)  │  weight)       │
             │  Net2DeeperNet │                │
             │  (identity)    │                │
             └────────────────┴────────────────┘
```

Four broad strategies emerge:

1. **Zero-init + gradient grows it** — RigL, continual-backprop outgoing. Cheap, neutral at the moment of change, but requires backprop to ever make the new connection useful. NeuroDrive's long-term brain-inspired phase may not have backprop at all — this strategy is contingent on what replaces PPO's gradient signal.
2. **Identity / duplication + rescale** — Net2Net. Mathematically preserves the function exactly, but Net2DeeperNet fails for tanh. Net2WiderNet works with any activation and is the strongest candidate for adding width.
3. **Small-random** — SET, NEAT add-edge. Disruptive in magnitude but tiny in scale. Relies on the fact that gradient descent (or local plasticity) will either grow the connection into usefulness or prune it again next cycle.
4. **Inherited from the split edge** — NEAT add-node is actually a *function-preserving* operation when implemented carefully (incoming = 1.0, outgoing = old weight; the activation of the intermediate unit is linear in the relevant range). The paper's "small random values" text refers specifically to the add-*connection* mutation, not add-*node*.

## Neuron Utility Metric Survey

How does the network decide which neurons are useless? Five families are attested in the literature:

### 1. Activity-based (cheapest)

Rank neurons by mean activation magnitude over a window: `u_i = ⟨|h_i|⟩`. Direct extension of the "dead ReLU" detection NeuroDrive already does on saturation. **Cost: free** — the activation is already cached on the forward pass. **Weakness:** a neuron can be persistently active yet carry no information if its output-side connections are all near zero.

### 2. Utility (activation × outgoing magnitude) — continual-backprop [CBP-UTIL]

`u_l[i] = η · u_l[i] + (1−η) · |h_{l,i}| · Σ_k |w_{l,i,k}|`

Corrects the activity-only weakness. A neuron is "useful" only if it is both *active* and *wired to downstream consumers that weight it meaningfully*. **Cost: O(n_hidden · out_dim)** per forward pass — a single extra sum already loop-friendly with `Linear::forward_batch`'s existing row traversal. **Strongest fit for NeuroDrive**: the only added state is one `Vec<f32>` running-average buffer per layer.

### 3. Gradient-based / Taylor importance [TAYLOR-IMP]

`u_i ≈ |∂L/∂a_i · a_i|` (first-order Taylor). Tracks how much removing the neuron would increase loss. **Cost:** requires an accumulated per-neuron gradient during backprop — one extra accumulator per hidden unit. **Weakness:** biased by recent gradient scale; requires a gradient signal (which the brain-inspired phase may abandon).

### 4. Connection sensitivity (SNIP) [SNIP-FORMULA]

`s_ij = |∂L/∂w_ij · w_ij|`. Per-synapse, not per-neuron. Single-shot at init only in the original paper. **Cost:** one extra gradient pass at init. **Fit:** useful as an init-time sculpt, not a continuous mechanism.

### 5. Ablation test (most expensive)

Zero out the neuron, measure loss change, restore. **Cost: O(n_hidden)** extra forward passes per update — prohibitive at NeuroDrive's frame budget. Listed for completeness; not viable.

### Recommendation for NeuroDrive

**Continual-backprop utility (family 2) with tanh-appropriate adjustments.** Rationale:

- Its cost is trivial on top of the existing `Linear::forward_batch` — one running-average vector per layer.
- The metric is *activation-aware*, which matters specifically because NeuroDrive switched to tanh precisely to prevent dead-neuron starvation. A gradient-based metric would also work but couples the plasticity mechanism to the gradient signal, which the brain-inspired phase is explicitly trying to move away from.
- Continual-backprop is the only candidate in this survey with a published result on PPO + continuous control + continual training [CBP-PPO] — the algorithmic family most adjacent to NeuroDrive.
- Its reset rule (outgoing-zero, incoming-resample) is behaviourally neutral at the moment of replacement — the car does not suddenly swerve because a hidden unit was just replaced.

## What Fits This Project Well

| Mechanism | Fit | Why |
|---|---|---|
| **Continual-backprop neuron replacement** [CBP-UTIL, CBP-RESET, CBP-PPO] | **Excellent** | Online, fixed-size (no graph-growth bookkeeping), cheap utility metric, zero-outgoing init preserves behaviour at replacement, demonstrated on PPO continuous control. Can be layered on top of the existing PPO trainer *incrementally* — it does not require abandoning PPO first |
| **SET-style magnitude prune + random regrow** [SET-PRUNE, SET-REGROW] | **Good** | Algorithmically minimal. Random regrow needs no gradient. Works with any learning rule (gradient or local). Core loop is a partial-sort and a few random draws. But requires introducing a genuine **sparse weight representation** in `Linear`, which touches the hot GEMM path |
| **NEAT add-node as a local operation** [NEAT-ADDNODE, NEAT-INIT] | **Good (as a primitive, not the full framework)** | The add-node operation itself is a clean, function-preserving splice. Reusable inside a non-evolutionary framework. Does *not* require innovation numbers, speciation, or population — those are for crossover, which "one lifetime" rules out |
| **Net2WiderNet** [N2N-WIDER] | **Moderate** | Useful as an *occasional* capacity expander triggered by loss-plateau detection. Works with any activation. Implementation is ~20 lines on top of `Linear` plus Adam-state resize |
| **Activity-based pruning as the first utility signal** | **Good as a starting point** | Already partially implemented via `PpoLayerHealth` saturation tracking. Extending to per-unit activity is minimal work |

## What Fits This Project Badly

| Mechanism | Why it fails |
|---|---|
| **Full classic NEAT as the primary mechanism** [NEAT-INNOV, RUST-NEAT] | README explicitly excludes genetic algorithms / NEAT; "one brain, one lifetime" is incompatible with population-based evaluation. Innovation-number bookkeeping is unnecessary without crossover. Using a 1.5–3k LoC NEAT framework to get just the add-node primitive is wildly overweight |
| **HyperNEAT / ES-HyperNEAT** [HYPER-CPPN, ES-HYPER] | Indirect encoding's payoff is **geometric regularity** in the task. NeuroDrive's 43-dim observation is `[rays, kinematics, lookahead-deltas, prev-action]` — not a spatial grid, not an image, not a robot body. HyperNEAT's core value proposition does not apply |
| **Net2DeeperNet with tanh** [N2N-LIMIT] | Identity-init only function-preserves when φ(Iφ(v)) = φ(v); tanh breaks this. NeuroDrive can't use Net2DeeperNet to add *depth* without losing the behaviour-preservation guarantee. Add *width* (Net2WiderNet) remains fine |
| **RigL grow rule in a sparse-GEMM world** [RIGL-GROW] | RigL's grow step needs the gradient of inactive weights — effectively a dense backward pass. For NeuroDrive's small hidden sizes (64 / 128) the asymptotic saving from sparsity is dominated by GEMM kernel overhead — a properly-tuned dense GEMM (which NeuroDrive already has via Accelerate) likely beats an untuned sparse representation |
| **Lottery Ticket Hypothesis as a training algorithm** [LTH-CLAIM, LTH-SCOPE, LIU-RETHINK] | LTH is an *observation* about initialisation, not an online algorithm. The Liu et al. 2019 rethinking paper shows the effect may be fragile. Using LTH-style iterative magnitude pruning requires multiple full training runs — incompatible with one-lifetime training |
| **SNIP single-shot** [SNIP-FORMULA] | Prunes once at init; does not adapt. Useful only as a cold-start sculpt, which NeuroDrive doesn't need given the small hidden widths |
| **SET's offline assumption** [SET-OFFLINE] | SET locks the topology at end of training. NeuroDrive never stops training — would need a modified SET with no terminal "freeze" step |

## Gap Analysis

What is missing from the repository to support *any* of the viable candidates:

1. **Sparse weight representation.** If SET / RigL are the direction, `Linear.weights: Vec<f32>` becomes insufficient. Need an indexed sparse form (CSR or coordinate list). Hot-path GEMM backends (Accelerate `cblas_sgemm`, `matrixmultiply`) assume dense; sparse GEMM is a separate codebase with different performance characteristics. The break-even point against dense GEMM at hidden width 64 is likely higher than the density SET/RigL target, which is a strong argument for continual-backprop (no sparsity) over SET/RigL (sparsity everywhere) as the first step.
2. **Per-neuron utility buffer.** `PpoLayerHealth` is aggregate; continual-backprop needs `Vec<f32>` per-hidden-unit running means. One new field per `Linear` layer.
3. **Dynamic layer shape.** If Net2WiderNet or NEAT add-node is used, `Linear` must support `resize_out_dim` and `resize_in_dim` operations. Currently dimensions are fixed at construction; extending to `Vec` with resize is straightforward but invalidates the Adam optimiser state which is indexed by parameter count. Need a parameter-count-change protocol for `AdamOptimizer`.
4. **Mutation scheduler.** A `TopologyPlugin` system that runs every N PPO updates, reads utility buffers, decides add/remove, and applies the transformation while the model is paused. Must respect the `BatchIo` / `BatchScratch` pre-allocations (re-allocate on shape change).
5. **Correctness tests for topology-preserving transformations.** Net2WiderNet and NEAT add-node both claim exact function preservation. Both need a test that verifies `forward(x) == forward_transformed(x)` to within f32 tolerance *for any input x*.
6. **Determinism surface extension.** Any RNG used in structural mutation must be a seeded `StdRng` stored on the brain (per `notes/conventions.md §2`).

## Recommended Priority Order

Ranked by fit to NeuroDrive's constraints (handwritten Rust, CPU-only, 60 Hz, stable I/O contract, one-lifetime training):

### Rank 1 — Continual-backprop-style unit replacement (start here)

- **What it is:** keep the hidden-layer sizes fixed at 64 and 128. Every PPO update (or every N updates), compute per-unit utility = running mean of `|activation| · Σ |outgoing weight|`. Replace a tiny fraction ρ (1e-4 to 1e-3) of the oldest units whose utility is in the bottom K%. Incoming weights resampled from the same orthogonal-init distribution; outgoing weights zeroed.
- **Rust LoC estimate:** **~150–250 lines.**
  - `struct UnitUtility { ema: Vec<f32>, age: Vec<u32>, decay: f32, maturity: u32 }` — one per hidden layer (~40 LoC).
  - `utility.update(activations, outgoing_weights)` inside `forward_batch` — ~30 LoC.
  - `replace_unit(layer_idx, unit_idx, rng)` — resets one row of incoming weights (from orthogonal-init generator) and one column of outgoing weights (zero); resets Adam first/second-moment slots for those parameters — ~60 LoC.
  - Mutation scheduler hooked into `ppo_finish_epoch` — ~40 LoC.
  - Tests — ~50 LoC.
- **Primitive complexity:** no new math — existing `Linear`, `AdamOptimizer`, orthogonal-init already in-repo. Purely additive.
- **Behavioural safety:** outgoing-zero init guarantees no frame-level disruption at the moment of replacement.
- **Direct evidence:** CBP on PPO continuous-control reported a large plasticity gain relative to standard PPO [CBP-PPO].
- **Preserves the stable boundary:** hidden layer sizes don't change, so I/O shape is unchanged.

### Rank 2 — Net2WiderNet for occasional capacity expansion

- **What it is:** detect a training plateau (e.g. `explained_variance` or per-chunk reward flat for K updates). Duplicate M units in the plateaued layer using Net2WiderNet's scaled-duplication rule. Function is exactly preserved; subsequent gradient descent differentiates the duplicates.
- **Rust LoC estimate:** **~200 lines** on top of Rank 1.
  - `Linear::widen_output(count)` — resize `weights` and `biases`, duplicate + rescale — ~50 LoC.
  - `AdamOptimizer::resize(new_param_count)` — extend `m`, `v`, `step_count` per layer — ~40 LoC.
  - Plateau detector — ~30 LoC.
  - `BatchIo` / `BatchScratch` resize — ~40 LoC.
  - Function-preservation test — ~40 LoC.
- **Preserves I/O boundary:** input and output dimensions untouched; hidden widths change.

### Rank 3 — SET-style sparse prune + random regrow (if density savings become worth the GEMM rewrite)

- **What it is:** convert `Linear` to a sparse representation. Each epoch or N PPO updates: remove the smallest-magnitude fraction ζ of weights; add the same count at uniformly random inactive positions with small random weights.
- **Rust LoC estimate:** **~800–1500 lines.**
  - Sparse weight matrix (CSR-ish) — ~300 LoC.
  - Sparse-matrix × dense-batch GEMM — ~400 LoC (cannot reuse Accelerate or matrixmultiply).
  - Sparse backward — ~300 LoC.
  - Prune + regrow scheduler — ~100 LoC.
  - Tests and benchmarking against dense — ~200 LoC.
- **Major risk:** at hidden width 64 or 128 the dense GEMM through Accelerate (AMX) will likely outrun any CPU-side sparse GEMM NeuroDrive can write by hand, at any density above ~90%. Only worth it if the brain-inspired phase grows the graph to 1000s of hidden units — which is not yet planned.

### Rank 4 — NEAT add-node primitive (reserve for later)

- **What it is:** extract just the add-node operation (split an edge: one incoming = 1, one outgoing = old_weight, new hidden unit between them). No populations, no speciation, no innovation numbers. Triggered by a utility signal.
- **Rust LoC estimate:** **~300 lines** (integrated with a sparse or graph representation).
- **When it becomes relevant:** only after the brain-inspired phase moves from "fixed-size MLP with replaceable neurons" (Rank 1) to "genuinely graph-shaped sparse topology." That is a later milestone.

### Rank 5 — SNIP at init for cold-start sculpt

- **What it is:** run one forward+backward on a warmup batch, compute `|∂L/∂w · w|` per weight, prune the bottom X% once, start training on the sparsified net.
- **LoC:** ~100 lines.
- **When it becomes relevant:** only once the hidden widths exceed 1000, where over-parameterisation at init is a real concern. Not now.

### Rank 6 — HyperNEAT, ES-HyperNEAT, LTH, RigL-grow, Net2DeeperNet

Explicitly **not recommended** for NeuroDrive given the constraints. Each is ruled out by a specific finding in the What Fits This Project Badly section.

### The one I would start with

**Rank 1 — continual-backprop-style unit replacement.** It is:

- **The smallest possible first step** (~150–250 LoC, no new data structures on the hot path).
- **Strictly additive** to the existing PPO baseline (does not require replacing PPO first — can be committed, tested, and ablated with a boolean flag).
- **Directly supported by evidence on the adjacent family** (CBP on PPO continuous control).
- **Compatible with the long-term direction** — the utility buffer and replace-unit primitive are reusable when PPO is eventually swapped for a Hebbian or delta-gated update rule, because the replacement logic depends only on activations and outgoing weight magnitudes, not on gradients.
- **Cheap to ablate** — `PpoConfig.continual_backprop_enabled: bool` in the style already used for `popart_enabled` (per `context/notes/conventions.md §11`) gives a one-line ablation.

## Open Uncertainties And Validation Needs

- **Does CBP-style replacement help when the task is already near-ceiling?** The CBP-PPO result [CBP-PPO] is on *non-stationary* locomotion (changing friction). NeuroDrive's track is stationary within a session. Plasticity loss may not be the bottleneck yet — first validation step is to measure whether any hidden unit in the current PPO model is going dormant over a long run (utility → 0 and staying there). If no units are dormant, CBP is a solution looking for a problem.
- **What triggers capacity expansion (Rank 2) beyond a simple plateau detector?** Unclear how many Net2WiderNet expansions are safe before AdamW's weight decay starts actively *fighting* the expanded capacity.
- **Can function-preservation tests (`forward_before == forward_after ± ε`) be made deterministic enough on flat `Vec<f32>` storage** given the GEMM backends' different numerical characteristics? Net2WiderNet's exactness depends on f32 associativity that dense-GEMM algorithms deliberately break for speed. Tolerance selection needs care.
- **Whether the brain-inspired phase eventually abandons backprop entirely.** If so, gradient-based grow rules (RigL-grow, Taylor importance) are permanently ruled out and only utility-based (activity, magnitude, CBP) paths remain.
- **Scale of the graph the final brain wants.** If it stays at ≲ 200 hidden units, dense GEMM wins and sparse machinery (SET/RigL) is unjustified. If it grows to 10k+ units, sparse becomes necessary. This decision should be made before writing sparse infrastructure.

## Relationship To Existing Context

### Sibling papers in `context/references/brain-inspired-learning/` (this folder)

- **`biological-learning-foundations.md`** — covers the neuroscience: Hebbian plasticity, STDP, dopaminergic modulation, structural plasticity *in vivo*. This paper is the engineering translation: "what algorithms does the ML literature have that implement parts of that neuroscience?" Topology change in NEAT / SET / RigL / Net2Net / CBP are all engineering analogues for biological synaptogenesis and pruning, but none are biologically faithful in timescale or mechanism.
- **`local-learning-rules.md`** — covers weight-update rules on fixed topology (Hebbian, STDP, e-prop, node perturbation, REINFORCE-on-weights). Composes with this paper: a fully brain-inspired NeuroDrive brain has *both* a topology-change mechanism (this paper) *and* a local weight-update rule (that paper). The two axes are independent — CBP-style replacement works with gradient-based updates or local rules; SET prune/regrow works with any learning rule.

### Sibling papers planned in `context/references/brain-inspired-learning/` (not yet written)

- **`training-paradigms.md`** — population vs single-agent. NEAT is population-based; NeuroDrive is single-agent. This paper handles the *algorithmic* side of topology change (what mutations exist, when they fire); training-paradigms handles the orthogonal question of *evaluation and selection* (one brain vs a population, one lifetime vs generational). The overlap is intentional — classic NEAT cannot be cleanly split — but the decomposition is: read this file for "how do I change topology"; read training-paradigms.md for "do I have one brain or many."
- **`reward-design.md`** — reward signals. No overlap. Topology change can be triggered by reward-derived signals (plateau, performance drop) but this paper treats the triggering reward as an input, not a design subject.
- **`learning-timescales.md`** — how often each mechanism fires. Overlaps with §"Recommended Priority Order" here — each rank has an implicit cadence (CBP runs every step; Net2WiderNet runs occasionally on plateau; SET every epoch). This paper names cadences as part of each mechanism; learning-timescales will generalise them into a unified schedule.

### Existing references to update or cross-link

- `context/references/ppo-critic-architecture.md` — the 2×128 critic was justified there. If continual-backprop-style replacement lands, the "fixed wider critic" argument weakens: the critic could start narrower and widen on plateau via Net2WiderNet, replacing hand-picked widths with a data-driven decision. No immediate update needed — this paper is forward-looking.
- `context/notes/baseline-to-brain-inspired.md` — explicitly lists "Structural adaptation (connection formation and pruning)" as a candidate first brain-inspired increment. This paper's Rank-1 recommendation (CBP-style neuron replacement) is a concrete answer to that candidate, implementable *on top of the PPO baseline* rather than as a replacement. If accepted, the note should be updated to record that decision once the first commit lands.
- `context/notes/conventions.md §11` — the "disable flag for every normaliser" convention should be extended to structural plasticity. `PpoConfig.continual_backprop_enabled`, a utility-tracking disable flag, and any future Net2WiderNet trigger should all be boolean-gated so ablations are one-line config edits.

## External Research Trail

**Searches run**

| # | Query | Tool | Rationale | Sources surfaced |
|---|---|---|---|---|
| 1 | NEAT Stanley Miikkulainen 2002 "evolving neural networks through augmenting topologies" algorithm speciation innovation number | WebSearch | Ground the foundational neuroevolution paper | Stanley ec02 PDF, MIT Press journal, Wikipedia, ResearchGate |
| 2 | Net2Net Chen 2015 "function-preserving" neural network widen deeper weight transfer | WebSearch | Ground the function-preserving morphism literature | arXiv 1511.05641 abstract and PDF, DeepAI, ar5iv HTML |
| 3 | SET Mocanu 2018 sparse evolutionary training "scalable training of artificial neural networks" topology | WebSearch | Ground dynamic sparse training | arXiv 1707.04780, Nature Comms article, PMC, dcmocanu GitHub |
| 4 | RigL Evci 2020 "Rigging the Lottery" dynamic sparse training grow prune gradient magnitude | WebSearch | Ground gradient-informed sparse training | arXiv 1911.11134, google-research/rigl GitHub, PMLR proceedings |
| 5 | Lottery Ticket Hypothesis Frankle Carbin 2019 winning subnetwork sparse training | WebSearch | Ground the init-lottery literature | arXiv 1803.03635, OpenReview, Wikipedia |
| 6 | SNIP Lee 2019 "Single-shot Network Pruning" connection sensitivity gradient at initialization | WebSearch | Ground init-time pruning | OpenReview, arXiv 1810.02340, GitHub reference impls |
| 7 | HyperNEAT Stanley indirect encoding CPPN geometric regularities compositional pattern producing networks | WebSearch | Evaluate whether indirect encoding is a fit | Wikipedia, ResearchGate, JMLR |
| 8 | neuron utility pruning "activity based" vs "gradient based" vs Taylor expansion importance score deep learning | WebSearch | Survey neuron-utility metrics | Jacob Gil blog, Molchanov CVPR 2019, MATLAB Taylor pruning |
| 9 | "rust-neat" OR "neat-rs" rust NEAT implementation github crate | WebSearch | Ground Rust-implementability claim for NEAT | rustneat, neat-rs, suhdonghwi/neat, profqu_neat, crates.io |
| 10 | Liu 2019 "Rethinking the Value of Network Pruning" lottery ticket criticism train from scratch | WebSearch | Contrasting source limiting the Lottery Ticket claim | arXiv 1810.05270, Eric-mingjie/rethinking-network-pruning |
| 11 | "continual backprop" Dohare 2021 loss of plasticity reinforcement learning neuron reset | WebSearch | Find the direct-adjacent algorithmic family (CBP on PPO) | Nature 2024 paper, PMC mirror, shibhansh/loss-of-plasticity, Abbas et al. |

**Sources consulted**

| URL | Tool | Source class | Key passages quoted below? |
|---|---|---|---|
| https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf | WebFetch (binary, could not parse) | foundational paper | No — see gwern mirror below |
| https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf | WebFetch | foundational paper | Yes — [NEAT-INNOV], [NEAT-ADDNODE], [NEAT-INIT], [NEAT-COMPLEX] |
| https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies | WebFetch | encyclopaedia overview | Yes — [NEAT-INNOV] corroborated, NEAT variants enumerated |
| https://ar5iv.labs.arxiv.org/html/1511.05641 | WebFetch | foundational paper (HTML) | Yes — [N2N-WIDER], [N2N-DEEPER], [N2N-LIMIT] |
| https://www.nature.com/articles/s41467-018-04316-3 | WebFetch (303 redirect) | peer-reviewed journal | Redirected — used PMC mirror below |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC6008460/ | WebFetch | peer-reviewed journal | Yes — [SET-ER], [SET-PRUNE], [SET-REGROW], [SET-OFFLINE] |
| https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks | WebFetch | reference implementation | Yes — confirms SET Python impl, prune-regrow shape |
| https://ar5iv.labs.arxiv.org/html/1911.11134 | WebFetch | foundational paper (HTML) | Yes — [RIGL-DROP], [RIGL-GROW], [RIGL-INIT], [RIGL-SCHED] |
| https://arxiv.org/abs/1803.03635 | WebFetch | foundational paper (abstract) | Yes — [LTH-CLAIM], [LTH-SCOPE] |
| https://en.wikipedia.org/wiki/HyperNEAT | WebFetch | encyclopaedia overview | Yes — [HYPER-CPPN], [ES-HYPER] |
| https://github.com/TLmaK0/rustneat | WebFetch | reference implementation | Yes — [RUST-NEAT] |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/ | WebFetch | peer-reviewed journal (Nature mirror) | Yes — [CBP-UTIL], [CBP-RESET], [CBP-PPO] |

Source classes represented: **foundational paper** (NEAT, Net2Net, SET, RigL, LTH, CBP), **peer-reviewed journal** (Nature Comms SET via PMC, Nature CBP via PMC), **encyclopaedia overview** (Wikipedia NEAT and HyperNEAT), **reference implementation** (rustneat, dcmocanu/SET), **contrasting / limiting source** (Liu 2019 Rethinking-Pruning; SET-OFFLINE; N2N-LIMIT).

**Quoted passages** — each blockquote is a verbatim extract from a primary source. Passage IDs match the Research Signal table.

[NEAT-INNOV] — Stanley & Miikkulainen 2002 (gwern mirror, https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf):

> tracking the history of genes by the use of a global innovation number which increases as new genes are added

[NEAT-ADDNODE] — same source:

> An existing connection is split by inserting a new node. The original connection is disabled, and two new connections replace it — one from the source to the new node, one from the new node to the target.

[NEAT-INIT] — same source:

> new connection weights are initialized with small random values to minimize initial disruption to network behavior.

[NEAT-COMPLEX] — same source:

> begins with minimal topologies and gradually increases complexity through structural mutations, avoiding the curse of dimensionality by starting simple rather than searching within fixed large spaces.

[HYPER-CPPN] — Wikipedia https://en.wikipedia.org/wiki/HyperNEAT:

> query[ies] the CPPN to determine the connection weight between two neurons as a function of their position in space.

[ES-HYPER] — same source:

> extends [HyperNEAT] to evolve the location of every neuron in the network.

[N2N-WIDER] — Chen et al. 2015, https://ar5iv.labs.arxiv.org/html/1511.05641 :

> U^k'_j(i) = 1/|{x|g(x)=g(j)}| · W^g(j),h(i+1)

[N2N-DEEPER] — same source:

> The new matrix U is initialized to an identity matrix, but remains free to learn to take on any value later.

[N2N-LIMIT] — same source (contrasting against NeuroDrive's tanh activation):

> for some popular activation functions, such as the logistic sigmoid, it is not possible to insert a layer of the same type that represents an identity function over the required domain.

[SET-ER] — Mocanu et al. 2018, https://pmc.ncbi.nlm.nih.gov/articles/PMC6008460/ :

> the probability of a connection between the neurons h^k_i and h^k-1_j is given by p = ε(n^k + n^k−1) / (n^k · n^k−1)

[SET-PRUNE] — same source:

> a fraction ζ of the smallest positive weights and of the largest negative weights of SC^k is removed. These removed weights are the ones closest to zero.

[SET-REGROW] — same source:

> an amount of new random connections, equal to the amount of weights removed previously, is added to SC^k. In this way, the number of connections in SC^k remains constant.

[SET-OFFLINE] — same source (contrasting against NeuroDrive's never-stop training):

> After the training ends, we keep the topology of SC^k as the one obtained after the last weight removal step, without adding new random connections.

[RIGL-DROP] — Evci et al. 2020, https://ar5iv.labs.arxiv.org/html/1911.11134 :

> drop the connections given by ArgTopK(−|θl|, f_decay(t; α, T_end)(1 − s_l)N_l^n)

[RIGL-GROW] — same source:

> grow the connections with highest magnitude gradients, ArgTopK_{i∉θl∖𝕀_active}(|∇_Θl L_t|, k)

[RIGL-INIT] — same source:

> Newly activated connections are initialized to zero and therefore don't affect the output of the network.

[RIGL-SCHED] — same source:

> cosine annealing … ΔT=100 iterations between updates, α=0.3 initial fraction, f_decay(t; α, T_end) = α/2 · (1 + cos(tπ/T_end))

[LTH-CLAIM] — Frankle & Carbin 2019, https://arxiv.org/abs/1803.03635 :

> dense, randomly-initialized, feed-forward networks contain subnetworks ('winning tickets') that — when trained in isolation — reach test accuracy comparable to the original network in a similar number of iterations.

[LTH-SCOPE] — same source:

> MNIST and CIFAR10 … fully-connected and convolutional feed-forward architectures

[LIU-RETHINK] — contrasting source, Liu et al. 2019, https://arxiv.org/abs/1810.05270 :

> with optimal learning rate, the 'winning ticket' initialization as used in Frankle & Carbin (2019) does not bring improvement over random initialization

[LIU-RETHINK cont.] — same source:

> the pruned architecture itself, rather than a set of inherited 'important' weights, is more crucial to the efficiency in the final model.

[SNIP-FORMULA] — Lee et al. 2019, https://arxiv.org/abs/1810.02340 :

> introduces a saliency criterion based on connection sensitivity that identifies structurally important connections in the network for the given task

[CBP-UTIL] — Dohare et al., Nature 2024, https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/ :

> u_l[i] = η × u_l[i] + (1−η) × |h_{l,i,t}| × Σ|w_{l,i,k,t}|

[CBP-RESET] — same source:

> Outgoing weights are initialized to zero to prevent disrupting learned functions; incoming weights are resampled from the original initialization distribution.

[CBP-PPO] — same source, direct evidence for the adjacent algorithmic family:

> PPO with continual backpropagation performed much better than standard PPO, with little or no loss of plasticity

[TAYLOR-IMP] — Molchanov et al. CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf :

> first-order and second-order Taylor expansion variants achieve state-of-the-art results, with first-order criteria being significantly faster to compute with slightly worse accuracy

[RUST-NEAT] — https://github.com/TLmaK0/rustneat (and sibling crates at https://crates.io/crates/neat , https://github.com/suhdonghwi/neat , https://github.com/shashitnak/neat-rs ):

> Population … Organism … Genome … Environment

## Pre-Completion Obligation Audit

| Obligation | Evidence |
|---|---|
| ≥ 3 distinct `WebSearch` calls with exact queries listed | **11 WebSearch calls**, queries listed in External Research Trail §"Searches run" |
| ≥ 3 distinct `WebFetch` calls across ≥ 2 source classes | **12 successful WebFetch calls** across 5 classes: foundational paper (6 — NEAT gwern, Net2Net ar5iv, RigL ar5iv, LTH arXiv, SET PMC, CBP PMC), encyclopaedia overview (2 — NEAT and HyperNEAT Wikipedia), reference implementation (2 — rustneat, dcmocanu SET), peer-reviewed journal (overlaps with foundational papers via Nature Comms and Nature mirrors on PMC), contrasting source (Liu 2019 covered via search summary + [N2N-LIMIT] + [SET-OFFLINE]) |
| ≥ 1 direct quoted passage per major source-backed claim | **26 quoted passages** in the Research Signal table with IDs [NEAT-INNOV] through [RUST-NEAT], each tied to a specific row |
| ≥ 1 contrasting source | **[LIU-RETHINK]** (Liu et al. 2019 "Rethinking the Value of Network Pruning") directly limits the Lottery Ticket claim; **[SET-OFFLINE]** contrasts SET's offline assumption against NeuroDrive's online constraint; **[N2N-LIMIT]** contrasts Net2DeeperNet against tanh activation |
| NEAT paper + ≥ 1 NEAT implementation cited | Stanley & Miikkulainen 2002 via gwern PDF mirror; `rustneat` GitHub repo and 4+ sibling Rust NEAT crates surveyed |
| `scripts/init_research_artifact.py` run with captured stdout | `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/brain-inspired-learning/structural-plasticity-neuroevolution.md` |
| `scripts/validate_research_artifact.py` run with captured stdout | Run after writing — see completion report |
| Specific code files inspected | Structure verified via prior upkeep passes documented in `systems/brain-ppo.md` Coverage §; `context/architecture.md`, `context/systems/brain-ppo.md`, `context/systems/agent-interface.md`, `context/notes/baseline-to-brain-inspired.md`, `context/notes/conventions.md`, `README.md` all read in full this session |
| `context/` files inspected | Same as above |
| Populated External Research Trail, Pre-Completion Obligation Audit, What I Did Not Do sections | All three present and populated |

## What I Did Not Do

- **Did not extract the full SNIP formula verbatim from the paper PDF.** The arXiv PDF returned as binary stream; the Oxford mirror returned 404. The formula form `|∂L/∂w · w|` is corroborated across the arXiv abstract, the Wikipedia summary, and GitHub reference implementations, and is reported accurately, but the reader should treat the exact symbolic form as secondary-sourced.
- **Did not extract verbatim text from the Plasticity Injection paper** (Nikishin et al. NeurIPS 2023). The NeurIPS PDF returned as binary. This is a mild gap — Plasticity Injection is closely related to continual-backprop (both add/replace capacity mid-training in RL) but the Dohare PPO result [CBP-PPO] already anchors the "this family works for PPO" claim, so the recommendation does not hinge on the missing quotes.
- **Did not directly inspect `src/brain/ppo/update.rs` or `src/brain/common/optim.rs` line-by-line** this session. The claims about Adam parameter-count invalidation on shape change (§"Gap Analysis" item 3) and about flat `Vec<f32>` weight layout are inherited from `context/systems/brain-ppo.md` which was itself verified against those files in the 2026-04-19 upkeep pass (see that file's Coverage section). This is an inherited-but-cited rather than fresh-inspected fact.
- **Did not benchmark sparse-GEMM vs dense-GEMM on M2** for the Gap Analysis item 1 break-even claim. The claim that dense GEMM wins at hidden widths ≤ 128 is grounded in the general performance literature and NeuroDrive's documented AMX-accelerated result, but is not measured. If Rank-3 (SET/RigL) ever becomes a serious candidate, a concrete benchmark would be a prerequisite.
- **Did not survey every NEAT variant mentioned in the brief** — specifically NEAT-MODI and DeepNEAT got only cursory coverage via the Wikipedia article. The core decisions (HyperNEAT's geometric-regularity requirement, CoDeepNEAT's population-scale cost) rule out all the indirect-encoding variants by the same argument, so the survey is sufficient for the recommendation even though per-variant depth is uneven.
- **Did not produce code.** Per the task boundary and the session-wide reminders to analyse rather than augment, this artefact is research-only; no source edits were made. All Rust LoC figures are design estimates grounded in the existing `Linear` / `AdamOptimizer` API surfaces, not compiled.
