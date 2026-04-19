# Brain-Inspired Learning — Overview

This is the "read this first" document for the `brain-inspired-learning/` research folder. It synthesises the seven deep-dive research papers in this folder into one coherent picture and ties them to the concrete v1 design decisions (captured in `context/notes/brain-v1-design.md`) and the project's guiding discipline (`context/notes/biology-first-principle.md`).

If you have time to read only one document in this folder, read this one. If you have time for more, each deep-dive paper is linked in §Relationship to Sibling Papers.

## Scope and Origin

Produced as the synthesis of a seven-paper parallel research round dispatched on 2026-04-19. Seven Opus-backed `project-research` agents investigated independent angles of the brain-inspired learning design space, each producing a rigorous reference artefact with ≥8 WebSearch calls, ≥5 WebFetch calls, contrasting sources, and falsification conditions.

The research round came after NeuroDrive's Milestone 5 (critic target-scaling for PPO) completed and was validated by `reports/analytics/run_1776556719.md`. The environment, observation contract, reward structure, and analytics pipeline are confirmed learnable. The project now transitions to the brain-inspired arc.

## The Seven Threads

| # | File | Question it answered |
|---|------|----------------------|
| 1 | `biological-learning-foundations.md` | What does a biological brain actually do when a human learns a motor skill? Which mechanisms are settled science? |
| 2 | `local-learning-rules.md` | What non-backprop learning algorithms exist and which are suitable for continuous-control racing? |
| 3 | `structural-plasticity-neuroevolution.md` | How do you have a neural network that grows, shrinks, and reshapes during training? |
| 4 | `training-paradigms.md` | 150-cars GA vs 3–8 cars with plasticity vs hybrid — which fits NeuroDrive? |
| 5 | `reward-design.md` | How should reward signals feed into brain-inspired plasticity? Compatibility with entertainment-first reward? |
| 6 | `learning-timescales.md` | Fast synaptic vs slow structural vs developmental timescales — which matter for NeuroDrive? |
| 7 | `transfer-and-curriculum.md` | Does the first brain-inspired implementation need curriculum / transfer / continual-learning support? |

## Headline Conclusions

The seven papers converged on a single consistent design. Every paper independently pushed the v1 scope **tighter and smaller** than the research round's starting hypothesis:

| Decision | Converged conclusion |
|----------|----------------------|
| Training paradigm | Lifelong single-agent plasticity in the 8-car runtime. **Not** evolution, **not** population-based. |
| Learning rule | Three-factor rate-coded plasticity with eligibility traces. |
| Reward signal | Unchanged. Velocity projection + centreline. Entertainment-first. |
| Neuromodulator | Per-tick reward directly. No value predictor in v1. |
| Timescales | Eligibility trace (seconds) + slow weight + critical-period hygiene. Nothing more. |
| Scope | Single task, single track, no curriculum, no transfer, no PPO-as-teacher. |
| Structural plasticity | Continual-backprop-style neuron replacement + plateau-triggered width growth. |
| Integration | New `AgentMode::BrainInspired`. F4 becomes three-way. Stable observation/action contracts. |

## The v1 Design in One Picture

```
┌──────────────────────────────────────────────────────────────────────┐
│   NEURODRIVE BRAIN-INSPIRED v1 (Milestone 6)                         │
│                                                                      │
│   Structure:        Sparse directed graph of rate-coded tanh         │
│                     neurons. Cyclic connections allowed. No layers.  │
│                                                                      │
│   Input:            43 reserved neurons bound to the observation     │
│                     contract (rays, kinematics, lookahead, etc.)     │
│                                                                      │
│   Output:           2 reserved neurons bound to steering + throttle. │
│                                                                      │
│   Hidden neurons:   ~15 at seed time. Grows via plasticity to a few  │
│                     hundred over training.                           │
│                                                                      │
│   Synapses:         Sparse edges with weight and eligibility trace.  │
│                     ~10% initial density. Pruned + sprouted over     │
│                     time.                                            │
│                                                                      │
│   Forward pass:     One step per tick. Each neuron sums weighted     │
│                     inputs from PREVIOUS-tick activations of its     │
│                     source neurons. No settling loop.                │
│                                                                      │
│   Learning rule:    Three-factor plasticity —                        │
│                       e_ij ← λ·e_ij + pre_i · post_j                 │
│                       δw_ij = η · M · e_ij                           │
│                     where M = per-tick reward (Option C, no critic). │
│                                                                      │
│   Homeostasis:      Synaptic scaling + intrinsic excitability        │
│                     homeostat. Both slow compared to plasticity.     │
│                                                                      │
│   Structure change: Continual-backprop adapted to graph —            │
│                     • Low-utility neurons replaced (edges rewired)   │
│                     • Plateau-triggered neurogenesis (Net2Wider)     │
│                     • Below-threshold synapses pruned                │
│                     • Co-active unconnected neurons sprout edges     │
│                                                                      │
│   Activation:       tanh throughout. ReLU rejected from PPO-era      │
│                     experience (34–57% dead neurons).                │
│                                                                      │
│   Performance:      At 500 neurons / 5000 synapses / 8 cars,         │
│                     forward pass is ~120 µs per tick. Well within    │
│                     the frame budget headroom from the PPO perf      │
│                     overhaul.                                        │
│                                                                      │
│   LoC estimate:     ~700–900 on top of existing Rust primitives.     │
└──────────────────────────────────────────────────────────────────────┘
```

## What Each Paper Contributed

### 1. Biological Learning Foundations

Established the biology baseline. Distilled decades of neuroscience into a **five-mechanism minimum viable substrate**:

1. Eligibility traces (per-synapse short-term memory)
2. Rate-coded Hebbian with BCM threshold
3. Dopamine-RPE-like gating of plasticity
4. Synaptic scaling
5. Intrinsic excitability homeostasis

All five are biologically settled (not speculative) and computationally tractable in a 60 Hz fixed-tick runtime. Spiking-STDP deferred because spike-timing precision < 1 tick is not representable.

Most impactful contribution: legitimised the **reuse PPO's δ as dopamine-RPE** recommendation. We ultimately rejected it in favour of Option C, but the paper's framing (dopamine IS the TD error computationally) is load-bearing for understanding how a future plastic value predictor should behave.

### 2. Local Learning Rules

Compared ten non-backprop algorithms (Hebbian, Oja's, STDP, predictive coding, forward-forward, feedback alignment, three-factor, echo-state reservoir, neuromodulated plasticity, meta-learned Hebbian) on biological plausibility, continuous-control track record, and implementation cost.

**Winner: three-factor rate-coded plasticity with eligibility traces.**

- Simple formula: `e ← λe + pre·post; δw = η·M·e`
- Well-studied in neuroscience (Frémaux & Gerstner 2015).
- Published RL success on maze navigation (Backpropamine, Uber AI 2020).
- Most importantly: **topology-agnostic** — works on layered or graph structures.
- ~400–600 LoC on top of existing primitives.

Second pick: echo-state reservoir (even cheaper, de-risks harness). Third pick: meta-learned Hebbian ABCD (Najarro/Risi 2020) — **has CarRacing-v0 demo**, directly analogous to NeuroDrive, but requires outer-loop ES meta-learning which conflicts with our "one brain, one lifetime" framing.

Flagged tension: pure-plasticity-from-scratch-without-outer-loop is empirically under-demonstrated. If v1 fails, the natural fallback (ES meta-learning of plasticity coefficients) is a form of evolution that conflicts with the README.

### 3. Structural Plasticity and Neuroevolution

Directly addressed the "NEAT-like growth, pruning" vision. Surveyed NEAT, HyperNEAT, neurogenesis methods (Net2Net, progressive nets), dynamic sparse training (SET, RigL), and pruning (SNIP, Lottery Ticket).

**Winner: continual backprop** (Dohare et al., Nature 2024).

- The only candidate in the survey with published PPO + continuous-control + continual-training results.
- ~150–250 LoC.
- Per-neuron utility metric; low-utility neurons replaced with zero-outgoing / resample-incoming. Behaviour-preserving at the moment of replacement.
- Generalises cleanly to graph topology (replace layered "column" with graph "edge set").

Second pick: Net2WiderNet for plateau-triggered width growth.

Classic NEAT **ruled out** because it's population-based (violates "one brain, one lifetime"). NEAT's add-node primitive is reusable conceptually.

Flagged tension: **Net2DeeperNet requires ReLU** — identity-preservation fails with tanh. NeuroDrive is tanh-committed, so we grow width, not depth. Depth stays fixed at whatever emerges from the seed graph.

### 4. Training Paradigms

Directly answered the "150 cars GA vs 3–8 with plasticity" question.

**Winner: lifelong single-agent plasticity in the existing 8-car runtime.** No evolution.

- PPO outperforms ES/NEAT on continuous control per the Frontiers 2020 benchmark (contrasting source cited against the ES hype).
- M2 MacBook Air has no GPU, erasing ES/GA's scaling advantage.
- Principled match to README's "one brain, one lifetime" framing.
- Reuses existing 8-car runtime; ~300 LoC runtime delta.

Deferred indefinitely: ES, NEAT population evolution, Population-Based Training, Deep GA. These would all require dozens-to-hundreds of parallel agents that NeuroDrive's real-time visual runtime cannot host efficiently on the M2.

Acknowledged: a headless training binary is cheap to add later (Bevy ships a headless example). But even headless, M2's single-CPU throughput doesn't make 150-agent evolution competitive with in-lifetime plasticity.

### 5. Reward Design

Answered "is the entertainment-first reward compatible with plasticity?" — **yes, unchanged**.

- Every existing reward term (velocity projection + centreline proximity) is compatible with three-factor plastic learning without modification.
- M6 is a pure learning-rule migration, not an environment change.

Concrete specifications added:

- Neuromodulator formula for a future plastic critic: `M = r + γV(s') − V(s)`.
- Eligibility decay `τ_e ≈ 2 s` (120 ticks at 60 Hz) — biologically measured window, self-consistent with γ=0.995's credit horizon.
- Multi-channel `Resource<Neuromodulator>` struct — dopamine channel populated in v1, novelty/salience channels reserved for M9.

Contrasting source honoured: Berridge's "dopamine is salience, not RPE" critique. We claim only that "an RPE-gated three-factor rule can learn racing", not that we are modelling dopamine biology exactly.

### 6. Learning Timescales

Answered "which timescales matter for NeuroDrive?" with a minimum-viable-stack of three:

1. **Eligibility trace** (seconds) — inside the three-factor rule.
2. **Slow weight** (minutes–hours within a session) — the plastic weights themselves.
3. **Critical-period hygiene** — keep plasticity hyperparameters stable in early training (Achille 2018: perturbations in critical periods have disproportionate impact).

**Deferred to later milestones:** experience replay, sleep-like consolidation, complex-synapse (Benna–Fusi) memory models.

The contrasting source (arXiv 2506.04377, "Replay can provably increase forgetting") was decisive — replay is not free. Combined with the transfer-and-curriculum paper's "we have one task, no forgetting to defend against", replay is unjustified in v1.

### 7. Transfer and Curriculum

Scope-protection paper. Argued explicitly for **one-shot single-task single-track v1** with:

- No curriculum (we have no multiple tracks to order)
- No transfer (we have no source task)
- No continual-learning defences (single-task means no forgetting)
- No meta-learning (meta-learning requires a task distribution)
- **No PPO-as-teacher** — explicitly rejected on four README-quoted grounds: "from scratch", "one persistent brain", "does not train against a single static dataset", "controlled experiment"

Contrasting source: Wu, Dyer, Neyshabur 2021 "When Do Curricula Work?" directly contradicts Bengio 2009 curriculum enthusiasm. Curricula help only in narrow conditions.

Three named triggers would legitimately reintroduce this machinery in future milestones:

1. Multi-track support lands → curriculum and transfer become real.
2. v1 fails despite correct design → PPO-as-teacher becomes a debugging tool, not a shortcut.
3. A task distribution emerges → meta-learning has something to meta-learn over.

None are active today.

## Two Genuinely Elegant Outcomes

### 1. PPO becomes a component of the brain-inspired learner (then later gets decoupled)

Although v1 rejects Option A (reusing PPO's critic as dopamine), the **reward-design paper's framing** that PPO's TD error IS the computational definition of dopamine informs M8's design: the plastic value predictor we build in M8 should mirror what PPO's critic does, but via plasticity instead of backprop. PPO's success becomes the existence proof that such a value predictor CAN learn this task — we just need to build one from local plasticity.

### 2. The "NEAT-like growth, pruning" vision is preserved, just implemented differently

User intent: "useless neurons pruned, new synapses form, neural net expands." This is v1 territory, not future work. Continual backprop + Net2WiderNet on graph topology = exactly this behaviour, without NEAT's population evolution wrapper. Growth and pruning happen **within one agent's lifetime**, not across generations.

## The Biology-First Principle

All seven papers implicitly respected the principle NeuroDrive explicitly articulated in `notes/biology-first-principle.md`: **when hitting a problem, the answer comes from biology, not the ML toolkit.**

Concrete examples from the research:

- Papers recommend synaptic scaling (biological) instead of weight decay (ML-toolkit).
- Papers recommend intrinsic excitability homeostasis (biological) instead of batch norm (ML-toolkit).
- Papers recommend eligibility traces (biological) instead of TD(λ) (RL-toolkit).
- Papers recommend continual backprop (biological utility-based replacement) instead of lottery-ticket pruning (ML-toolkit).

This is not cosmetic — every recommendation is traceable to a biological mechanism, not an engineering convenience.

## Open Uncertainties

The research converged remarkably well, but some questions remain:

1. **Will pure-plasticity-from-scratch work?** The strongest published results (Najarro 2020, CarRacing-v0) use outer-loop ES meta-learning. Our stance is purer but empirically under-demonstrated.
2. **Shared substrate vs 8 independent brains for the 8-car fleet?** Both workable. Shared is cheaper, independent gives within-run diversity. Defer to M6 plan file.
3. **Initial graph density and size?** Rough starting point is ~15 hidden neurons at ~10% density. Will need tuning based on v1 runs.
4. **When to trigger structural plasticity?** Per-episode, per-N-episodes, per-plateau? Likely plateau-triggered, but the exact metric needs design in the M6 plan.
5. **Per-neuron utility metric for continual backprop.** Several biological and computational candidates; the structural-plasticity paper recommended "mean absolute output × mean absolute sum of outgoing weights", which is a reasonable starting heuristic.

## Relationship to Sibling Papers

| Paper | Primary role in the design |
|-------|----------------------------|
| `biological-learning-foundations.md` | Defines what's biologically settled vs speculative; names the five-mechanism minimum viable substrate |
| `local-learning-rules.md` | Selects the specific learning rule (three-factor plasticity with eligibility traces) |
| `structural-plasticity-neuroevolution.md` | Selects the structural-plasticity algorithm (continual backprop) |
| `training-paradigms.md` | Rules out population/evolution paradigms; confirms single-agent lifelong plasticity |
| `reward-design.md` | Confirms existing reward is compatible unchanged; specifies τ_e and multi-channel Neuromodulator |
| `learning-timescales.md` | Constrains the minimum viable timescale stack to three |
| `transfer-and-curriculum.md` | Protects scope; explicitly rules out all curriculum / transfer / meta-learning / PPO-as-teacher for v1 |

## Recommendation for NeuroDrive

**Milestone 6 (brain-inspired v1)** should ship exactly the design in the ASCII box above. Nothing more, nothing less.

Subsequent milestones (M7 visualisation, M8 plastic value predictor, M9 multi-neuromodulator, and the Long-Term Plan items — Dale's law, synaptic delays, short-term dynamics, multiple neuron types, sleep/replay, spiking+STDP) are not urgent and will be pulled forward as specific pathologies emerge. **The biology-first principle governs which Long-Term Plan item addresses which pathology.**

Concretely, the v1 implementation plan should cover:

1. Data structures (`Neuron`, `Synapse`, graph storage).
2. Forward pass (one-step propagation, handling cycles).
3. Three-factor plasticity update.
4. Homeostasis update schedule (slower than plasticity).
5. Structural plasticity triggers and operations.
6. Integration with `AgentMode`, `ObservationVector`, `ActionState`.
7. Test strategy (deterministic unit tests for each rule; regression tests for graph integrity).
8. Analytics integration (adapt existing capture to the new controller type).

That plan belongs in `context/plans/brain-inspired-v1.md`, to be drafted when you decide to commit to the implementation.

## What This Paper Is Not

- It is not a replacement for the seven deep-dive papers — if you need the citations, contrasting sources, or verbatim quotes, go to the source paper.
- It is not a v1 implementation plan — that lives in `context/plans/brain-inspired-v1.md` when written.
- It is not a commitment to Milestone 6 starting immediately — the transition is decided in `notes/baseline-to-brain-inspired.md`.
- It is not neutral on design — it commits to Option C (no critic in v1), graph topology (not layered), continual backprop (not NEAT). Those commitments reflect the seven-paper consensus as filtered through the biology-first principle.

## External Research Trail

All quoted passages and URLs live in the seven deep-dive papers in this folder. This overview does not re-quote them. Total research footprint across the seven papers: ~75 WebSearch calls, ~55 successful WebFetch calls, ≥7 contrasting sources, ≥40 direct quoted passages across ≥7 source classes (foundational papers, reference implementations, peer-reviewed evaluations, author-of-record blogs, contrasting critiques, encyclopedic summaries, primary experimental papers).
