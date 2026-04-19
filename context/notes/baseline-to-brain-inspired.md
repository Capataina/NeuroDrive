# Baseline → Brain-Inspired

Captures the transition from Milestone 5 (PPO baseline with critic target-scaling, validated) to Milestone 6 and beyond (the brain-inspired arc). Last refreshed 2026-04-19 after the seven-paper research round landed and the v1 design was settled.

## Current Understanding

As of 2026-04-19, NeuroDrive has **completed Milestones 1–5**:

- M1 — Environment + keyboard controller
- M2 — PPO baseline from scratch (actor-critic, GAE, clipped surrogate)
- M3 — Multi-car vectorisation + comprehensive analytics pipeline
- M4 — Performance overhaul (dual GEMM backend, batched actor, 21× frame-time improvement)
- M5 — Critic target-scaling (PopArt, γ=0.995, observation normalisation, target-KL early stop)

M5's validation (`reports/analytics/run_1776556719.md`) showed all 8 cars completing the full track loop, fleet max-progress spread 1.1%, crash rate falling from 100% to 56% in the best chunk, and pre-crash analytics confirming the policy anticipates (96% of crashes had throttle released > 0.25 s before impact). The environment, observation contract, and reward shaping are **confirmed learnable**.

This was the goal of the baseline phase: prove the environment is learnable before committing to the harder biology-first learning-rules work. **That proof is complete.** The project now transitions to the brain-inspired arc.

## The Transition Framing

The PPO work was never the destination — it was environment validation. The project's stated intent per `README.md` has always been brain-inspired local plasticity. The baseline phase existed specifically to:

- Validate that a 43-dim observation + 2-dim action contract is sufficient for learning this task.
- Validate that the velocity-projection + centreline reward produces entertaining driving without reward hacks.
- Provide a permanent diagnostic baseline for future environment changes — if the brain-inspired learner fails and PPO still works, we know the environment is fine.

**PPO stays permanently live** as a diagnostic controller (per the three-way `AgentMode` toggle: Keyboard / PPO / Brain-Inspired). It is not being retired. It is not being replaced. The brain-inspired learner is **additive**, not a replacement.

## Research Round (2026-04-19)

Seven parallel project-research agents on Opus produced deep analysis across the brain-inspired design space. All seven outputs live in `context/references/brain-inspired-learning/`:

1. `biological-learning-foundations.md` — the neuroscience baseline
2. `local-learning-rules.md` — computational non-backprop algorithms
3. `structural-plasticity-neuroevolution.md` — topology change
4. `training-paradigms.md` — population vs lifelong-plasticity
5. `reward-design.md` — how reward signals feed into plasticity
6. `learning-timescales.md` — fast synaptic vs slow structural dynamics
7. `transfer-and-curriculum.md` — scope protection

All seven converged on a consistent design and it is captured in `overview.md` and in `notes/brain-v1-design.md`.

## Decisions Settled in This Transition

### The Biology-First Principle

The single most important decision of the transition: **when we hit a problem, the answer comes from biology, not the ML toolkit.** See `notes/biology-first-principle.md` for the full articulation.

This changes how every future decision is made. It rules out reaching for dropout, batch norm, experience replay, or other ML-toolkit defaults as responses to problems unless they have a direct biological analogue.

### The v1 Substrate

Concrete v1 design captured in `notes/brain-v1-design.md`:

- **Graph topology** (not layered). Sparse edge list, cyclic allowed, one-step propagation per tick.
- **Three-factor plasticity with eligibility traces.** `e ← λe + pre·post; δw = η·M·e`.
- **Raw per-tick reward as the modulator (Option C).** No value predictor in v1. No borrowed critic from PPO.
- **Homeostatic scaling + intrinsic excitability** — two settled biological mechanisms running alongside plasticity.
- **Structural plasticity via continual backprop** adapted to graph form. Low-utility neurons replaced, plateau-triggered neurogenesis, synapse pruning + sprouting.
- **Tanh activation** (matches PPO baseline; avoids dead-neuron failure seen with ReLU).
- **Fixed depth, variable width** (Net2DeeperNet is incompatible with tanh; we grow the graph's width, not depth).
- **Reserved I/O neurons** preserving the stable 43-dim observation and 2-dim action contracts.

### What Does NOT Change

- **Reward structure:** unchanged. Velocity projection + centreline proximity, no crash penalty, entertainment-first. Compatible with plasticity unchanged.
- **Observation contract:** 43-dim, stable. Both PPO and brain-inspired consume it identically.
- **Action contract:** 2-dim (steering, throttle), stable.
- **Agent mode boundary:** still a drop-in swap at the controller level.

### What Carries Forward From PPO Work

- All analytics infrastructure (`src/analytics/`) — episode tracking, reward decomposition, crash forensics all apply to brain-inspired episodes.
- The visualisation / HUD culture — debugging the brain-inspired learner will rely on the same observability habits.
- The `EpisodeState.current_tick_reward` pipeline — the modulator signal for v1 reads directly from it.
- The env-tagged rollout pattern (if we need buffered experience for later milestones, the infrastructure exists).
- The performance-tuning lessons — tanh, flat weight storage, pre-allocated scratch, disjoint-field borrows. Graph storage is different (sparse edges) but the discipline transfers.

## Updated Milestone Structure (2026-04-19)

The old "Milestone 1 of 9" framing is replaced by the 11-milestone structure in `README.md`:

```
M1  Environment + keyboard controller                    ✓
M2  PPO baseline from scratch                            ✓
M3  Multi-car + analytics pipeline                       ✓
M4  Performance overhaul                                 ✓
M5  Critic target-scaling                                ✓
M6  Brain-inspired v1 — the substrate                    ← next
M7  Brain visualisation
M8  Brain-inspired v2 — plastic value predictor (Option B)
M9  Multi-neuromodulator refinement

Long-Term Plan (biological-realism arc, ordering flexible):
    • Dale's law
    • Synaptic delays
    • Short-term synaptic dynamics (Tsodyks-Markram)
    • Multiple neuron types (pyramidal + interneurons)
    • Sleep/replay consolidation
    • Spiking neurons + STDP

M10 Evaluation (multi-track, transfer, curriculum)
M11 Writeup / release preparation

Research Frontier (explicitly out of scope, not forgotten):
    • Dendritic compartments
    • Glial cells
    • Multi-region architecture
    • Developmental programs / critical periods
    • Embodied proprioception
    • Evolutionary priors
    • Full-scale neuron counts
```

Each future milestone names a biological mechanism and a pathology it addresses. Long-Term Plan ordering is flexible — we pull items forward as we encounter the pathologies they address.

## Guiding Principles for the Transition

- **Preserve the stable boundary.** `CarAction` ↔ `ActionState`, `ObservationVector` 43-dim. Any new learning rule consumes and produces identical contracts.
- **PPO stays until retired deliberately, not by default.** `AgentMode::BrainInspired` is a third option, not a replacement. Retire PPO only if the brain-inspired learner has been shown to work end-to-end.
- **Entertainment-first reward carries forward.** `notes/reward-and-entertainment.md` applies regardless of learning rule.
- **Biology-first discipline applies to every decision from here forward.** See `notes/biology-first-principle.md`.
- **Visualisation matters.** The M7 brain inspector is the emotional core of the project. Graph topology makes it possible.

## What Was Tried During the Baseline (Preserved)

Durable lessons from the PPO baseline that inform the brain-inspired design:

- **ReLU → tanh switch** because ReLU produced 34–57% dead neurons. The brain-inspired v1 also uses tanh for this reason.
- **Braking axis reverted** because the policy collapsed into an idle basin. The brain-inspired learner inherits the throttle-only `[0, 1]` action space.
- **Crash penalty never shipped** because any crash penalty produces boring safe play. Entertainment-first reward philosophy preserved.
- **Progress-bonus reward superseded by velocity projection.** Dense per-tick reward lets the eligibility trace do credit assignment — relevant for v1's Option C choice.
- **Asymmetric actor/critic** (2×64 + 2×128) for PPO. Doesn't directly apply to brain-inspired since there's no layered structure, but the underlying lesson — value estimation may need more capacity than action selection — may inform M8's plastic value predictor design.
- **Wider critic + AdamW + log_std floor** (round 1) + **PopArt + observation norm + γ + target-KL** (round 2) — the PPO-specific fixes don't transfer directly, but the research methodology (parallel project-research agents, falsification criteria, analytics-first debugging) does.

## Constraints That Persist

- Rust from scratch, no external ML libraries.
- M2 MacBook Air (Apple Silicon, AMX via Accelerate, no GPU).
- Bevy 60 Hz fixed tick, 8 cars (configurable).
- 43-dim observation, 2-dim action — stable.
- PPO coexists permanently as diagnostic.

## References

- `notes/biology-first-principle.md` — the discipline.
- `notes/brain-v1-design.md` — concrete v1 design.
- `notes/reward-and-entertainment.md` — reward philosophy.
- `notes/normalisation-layers.md` — the three normalisations in PPO, relevant if we ever reference PPO as a comparison.
- `notes/conventions.md` — project-wide conventions.
- `context/references/brain-inspired-learning/overview.md` — seven-paper synthesis.
- `README.md` — project intent and current milestone structure.
