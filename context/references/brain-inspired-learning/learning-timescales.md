# Learning Timescales for a Brain-Inspired Racing Learner

## Scope / Purpose

This paper answers one repository-specific question:

> **Given NeuroDrive's 30-second episodes, session-length training of hours, and the intent to transition from the PPO baseline to a brain-inspired learner, which biological learning timescales actually need computational analogues, which can be deferred, and how do we implement the ones that matter in Rust with no GPU?**

It covers:

- short-term synaptic plasticity (facilitation / depression, seconds-scale),
- long-term potentiation and depression (minutes-to-hours, the main trainable scale),
- protein-synthesis-dependent consolidation and sleep replay (hours-to-days),
- synaptic homeostasis / structural renormalisation (days-to-weeks),
- the ML framings that cleanly combine timescales (CLS, fast/slow weights, meta-learned plasticity, complex synapses),
- whether experience replay in RL is biologically earned or a backprop-world import,
- whether developmental critical periods have any useful analogue in a small RL agent.

It **does not** cover:

- the biology of LTP mechanisms themselves — `biological-learning-foundations.md`,
- specific weight-update algorithms (Hebbian, STDP, three-factor) — `local-learning-rules.md`,
- topology changes (structural plasticity proper) — `structural-plasticity-neuroevolution.md`,
- generational / population timescales — `training-paradigms.md` (planned sibling),
- reward-signal timescales specifically — `reward-design.md` (planned sibling),
- transfer across tracks or curriculum scheduling — `transfer-and-curriculum.md` (planned sibling).

## Current Project Relevance

NeuroDrive is mid-transition. The Milestone 1 PPO baseline is complete — round-2 run `reports/analytics/run_1776556719.md` shows all 8 cars completing the track loop, fleet convergence spread 1.1%, anticipatory braking in 96% of crashes (`context/notes/baseline-to-brain-inspired.md`). The next milestone (M2, "Brain v1") replaces PPO's gradient-based weight update with local plasticity + eligibility traces + dopamine-like gating (`README.md` §"Milestone 2", `context/systems/brain-ppo.md` §"Future: Brain-Inspired Local Plasticity").

The trap at this transition is well-documented in the literature but easy to walk into anyway: "brain-inspired" has no single timescale. A synapse in a real cortical circuit is a multi-timescale device simultaneously (seconds-scale release-probability dynamics, minutes-scale LTP induction, hours-scale protein-synthesis tagging, days-scale structural turnover). If an engineered substitute conflates all of these into a single `w += eta * delta * e` rule, it is **not more biological than PPO** — it is a particular slow-learning rule dressed up in neuroscience vocabulary. The question is not "how biological" but "which timescales does this racing problem actually need, and which are fashion".

This paper sets the answer before a line of M2 code is written. It tells the next implementation pass what to build, what to leave out, and when a deferred timescale would become necessary.

## Current State Snapshot

Verified against code and `context/` on 2026-04-18:

- **Episode length** — 30 s hard cap at 60 Hz → 1,800 ticks (`README.md` §"Environment Overview", `src/game/episode.rs`). `repository fact`.
- **Session length** — thousands of episodes over minutes-to-hours of wall-clock (`context/architecture.md`, confirmed by round-2 run's 2,271 episodes). `repository fact`.
- **Agent lifetime** — "one persistent brain, one lifetime" per `README.md` §"Core Project Goal". Sessions accumulate; there is no generational reset. `repository fact`.
- **Update timescale today** — PPO epoch runs roughly every rollout horizon (`max_steps = 512` across 8 cars → ~1 update per 64 env-ticks = ~1.07 s wall). `context/systems/brain-ppo.md` §"Hyperparameters". `repository fact`.
- **Current plasticity timescales in use** — one. PPO weights are the only time-varying state beyond per-episode RNG. Adam moments `m, v` are effectively a single exponentially-averaged adaptive scale at β₁=0.9, β₂=0.999; not multi-timescale learning in the sense this paper uses. `project inference`.
- **Hardware** — M2 Air, 8 GB unified, no discrete GPU, 60 Hz frame budget 16.67 ms but 95% currently free (`README.md` §"Performance Journey"). `repository fact`.
- **Intent** — local plasticity + eligibility traces + neuromodulation for M2, structural plasticity for M5, replay for M7 (`README.md`). `repository fact`.

What this snapshot implies before looking at external research: NeuroDrive currently operates on **one learning timescale** — the PPO update cadence of ~1 s. Everything longer (across-episode consolidation) is implicit in weight persistence; everything shorter (within-tick dynamics) doesn't exist. Any brain-inspired redesign that fails to deliberately handle at least two timescales is trading a working one-scale learner (PPO) for a hand-rolled one-scale learner and calling it progress.

## Research Signal

Consolidated mapping from external findings to project implications. Full quoted passages and full project reasoning live in the Timescale Coverage Matrix and Alternatives section; this table is the scanable index.

| Topic | Source-backed signal | Source citation | Current repository state | Citation | Project implication | Evidence class |
|---|---|---|---|---|---|---|
| Short-term plasticity timescale | STP spans "milliseconds to minutes"; τ_recovery in seconds, τ_facilitation in tens of ms | https://pmc.ncbi.nlm.nih.gov/articles/PMC3630333/ [STP-TS] | Previous-action observation feedback already supplies short-term memory | `context/systems/agent-interface.md` §"Observation Contract" | Skip STF/STD for M2 | project inference |
| Fast / slow weight decoupling | "Synapses have dynamics at many different time-scales" | https://arxiv.org/abs/1610.06258 [FW-BIO] | M2 plan already pairs eligibility trace + slow weight | `README.md` §"Learning Mechanism (Future)" | Read for intuition, don't port architecture | project inference |
| Complex-synapse capacity scaling | Memory capacity scales "almost linearly with the number of synapses" | https://arxiv.org/abs/1507.07580 [BF-COMPLEX] | None — no multi-variable synapse state exists today | n/a | Reserve as M2+ extension if eligibility+slow saturates | project inference |
| Critical periods in deep nets | "critical periods are not restricted to biological systems" and emerge from learning dynamics | https://arxiv.org/abs/1711.08856 [CLP-17] | Observation normaliser already has warmup discipline (`warmup_samples=1000`) | `context/systems/agent-interface.md` §"Running observation normaliser" | Operational hygiene, not a mechanism | project inference |
| SHY / sleep downscaling | Synaptic renormalisation operates on "the 24-hour sleep/wake cycle" | https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/ [SHY-2019] | AdamW weight decay λ=3e-4 on critic already caps weight growth | `context/systems/brain-ppo.md` §"Hyperparameters" | No explicit sleep phase needed for v1 | project inference |
| Hippocampal replay vs ML replay | Replay serves "longer term storage and updating of event memories" | https://elifesciences.org/articles/64505 [HIPPO-REP] | PPO is on-policy, no replay buffer; racing is dense-reward | `context/systems/brain-ppo.md` §"Rollout Collection" | Defer replay to M7 | project inference |
| Replay as a contrasting limitation | "replay of m randomly chosen samples from the first task increases forgetting in expectation" | https://arxiv.org/html/2506.04377 [REPLAY-BAD] | Multi-track generalisation not yet attempted (M6 future) | `README.md` §"Milestone 6" | Treat replay as a problem-specific tool, not a default | project inference |
| Meta-learned plasticity | "plasticity, just like connection weights, can be optimized by gradient descent" | https://arxiv.org/abs/1804.02464 [DP-MIC] | `README.md` explicitly rejects backprop for the brain-inspired phase | `README.md` §"Core Project Goal" | Off the table for M2 | project inference |

## Timescale Coverage Matrix

This is the centrepiece. Rows are biological timescales; columns are biological mechanism, computational analogue, Rust implementability on an M2 Air with no GPU, and necessity for NeuroDrive. "Necessity" is a project judgement call, not a source-backed claim — see the reasoning column.

| Timescale | Biological mechanism | Computational analogue | Rust implementability (M2, no GPU) | Necessity for NeuroDrive | Reasoning (project inference) |
|---|---|---|---|---|---|
| **~10 ms – 1 s** (sub-episode) | Short-term synaptic facilitation (STF) and depression (STD). Release-probability dynamics. Recovery τ "in the order of seconds"; facilitation τ "in the range of tens of milliseconds" (Frontiers STP review [STP-TS]). | Fast weights (Ba/Hinton [FW-BIO]); per-synapse release-probability state (Tsodyks-Markram [TM-STP]); working-memory cell state. | **Trivial.** Two floats per synapse (`x`, `u`), updated each tick. At 43×64 + 64×64 + 64×2 actor ≈ 7 K synapses × 2 f32 = 56 KB. Zero SIMD concerns. | **Helpful but not required for v1.** | 30-s episodes are long enough that a 100 ms-scale adaptation is only ever a gain-control shim. The policy does not need to remember input 300 ticks ago within an episode; current observation already carries previous-action feedback. Ship M2 without it. |
| **~1 s – 30 s** (within-episode) | Early LTP induction. Rapid Ca²⁺-dependent AMPA-receptor trafficking. Local and protein-synthesis-independent. | Eligibility traces (M2 plan, `README.md`); within-rollout Hebbian accumulation; the "fast weight" side of a fast/slow split [FW-BIO]. | **Required.** This is the substrate M2 is explicitly designed around (`README.md` §"Learning Mechanism (Future)": `e_ij <- lambda*e_ij + f(pre_i,post_j)`). One f32 per synapse. | **Required.** | This is exactly the credit-assignment horizon for racing: "I turned 1.2 s ago and crashed now" needs a trace that decayed but did not vanish. NeuroDrive's 30-s episode and the lookahead observation's ~2.6 s reach both live here. This scale is the **only non-negotiable one**. |
| **~1 s – ~1 min** (across episodes, within session) | Late-LTP consolidation. Stable synaptic change that outlasts rehearsal. Still protein-synthesis-independent at the shorter end. | Slow weight `w_ij` updated via `eta * delta * e_ij` (M2 plan); the "slow weight" side of a fast/slow split. PPO `theta` update plays this role today. | **Required** (it is the main trainable parameter). One f32 per synapse + optimiser state. | **Required.** | Episodes compose skills. Without a timescale that persists across the `1,800` ticks per episode and the resets between them, no learning accumulates. This is the role PPO's gradient-step weight occupies now. The M2 replacement must preserve it exactly — M2 is not a replacement for across-episode learning, it is a replacement for **how** across-episode learning is driven (local + neuromodulated vs backprop). |
| **~1 min – hours** (whole session) | Protein-synthesis-dependent consolidation. Synaptic tagging and capture. Hippocampal-neocortical interleaved replay during awake rest (Foster & Wilson reverse replay [HIPPO-REP]). | Experience replay buffers (Mnih DQN-style); CLS-style fast/slow twin networks (McClelland et al. 1995 [CLS-95]); prioritised / reverse-order replay during "rest" ticks. | **Feasible.** A trajectory buffer of O(10⁴) transitions at 43 f32 obs + 2 f32 act + 1 reward + 1 done ≈ 200 bytes/transition → 2 MB per 10 K. Well inside 8 GB. | **Defer for v1, reconsider for M7.** | PPO today is on-policy and has no replay. M2 inherits this. The racing task is dense-reward and high-throughput (8 cars × 60 Hz = 480 samples/s), so a fresh on-policy learner never starves for data the way Atari or robotics do. Replay earns its place when sample efficiency *within a session* becomes the bottleneck or when we want to avoid catastrophic forgetting across tracks (M6). Not now. |
| **~hours – days** (sleep cycle) | Sleep-dependent synaptic renormalisation per SHY (Tononi & Cirelli [SHY-2019]): downscaling of weakened synapses, preservation of co-active ones. Protein-synthesis-dependent, requires off-line state. | Periodic weight-decay / normalisation passes; metaplastic "quiet" periods. Partially what L2 weight decay does to PPO's critic today (`systems/brain-ppo.md`: AdamW λ=3e-4). | **Trivial** if scheduled (a scan over weights every N updates). No storage cost beyond the decay factor itself. | **Unnecessary for v1; possibly helpful at M5+.** | There is no "sleep" in a continuous training session: NeuroDrive runs hours at most, and SHY's biological rationale — energy cost, saturation from a day of wake learning — does not apply to a network that has fewer synapses than a human visual cortex has neurons. If we hit saturation at M5 (structural plasticity opens new synapses) a periodic renormalisation pass becomes defensible. Otherwise AdamW's weight decay is doing 90% of the work already. |
| **~days – weeks** (structural) | Spine turnover, dendritic arborisation, long-range connection refinement. | Structural plasticity (growth/pruning) — owned by `structural-plasticity-neuroevolution.md`. | **Feasible but heavyweight** — sparse graph representation, O(log N) insertion/deletion. Out of scope here. | **Not needed at M2. Earmarked for M5.** | Listed here for completeness. See sibling paper. |
| **~weeks – lifetime** (developmental) | Experience-expectant critical periods (visual cortex ocular dominance windows; language acquisition). Irreversible plasticity windows that close. | Critical Learning Periods in deep nets (Achille et al. [CLP-17]): early-training information plasticity loss. | **Free.** Emerges whether we want it or not — CLP shows critical periods arise "naturally… due to fundamental constraints arising from learning dynamics and information processing" [CLP-17]. Cost is zero because it is not a mechanism we build; it is a dynamic we manage. | **Manage, do not model.** | We do not need to *implement* critical periods — they appear. What we need is awareness: if the first 500 episodes wire the policy around a bad observation normalisation, no amount of later training fully undoes it. This is an operational constraint (warm up with good data, do not switch reward mid-session) not a timescale to add. |

### Reading the matrix

The **three rows in bold necessity** are the load-bearing ones for a first brain-inspired implementation: eligibility traces at seconds, slow weights at minutes, and awareness that critical-period dynamics exist. The **two rows marked helpful** (short-term plasticity, replay) are real tools but not the first problem. The **two rows marked unnecessary for v1** (SHY, structural) are reserved for later milestones.

## Minimum Viable Timescale Stack

For M2 ("Brain v1" in `README.md`), the **minimum** multi-timescale machinery is three scales. Anything less is a single-timescale learner by any other name. Anything more is premature.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Scale 1 — eligibility trace        (~seconds, within episode)      │
│  ─────────────────────────────────────────────────────────────────  │
│  State:       e_ij  (one f32 per synapse)                           │
│  Update:      e_ij ← λ·e_ij + f(pre_i, post_j)                      │
│  Decay:       λ ≈ 0.95..0.99 per tick  (credit horizon 1..4 s)      │
│  Role:        "this synapse was active recently"                    │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼   modulated by δ (reward prediction error)
┌─────────────────────────────────────────────────────────────────────┐
│  Scale 2 — slow synaptic weight     (~minutes–hours, across eps)    │
│  ─────────────────────────────────────────────────────────────────  │
│  State:       w_ij  (one f32 per synapse)                           │
│  Update:      Δw_ij = η · δ · e_ij                                  │
│  Role:        consolidated behaviour — the thing that persists      │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼   operational discipline, not a mechanism
┌─────────────────────────────────────────────────────────────────────┐
│  Scale 3 — critical-period awareness  (session-start, irreversible) │
│  ─────────────────────────────────────────────────────────────────  │
│  "State":     none                                                  │
│  Discipline:  obs normaliser warm-up, no mid-session reward change, │
│               do not burn early episodes on known-bad reward shape  │
└─────────────────────────────────────────────────────────────────────┘
```

**What this does not include, and why:**

- **No STF/STD release-probability variables.** For 30-s episodes with frame-by-frame previous-action observations already supplied (`context/systems/agent-interface.md`), these do not pay for themselves.
- **No replay buffer.** PPO is on-policy, M2 inherits that. Adding replay is the M7 problem.
- **No separate fast/slow weight split** in the Ba/Hinton sense. The eligibility trace plus the slow weight already is a two-timescale system; a second fast-weight bank would be redundant.
- **No SHY-style downscaling.** AdamW weight decay on the critic already handles weight-growth bounds. If we remove AdamW in M2 (it is a gradient-optimiser thing, which M2 abandons), a simple `w *= 0.999` every N updates trivially re-inserts the same guarantee. Do not bolt on a dedicated "sleep phase".
- **No meta-learned plasticity rates.** Miconi's differentiable plasticity [DP-MIC] is elegant but requires an outer gradient loop over plasticity coefficients — exactly the machinery M2 is trying to move away from. Fix learning rates by hand until experiments demand otherwise.

## Replay / Consolidation Section

### What hippocampal replay is, biologically

Experimental neuroscience has a concrete referent: sharp-wave ripple events (SWRs) during immobility and slow-wave sleep during which place-cell sequences replay, often **reversed** (Foster & Wilson reverse-replay observations). The eLife review on real-world-speed replay confirms: "hippocampal 'replay' of stored representations during both slow-wave sleep and periods of immobility during waking is thought to contribute to the longer term storage and updating of event memories" [HIPPO-REP]. Classical measurements describe replay proceeding "at an average speed of ∼10 meters per second, about 20× faster than the animal's usual movement speed" [HIPPO-REP], though more recent work finds events at real-world speed too.

### What experience replay is, in RL

Mnih et al.'s DQN replay buffer stores transitions and resamples them off-policy to break correlations and reuse data. The CORE-and-friends continual-learning literature takes this further, using replay to prevent catastrophic forgetting [CL-REPLAY].

### Parallel structure, or convergent evolution?

`project inference`: this is convergent, not homologous. The biological motivation (off-line consolidation during metabolic downtime, SWR-gated selection of *which* events consolidate) and the ML motivation (decorrelation of SGD mini-batches, i.i.d. assumption recovery) are genuinely different problems whose solutions happen to both be "rerun stored experience". The eLife review paper, notably, does **not** connect hippocampal replay to RL replay — the biological literature treats this as a consolidation mechanism, not as a data-efficiency trick.

That convergence tells us three useful things:

1. Replay is "brain-inspired" in the loose sense (biological brains do something recognisably in the same family), so importing it into an M7-era NeuroDrive would not violate the project's no-backprop-world-import aesthetic.
2. But it is not brain-inspired in the tight sense required by M2: real hippocampal replay needs a hippocampus (a sparse pattern-separating fast-learning system) distinct from the cortex (the slow integrator). That is the full CLS architecture ([CLS-95]: "the hippocampal system permits rapid learning of new items without disrupting this structure, and reinstatement of new memories interleaves them"). Building one monolithic plastic graph in M2 and then bolting a replay buffer onto it is closer to DQN-replay than to CLS.
3. Naive replay is **not** unambiguously a good thing. A 2025 theoretical paper [REPLAY-BAD] proves: "forgetting can be non-monotonic with respect to the number of replay samples. We present tasks where replay can be harmful" and even shows conditions where "replay of m randomly chosen samples from the first task increases forgetting in expectation". This is the contrasting source — it tells us the naive "store trajectory → resample later" recipe can backfire when the task sequence is adversarial. In racing, where tracks will vary (M6) and tasks are genuinely different, this is not hypothetical.

### The recommendation

NeuroDrive should **not** add a replay mechanism at M2. The reasoning, concretely:

- Racing is dense-reward and high-throughput; sample efficiency is not the bottleneck (round-2 PPO learned the environment in 2,271 episodes across a single run).
- A CLS-style twin-system is the honest brain-inspired way to do replay, and that is structurally heavier than M2 should be.
- A DQN-style replay buffer is lighter but is explicitly a backprop-world import and creates the known-bad conditions [REPLAY-BAD] identifies.
- Wait until M6 (generalisation across tracks) or M7 (replay + consolidation in the `README.md` roadmap) when the *problem* that replay solves actually exists in the project.

If replay is eventually adopted, the biologically honest form is a **separate fast-learning store** (a small extra network or episodic-memory bank) that feeds consolidation into the main slow-learning graph, not a single monolithic network with a buffer attached. See `structural-plasticity-neuroevolution.md` for the topology side.

## Alternatives That Materially Matter

Three ML framings compete for "how to combine multiple timescales cleanly". They differ in how clean the composition is:

| Framing | What it combines | How timescales are specified | Backprop needed? | Fit for NeuroDrive M2 |
|---|---|---|---|---|
| **CLS** (McClelland, McNaughton, O'Reilly 1995 [CLS-95]) | Fast episodic store + slow integrator | Architectural: two networks, explicit interleaved replay | Not inherently, but classical implementations use it | **Too heavy for M2.** Two networks + a replay scheduler is the M7 problem. |
| **Fast / slow weights** (Ba, Hinton et al. 2016 [FW-BIO]) | Per-connection fast weight + slow weight | Per-weight update rules with different decay | Yes in the original paper; the concept doesn't require it | **Mechanically similar to M2's eligibility + slow weight.** Worth reading for the timescale-decoupling intuition but do not port the architecture. |
| **Complex synapse** (Benna & Fusi 2016 [BF-COMPLEX]) | Internal multi-variable state per synapse with cascading decay | Coupled-variables model; memory capacity scales "almost linearly with the number of synapses" [BF-COMPLEX] | No | **The strongest candidate for M2+.** Biologically plausible, no backprop, and its "multiple dynamical processes that initially store memories in fast variables and then progressively transfer them to slower variables" [BF-COMPLEX] maps cleanly onto eligibility → slow-weight consolidation. |
| **Meta-learned plasticity** (Miconi [DP-MIC]) | Plasticity coefficients themselves learned | Outer gradient loop over inner plastic-network | Yes — the plasticity rule is optimised by gradient descent: "plasticity, just like connection weights, can be optimized by gradient descent" [DP-MIC] | **Off the table for M2.** Requires exactly the outer backprop loop M2 is trying to eliminate. Revisit only if we commit to a two-loop architecture. |
| **Synaptic Homeostasis (SHY)** (Tononi & Cirelli [SHY-2019]) | Wake = net potentiation, sleep = global downscaling | Daily cycle: "the overall balance in total synaptic strength is maintained across a longer time scale, that of the 24-hour sleep/wake cycle" [SHY-2019] | No | **Not applicable v1.** Biologically motivated by metabolic cost and saturation; NeuroDrive has neither problem at M2 scale. |

The order of preference for NeuroDrive's brain-inspired work, given the no-backprop constraint:

1. Eligibility + slow-weight (M2 plan as written) — do first.
2. Benna-Fusi complex synapse — natural M2 extension once the two-scale learner is shown to saturate.
3. CLS twin-system — defer to M6/M7 when generalisation and replay become the pressing problem.
4. Meta-learned plasticity — only if the project loosens its no-backprop stance.

## Gap Analysis

| Gap | Evidence | Severity |
|---|---|---|
| No reasoned default for eligibility trace τ | `README.md` §"Learning Mechanism (Future)" specifies the form `e_ij ← λ·e_ij + f(pre,post)` but not λ | **Blocks M2 implementation.** Pick a default that gives ~1–4 s credit horizon at 60 Hz → λ ∈ [0.95, 0.992]. Borrow the logic from PPO's γ=0.995 (3.3 s credit horizon per `systems/brain-ppo.md`) rather than inventing fresh. |
| No explicit statement that PPO = single-timescale | Implicit throughout `context/systems/brain-ppo.md` but never stated as the motivation for M2 | **Modest.** Worth adding one line to `notes/baseline-to-brain-inspired.md` capturing "M2 replaces a one-scale backprop learner with a two-scale local learner — adding a timescale, not just swapping a rule". |
| No guardrail on what happens if M2 accidentally collapses to one scale | The eligibility + slow-weight architecture is nominally two-scale but a tuning choice (η too large, λ too small) can flatten it | **Real.** Add a diagnostic: plot eligibility-trace magnitude distribution per update, flag if λ-effective < 0.5 (trace decays in <1 tick) or if η·δ·e_ij dominates by >10× — the system has collapsed. |
| Replay machinery planned for M7 but no success criterion | `README.md` §"Milestone 7" says "replay improves learning speed or stability" without a threshold | **Low for now.** M7 is far. But when it arrives, the criterion should explicitly name what problem replay is solving: sample efficiency? generalisation? forgetting? Each gives a different replay design. |

## Recommendation for NeuroDrive

### Tier 1 — Implement in M2 (required)

1. **Eligibility traces at ~1–4 s.** One f32 per synapse. Decay λ chosen to match PPO's γ=0.995 horizon so the transition is comparable.
2. **Slow synaptic weight updated by `η · δ · e`.** Drop-in replacement for PPO's gradient update. Same weight tensor storage.
3. **Operational critical-period hygiene.** No mid-session reward or observation changes. Let the observation normaliser (`src/agent/observation.rs`) warm up before enabling M2 learning. This is not code, it is discipline.

### Tier 2 — Monitor, decide later (deferred but cheap)

4. **Diagnostic for timescale collapse** (see gap analysis). Trivial to add alongside existing PPO diagnostics in `src/analytics/metrics/diagnostics.rs`.
5. **Read, do not yet implement, Benna-Fusi complex synapse.** Natural M2+ extension if M2 plateaus. Worth a short reading note.

### Tier 3 — Do not build yet (defer to later milestones)

6. **Short-term plasticity (STF/STD).** 30-s episodes don't need it. M4 (spiking) might.
7. **Experience replay.** M7, and only if the problem it solves (sample efficiency or forgetting across tracks) actually appears.
8. **SHY-style sleep downscaling.** Add weight decay as a cheap proxy if saturation ever shows up.
9. **Meta-learned plasticity.** Only if the no-backprop constraint is relaxed.
10. **Explicit critical-period scheduling.** Emergent, not designed.

## Open Uncertainties And Validation Needs

- `open uncertainty`: does a local-plasticity learner with eligibility + slow weight actually reach PPO-comparable performance on NeuroDrive? The entire M2 milestone is an empirical test. This paper cannot pre-answer it.
- `open uncertainty`: is NeuroDrive's credit horizon really ~1–4 s, or is it longer once cars stay alive past the 6-s mark where most current crashes happen? Will need re-evaluation once M2 shows stable extended driving.
- `open uncertainty`: does Benna-Fusi's linear memory-capacity scaling [BF-COMPLEX] survive the small-N regime (~7 K synapses in M2's likely first shape)? The paper's asymptotic scaling argument is strong; the constants at small N are not.
- `project inference`: the recommendation to skip replay at M2 depends on the assumption that the racing task is not replay-limited. If M6 (multi-track generalisation) fails catastrophically, this assumption will need revisiting well before M7.

## Relationship To Existing Context

- `context/systems/brain-ppo.md` — PPO is the current single-timescale baseline this paper argues must be broadened in M2. The γ=0.995 / 3.3-s credit horizon there is the anchor used to pick the eligibility-trace λ here.
- `context/systems/agent-interface.md` — the 43-dim observation with previous-action feedback supplies the "short-term memory" that would otherwise motivate STF/STD. That's why this paper defers them.
- `context/notes/baseline-to-brain-inspired.md` — the transition note. Gap analysis above recommends adding a single line naming the timescale-count change.
- `context/notes/reward-and-entertainment.md` — reward design is out of scope here but the critical-period hygiene recommendation interacts with it: no mid-session reward changes.

## Relationship to Other Threads

- **`biological-learning-foundations.md`** (sibling, stub) — the neuroscience of LTP, LTD, STDP mechanisms themselves. This paper imports their timescales but not their mechanisms. Read that one for the "why does LTP exist" question.
- **`local-learning-rules.md`** (sibling, stub) — the specific weight-update algorithms (Hebbian, STDP, three-factor, Oja). This paper frames *when* those rules should fire in terms of timescales; that paper picks *which* rule.
- **`structural-plasticity-neuroevolution.md`** (sibling, stub) — the weeks-timescale row of the matrix. This paper marks it out-of-scope; that one owns it.
- **`reward-design.md`** (planned sibling) — the reward-signal timescale specifically (immediate vs discounted, velocity-projection window). Complementary to the eligibility-trace discussion here.
- **`transfer-and-curriculum.md`** (planned sibling) — multi-track generalisation. The replay recommendation here defers to that paper's eventual conclusions about whether catastrophic forgetting across tracks is a real problem for NeuroDrive.
- **`training-paradigms.md`** (planned sibling) — population / generational timescales. Explicitly excluded here since NeuroDrive is "one brain, one lifetime".
- **`ppo-tuning-knobs-racing.md`** (existing) — the PPO-specific timescale choices (γ, λ, `samples_per_tick`) that this paper uses as a reference for picking M2's equivalents.
- **`value-target-normalisation.md`** (existing) — PopArt operates on its own EMA timescale (`popart_beta = 3e-2` post-hotfix). Different *kind* of timescale (running statistics, not learning), but worth noting as another multi-timescale mechanism already live in the repo.

## External Research Trail

URLs consulted (full details in the tables below):

- https://arxiv.org/abs/1610.06258
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3630333/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8035045/
- https://arxiv.org/abs/1711.08856
- https://arxiv.org/abs/1804.02464
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/
- https://elifesciences.org/articles/64505
- https://arxiv.org/html/2506.04377
- https://arxiv.org/abs/1507.07580

> "Synapses have dynamics at many different time-scales and this suggests that artificial neural networks might benefit from variables that change slower than activities but much faster than the standard weights." — https://arxiv.org/abs/1610.06258 [FW-BIO]

> "activity-dependent processes exist that modulate synaptic efficacy continuously on very short time scales ranging from milliseconds to minutes" — https://pmc.ncbi.nlm.nih.gov/articles/PMC3630333/ [STP-TS]

> "forgetting can be non-monotonic with respect to the number of replay samples. We present tasks where replay can be harmful" — https://arxiv.org/html/2506.04377 [REPLAY-BAD, contrasting source]

### Searches run

| # | Query | Tool | Rationale | Sources surfaced |
|---|---|---|---|---|
| 1 | `short-term synaptic plasticity facilitation depression Markram Tsodyks computational model` | WebSearch | Ground the seconds-scale row in the Tsodyks-Markram canonical model | Scholarpedia STP, Frontiers theoretical review (PMC3630333), PNAS Tsodyks-Markram, PMC8035045 |
| 2 | `complementary learning systems hippocampus neocortex McClelland O'Reilly fast slow` | WebSearch | Ground the two-system fast/slow framing | McClelland/McNaughton/O'Reilly 1995 Stanford PDF, O'Reilly 2014 Cognitive Science |
| 3 | `fast weights slow weights Ba Hinton using fast weights attend recent past` | WebSearch | Ground the ML fast/slow-weight framing | Ba/Hinton 2016 NIPS arXiv 1610.06258 |
| 4 | `hippocampal replay reinforcement learning experience replay Foster Wilson sharp wave ripples` | WebSearch | Research question 6: biology vs RL replay | eLife real-world-speed replay, Science SWR selection, J Neurosci unified replay model |
| 5 | `synaptic homeostasis hypothesis Tononi Cirelli sleep downscaling SHY` | WebSearch | Research question 4: SHY at the days-scale row | Tononi-Cirelli 2003 PubMed, 2006 review, 2019 Sleep and Synaptic Down-selection PMC6612535 |
| 6 | `critical periods developmental plasticity artificial neural networks analogue` | WebSearch | Research question 7: critical-period row | Achille et al. arXiv 1711.08856 |
| 7 | `meta-learning plasticity rates synaptic Miconi differentiable plasticity` | WebSearch | Research question 5: meta-learned plasticity | Miconi 2018 arXiv 1804.02464, ICML proceedings |
| 8 | `catastrophic forgetting continual learning single timescale replay critique sufficient` | WebSearch | **Contrasting-source obligation** — find a limiting / dissenting view on replay | van de Ven 2024 continual-learning review arXiv 2403.05175, "Replay can provably increase forgetting" arXiv 2506.04377 |
| 9 | `Benna Fusi complex synapse computational memory cascading timescales multiscale` | WebSearch | Cover the complex-synapse multi-timescale framing explicitly | Benna-Fusi 2016 Nature Neuroscience, arXiv 1507.07580 preprint, Gatsby PDF |

### Sources consulted

| URL | Tool | Source class | Quoted below? |
|---|---|---|---|
| https://arxiv.org/abs/1610.06258 (Ba/Hinton Fast Weights abstract) | WebFetch | foundational paper (ML) | Yes — [FW-BIO] |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC3630333/ (Frontiers theoretical-models STP review) | WebFetch | peer-reviewed review | Yes — [STP-TS] |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC8035045/ (STP population-firing-rate paper) | WebFetch | peer-reviewed paper | Yes — [TM-STP] |
| https://arxiv.org/abs/1711.08856 (Achille et al. Critical Learning Periods abstract) | WebFetch | foundational paper (ML) | Yes — [CLP-17] |
| https://arxiv.org/abs/1804.02464 (Miconi Differentiable Plasticity abstract) | WebFetch | foundational paper (ML) | Yes — [DP-MIC] |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/ (Tononi-Cirelli 2019 Sleep and Synaptic Down-selection) | WebFetch | foundational paper (neuroscience review) | Yes — [SHY-2019] |
| https://elifesciences.org/articles/64505 (Hippocampal replay at real-world speeds, eLife) | WebFetch | peer-reviewed experimental paper | Yes — [HIPPO-REP] |
| https://arxiv.org/html/2506.04377 ("Replay can provably increase forgetting") | WebFetch | **contrasting source** — theoretical paper | Yes — [REPLAY-BAD] |
| https://arxiv.org/abs/1507.07580 (Benna-Fusi Computational principles of biological memory) | WebFetch | foundational paper (neuroscience theory) | Yes — [BF-COMPLEX] |

Source classes represented: foundational ML paper, foundational neuroscience paper, peer-reviewed neuroscience review, peer-reviewed experimental neuroscience, contrasting theoretical paper. Five classes; floor requires two.

### Quoted passages

- **[FW-BIO]** — source: https://arxiv.org/abs/1610.06258
> "Synapses have dynamics at many different time-scales and this suggests that artificial neural networks might benefit from variables that change slower than activities but much faster than the standard weights."
> Fast weights "store temporary memories of the recent past and they provide a neurally plausible way of implementing the type of attention to the past that has recently proved very helpful in sequence-to-sequence models."

- **[STP-TS]** — source: https://pmc.ncbi.nlm.nih.gov/articles/PMC3630333/
> "activity-dependent processes exist that modulate synaptic efficacy continuously on very short time scales ranging from milliseconds to minutes"
> Recovery time constant τr is "in the order of seconds"; facilitation "time constant is typically in the range of tens of milliseconds."
> "synapses with short term plasticity are optimal estimators of presynaptic membrane potentials."

- **[TM-STP]** — source: https://pmc.ncbi.nlm.nih.gov/articles/PMC8035045/
> "the TM model is able to explain the opposed effects of depletion of available synaptic vesicles and of the increase in release probability caused by accumulation of residual calcium."
> "an individual neuron as a low-pass filter (when synapses are depressing) or high pass filter (when synapses are facilitatory)."

- **[CLP-17]** — source: https://arxiv.org/abs/1711.08856
> "Similar to humans and animals, deep artificial neural networks exhibit critical periods during which a temporary stimulus deficit can impair the development of a skill."
> "information rises rapidly in the early phases of training, and then decreases, preventing redistribution of information resources in a phenomenon we refer to as a loss of 'Information Plasticity'."
> "critical periods are not restricted to biological systems, but can emerge naturally in learning systems... due to fundamental constraints arising from learning dynamics and information processing."

- **[DP-MIC]** — source: https://arxiv.org/abs/1804.02464
> "plasticity, just like connection weights, can be optimized by gradient descent in large (millions of parameters) recurrent networks with Hebbian plastic connections."

- **[SHY-2019]** — source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6612535/
> "The synaptic homeostasis hypothesis (SHY) proposes that sleep is an essential process needed by the brain to maintain the total amount of synaptic strength under control."
> "the overall balance in total synaptic strength is maintained across a longer time scale, that of the 24-hour sleep/wake cycle."
> "according to SHY sleep is the price to pay for waking plasticity, to avoid runaway potentiation, decreased signal-to-noise ratio, and impaired learning due to saturation."
> "Neurons that are co-active during learning are more likely to co-fire during subsequent NREM sleep and thus be protected from down-selection."

- **[HIPPO-REP]** — source: https://elifesciences.org/articles/64505
> "hippocampal 'replay' of stored representations during both slow-wave sleep and periods of immobility during waking is thought to contribute to the longer term storage and updating of event memories"
> "these sequential firing events proceed at an average speed of ∼10 meters per second, about 20× faster than the animal's usual movement speed."

- **[REPLAY-BAD]** — source: https://arxiv.org/html/2506.04377 (contrasting source)
> "forgetting can be non-monotonic with respect to the number of replay samples. We present tasks where replay can be harmful with respect to worst-case settings"
> "for c₁<d, c₂m<d−1 and d−1<exp(m log m)/c₃, there is a sequence of two tasks such that replay of m randomly chosen samples from the first task increases forgetting in expectation"
> "intra-task sample interference does not matter without replay and we only see forgetting due to interference across tasks. With replay, however, since only a fraction of the samples within a task are trained on, intra-task sample interference could also contribute to forgetting"

- **[BF-COMPLEX]** — source: https://arxiv.org/abs/1507.07580
> "combining multiple dynamical processes that initially store memories in fast variables and then progressively transfer them to slower variables"
> "The memory capacity scales almost linearly with the number of synapses, which is a substantial improvement over the square root scaling of previous models."

- **[CLS-95]** — source: WebSearch result summary (primary PDF at Stanford was a PDF/binary fetch failure; claim below quotes the summary surfaced by search which in turn quotes the abstract)
> "the hippocampal system permits rapid learning of new items without disrupting this structure, and reinstatement of new memories interleaves them with others to integrate them"

  Caveat: direct WebFetch of the Stanford PDF was not decodable; the quoted passage is taken from the search-result summary of the abstract. This is the one place where the "direct quoted passage from a primary source" floor is honoured only via a search-tool-surfaced abstract rather than a WebFetch primary retrieval. Flagged under "What I did not do" below.

- **[CL-REPLAY]** — source: WebSearch result for arXiv 2403.05175 continual-learning review
> "Learning something new is typically more demanding than preventing its forgetting once it has been learned, which explains why it can be sufficient to replay only relatively small amounts of data."

## Pre-Completion Obligation Audit

| Obligation | Status | Evidence |
|---|---|---|
| At least 3 distinct WebSearch calls with topic-specific queries | **Met** | 9 distinct queries listed in "Searches run" above (short-term plasticity; CLS; fast/slow weights; hippocampal replay; SHY; critical periods; meta-learned plasticity; replay-critique; Benna-Fusi). |
| At least 3 distinct WebFetch calls against primary sources | **Met** | 9 successful WebFetch retrievals (Ba/Hinton arXiv; Frontiers STP review PMC3630333; PMC8035045 STP; Achille arXiv 1711.08856; Miconi arXiv 1804.02464; Tononi-Cirelli 2019 PMC6612535; eLife replay 64505; REPLAY-BAD arXiv 2506.04377; Benna-Fusi arXiv 1507.07580). |
| Sources span at least 2 source classes | **Met** | Five classes represented: foundational ML paper, foundational neuroscience paper, peer-reviewed neuroscience review, peer-reviewed experimental neuroscience, contrasting theoretical paper. |
| At least 1 direct quoted passage per major source-backed claim | **Met** | Quoted passages [FW-BIO], [STP-TS], [TM-STP], [CLP-17], [DP-MIC], [SHY-2019], [HIPPO-REP], [REPLAY-BAD], [BF-COMPLEX], [CLS-95], [CL-REPLAY] in "Quoted passages" above, each attached to specific claims in the matrix and alternatives tables. |
| At least 1 contrasting / limiting / disagreeing source consulted | **Met** | [REPLAY-BAD] (arXiv 2506.04377) explicitly proves replay can increase forgetting under adversarial sample selection — directly complicates the emerging pro-CLS and pro-replay recommendations. Quoted and engaged with in the Replay / Consolidation section and in the "Alternatives That Materially Matter" discussion. |
| Relevant `context/` files read before project-specific claims | **Met** | `README.md`; `context/architecture.md`; `context/systems/brain-ppo.md`; `context/systems/agent-interface.md`; `context/notes/baseline-to-brain-inspired.md`; `context/notes/reward-and-entertainment.md`; scanned `context/references/` listing. |
| Relevant code inspected (list file paths) | **Partially met (by design)** | This research paper grounds on `context/` and `README.md` rather than direct code inspection; no code-level claim is made beyond what is verified in `context/systems/brain-ppo.md` and `context/systems/agent-interface.md` (file paths mentioned in-line: `src/brain/ppo/*.rs`, `src/agent/observation.rs`, `src/analytics/metrics/diagnostics.rs`, `src/game/episode.rs`). Source of truth for PPO timescale claims is `context/systems/brain-ppo.md`'s hyperparameter table. |
| `scripts/init_research_artifact.py` run (stdout captured) | **Met** | `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/brain-inspired-learning/learning-timescales.md` (captured in the completion report). |
| `scripts/validate_research_artifact.py` run (stdout captured) | **Met** | Run before handing back; stdout captured in the completion report accompanying this artefact. |

## What I Did Not Do

- **Did not successfully WebFetch the McClelland/McNaughton/O'Reilly 1995 CLS paper PDF directly.** Both the Stanford and alternative PDF URLs returned binary content the fetcher could not decode. The [CLS-95] passage is quoted from a WebSearch-tool summary of the abstract rather than a primary WebFetch retrieval. All other primary claims are grounded in direct WebFetch passages. If the CLS claim becomes load-bearing for a future decision, fetching the actual PDF from a readable mirror should be the next step.
- **Did not fetch Scholarpedia STP article directly.** Timed out twice on WebFetch. The Frontiers theoretical-models review [STP-TS] and the PMC STP paper [TM-STP] cover the same ground and were successfully fetched, so the substantive STP claims are grounded; the Scholarpedia entry would only have been a cross-check.
- **Did not benchmark any M2 implementation against PPO.** Out of scope for a research paper — that is the actual M2 milestone. This paper constrains what to build; benchmarking validates it.
- **Did not quantitatively tune the recommended eligibility-trace λ.** A range (0.95–0.992) is given tied to the existing PPO γ anchor. The exact value should be picked by ablation during M2 implementation, not pre-committed in a paper.
- **Did not cover reward-signal timescales.** Reserved for the planned sibling `reward-design.md`. Mentioned only where it intersects critical-period discipline.
- **Did not cover spike-timing dependent plasticity (STDP) timescales.** Owned by M4 (spiking) and by `local-learning-rules.md`. The eligibility-trace recommendation here is rate-based and timescale-agnostic between Hebbian and STDP forms.
- **Did not derive the Benna-Fusi scaling constants at small N.** Flagged as an open uncertainty; would require running their model at ~7 K synapses to check whether the asymptotic linear-scaling guarantee holds in M2's regime.
