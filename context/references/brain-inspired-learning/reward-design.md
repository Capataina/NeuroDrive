<!-- Research paper: populated per project-research skill. -->

# Reward Design for Brain-Inspired Plasticity in NeuroDrive

## Scope / Purpose

- Answer the repository-specific question: **how should reward signals feed into brain-inspired plasticity in NeuroDrive, and is the existing velocity-projection + centreline-proximity reward compatible with biologically plausible learning rules — such that PPO coexistence and the entertainment-first philosophy are preserved?**
- Map the three dominant plasticity paradigms (vanilla Hebbian, three-factor, actor-critic-with-plasticity) onto the reward signal each consumes, with timescales, granularity, and eligibility window made explicit.
- Audit the *current* reward implementation term-by-term against those paradigms and surface any term that only works for gradient-based learners.
- Issue a concrete reward + neuromodulator recommendation for the first brain-inspired increment that preserves both PPO compatibility and the entertainment-first constraint.
- Cross-link to the sibling papers in `brain-inspired-learning/` rather than duplicate their content.

### Out of scope (owned by sibling papers)

| Thread | Owner |
|---|---|
| Neuroscience of neuromodulators in full biological detail | `biological-learning-foundations.md` |
| Weight-update math of three-factor rules in detail | `local-learning-rules.md` |
| Topology growth / pruning | `structural-plasticity-neuroevolution.md` |
| Population vs single-brain paradigms | `training-paradigms.md` |
| Consolidation, replay, offline learning | `learning-timescales.md` |
| Curriculum and transfer across tracks | `transfer-and-curriculum.md` |

This paper is strictly about **reward as an algorithmic signal that gets into synaptic weight updates**, not about the neuromodulator biology or the learning-rule math.

---

## Current Project Relevance

NeuroDrive's Milestone 1 PPO baseline is complete (`context/notes/baseline-to-brain-inspired.md`), and Milestone 2 — rate-based local plasticity with eligibility traces and dopamine-like gating — is the next active work area. The transition only succeeds if the **existing 43-dim observation contract and the velocity-projection + centreline reward continue to work unchanged**; otherwise the environment validation has to be re-done. The question is therefore not "what reward would a brain-inspired learner like in principle?" but "what is the smallest reward-side change (if any) that makes the existing reward consumable by a plasticity learner without breaking PPO coexistence or the entertainment-first philosophy?"

The entertainment-first constraint is non-negotiable (`context/notes/reward-and-entertainment.md` §"Core Principle"): cars must drive aggressively and crash until they learn to survive. No crash penalty, no survival bonus, no centreline term large enough to dominate. Any reward-side proposal that even mildly violates this gets rejected on entertainment grounds regardless of its learning-theoretic elegance.

---

## Current State Snapshot

Verified by direct inspection.

### Reward implementation (repository fact)

```text
Per-tick reward (src/game/episode.rs:268–291):
  progress_reward   = dot(velocity, tangent) / speed_reward_reference
                    * velocity_reward_scale                       (line 269–271)
  centreline_reward = centreline_reward_coef
                    * (1 - (distance / max_distance)^2)           (line 274–275)
  time_penalty      = -0.005 per tick                             (config default)
  terminal_reward   = 0.0 on crash                                (line 284)
  tick_reward       = progress_reward + centreline_reward
                    + time_penalty + terminal_reward              (line 291)

Defaults (src/game/episode.rs:57–59):
  velocity_reward_scale    = 1.0
  speed_reward_reference   = 200.0
  centreline_reward_coef   = 0.3
  centreline_reward_max_distance = 50.0

Termination: crash (Collided marker) OR 30 s timeout.
```

### How the reward currently flows to the learner (repository fact)

```text
src/game/episode.rs:293                  # writes EpisodeState.tick.reward
         │
         ▼
src/brain/ppo/mod.rs (collect system)    # ppo_collect_rewards_all_cars_system
         │                                 appends per-car reward_t, done_t to buffer
         ▼
src/brain/ppo/buffer.rs                  # per-env GAE: δ_t = r_t + γV(s_{t+1}) - V(s_t)
         │                                 A_t computed from δ via exponential sum
         ▼
src/brain/ppo/update.rs                  # advantage * ∂log π / ∂θ → policy gradient
                                         # (returns = A + V) → value regression target
```

The reward is **scalar, per-tick, dense, and bounded in typical driving** (roughly [−0.3, +4.5] per tick at terminal velocity with perfect heading; negative when going backward). `done_t` is the only terminal signal — there is no explicit crash penalty. The PPO critic learns `V(s)` online, and GAE folds reward-plus-bootstrap into the advantage.

### What a brain-inspired learner will look like (project inference, guided by `README.md` + `baseline-to-brain-inspired.md`)

- A sparse directed graph of neurons with per-synapse weight `w_ij` and per-synapse eligibility trace `e_ij`.
- A local pre×post plasticity rule that accumulates into `e_ij` without ever seeing reward.
- A **global scalar neuromodulator** `M(t)` derived from the reward stream, broadcast to every synapse.
- Weight updates of the form `Δw_ij = η · M(t) · e_ij`.
- External boundary preserved: same 43-dim observation in, same 2-dim `ActionState.desired` out.

The question this paper answers is what `M(t)` should be computed from, given the reward stream NeuroDrive already produces.

---

## The Reward-to-Plasticity Mapping Table

For each major plasticity paradigm, what is the reward signal's role, and what must be true about it for learning to occur?

| Paradigm | Consumes reward? | Weight-update form | Required reward property | Timescale of credit | Eligibility window | Works for NeuroDrive? |
|---|---|---|---|---|---|---|
| **Vanilla Hebbian / STDP** (two-factor) | **No.** Reward is ignored. | `Δw_ij = η · f(pre_i, post_j)` | None — reward doesn't enter | Instantaneous correlation only | None needed | **No.** Cannot distinguish good driving from bad. Hebbian alone rewards any repeating correlation, including repetitive crashes. See **[FG-16-HEB]**. |
| **Three-factor (neuromodulated Hebbian / R-STDP)** | **Yes.** Scalar `M(t)` gates consolidation. | `ė_ij = −e_ij/τ_e + f(pre_i, post_j)`; `Δw_ij = η · M(t) · e_ij` | Scalar broadcast signal; sign + magnitude carry credit | Single global signal arriving within eligibility window (seconds) | **200 ms – 2 s** behaviourally, **5 s** max for LTP in cortex under NE (**[GB-18-WIN]**) | **Yes**, with qualifications. The eligibility window (≤5 s) matches NeuroDrive's 30 s episode and ≤2.17 s observation lookahead. |
| **Reward-modulated STDP (phasic RPE)** | **Yes.** `M(t) = δ_t = r_t + γV(s') − V(s)`. | Same as three-factor but with TD-error in place of raw reward | Requires *some* predictor `V(s)`; bias-free if the baseline is accurate | Per-tick (or per-event in SNNs) | Same as three-factor | **Yes**, but requires building a value predictor alongside the plastic network. This is the Frémaux & Gerstner actor-critic-with-plasticity pattern. |
| **Actor-critic with plastic actor, gradient critic** | **Yes.** Critic is still trained by gradient descent; its output feeds the actor's plasticity rule | Actor: `Δw = η · δ · e_ij`; Critic: gradient descent on `(r + γV(s') − V(s))²` | Raw scalar reward at the critic; TD-error `δ` at the actor | Per-tick | Same | **Yes** and is the most conservative migration path for NeuroDrive: reuse the existing PPO critic as the `V(s)` predictor while replacing the actor's gradient updates with plastic ones. PPO coexistence is automatic because the critic stays shared. |
| **Curiosity / intrinsic motivation** (Pathak et al. 2017) | **Yes**, but the reward is *generated* by the agent itself from prediction error of its own forward model | Augments `M(t) = r_extrinsic + β · r_intrinsic`, where `r_intrinsic = ‖φ(s_{t+1}) − f̂(φ(s_t), a_t)‖²` | None extra — compatible with any of the above | Per-tick | Same | **Partial fit.** Good for breaking exploration plateaus; risky ("noisy-TV problem") in a visually stochastic environment. Our environment is deterministic given actions, so the noisy-TV risk is low. See **[ICM-17]**. |

### ASCII summary of where reward enters plasticity

```text
          pre spike / activity ─┐
                                ├──► eligibility trace  e_ij(t)
          post spike / activity ┘         │
                                          │  (local, no reward needed)
                                          │
  reward r(t) ──► modulator M(t) ─────────┴───► Δw_ij = η · M(t) · e_ij
                                                   (global broadcast)
```

### Reward granularity: scalar vs vector

Frémaux & Gerstner argue the third factor is most commonly modelled as **a single scalar `M(t)`** broadcast uniformly, but note explicitly that "synapses may react to the specific mix of neuromodulators" and that "the time course of dopamine could contain information on a mixture of 'reward compared to expected reward' and 'novelty'" (**[FG-16-MIX]**). Biologically, dopamine, serotonin, noradrenaline, and acetylcholine coexist and can differentially gate different neuron populations — so a vector is closer to reality. **Practical implication for NeuroDrive:** start with one scalar, but design the plumbing so a second (or third) neuromodulator channel can be added without touching the synaptic-update kernel. This is cheap up-front and avoids a refactor if curiosity or a separate "stay-on-track" modulator is introduced later.

---

## Sub-Question Walk-Through

### 1. Dopamine and reward prediction error (Schultz 1997)

**Source-backed finding.** Schultz, Dayan & Montague (1997) established the mapping from midbrain dopamine neuron firing to the TD prediction error: "The dopamine prediction error signal with reward-predicting stimuli corresponds well to the teaching term of temporal difference (TD) learning, a derivative of the Rescorla-Wagner model" (**[SDM-97]**, via search summary of *A Neural Substrate of Prediction and Reward*, Science 1997).

**Does every synapse need access to a global RPE signal?** The canonical three-factor framework says yes — the signal is broadcast through ramified dopaminergic projections. But it is a *scalar broadcast*, not a per-synapse signal. Every synapse reads the same `M(t)`; what differs across synapses is the eligibility trace they bring to the multiplication. This is computationally cheap: one scalar per tick, multiplied into an `O(|synapses|)` kernel. It maps cleanly onto a shared-memory resource in Bevy.

**Project implication.** NeuroDrive can compute `δ_t = r_t + γV(s_{t+1}) − V(s_t)` exactly as PPO already does and expose it as a `Resource<Neuromodulator>` for the plastic layer to read. The existing PPO critic network is the natural `V(s)` estimator — reusing it as a shared service is the conservative default (see Recommendation).

### 2. Three-factor learning rules — what counts as "modulator"

**Source-backed finding.** Frémaux & Gerstner (2016): "neuromodulators that would 'gate' Hebbian plasticity such that up- or down-regulation of synapses happens at appropriate moments in time" — the common candidates being **dopamine, acetylcholine, and noradrenaline** (**[FG-16-MOD]**). The third factor can also encode "a mixture of novelty and success" (**[FG-16-MIX]**).

Dopamine is the classical "success" signal (RPE). Acetylcholine is more tied to attention / uncertainty. Noradrenaline to arousal / novelty. Serotonin, controversially, to patience / long-horizon valuation.

**Project implication.** For NeuroDrive's first brain-inspired increment, use a *single* scalar `M(t) = δ_t` (RPE-like). This is the tightest analogue to the PPO advantage signal and keeps the plumbing minimal. Do not design for multi-modulator channels before there is a concrete learning deficit that demands one.

### 3. Eligibility traces — how long does credit persist?

**Source-backed finding.** Gerstner et al. (2018), *Eligibility Traces and Plasticity on Behavioral Time Scales*: "the eligibility trace for action learning should be in the range of a typical elementary action, about 200 ms to 2 s" (**[GB-18-WIN]**). Experimentally: "LTP can be induced if the neuromodulator NE (third factor) arrived with a delay of 5 s or less" in cortex (**[GB-18-CTX]**); striatal dopamine acts "only if phasic dopamine was given in a narrow time window during or immediately after the 1 s-long STDP protocol" (**[GB-18-STR]**); hippocampal eligibility decays over ~2 s (**[GB-18-HIP]**). The paper also notes: "delaying a reinforcer by 10 s during ongoing actions decreases learning compared to immediate reinforcement" (**[GB-18-10S]**).

**Project implication.** NeuroDrive's 30 s episode is well outside any biologically plausible eligibility window, but this is a non-problem because the reward is already dense (every tick, 60 Hz). The credit-assignment span that actually matters is the **action-to-consequence span within an episode** — typically 100 ms to ~3 s between deciding to lift off throttle and the car reaching the corner. That fits inside the 200 ms – 5 s window reported above. A trace decay of **τ_e ≈ 1–3 s** (60–180 ticks at 60 Hz) is the defensible starting point. This is close to NeuroDrive's current discount horizon (γ=0.995 ⇒ ~3.3 s), so the two timescales agree without tuning.

### 4. Sparse vs dense reward in biological learning

**Source-backed finding.** Research on infant motor learning (**[TODDLER-25]**, arXiv 2501.17842): "Biological systems, such as human toddlers, naturally navigate the balance between exploration and goal-directed behavior by transitioning from free exploration with sparse rewards to goal-directed behavior guided by increasingly dense rewards." Infants can learn from binary rewards from very young ages (nine-week-old kicking-mobile paradigm), but "three to eight-year-old children learned best with discrete targets and in a deterministic reward landscape."

**Nuance.** Biological skill acquisition actually uses a *mixture* — sparse signals at task completion (success/failure), dense continuous signals from proprioception and interoception (am I tense? am I off-balance?), and very sparse explicit reward (occasional feedback from a coach). The mythology of "biology = sparse reward" is an oversimplification. The brain's own dense signals are largely *self-generated* from internal models.

**Project implication.** NeuroDrive's dense velocity-projection reward is *not* biologically implausible — it is exactly the kind of dense proprioceptive signal a driving brain would generate from its own velocity and heading estimates. Keeping the reward dense is both entertainment-correct and biology-correct. The sparse alternative (reward only on lap completion or crash) would be worse for both axes.

### 5. Intrinsic motivation — curiosity, novelty, empowerment

**Source-backed finding.** Pathak et al. (ICM, ICML 2017): intrinsic curiosity is computed as "the error in an agent's ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model" (**[ICM-17]**). They explicitly note that curiosity-driven agents can solve tasks "with no extrinsic reward" in some domains.

**Contrasting / limiting signal.** The "noisy-TV problem" (search summary): "Curiosity-driven agents often fall prey to the 'noisy TV problem', getting distracted by random, uncontrollable stimuli that offer no meaningful opportunities for mastery." Pathak specifically designs ICM to avoid this by working in a learned inverse-dynamics feature space rather than raw pixels.

**Project implication.** Biological learners do use novelty-driven exploration, and this is usually mediated by noradrenaline / acetylcholine rather than dopamine. For NeuroDrive specifically, the environment is **deterministic given actions** — so the noisy-TV problem largely vanishes, and an ICM-style bonus based on forward-model prediction error would be a plausible second neuromodulator channel if exploration collapses under pure RPE. But it is premature: PPO already solves the task with entropy-bonus exploration alone, and adding curiosity before the plastic learner has been benchmarked would confound a diagnosis.

### 6. Reward shaping for plastic agents vs PPO — do they need the same signal?

**Project inference (guided by repository facts).** Yes, with one clarification: the raw `r_t` that leaves `episode.rs:293` can be consumed by both learners identically. What differs is the *derived* signal each constructs from `r_t`:

- PPO derives `A_t = Σ (γλ)^k · δ_{t+k}` (GAE advantage) over an on-policy rollout.
- A plastic three-factor learner derives `M(t) = δ_t` (instantaneous TD-error) and multiplies into eligibility traces.

Both are functions of the same `r_t` stream. No change to the environment is required. The plastic learner gets its own `V(s)` predictor (either a shared PPO critic or its own small network).

### 7. Multi-dimensional reward signals

**Source-backed finding.** Frémaux & Gerstner (2016): "synapses may react to the specific mix of neuromodulators" and "there is no need of a one-to-one mapping from neuromodulators to specific functions, but a mixed coding scheme would be sufficient" (**[FG-16-MIX]**). The brain uses vector neuromodulation; models usually approximate it with a scalar.

**Project implication.** Design the `Resource<Neuromodulator>` as a struct with named scalar fields (`dopamine: f32, novelty: f32, ...`) from the start, even if only `dopamine` is written for now. This is a free architectural decision — the weight-update kernel can ignore unused fields — and avoids a painful refactor if a second modulator ever becomes necessary.

### 8. Compatibility audit — does every current reward term work as a plasticity modulator?

See the dedicated **Compatibility Audit** section below. Summary: **yes, every current term is compatible.** None of the terms rely on gradient flow through the reward itself; they are all scalar contributions to `r_t` that a plastic learner can consume via `M(t) = δ_t`.

---

## Contrasting Perspective — Is Dopamine Actually RPE?

The dopamine = RPE story is dominant but not uncontested. Two strands of critique are directly relevant to NeuroDrive's design choices.

**Berridge's incentive-salience account.** "Robinson and Berridge (2008) suggested that mesolimbic dopamine is selectively involved in attributing incentive salience to potential objects or options to guide approach behavior, and that it has no role in RPE coding" (**[BER-07]**, via PMC7804370 review). The claim is that dopamine signals *wanting* rather than *learning*.

**Redgrave / Gurney salience critique.** Dopamine neurons also fire to "physically salient sensory stimuli, such as tones and lights" with a 50–110 ms latency, before reward value is even resolved (**[DPE-BEY]**). This short-latency salience burst precedes the reward identity being computed and does not look like an RPE.

**Why this matters for NeuroDrive.** If the plastic learner's sole modulator is `M(t) = δ_t` and we are wrong that this is what dopamine does, the learner will still converge on driving — because `δ_t` is a valid RL teaching signal mathematically regardless of whether it maps onto biology. The biological-plausibility argument is about whether the *implementation matches the brain's computation*, not whether the *mathematics is valid for credit assignment*. NeuroDrive is a research-grade learning laboratory, not a neuroscience model, so using the RPE signal is pragmatically safe. But the project should not claim the simulation demonstrates something about dopamine specifically — only that an RPE-gated three-factor rule can learn racing.

This is the single most important honesty point when writing up future NeuroDrive findings.

---

## Compatibility Audit — Current Reward Terms vs Plasticity Paradigms

For each term in the existing per-tick reward, audit: is it consumable by a plastic learner unchanged? Does it require a gradient? Does it violate any timescale constraint?

| Term | Current value | Signal shape | Plastic-learner compatibility | Timescale compatibility | Entertainment-first compatibility | Verdict |
|---|---|---|---|---|---|---|
| `velocity_projection = dot(v, tangent) / 200 · 1.0` | Dense, signed, unbounded above (~4.5/tick at terminal v), negative when reversing | Pure scalar bolt-on to `r_t` | Consumable unchanged. Feeds `M(t) = δ_t` directly. No gradient needed in the reward computation itself. | Per-tick (16.67 ms) is well inside the 200 ms – 2 s eligibility window. | Makes cars go fast. | **Keep as-is.** |
| `centreline_reward = 0.3 · (1 − (d/50)²)` | Dense, non-negative, bounded [0, 0.3] | Pure scalar bolt-on | Consumable unchanged. | Per-tick. | Gentle — coefficient 0.3 is intentionally below dominance threshold (`notes/reward-and-entertainment.md`). | **Keep as-is.** |
| `time_penalty = -0.005` | Dense, constant, small | Pure scalar bolt-on | Consumable unchanged. | Per-tick. | Too small to create "do nothing" optimum when velocity reward exists. | **Keep as-is.** |
| `terminal_reward = 0.0 (on crash)` | Sparse, zero | Nominal terminal | The `done_t` flag does the work — critic learns opportunity cost. Plastic learner sees `done_t` identically. | One-off on episode end. | Entertainment rule satisfied (no penalty). | **Keep as-is.** |
| Episode termination at 30 s / crash | Sparse event | `done_t` flag | Three-factor learners use `done_t` to truncate bootstrap (`V(s_{terminal+1}) = 0`) exactly like PPO does. | Episode timescale unrelated to eligibility timescale. | Dies-is-the-penalty is the philosophy. | **Keep as-is.** |

### What would *not* be compatible (counterfactual check)

| Hypothetical reward term | Why it would break plasticity compatibility |
|---|---|
| A term that depends on `∂π/∂θ` (policy gradient) | Plastic learners have no access to `∂π/∂θ`. Would be gradient-only and PPO-only. |
| A term that requires off-policy correction (IS weights) | Plastic learners are online by design; importance sampling has no natural analogue. |
| A crash penalty of −5 | Violates entertainment-first (`notes/reward-and-entertainment.md`). Also known empirically to produce "sit still" behaviour (`README.md` §Design Decisions). |
| A survival bonus | Same failure mode — incentivises passivity. |
| A reward term that activates only at episode end (pure sparse reward) | Plastic eligibility traces decay in seconds; 30 s delay would wipe all credit. Would need n-step bootstrapping from a value predictor, which is exactly what plastic rules try to avoid doing by hand. |

**Conclusion of the compatibility audit:** every term currently in `episode.rs` is consumable by a three-factor plastic learner without modification. No reward-side change is required to enable Milestone 2. The transition is a *learning-rule* migration, not a *reward* migration.

This is the strongest possible finding for the project plan — it means Milestone 2 can focus entirely on the plasticity implementation (eligibility traces, neuromodulator broadcast, synaptic update kernel) without reopening the reward-design debate.

---

## Recommendation for NeuroDrive

### The reward + neuromodulator pattern to start with

1. **Do not change the reward stream.** Keep `velocity_projection + centreline_reward + time_penalty`, crash penalty 0.0, termination on crash or 30 s. The compatibility audit confirms this works for both paradigms.

2. **Compute a scalar neuromodulator once per tick, in a new `Resource<Neuromodulator>`.** Definition:

   ```text
   M_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
   ```

   This is exactly the PPO one-step TD error. Reuse the PPO critic as `V(s)` to start — the critic is already in memory, already returns denormalised values via PopArt, and already participates in every tick. The cost is one extra `forward_critic` call per car per tick, which is well within the current 4.4% budget utilisation (`context/architecture.md` §Structural Notes).

3. **Design the `Neuromodulator` resource with multi-channel shape even if only one channel is written.** A struct with named fields `dopamine: f32, novelty: f32 (unused), ...` costs nothing now and avoids a refactor later. Set a convention: `dopamine` is TD error; `novelty` would be a forward-model prediction error if curiosity is added.

4. **Set eligibility trace decay to `τ_e ≈ 2 s` (120 ticks at 60 Hz).** This is inside the biologically measured 200 ms – 5 s window (**[GB-18-WIN]**, **[GB-18-CTX]**) and aligned with NeuroDrive's existing γ=0.995 credit horizon (~3.3 s). Expose as a config field so ablations can sweep it.

5. **Keep PPO coexistence by not retiring the critic.** `AgentMode` becomes three-way: `Keyboard` / `Ai` (PPO) / `Brain` (plastic actor + PPO critic for `V(s)`). This lets the project run PPO and the plastic learner side-by-side on the same environment and same reward, producing a direct learning-curve comparison. The stable-boundary constraint from `baseline-to-brain-inspired.md` is preserved automatically.

6. **Preserve entertainment-first.** No change needed — the reward terms that enforce it (velocity projection dominant, centreline gentle, zero crash penalty) are already in place and compatibility-audited.

### ASCII of the proposed coexistence

```text
                                          Shared reward
          ┌──────── episode_loop_system ──► r_t ─┬─────────────┐
          │                                      │             │
          ▼                                      ▼             ▼
  ┌─────────────┐                        ┌──────────────┐ ┌──────────────┐
  │   Critic    │◄─── V(s), V(s')        │   PPO Actor  │ │ Plastic Actor│
  │  (shared)   │                        │  (unchanged) │ │ (new, M2)    │
  └──────┬──────┘                        └──────┬───────┘ └──────┬───────┘
         │  δ_t                                 │                │
         └──────── Resource<Neuromodulator> ────┼────────────────┤
                        dopamine = δ_t          │                │
                                         chosen │                │
                                   by AgentMode ▼                ▼
                                          ActionState.desired
```

### Why this is the right starting shape

- **Reuses everything that works.** The environment, observation contract, reward, critic, and analytics are all untouched. The only new engineering is the eligibility-trace kernel and the `Neuromodulator` resource.
- **PPO coexistence is free.** Both learners share the same `r_t` and the same `V(s)`.
- **Biologically honest enough.** The scalar-RPE broadcast matches Frémaux & Gerstner's canonical three-factor rule. It is not the most biologically accurate possible choice, but the more accurate alternatives (vector modulation, multi-timescale traces, distributional RPE) are premature before the scalar version has been shown to learn at all.
- **Entertainment-first preserved automatically.** The reward philosophy lives in `episode.rs`; it doesn't care which learner consumes `r_t`.

### Assumption that needs stronger evidence

The dominant assumption is that **a scalar TD-error broadcast is sufficient for a plastic actor to learn racing at NeuroDrive's current speeds.** This is empirically true for simple tasks (cart-pole, LunarLander per **[TF-SNN-25]**) but unvalidated at 43-dim observation + 2-dim continuous action in 30 s episodes with delayed consequences (throttle decision now, crash in 2 s). If the plastic actor fails to learn corner anticipation, the first diagnostic should be whether the critic's `V(s)` is accurate enough — not reward-side changes.

### Failure mode to watch

**Eligibility-trace decay misalignment.** If `τ_e` is too short (<500 ms), credit never bridges the action-to-consequence span for cornering. If too long (>5 s), traces from old actions contaminate current updates and learning becomes noisy. Monitor the histogram of `|e_ij|` across synapses during training — if it is dominated by a few persistently-large traces that never decay, `τ_e` is too long.

### What would flip the recommendation

- If early experiments show the plastic actor cannot learn from scalar RPE alone despite a healthy critic, the first thing to try is **a second modulator channel for novelty/curiosity** (ICM-style forward-model error), not a reward shape change. Reward shape is already compatibility-audited.
- If the shared PPO critic becomes a bottleneck for the plastic learner (e.g., its value estimates are too coarse for the three-factor rule's tighter eligibility timescales), the fallback is a **dedicated small critic trained by gradient descent alongside the plastic actor**, still on the same `r_t`. This still does not require a reward change.

---

## Gap Analysis — What's Missing to Ship Milestone 2

| Gap | Blocker for plastic learner? | Severity |
|---|---|---|
| `Resource<Neuromodulator>` with scalar `dopamine` channel | Yes | Must ship |
| Shared access path from PPO critic's `V(s)` into `M_t` computation | Yes | Must ship |
| Per-synapse `eligibility_trace: f32` field in the plastic-graph data structure | Yes | Must ship — see sibling `local-learning-rules.md` |
| Config field for `eligibility_trace_decay_seconds` (τ_e) | Yes | Must ship |
| Three-way `AgentMode` (Keyboard / PPO / Brain) | Not for learning, yes for coexistence | Should ship |
| Instrumentation: `M_t` histogram, `|e_ij|` histogram, weight-change rate per tick | Not for correctness; yes for debugging | Should ship before claiming the learner works |
| Multi-channel neuromodulator plumbing (even if only `dopamine` is live) | No | Cheap to include; include to avoid refactor |

Nothing on this list is a reward-side change. The entire Milestone 2 work plan is on the learning-rule side, which is exactly the decoupling the baseline validation was designed to produce.

---

## Open Uncertainties And Validation Needs

1. **Is the PPO critic's `V(s)` stable enough to feed a three-factor rule?** The critic already uses PopArt and target-KL early-stop, which makes it stable for PPO. But three-factor rules consume `δ_t` directly (not the smoothed GAE advantage), so noise in `V(s)` shows up one-to-one in the modulator. **Validation:** log the std of `δ_t` over a window; compare to PPO's advantage-normalisation scale. If σ(δ_t) > a few times typical `|r_t|`, smooth with a short EMA before broadcasting.

2. **Is the eligibility window of 2 s optimal?** The biologically measured range is 200 ms – 5 s. NeuroDrive's action-to-consequence gap for cornering is ~0.5 – 3 s. These overlap, but the exact value is a free parameter. **Validation:** sweep `τ_e ∈ {0.5, 1, 2, 3, 5}` seconds in an ablation once the learner is running.

3. **Does scalar RPE suffice, or is vector modulation needed?** See Recommendation §Assumption. Empirical question, answerable only after the scalar version is benchmarked.

4. **Does the current dense reward give enough signal for a plastic actor?** PPO learns from it, but PPO has the advantage-normalisation and GAE smoothing that a plastic actor lacks. **Validation:** report `M_t` distribution and zero-crossing rate in the same analytics report that reports PPO stats.

---

## Relationship To Existing Context

| Sibling | What it owns | What this paper cross-links on |
|---|---|---|
| `biological-learning-foundations.md` | Full neuroscience of neuromodulators, dopamine biology, STDP substrate | Anchors the "dopamine broadcast" concept; this paper uses the algorithmic shape, that paper owns the biology |
| `local-learning-rules.md` | The weight-update math (R-STDP, R-max, Frémaux/Gerstner equations) | Consumes the scalar `M(t)` this paper recommends; owns the `ė = -e/τ + STDP(...)` kernel |
| `structural-plasticity-neuroevolution.md` | Growth / pruning rules | Will consume a dedicated modulator channel (e.g. co-activity-driven growth); multi-channel `Neuromodulator` here makes that cheap |
| `training-paradigms.md` | Population vs single-brain, episode-based vs lifetime | Orthogonal — reward design is agnostic to how many brains consume it |
| `learning-timescales.md` | Consolidation, replay, sleep phases | Shares the "what timescale to credit" question but at hours-to-days horizon rather than seconds |
| `transfer-and-curriculum.md` | Multi-track transfer | May add curriculum-dependent modulator shaping later |

Existing on-track references also cross-link:
- `context/references/reward-structure-design.md` — the original reward research that justified velocity projection + centreline + zero crash penalty. **This paper does not re-litigate reward-term choice; it validates the existing choices against a new consumer.**
- `context/references/ppo-critic-architecture.md`, `value-target-normalisation.md` — the critic is now shared machinery; those papers document why it works.
- `context/notes/reward-and-entertainment.md` — entertainment-first constraint that this paper is explicitly designed not to violate.

---

## External Research Trail

**Searches run**

| # | Query | Tool | Rationale | Sources surfaced (kept) |
|---|---|---|---|---|
| 1 | `three-factor learning rule dopamine neuromodulation eligibility trace pre post review 2024` | WebSearch | Canonical three-factor framework | Frémaux & Gerstner 2016 (Frontiers / PMC), Gerstner et al. 2018 (Frontiers / PMC), Patterns 2025 SNN review |
| 2 | `Schultz 1997 dopamine reward prediction error neuron TD learning original` | WebSearch | Foundational dopamine = RPE paper | Schultz/Dayan/Montague 1997 *Science*; PubMed 9054347 / PMC 4826767; Bornlab 1998 Nature Neuro |
| 3 | `dopamine is not reward prediction error salience novelty criticism Berridge Redgrave` | WebSearch | Contrasting source (Sufficiency Floor) | Berridge 2007 *Psychopharmacology*; Robinson & Berridge 2008; PMC 7804370 "Dopamine, Prediction Error and Beyond" |
| 4 | `Fremaux Gerstner neuromodulated STDP review reward-modulated plasticity eligibility trace` | WebSearch | Primary three-factor review | Frémaux & Gerstner 2016 (Frontiers Neural Circuits); PMC 4717313 |
| 5 | `curiosity intrinsic motivation reinforcement learning Pathak Schmidhuber sparse reward racing` | WebSearch | Intrinsic-motivation sub-question | Pathak et al. 2017 ICML; project page pathak22.github.io |
| 6 | `sparse versus dense reward biological skill acquisition motor learning human infant` | WebSearch | Sub-question 4 — dense vs sparse in biology | arXiv 2501.17842 "Toddler-inspired Reward Transition"; multiple Frontiers motor-learning papers |

**Sources consulted (WebFetch)**

| URL | Tool | Source class | Key passages quoted below? |
|---|---|---|---|
| https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full | WebFetch | Foundational review (Frémaux & Gerstner 2016) | Yes — **[FG-16-MOD]**, **[FG-16-HEB]**, **[FG-16-MIX]** |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/ | WebFetch | Foundational review (Gerstner et al. 2018) | Yes — **[GB-18-WIN]**, **[GB-18-CTX]**, **[GB-18-STR]**, **[GB-18-HIP]**, **[GB-18-10S]** |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC7804370/ | WebFetch | Contrasting / limiting review ("Dopamine, Prediction Error and Beyond") | Yes — **[DPE-BEY]**, **[BER-07]** |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/ | WebFetch | Foundational review (Frémaux & Gerstner PMC mirror) | Yes — explicit equations `ẇ = M × e` and `ė = -e/τ_e + STDP(pre,post)` |
| https://pathak22.github.io/noreward-rl/ | WebFetch | Reference implementation / project page (ICM) | Yes — **[ICM-17]** |
| https://www.sciencedirect.com/science/article/pii/S2666389925002624 | WebFetch | Recent SNN review (Cell *Patterns* 2025) | Yes — **[TF-SNN-25]** |
| https://sites.lsa.umich.edu/berridge-lab/.../Berridge-2007-Debate-over-dopamine...pdf | WebFetch (PDF — failed) | Primary contrasting paper | No verbatim (binary PDF unreadable). Berridge's position is represented via PMC7804370's quoted summary instead. |
| https://arxiv.org/pdf/1705.05363 | WebFetch (PDF — failed) | ICM original paper | No verbatim (binary PDF unreadable). Claim represented via the project-page fetch **[ICM-17]** and search-summary quote. |
| https://arxiv.org/pdf/2504.05341 | WebFetch (PDF — failed) | SNN three-factor primary | No verbatim. Represented via Patterns review **[TF-SNN-25]**. |

Six substantive fetches succeeded; three PDF fetches failed on binary encoding and their content is covered by alternative HTML sources on the same topic. Per `references/script-fallbacks.md` principles, the gap is documented rather than silently skipped.

**Quoted passages**

- **[FG-16-HEB]** — source: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full
> "Hebbian learning, STDP, as well as other unsupervised learning rules neglect, by design, any information regarding 'reward,' 'success,' 'punishment,' or 'novelty.'"

- **[FG-16-MOD]** — source: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full
> "neuromodulators that would 'gate' Hebbian plasticity such that up- or down-regulation of synapses happens at appropriate moments in time."

- **[FG-16-MIX]** — source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/
> "synapses may react to the specific mix of neuromodulators ... there is no need of a one-to-one mapping from neuromodulators to specific functions, but a mixed coding scheme would be sufficient."
> "the time course of dopamine could contain information on a mixture of 'reward compared to expected reward' and 'novelty.'"

- **[GB-18-WIN]** — source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/
> "the eligibility trace for action learning should be in the range of a typical elementary action, about 200 ms to 2 s."

- **[GB-18-CTX]** — same source
> "LTP can be induced if the neuromodulator NE (third factor) arrived with a delay of 5 s or less" (cortex).

- **[GB-18-STR]** — same source
> "dopamine promoted spine enlargement only if phasic dopamine was given in a narrow time window during or immediately after the 1 s-long STDP protocol" (striatum).

- **[GB-18-HIP]** — same source
> "the synaptic flag set by the induction protocol leaves an eligibility trace which decays over 2 s" (hippocampus).

- **[GB-18-10S]** — same source
> "delaying a reinforcer by 10 s during ongoing actions decreases learning compared to immediate reinforcement."

- **[SDM-97]** — Schultz, Dayan & Montague 1997, via search summary of *A Neural Substrate of Prediction and Reward*, Science 275:1593–1599
> "The dopamine prediction error signal with reward-predicting stimuli corresponds well to the teaching term of temporal difference (TD) learning, a derivative of the Rescorla-Wagner model."

- **[BER-07]** — Berridge / Robinson via https://pmc.ncbi.nlm.nih.gov/articles/PMC7804370/
> "Robinson and Berridge (2008) suggested that mesolimbic dopamine is selectively involved in attributing incentive salience to potential objects or options to guide approach behavior, and that it has no role in RPE coding."

- **[DPE-BEY]** — "Dopamine, Prediction Error and Beyond", PMC 7804370
> "Physically salient sensory stimuli, such as tones and lights, evoke very rapid (50-110 ms), phasic excitations in dopamine neurons."
> "The main question though is whether dopamine uniquely codes RPEs, or whether this is one of the (many) functions of dopamine."

- **[ICM-17]** — Pathak et al. 2017, via https://pathak22.github.io/noreward-rl/
> curiosity is "the error in an agent's ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model."

- **[TF-SNN-25]** — ScienceDirect S2666389925002624 (Cell *Patterns* 2025 review)
> the neuromodulatory signal is "analogous to the function of dopamine in the brain ... a third, global signal [that] modulates synaptic plasticity based on global information, thereby facilitating more effective credit assignment."

---

## Pre-Completion Obligation Audit

| Obligation | Status | Evidence |
|---|---|---|
| At least 3 distinct WebSearch calls with topic-specific queries | Met | 6 distinct queries run — see "Searches run" (three-factor rules, Schultz 1997, Berridge/Redgrave critique, Frémaux/Gerstner review, curiosity/ICM, infant sparse vs dense reward) |
| At least 3 distinct WebFetch calls against primary sources | Met | 6 substantive fetches succeeded: Frontiers Frémaux&Gerstner 2016, PMC 6079224 Gerstner 2018, PMC 7804370 Dopamine-Beyond, PMC 4717313 Frémaux&Gerstner mirror, pathak22.github.io ICM project page, ScienceDirect Patterns 2025 |
| Sources span at least 2 source classes | Met | Foundational peer-reviewed reviews (Frontiers, PMC), contrasting / limiting review (PMC7804370), reference-implementation project page (pathak22.github.io), recent survey (Cell Patterns 2025) — four classes |
| At least 1 direct quoted passage per major source-backed claim | Met | 13 labelled passages **[FG-16-*]**, **[GB-18-*]**, **[SDM-97]**, **[BER-07]**, **[DPE-BEY]**, **[ICM-17]**, **[TF-SNN-25]** cited in-line |
| At least 1 contrasting / limiting / disagreeing source consulted | Met | PMC 7804370 "Dopamine, Prediction Error and Beyond" and the Berridge incentive-salience account **[BER-07]**, represented in the "Contrasting Perspective" section. Plus the ICM noisy-TV limitation as a bounded limitation on the curiosity recommendation. |
| Relevant `context/` files read before project-specific claims | Met | `README.md`, `context/architecture.md`, `context/systems/brain-ppo.md`, `context/systems/environment.md`, `context/notes/reward-and-entertainment.md`, `context/notes/baseline-to-brain-inspired.md`, `context/references/reward-structure-design.md` all read end-to-end before writing. Sibling scaffolds under `brain-inspired-learning/` also inspected. |
| Relevant code inspected (list file paths) | Met | `src/game/episode.rs:57–59, 250–312, 461–463` verified via Grep + Read; reward composition quoted from the file lines. Architecture inventory cross-checked against `context/architecture.md`. |
| `scripts/init_research_artifact.py` run (stdout captured) | Met | Ran `brain-inspired-learning/reward-design --title "Reward Design..." --kind file`. Stdout: `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/brain-inspired-learning/reward-design.md`. Prior folder-kind scaffold was removed before re-running in file mode. |
| `scripts/validate_research_artifact.py` run (stdout captured) | Met | All hard checks pass: title OK, all required sections OK, 9 URLs / 6 unique domains in External Research Trail, 13 quoted passages, 3/4 evidence-label classes, no exhortation adverbs. Two remaining warnings are advisory (suggested `## Research Signal` section; see Research-Signal-style rows already embedded in sub-question walk-through + compatibility audit). |

## What I Did Not Do

- **Did not quote from the Berridge 2007 *Psychopharmacology* PDF directly.** The WebFetch on the LSA/Berridge-Lab PDF returned only binary/compressed bytes. The contrasting-source obligation is satisfied via the PMC 7804370 review which quotes Berridge & Robinson 2008 directly; however, a reader who wanted the verbatim 2007 passage will not find it here. Remedy if it matters: fetch the *Psychopharmacology* published HTML version (via DOI 10.1007/s00213-006-0578-x) and re-run the extraction.
- **Did not quote from the ICM 2017 arXiv PDF directly** for the same reason. The project-page fetch covers the essential claims; deeper methodological detail (the feature-space inverse-dynamics loss) is not reproduced here. Remedy: fetch the HTML abstract page on arXiv or the ICML proceedings HTML.
- **Did not benchmark the recommended reward + modulator scheme empirically.** This is by design — research paper, not implementation. The first empirical test is Milestone 2's first plastic learner run, at which point an Update pass to this paper is appropriate.
- **Did not survey non-dopaminergic modulators in depth.** Acetylcholine, noradrenaline, serotonin are mentioned but not unpacked — the Recommendation is scalar-dopamine-first. Further biology on the multi-modulator story lives in `biological-learning-foundations.md`.
- **Did not assess reward-weighted regression or KL-regularised plasticity alternatives.** These are outside the three families in the mapping table and are future work if the baseline three-factor recommendation fails.
- **Did not propose reward changes that would help plasticity but hurt PPO.** Explicit constraint from the user spec; honoured throughout.
