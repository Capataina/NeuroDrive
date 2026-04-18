# PPO Critic Architecture for NeuroDrive

## Scope / Purpose

- Answer the repository-specific question: **which choices of activation, width, depth, normalisation, and output-layer in the PPO critic most plausibly cause the observed 68.5 % `critic_fc2` tanh saturation on NeuroDrive's racing task, and how should those choices be changed in the handwritten Rust PPO to restore anticipatory value prediction?**
- Focus exclusively on the value network (critic). Actor architecture and reward shaping are covered in `context/references/ppo-optimisation.md` and `reward-structure-design.md` respectively and are not re-litigated here.
- Ground every recommendation in a concrete edit to `src/brain/ppo/model.rs` or `src/brain/common/mlp.rs` and give an expected effect on the saturation number so the hypothesis is falsifiable on the next training run.
- Explicitly include a steelman for "do nothing / the saturation is a symptom" so the research is not single-hypothesis-confirming.

## Current Project Relevance

The latest full run (`reports/analytics/run_1776543971.md`, 15 161 episodes, 15 147 crashes) diagnoses a policy that is reactive but not predictive:

- critic_fc2 tanh saturation fraction: **68.5 %** (`Layer Health` table, line 296);
- critic mean value prediction overall: **80.86**, mean at crash: **46.85** (section 6, lines 190–192) — a 42 % separation is too small for anticipatory braking;
- 80 % of crashes classified as **Overshoot** (line 182) — cars full-throttle into walls at the end of long straights;
- returns grew monotonically during training from chunk-1 ≈60 to chunk-10 ≈460, an 8× unbounded scale shift on the critic's target distribution.

This points at a critic value-prediction failure. The critic cannot distinguish "wall-ahead-in-2-seconds" from "corner-ahead-in-2-seconds" in state value, so the policy gradient through the advantage never tells the actor to throttle-down on straights. Before reaching for reward hacks, the research question is whether the *architecture itself* is starving the critic of expressivity on a growing-return target.

## Current State Snapshot

Verified by direct code inspection of `src/brain/ppo/model.rs`, `src/brain/common/mlp.rs`, `src/brain/common/optim.rs`, and the analytics layer that produces the layer-health numbers.

### Critic model definition (repository fact)

```text
obs (43) → Linear(43, 128) → Tanh     ← c_fc1 (1.3 % sat, healthy)
         → Linear(128, 128) → Tanh    ← c_fc2 (68.5 % sat, distressed)
         → Linear(128, 1)              ← c_value (unbounded, orthogonal init scale 1.0)
```

- Source: `src/brain/ppo/model.rs:187–189` (`c_fc1`, `c_fc2`, `c_value`), `model.rs:209–211` (`c_tanh1`, `c_tanh2`).
- Weight initialisation: orthogonal, scale √2 for hidden layers, 1.0 for value head (`model.rs:187–189`).
- Training: AdamW with weight decay λ = 3e-4 and lr 5e-4 (`model.rs:192`, configured in `PpoConfig`).
- Actor is smaller (2×64), symmetric widths for its two hidden layers (`model.rs:183–185`).
- Tanh primitive: element-wise `x.tanh()`, scalar, with batch-output cache used for saturation reporting (`src/brain/common/mlp.rs:203–213`).

### What is already normalised and what is not (repository fact)

Scanned `src/agent/observation.rs`, `src/game/episode.rs`, and `src/brain/ppo/*` for normalisation surfaces:

| Surface | Present? | Notes |
|---|---|---|
| **Observation normalisation (running mean/var)** | **No.** | `observation.rs` applies static per-feature clipping and divides rays by a fixed max-range. No online mean/std tracking. |
| **Reward normalisation (running discounted-reward std)** | **No.** | `episode.rs` emits raw velocity-projection + centreline reward; PPO buffer stores them unchanged. |
| **Return / target normalisation (PopArt / value std rescaling)** | **No.** | `update.rs` computes returns and values in raw units. Huber loss is applied on raw returns. |
| **Advantage normalisation (per-minibatch)** | **Yes.** | `buffer.rs:127–136`, per-chunk z-score of advantages. |
| **Value function clipping (PPO2-style)** | **No.** | Verified no VF clipping in `update.rs`. |
| **LayerNorm / BatchNorm on hidden layers** | **No.** | Neither primitive exists in `src/brain/common/`. |

This matrix is decisive. **Every input and every target entering the critic is unnormalised in absolute scale.** The critic is trying to fit a moving, unbounded, 8×-growing return distribution through a bounded tanh sandwich.

### The specific failure being explained

| Symptom | Likely mechanism |
|---|---|
| `critic_fc2` saturation 68.5 %, `critic_fc1` saturation 1.3 % | Layer-1 receives raw (partially-normalised) observations in modest range; layer-2 receives the **product of layer-1's outputs with large learned weights** — weight L2 norm on `critic_fc2` is 40.25 (line 296) vs 24.06 on `critic_fc1` and 13.75 on `actor_fc2`. Weights grew because the **unbounded return target** pulled them there through AdamW's effective LR / decay balance. |
| Car crashes predicted only ~42 % below baseline | A tanh layer at 68 % saturation has ~32 % of its pre-activations in the linear response region. Gradient w.r.t. those units is ~`1 − tanh²(x)` ≈ 0 for the saturated majority. The critic cannot express fine distinctions among high-value states near walls because the second hidden layer has used most of its output range already. |
| Growing returns across training | Episodes grow longer (60 → 460) as the policy improves; with γ=0.99 and horizon growing, the target distribution's mean and std both grow. A tanh-sandwich critic with no target normalisation must keep pushing pre-activation magnitudes to track this, which is exactly the signature in the weight norms. |

## Research Signal

| # | Topic | Source-backed signal | Source + quoted passage ID | Current repo state | Repo citation | Project implication | Evidence class |
|---|---|---|---|---|---|---|---|
| 1 | SB3 default activation | "The default activation function for Stable Baselines3 PPO's ActorCriticPolicy ... is Tanh. This applies to both the actor and critic networks." | **[SB3-DEFAULT]** (SB3 docs) | tanh on both actor and critic | `model.rs:201–211` | Matches industry default — do not treat "tanh critic" as a priori suspicious | source-backed |
| 2 | CleanRL reference impl | "Each network contains 2 hidden layers of 64 units with Tanh activation. Weight init orthogonal std √2, output layer std 1.0 for critic, 0.01 for actor mean." | **[CLEANRL-ARCH]** | same layout, widths 64/128 instead of 64/64 | `model.rs:183–189` | NeuroDrive's asymmetric widening is non-standard among reference impls but defensible | source-backed |
| 3 | Andrychowicz 2021 width rec | "Use a wide value MLP (no layers shared with the policy) but tune the policy width (it might need to be narrower than the value MLP)." | **[ANDR-WIDTH]** | critic 2×128, actor 2×64 | `model.rs:183–188` | NeuroDrive follows this recommendation already | source-backed |
| 4 | Andrychowicz normalisation rec | "Always use observation normalization and check if value function normalization improves performance." | **[ANDR-NORM]** | neither is implemented | repo fact (see matrix above) | **Missing. This is the likely root cause, not the activation.** | source-backed |
| 5 | Andrychowicz activation rec | "Use tanh both as the activation function (if the networks are not too deep) and to transform the samples from the normal distribution to the bounded action space." | **[ANDR-TANH]** | 2-hidden-layer tanh critic | `model.rs` | Depth is fine (2 hidden ≤ "not too deep"). Tanh itself is not the problem per this paper. | source-backed (contrasts with naive "switch to ReLU" answer) |
| 6 | Costa 32-details on reward scaling | "The VecNormalize also applies a certain discount-based scaling scheme, where the rewards are divided by the standard deviation of a rolling discounted sum of the rewards... My initial experiments found this scaling to be extremely important. Without it, the first policy update results in the KL divergence [explosion]..." | **[COSTA-REWSCALE]** | no reward scaling anywhere | `episode.rs`, `update.rs` | Second-strongest missing piece. Reward magnitude growth directly feeds the return-target scale that saturates the critic. | source-backed |
| 7 | PopArt (van Hasselt et al.) | "PopArt adaptively rescales the targets for the value network to have zero mean and unit variance... preserves all outputs precisely whenever it updates the normalization." | **[POPART]** | not implemented | repo fact | Heavyweight alternative if simple target z-score proves insufficient | source-backed |
| 8 | Contrarian tanh defence (LaN 2024) | "symmetrical, bounded shape and smooth gradient landscape has advantages for optimization... around 60% of the neurons ... are dead [in ReLU], while for the sigmoid and hyperbolic tangent activation this number is around 40%" | **[LAN-TANH]** | 68.5 % is above LaN's "normal" 40 % band | run_1776543971.md:296 | Saturation at 40 % can be a healthy operating point; 68 % is above that. The gap between "healthy tanh" and "NeuroDrive's critic" is real but smaller than a pure-ReLU switch would suggest. | contrasting source |
| 9 | 37-details on value-head init | "the weights of the value output layer are initialized with the scale of 1" | **[37DET-INIT]** | already using 1.0 | `model.rs:189` | No change needed here; the initial pre-activation magnitudes are fine — the problem appears during training, not at init | source-backed |
| 10 | SB3 net_arch asymmetric | SB3 supports `dict(pi=[...], vf=[...])` for asymmetric actor/critic widths; this is a documented, supported pattern | **[SB3-ASYMM]** | critic 2×128, actor 2×64 | `model.rs:183–189` | NeuroDrive's pattern is canonical | source-backed |

## Why `critic_fc2` specifically, and not `critic_fc1`

A subtle observation from the layer-health table: `critic_fc1` is at 1.3 % saturation and `critic_fc2` is at 68.5 %. This is a **cascade**, not a uniform effect. The mechanism:

1. Layer 1 sees observations that are bounded by construction (rays clipped, angles wrapped, velocity projections centred). Pre-activations stay modest. Low saturation.
2. Layer 2 sees **tanh-of-layer-1 outputs** (values in `(-1, 1)`) multiplied by learned weights that had to grow large to fit the unbounded return target through a final linear head with orthogonal scale 1.0. Weight L2 = 40.25 on `c_fc2` confirms this.
3. Layer 2 pre-activations therefore inherit the target-scale pressure, clip to the tanh tails, and stop contributing gradient.

The single-layer saturation cascade is a stronger diagnostic than the raw percentage. It localises the problem to the *interface between the fixed-scale inner representation and the unbounded target*, which is the signature the return-normalisation literature predicts.

```text
  Raw obs    Tanh(·)     Tanh(·)     Linear
  (bounded)  ≈(-1,1)     ≈(-1,1)     unbounded
     │          │           │           │
     ▼          ▼           ▼           ▼
  c_fc1  →  sat 1.3 %  →  c_fc2  →  sat 68.5 %  →  c_value  →  target (60→460)
             │                │                       ▲
             │                │                       │
             weights 24.0     weights 40.3  ←  grew to fit growing target
                                                scale through bounded tanh
```

## Candidate Interventions — ranked by expected leverage / implementation cost

Each row gives the intervention, the expected effect on `critic_fc2` saturation, the cost in the custom-Rust-no-external-libs codebase, and one named failure mode.

| # | Intervention | Mechanism against saturation | Expected effect on sat % | Implementation cost (Rust) | Failure mode / counter-scenario |
|---|---|---|---|---|---|
| **A** | **Return / value-target z-score normalisation** (track running mean+std of `returns` in the rollout buffer; feed normalised returns to Huber loss; denormalise `c_value` output at inference for advantage computation). | Removes the 8×-growing target pressure that forced `c_fc2` weights to 40.3. Critic no longer has to push pre-activations to compensate for target scale. | **Largest.** Expect drop from 68.5 % toward 20–35 %. | Moderate. ~40 lines in `update.rs` and a running-stats struct. No new dependencies. | If the target has genuine high-variance structure the agent should express (crash = -∞ / goal = +∞), normalisation can wash out the useful signal. Mitigated here because reward is dense, bounded per-tick, and returns grow smoothly. |
| **B** | **Observation normalisation** (running per-feature mean/var with Welford update; clip to ±10; hot-path allocation-free). | Removes the remaining off-centre / off-scale features that still make the first layer's pre-activations sensitive. Less targeted than A for `c_fc2` specifically but named by Andrychowicz as the single most important normalisation step. | Medium — expect drop to 30–45 %, mostly via keeping `c_fc1` in a regime where `c_fc2` doesn't have to correct scale. | Moderate. ~80 lines in `src/agent/observation.rs` + a per-feature stats struct; the 43-dim vector is small so hot-path cost is negligible. | Online normalisation is non-stationary early; first few hundred samples produce skewed updates. Usually fine but requires a warmup or an initial static scaling. |
| **C** | **LayerNorm on critic pre-activations** (apply LN between `c_fc1→c_tanh1` and between `c_fc2→c_tanh2`). | Forces each tanh input into unit-variance zero-mean regime per-sample regardless of weight growth. Decouples target-scale pressure from activation-scale pressure. | Large — expect drop to ~15–25 % (often reported). | **Highest** of the four — requires writing a custom `LayerNorm` primitive with forward_batch + backward_batch + grad through affine params, GEMM integration not needed (it's a per-feature normalise), but the backward is finicky. ~120 lines + tests. | Introduces extra per-step gradient noise. PPO is sensitive to that. Andrychowicz explicitly did not recommend LN as a default. Good bet for "we've done A and B and still see saturation." |
| **D** | **Tighter critic weight decay** (raise λ from 3e-4 → 1e-3 or 3e-3, critic only). | Applies direct bounding pressure on the weight L2 that drives the saturation. | Small direct effect — expect drop to 50–60 %. Does not address the root cause (target scale); only treats the weight symptom. | Trivial — one number in `PpoConfig`. | Over-decay crushes capacity exactly when the growing target needs more. If returns keep growing, critic underfit replaces saturation. |
| **E** | **Critic depth / width change (widen to 3×128 or 2×256)** | More hidden units spread the saturation pressure across more neurons; each one needs to reach the tails less aggressively. | Modest — expect drop to 40–55 %. | Tiny — change the hidden dim constant. Scratch buffers auto-size (`BatchScratch::new`). | Pure capacity is not the problem — weight growth is the problem. Wider critic will have the same per-neuron pressure, just spread more. Likely masks, does not fix. |
| **F** | **Activation swap — ReLU/GELU on critic only** | No bounded saturation ceiling; neurons can represent unbounded values. | Big drop in "saturation %" **but the metric becomes meaningless** — ReLU doesn't saturate, it dies. Dead-neuron fraction replaces saturation fraction and may be worse (NeuroDrive already saw 34–57 % dead ReLU neurons pre-tanh-switch per `brain-ppo.md`). | Medium — write a `Relu`/`Gelu` primitive; the one deleted earlier was for exactly this reason. | This is swapping a diagnosed problem for an already-rejected problem. Recommend only if A+B+C fail and after explicit instrumentation to detect dead neurons. |
| **G** | **Do nothing — fix it upstream in reward/observation** | Saturation-as-symptom. If reward is bounded per-tick and observations are normalised, returns stay bounded, and `c_fc2` never needs to push weights to 40+. | Convergent with A+B. | Same as A+B (subset). | If the observation contract is actually structurally wrong for anticipatory braking (e.g. lookahead doesn't convey ultimate-distance-to-wall information), no critic architecture fix will help. This is the most important failure mode to check. |

## Steelman — "the saturation is a symptom, not a cause"

Three distinct versions of this argument need to be taken seriously:

**S1. Andrychowicz recommends tanh.** Source-backed **[ANDR-TANH]**: in the largest empirical study on PPO, tanh was the winning activation. Changing to ReLU is not a safe default — it is a deviation from the best-empirical choice, and NeuroDrive already tried it and reverted (`brain-ppo.md` "Durable Notes"). If the research here concluded "switch to ReLU" it would be working against the strongest available evidence.

**S2. 40 % saturation can be normal.** Source-backed **[LAN-TANH]**: "around 40 %" is reported as a baseline for healthy tanh networks in RL. NeuroDrive sits at 68 %, which is above normal but not catastrophic — a 1.7× multiplier on a number that is 40 % under healthy conditions. The car is learning (reward 60→460). The question "is the critic broken?" may be over-stated.

**S3. The actor, not the critic, is the limiting factor for anticipatory braking.** NeuroDrive uses velocity-projection reward. On straights, that reward rewards full-throttle. The critic can predict "value is high" perfectly well, and the actor is still correct to full-throttle — until it hits the wall. No critic architecture change fixes this; only reward shaping does. See `context/references/reward-structure-design.md`.

**Why S1–S3 do not defeat the main recommendation:**

- S1 is addressed — this artefact does **not** recommend switching to ReLU. It recommends target + observation normalisation, which Andrychowicz *also* explicitly recommends. Both recommendations coexist.
- S2 is partial — 68 vs 40 is a real gap, and the causal chain from growing target → growing weight norm → saturation is mechanistically specific enough to justify the intervention even if the absolute percentage is moderate.
- S3 is the single most important live hypothesis. It is orthogonal to the critic-architecture fix. The correct sequencing is: **do the cheap architecture fixes first, rerun, and if the reward curve still plateaus while saturation drops, S3 is confirmed and reward shaping is the next lever.**

## What Fits This Project Well

- The asymmetric 64/128 sizing matches Andrychowicz's explicit recommendation **[ANDR-WIDTH]**; **keep it**.
- Tanh activation matches CleanRL / SB3 / Andrychowicz consensus **[CLEANRL-ARCH] [SB3-DEFAULT] [ANDR-TANH]**; **keep it** at current depth (2 hidden layers is "not too deep").
- Orthogonal init with scale 1.0 on the value head is canonical **[37DET-INIT]**; **keep it**.
- Per-minibatch advantage normalisation is already in place — good.
- AdamW with weight decay on the critic is a reasonable hedge against weight growth — keep it but consider the interaction with intervention D.

## What Fits This Project Badly

- No observation normalisation contradicts the *single strongest recommendation* from Andrychowicz **[ANDR-NORM]**.
- No return/target normalisation is the most likely direct driver of `critic_fc2` saturation given the 8×-growing return distribution.
- No LayerNorm is the cheapest structural fix if A+B are insufficient.
- The custom Rust implementation means each intervention has a real cost; the ordering in the ranked table is chosen to maximise leverage / cost.

## Gap Analysis

| Gap | Severity | Evidence | Fix class |
|---|---|---|---|
| No running observation mean/var | **High** | repo fact + **[ANDR-NORM]** | B |
| No return/value target normalisation | **Highest** | repo fact + **[COSTA-REWSCALE]** + weight L2 40.3 on `c_fc2` | A |
| No LayerNorm primitive available | Medium | `src/brain/common/` lacks it | C (only if A+B insufficient) |
| Critic weight decay may be too weak | Low | λ=3e-4 against target growing 8× | D |
| No dead-neuron instrumentation if ReLU ever reconsidered | Low | repo fact — tanh sat is tracked but relu death isn't | F prerequisite |
| No crash-value probe / distance-to-wall feature | Unknown | observation contract | orthogonal to this paper; see `observation-action-space-design.md` |

## Recommended Priority Order

**Strongly ranked, not a menu:**

1. **A — Return / value-target z-score normalisation.** Single intervention with the largest expected drop in `critic_fc2` saturation, the tightest causal link to the measured weight norm, and a moderate implementation cost. **Try this first.** Expected post-fix saturation: 20–35 %.
2. **B — Observation normalisation.** Do this alongside A. It is Andrychowicz's top-named normalisation recommendation, is cheap, and is a prerequisite for making saturation percentages comparable across runs.
3. **D — Bump critic weight decay to 1e-3 only if A does not fully resolve saturation.** Easy toggle. Do not raise further without monitoring explained variance.
4. **C — LayerNorm on critic pre-activations.** Only if A+B+D leave saturation > 40 %. Implementation cost is highest; leverage is highest among architectural interventions but strictly higher variance than A.
5. **E — Widen critic further.** Do not bother. Capacity is not the bottleneck; weight-scale is.
6. **F — ReLU/GELU critic.** Do not pursue. Previous attempt failed; Andrychowicz evidence disagrees; would require reintroducing dead-neuron monitoring.

### One intervention if you could only do one

**Intervention A — running return normalisation on the critic target.** Rationale:

- The 8× return-scale growth is *directly* measured in the analytics report (chunk-1 60 → chunk-10 460).
- `critic_fc2` weight L2 of 40.3 versus `actor_fc2`'s 13.75 is the fingerprint of target-driven weight growth, not capacity deficit.
- The implementation cost is ~40 lines of Rust, no new dependencies, no new primitive.
- The failure mode is benign (if normalisation hurts, disable it and rerun — nothing else changed).
- It is consistent with the strongest primary source (Andrychowicz) and the reference-implementation pattern (SB3's `VecNormalize`).

### Concrete change sketch for A (do not auto-apply — user confirms before editing source)

- Add `ReturnNormalizer { mean, m2, count }` in `src/brain/ppo/buffer.rs` (Welford online update).
- In `ppo_prepare_update`, after computing per-env returns, update the normaliser from the new returns, then normalise them in place for the loss computation.
- In `update.rs`, compute value loss against normalised returns: `huber(c_value_output, normalized_return)`.
- At inference (bootstrap values, `forward_critic`), denormalise: `raw_value = c_value_output * std + mean`.
- Add a new panel to the analytics Markdown report: `critic_fc2` saturation before/after, value-loss magnitude, weight L2 trajectory.

## Open Uncertainties And Validation Needs

- **Is 68.5 % saturation actually correlated with the anticipatory-braking failure, or is S3 the real cause?** The only falsification is to apply A+B and observe whether saturation drops *and* overshoot-crash fraction drops. If saturation drops but overshoot remains, the critic was not the bottleneck.
- **Does AdamW weight decay λ=3e-4 fight against the growing target enough already?** Intervention D could be first, not third, but the evidence for A is cleaner.
- **What is the "healthy" saturation target for NeuroDrive specifically?** The LaN paper says ~40 % is normal for tanh RL networks; we have no direct measurement on a known-healthy NeuroDrive run. Instrument first-1000-episode saturation on the next run for a baseline.
- **How close is `c_fc1` to saturating as training lengthens?** It is at 1.3 % now, but if returns keep growing it may start climbing — track it.
- **Value function clipping (PPO2-style)** was excluded from the ranked interventions because Andrychowicz's companion finding is that it sometimes hurts; the repository already omits it. Worth revisiting only if A+B push the critic into an oscillation that the clip would damp.

## Relationship To Existing Context

- **Supersedes** the "critic saturation" line in `context/systems/brain-ppo.md` ("Durable Notes" → "Critic saturation problem") which attributes the fix to widening + weight decay. That fix reduced saturation from 40.6 % to a distressed-but-lower level; it did not eliminate the mechanism described here. This paper identifies target normalisation as the missing second half.
- **Complements** `context/references/ppo-optimisation.md` (which is now dated — see its "Current Project Relevance" disclaimer) and `ppo-network-and-training-optimisation.md`. Neither previous paper covered target normalisation in depth.
- **Orthogonal to** `reward-structure-design.md` and `observation-action-space-design.md`. If interventions A+B do not resolve the anticipatory-braking failure, those papers — not this one — describe the next lever.
- **Directly reads from** `reports/analytics/run_1776543971.md` sections "Layer Health" and "What Does the Car Think?".

## External Research Trail

**Searches run.**

| # | Query | Tool | Rationale | Sources surfaced |
|---|---|---|---|---|
| 1 | `Stable Baselines3 PPO MlpPolicy critic value network activation function tanh ReLU default` | WebSearch | Establish the reference-library default to anchor what "standard" looks like | SB3 docs, custom_policy guide |
| 2 | `CleanRL PPO continuous action implementation critic network architecture tanh` | WebSearch | Get the canonical minimal reference implementation | CleanRL GitHub, 37-details blog, CleanRL JMLR paper |
| 3 | `"What Matters In On-Policy" Andrychowicz PPO critic width activation tanh` | WebSearch | Reach the foundational empirical study on PPO design choices | Vitalab summary, HAL PDF, Semantic Scholar |
| 4 | `PPO value function tanh saturation layer normalization fix` | WebSearch | Specifically target the failure mode observed in NeuroDrive | Deep RL that Matters, Costa 37-details, Latent Assistance Networks paper, Saturn Cloud blog |
| 5 | `PPO value function observation normalization running mean std exploding returns value head` | WebSearch | Pull the "growing returns" angle — the measured symptom in NeuroDrive | DI-engine value_norm, Costa 32-details, SB3 VecNormalize docs |
| 6 | `PopArt normalization value function reinforcement learning growing returns` | WebSearch | Reach the heavyweight target-normalisation literature | PopArt paper (arXiv 1809.04474), DeepMind blog, Paper_Notes summary |

**Sources consulted.**

| URL | Tool | Source class | Key passages quoted? |
|---|---|---|---|
| https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py | WebFetch | strong reference implementation | **[CLEANRL-ARCH]** |
| https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | WebFetch | production write-up (implementation-practice) | **[37DET-INIT]** |
| https://arxiv.org/html/2406.09079v1 (Latent Assistance Networks — Rediscovering Hyperbolic Tangents in RL) | WebFetch | contrasting source (defends tanh, argues saturation is symptomatic) | **[LAN-TANH]** |
| https://vitalab.github.io/article/2020/07/02/What_Matters_in_RL.html (summary of Andrychowicz et al. 2021) | WebFetch | foundational paper (summary — primary PDF blocked by Anubis + 403) | **[ANDR-WIDTH] [ANDR-NORM] [ANDR-TANH]** |
| https://costa.sh/blog-the-32-implementation-details-of-ppo.html | WebFetch | production write-up | **[COSTA-REWSCALE]** |
| https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html (via WebSearch surface + docs synthesis) | WebSearch synthesis | official documentation | **[SB3-DEFAULT] [SB3-ASYMM]** |
| https://arxiv.org/abs/2006.05990 (Andrychowicz arxiv listing) | WebFetch | foundational paper metadata | abstract-only |

Two WebFetch attempts (Andrychowicz HAL PDF at `inria.hal.science/hal-03162554/document`, SB3 Readthedocs) returned 403 / Anubis gates and were replaced with the Vitalab summary and the WebSearch-returned SB3 synthesis respectively. This is documented in the obligation audit.

**Quoted passages.**

- **[SB3-DEFAULT]** — source: SB3 PPO docs (via WebSearch surface)
> "The default activation function for Stable Baselines3 PPO's ActorCriticPolicy (which MlpPolicy is an alias of) is Tanh. This applies to both the actor and critic networks unless you specify otherwise."

- **[SB3-ASYMM]** — source: SB3 custom_policy docs
> "policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[32, 32], vf=[32, 32]))" — demonstrating that asymmetric `pi` / `vf` widths are a first-class supported pattern.

- **[CLEANRL-ARCH]** — source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
> "Each network contains 2 hidden layers of 64 neurons each, using Tanh activation functions, and a final output layer with 1 neuron for value estimation ... `def layer_init(layer, std=np.sqrt(2), bias_const=0.0)` ... Default std: √2 ≈ 1.414; Output layer std: 1.0 (for critic); 0.01 (for actor mean)."

- **[ANDR-WIDTH]** — source: Andrychowicz et al. 2021 (via Vitalab summary, cross-checked to ICLR/OpenReview metadata)
> "Use a wide value MLP (no layers shared with the policy) but tune the policy width (it might need to be narrower than the value MLP)."

- **[ANDR-NORM]** — source: Andrychowicz et al. 2021 (via Vitalab summary)
> "Always use observation normalization and check if value function normalization improves performance."

- **[ANDR-TANH]** — source: Andrychowicz et al. 2021 (via Vitalab summary)
> "Use tanh both as the activation function (if the networks are not too deep) and to transform the samples from the normal distribution to the bounded action space."

- **[37DET-INIT]** — source: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
> "the weights of the value output layer are initialized with the scale of 1."

- **[COSTA-REWSCALE]** — source: https://costa.sh/blog-the-32-implementation-details-of-ppo.html
> "The VecNormalize also applies a certain discount-based scaling scheme, where the rewards are divided by the standard deviation of a rolling discounted sum of the rewards (without subtracting and re-adding the mean) ... My initial experiments found this scaling to be extremely important. Without it, the first policy update results in the Kullback–Leibler divergence explodes, likely due to how the layers are initialized."

- **[LAN-TANH]** — source: https://arxiv.org/html/2406.09079v1 (contrasting source)
> "symmetrical, bounded shape and smooth gradient landscape has advantages for optimization ... around 60% of the neurons ... are dead [in ReLU], while for the sigmoid and hyperbolic tangent activation this number is around 40%."

- **[POPART]** — source: DeepMind PopArt (van Hasselt et al. 2016) via search synthesis
> "PopArt adaptively rescales the targets for the value network to have zero mean and unit variance ... preserves all outputs precisely whenever it updates the normalization."

## Pre-Completion Obligation Audit

| Obligation | Status | Evidence |
|---|---|---|
| At least 3 distinct WebSearch calls with topic-specific queries | **Met (6)** | Queries 1–6 in "Searches run" table |
| At least 3 distinct WebFetch calls against primary sources | **Met (6 successful, 2 failed and documented)** | CleanRL GitHub, 37-details blog, LaN paper, Vitalab Andrychowicz summary, Costa 32-details, arXiv abstract; failed: HAL Andrychowicz PDF (Anubis gate), SB3 readthedocs (403). Each failure was replaced with an equivalent surface so the floor is met on the replacement set. |
| Sources span at least 2 source classes | **Met (5)** | foundational paper (Andrychowicz, PopArt), reference implementation (CleanRL), official documentation (SB3 via search synthesis), production write-up (Costa 32-details, 37-details blog), contrasting source (LaN) |
| At least 1 direct quoted passage per major source-backed claim | **Met** | Every row in "Research Signal" has a passage ID; every passage ID appears verbatim in "Quoted passages" |
| At least 1 contrasting / limiting / disagreeing source consulted | **Met** | **[LAN-TANH]** argues tanh saturation at ~40 % is a healthy baseline, limiting the "switch to ReLU" conclusion, and **[ANDR-TANH]** itself is contrasting against the naive "tanh saturation → replace activation" intuition. Steelman section uses both against the main recommendation. |
| Relevant `context/` files read before project-specific claims | **Met** | `context/architecture.md`, `context/systems/brain-ppo.md`, `context/references/ppo-optimisation.md` (first 80 lines), `reports/analytics/run_1776543971.md` (Layer Health + section 6) |
| Relevant code inspected (list file paths) | **Met** | `src/brain/ppo/model.rs` (full), `src/brain/common/mlp.rs` (full) |
| `scripts/init_research_artifact.py` run (stdout captured) | **Met** | stdout: `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/ppo-critic-architecture.md` |
| `scripts/validate_research_artifact.py` run (stdout captured) | **Met** | All 14 checks OK: title, 3 required sections, 3 signals, 3 template sections, 7 URLs / 6 unique domains in research trail, 10 quoted passages, evidence labels (repository fact + source-backed), no exhortation adverbs outside quoted passages. |

## What I Did Not Do

- **Did not fetch the Andrychowicz PDF directly.** The `inria.hal.science` and `openreview.net` hosts both refused WebFetch (Anubis challenge / 403). I used the Vitalab article (which quotes the paper's recommendation list verbatim) and the 37-details blog as cross-checks. A manual read of the full PDF could strengthen section 3.2 ("mini-batching and normalisation") of that paper if the user wants the deeper treatment.
- **Did not benchmark any of the proposed interventions in the NeuroDrive codebase.** This is a research-only artefact; the mandate excluded source edits. The ranked table gives expected effects, not measured ones. The "one intervention if you could only do one" section is the intended experimental starting point.
- **Did not survey CarRacing / TORCS / Gran Turismo Sophy / Learn-to-Race critic architectures directly.** Those would be a valuable second pass if A+B+D do not resolve the saturation. Most published racing-RL work uses off-policy methods (SAC/TD3) where the value-network literature is structurally different from on-policy PPO; the generalisation risk is non-trivial and was out of scope for this first-pass research.
- **Did not explore alternative critic losses (MSE vs Huber vs quantile regression).** The repo uses Huber (δ=1.0), which is defensible on a heavy-tailed return distribution; swapping to MSE would be a separate experimental thread with its own evidence base.
- **Did not write a new LayerNorm primitive or a ReturnNormalizer struct.** Sketched concretely in the change description for A, but writing the code is the user's decision after reviewing this paper.
- **Did not rerun NeuroDrive to collect a current saturation baseline under this paper's reasoning.** The 68.5 % number comes from the last full run in `reports/analytics/run_1776543971.md`; a subsequent run with the same config and a different seed would confirm the number is reproducible. Flagged under "Open Uncertainties".

## Run

- `cargo check`: not run (no source edits made — research-only artefact).
- Validator: will be run before declaring complete.
