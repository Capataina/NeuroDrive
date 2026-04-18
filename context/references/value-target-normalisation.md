# Value-Target Normalisation for NeuroDrive's PPO Critic

## Scope / Purpose

- Answer the repository-specific question: **what target-scaling or return-normalisation technique will most directly reduce NeuroDrive's critic fc2 tanh saturation (68.5% in `run_1776543971`), given that this saturation is being driven by growing unbounded value targets rather than by architectural capacity?**
- Cover PopArt, SB3-style running-return reward scaling (`VecNormalize`), per-minibatch advantage normalisation (already implemented), and value-function loss clipping (`clip_range_vf`).
- Produce a Rust-grounded implementation sketch for the recommended technique that fits NeuroDrive's existing `src/brain/common/` primitives — no external ML dependencies.
- Stay narrow: this paper is about **scale management for value regression**. It does not re-open reward-shape design (see `context/references/reward-structure-design.md`), critic capacity (see `context/references/ppo-critic-architecture.md`'s scaffold), or exploration mechanics.

## Current Project Relevance

The critic has a measured pathology in the most recent training run:

| Metric | Value | Source |
|---|---|---|
| Critic fc2 tanh saturation | **68.5%** | `reports/analytics/run_1776543971.md:296` |
| Critic fc2 weight L2 norm | **40.25** | `run_1776543971.md:296` |
| Critic value weight L2 norm | **19.22** | `run_1776543971.md:297` |
| Chunk-1 mean reward | 84.22 (per episode) | `run_1776543971.md:table §2` |
| Chunk-10 mean reward | 563.35 (per episode) | `run_1776543971.md:table §2` |
| Episode length | 1.2 s → 6.4 s over ~1.9k episodes | `run_1776543971.md:table §2` |
| Critic's crash-vs-mean gap | 46.9 vs 80.9 | `run_1776543971.md:194` |

`repository fact` — the returns the critic is regressing to grew ~**6.7×** across the run, and the critic's fc2 tanh hidden units are now saturated two-thirds of the time. The previous intervention (`context/references/ppo-network-and-training-optimisation.md`) raised saturation from its target but did not eliminate the underlying driver — it widened the critic to 2×128 and added AdamW weight decay λ=3e-4 on the critic to cap weight L2, but **did not address target-side scaling**. The fc2 weight L2 has nevertheless climbed to 40.25, and saturation has risen from 40.6% to 68.5%.

`project inference` — the symptom pattern (unbounded targets, growing weight norm *despite* weight decay, hidden-layer saturation increasing with return magnitude) is the textbook signature of a value regression chasing a non-stationary target whose magnitude outpaces both the learning rate and the decoupled decay coefficient. This is exactly the failure mode PopArt was introduced to solve in 2016.

## Current State Snapshot

**Verified architecture** (`src/brain/ppo/model.rs:29–37, 187–189, 229–237`):

```text
Critic: obs (43) → Linear(43×128) → tanh → Linear(128×128) → tanh → Linear(128×1)
                                   c_fc1  c_tanh1           c_fc2  c_tanh2     c_value
                                                            [68.5% sat]        [linear, no tanh]
```

The value head `c_value` is **linear** (no tanh on its output), so the unbounded-target problem does not drive saturation *at the output*. It drives saturation one layer upstream: `c_fc2`'s pre-activation magnitudes must grow to produce the large linear combination `c_value(tanh(fc2(…)))` that matches returns ~460. Because `c_tanh2` is a bounded nonlinearity, the only way for `c_value @ c_tanh2_out + b` to reach a target of 460 is for `c_value`'s weights to be large AND for `c_tanh2_out` to have many coordinates pinned at ±1 (sparse-but-extreme code). `repository fact`: `c_value`'s weight L2 is 19.22; by the L2 bound, 128 hidden units pinned at ±1 give a max predictable value of ~19.22·√128 ≈ 217 — consistent with the observed critic mean around 80.9 but far below episode returns of 460. The critic is **capacity-limited on the upper tail** precisely because the upstream code has collapsed to saturating hidden activations.

**Verified optimisation** (`src/brain/common/optim.rs:22–89`):

- AdamW with decoupled weight decay (`weight_decay` applied post-Adam-step, line 72–74).
- Critic uses `weight_decay = 3e-4` (`src/brain/ppo/model.rs:192`); actor uses 0.0.
- Weight decay **on biases is deliberately zero** (line 77 comment).

**Verified loss** (`src/brain/ppo/update.rs:210–222`):

- Huber value loss on raw `returns` vs raw `values` with `value_huber_delta`.
- No value-function clipping (no `clip_range_vf` analogue).
- **Per-minibatch advantage normalisation** already implemented (`update.rs:136–140` — mean/std of chunk advantages, used only to normalise the advantage weighting the policy loss).

**Verified GAE** (`src/brain/ppo/buffer.rs:130–188`):

- Returns are `return[t] = advantage[t] + value[t]`.
- Advantages are returned un-normalised; normalisation happens per-chunk in `update.rs`.
- Bootstrap value for non-terminal tails comes from `forward_critic()` single-sample call.

`open uncertainty` — the code does not record a running min/max/mean of `returns` across updates, so the exact distribution of targets seen by the critic across this run is not preserved in analytics. We are inferring "returns grew 6.7×" from episode-level reward (`reward/s × life(s)`), not from logged return statistics. A quick instrumentation pass on `PreparedUpdate.returns` would sharpen this.

## Research Signal

| Topic | Source-backed signal | Source citation (URL + passage ID) | Current repository state | Citation (file:line) | Project implication | Evidence class |
|---|---|---|---|---|---|---|
| PopArt mechanism | ART normalises targets to zero mean / unit variance; POP rescales output-layer W, b so previous outputs are preserved under the new scale. | `[DM-ART-POP]` https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/ | NeuroDrive has no running target statistics; critic regresses to raw returns directly. | `update.rs:211` | Value targets are unbounded and non-stationary — textbook PopArt setup. | source-backed |
| PopArt rescaling math | `W' = (W^T · old_σ / new_σ)^T`; `b' = (old_σ · b + old_μ - new_μ) / new_σ`. | `[AL-CODE]` aluscher/torchbeastpopart popart.py and `[OD-CODE]` opendilab/PPOxFamily popart.py | No analogue exists. | — | ~40 lines of Rust around `c_value` layer would add this. | source-backed |
| Running-return reward scaling | SB3 `VecNormalize` keeps a running discounted return `R_t = γR_{t-1} + r_t`, then divides each reward by √Var(R) before the PPO pipeline sees it. Rewards are **not centered**. | `[SB3-CODE]` L273–299 `normalize_reward` | Not present. Rewards feed raw into GAE. | `update.rs:211`, `buffer.rs:80` | Scaling rewards at ingest shrinks return magnitudes at the source, taking saturation pressure off the critic without touching the target contract. | source-backed |
| Advantage normalisation is orthogonal | "PPO normalizes the advantages by subtracting their mean and dividing them by their standard deviation … at the minibatch level." Andrychowicz et al. "find per-minibatch advantage normalization to not affect performance much." | `[ICLR37]` https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | Already implemented. | `update.rs:138–140` | Advantage normalisation helps the **policy** gradient; it does nothing for the critic regression target scale. | source-backed |
| Value clipping is contested | SB3 formula `L^V = max[(V − V_targ)², (clip(V, V_old±ε) − V_targ)²]`. "Engstrom, Ilyas, et al., (2020) find no evidence that the value function loss clipping helps with the performance. Andrychowicz, et al. (2021) suggest value function loss clipping even hurts performance." | `[ICLR37]`, `[SB3-VF]` https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html | Not implemented. | — | Not worth adding. Two independent large empirical studies find it neutral or harmful. | source-backed (contrasting) |
| SB3 `clip_range_vf` depends on reward scaling | "This clipping depends on the reward scaling." | `[SB3-VF]` | Neither present. | — | Adding value clipping without fixing target scale would not help saturation. | source-backed |
| PPO implementation details are load-bearing | "Much of the performance of PPO over TRPO comes from code-level optimization and not the original paper's main selling points." | `[VITA]` https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html | PPO implementation has many of the 9 Engstrom optimisations (orthogonal init, tanh, advantage norm, grad clip, AdamW) but is **missing reward/return scaling**. | — | The one class of implementation detail most relevant to the saturation failure mode is the one not yet present. | source-backed |
| Reward normalisation redundancy argument | "the effect of reward whitening can be emulated by properly learning the value function and adjusting the hyperparameters." | `[LIU]` https://liujch1998.github.io/2023/04/16/ppo-norm.html | n/a | — | A theoretical contrasting source — reward scaling may be absorbable into learning rate and value loss coefficient. In NeuroDrive it is not: the critic *is not learning well*, so the "properly learning the value function" precondition fails. | source-backed (contrasting) |
| Tanh saturation mechanics | "if targets contain these boundary values, the network is forced to output extremely large/small inputs so that tanh gets as close to -1 or 1 as possible. This can cause activations in layers to explode during training." | `[DANS]` https://dans.world/Bounded-output-networks/ | fc2 tanh at 68.5% sat, fc2 weight L2 = 40.25, rising. | `run_1776543971.md:296` | The observed pattern matches — large upstream activations are forcing the hidden tanh into saturation to encode large linear-head outputs. | source-backed |

(Verbatim quoted passages now live inside the External Research Trail section so the validator can attribute them; see "Quoted passages" under that section.)

## Technique Comparison

| # | Technique | What it normalises | Math | Rust impl cost | Expected effect on fc2 saturation | Load-bearing in literature? |
|---|---|---|---|---|---|---|
| 1 | **PopArt** (value-head normalisation) | Value targets (returns). Output-layer weights rescaled to preserve predictions. | Running `μ, σ` updated per batch; `ỹ = (y-μ)/σ`; POP step: `W' = W · σ/σ_new`, `b' = (σ·b + μ − μ_new)/σ_new`. Output returned as `σ·(Wh+b)+μ`. | **~40–60 LoC** in `src/brain/common/mlp.rs` (subclass/wrap `Linear` for the critic output) + ~15 LoC hooks in `update.rs` for the μ/σ update call; re-uses existing `Vec<f32>` weight storage. | **Direct.** Targets seen by the regression become `~N(0, 1)` regardless of return magnitude, so fc2 pre-activations do not need to grow to produce large outputs. Saturation pressure removed at the source. | Yes — foundational NIPS 2016 paper; load-bearing in IMPALA multi-task Atari |
| 2 | **Running-return reward scaling** (SB3 `VecNormalize`-style) | Rewards, divided (not centred) by std of discounted running sum. | `R_t = γR_{t-1} + r_t` per env; `r̃_t = r_t / √(Var(R) + ε)`, clipped to ±10. | **~30 LoC** — one `VecReturnNormaliser` resource on `PpoBrain` holding per-env `R` and a `RunningMeanStd`; divide `EpisodeState.current_tick_reward` before it reaches `ppo_collect_rewards_all_cars_system`. | **Indirect but strong.** Shrinks return magnitudes at the ingress so GAE returns land on an O(1) scale; critic never sees targets of magnitude 460. | Yes — one of Engstrom's 4 ablated optimisations; SB3 default for continuous control |
| 3 | **Value-function loss clipping** (`clip_range_vf`) | Per-sample value prediction update clipped within ε of old value. | `L_V = max((V−V_targ)², (clip(V, V_old±ε)−V_targ)²)` | ~15 LoC in `update.rs:210`; requires storing `V_old` per sample in buffer. | **None expected.** Clips the **update** magnitude, not the target magnitude. Does not reduce the scale the critic must represent. | **Contested** — Engstrom 2020 and Andrychowicz 2021 find neutral-to-harmful; SB3 notes it "depends on reward scaling" |
| 4 | **Per-minibatch advantage normalisation** | Advantages, per chunk. | `Ã = (A − mean_chunk)/std_chunk` on the policy-loss weighting only. | Already implemented (`update.rs:138`). | **Zero.** Acts on policy gradient weighting, not critic target. | Andrychowicz: "not affect performance much" |

### Orthogonality map (why you want #1 or #2, not #3 or #4)

```text
             │  Fixes target         │  Fixes update      │  Fixes policy
             │  magnitude (critic    │  magnitude (value  │  variance (actor
             │  saturation)          │  step size)        │  gradient noise)
─────────────┼───────────────────────┼────────────────────┼──────────────────
 PopArt      │         YES           │        —           │        —
 Reward norm │         YES           │        —           │        —
 VF clip     │         —             │       YES          │        —
 Adv norm    │         —             │        —           │       YES
```

The saturation problem lives in column 1. Advantage normalisation (column 3) is already in place and does nothing for column 1 — a common misattribution in RL debugging.

## What Fits This Project Well

- **PopArt fits structurally.** The critic is small (128→1 head), the weight storage is already flat `Vec<f32>` row-major (`mlp.rs` `Linear.weights`), and the value head is *already* the only layer that needs to know about de-normalisation. The rescaling math is 6 scalar ops per weight after each batch μ/σ update — fully implementable with the existing `src/brain/common/` primitives. The per-env bootstrap call in `forward_critic()` also denormalises cleanly if the PopArt state lives on `c_value`.
- **Reward normalisation fits operationally.** Per-env discounted running return is trivial — each car already has an `EnvInstanceId` and `EpisodeState`. A single `Vec<f32>` of running returns keyed by `env_id` plus one `RunningMeanStd` struct covers it. Integration point is `ppo_collect_rewards_all_cars_system` (before buffer push).
- **Both techniques compose with AdamW weight decay.** They reduce the need for the decay to fight target growth, so decay can be loosened if needed.

## What Fits This Project Badly

- **Value-function loss clipping.** Two caveats: (1) the empirical case against it is strong (Engstrom 2020, Andrychowicz 2021). (2) SB3's own doc says "this clipping depends on the reward scaling" — i.e. you cannot set ε sensibly when returns span 60→460. Adding it before fixing scale would be backwards.
- **Observation normalisation (`VecNormalize` for obs).** Not in scope here. NeuroDrive's 43-dim observation is hand-engineered with physical-unit normalisation already (rays, v_forward/v_lateral, speed_delta), so the motivation for running-obs-stats is weaker than in raw Atari/MuJoCo settings. Flag for later; this paper does not recommend it.
- **Global advantage normalisation** (as opposed to per-minibatch). Present approach is already the empirically-preferred one.

## Gap Analysis

```text
 Current                                   Missing (ordered by expected saturation impact)
 ────────                                  ─────────────────────────────────────────────
 ✓ Per-minibatch adv norm     update.rs    ✗ PopArt on c_value head           HIGH impact
 ✓ Orthogonal init             model.rs    ✗ Running-return reward scaling    HIGH impact
 ✓ AdamW decay on critic       optim.rs    ✗ Running return/target stats      INSTRUMENTATION
 ✓ Tanh (+saturation tracking) mlp.rs      ✗ Value loss clipping              NOT RECOMMENDED
 ✓ Grad clip at L2 = 0.5       update.rs
 ✓ Huber value loss            update.rs
```

## Recommended Priority Order

1. **PopArt on `c_value` (highest priority, highest leverage).** Directly addresses the saturation driver: the critic becomes a regressor over `~N(0,1)` targets. The POP step preserves existing predictions across the scale update so training does not take a step back each time statistics shift. Falsifiable: if after 1k episodes `c_fc2` saturation does not drop below 20% and `c_fc2` weight L2 does not plateau below 15, the problem was not target scale.
2. **Running-return reward scaling (second priority, independent of #1, overlaps in mechanism).** Cheaper to implement and more familiar to the SB3/CleanRL community. Delivers most of PopArt's saturation relief if returns are roughly stationary in shape once divided by magnitude. Less principled than PopArt because it cannot preserve previous predictions when Var(R) changes — the critic is silently regressing to moving targets during the warm-up phase.
3. **Add analytics instrumentation** for `PreparedUpdate.returns` distribution (min/max/mean/std per update) regardless of which fix is adopted. Without this, the next session cannot discriminate between "saturation is falling because targets are normalised" and "saturation is falling for some unrelated reason."
4. **Do not add value-function loss clipping.** Two strong empirical studies find it neutral or harmful; SB3 docs flag it as reward-scale-dependent; it does not address saturation mechanistically.

### Recommendation: PopArt with a minimal Rust sketch

`project inference` — the following is a code sketch that respects the verified file structure. It is not a complete implementation.

**New state on `c_value`-adjacent storage** (cheapest place: fields next to `c_value` on `ActorCritic`, not a subclass):

```rust
// In src/brain/ppo/model.rs alongside c_value
pub struct ValueNorm {
    pub mu: f32,
    pub sigma: f32,
    pub nu: f32,      // second raw moment, running
    pub beta: f32,    // EMA decay for running stats (e.g. 3e-4 to match SB3 RMS defaults)
    pub epsilon: f32, // numerical floor on sigma (e.g. 1e-4)
}
```

**Update step** — call once per `PreparedUpdate` before the first `ppo_process_chunk`, before any backward pass touches `c_value`. Input: the `returns` slice from `PreparedUpdate`.

```rust
// In src/brain/ppo/update.rs, called once in ppo_prepare_update after compute_gae_per_env
fn popart_update(c_value: &mut Linear, vn: &mut ValueNorm, returns: &[f32]) {
    let n = returns.len() as f32;
    if n < 1.0 { return; }
    let batch_mu  = returns.iter().sum::<f32>() / n;
    let batch_nu  = returns.iter().map(|r| r*r).sum::<f32>() / n;

    // ART: EMA running statistics (exactly the aluscher/torchbeastpopart form)
    let new_mu  = (1.0 - vn.beta) * vn.mu + vn.beta * batch_mu;
    let new_nu  = (1.0 - vn.beta) * vn.nu + vn.beta * batch_nu;
    let new_sig = (new_nu - new_mu * new_mu).max(vn.epsilon * vn.epsilon).sqrt();

    // POP: rescale c_value weights and bias so current predictions are preserved
    // c_value is Linear(128 -> 1); weights are [1 * 128] row-major, bias is [1].
    let ratio = vn.sigma / new_sig;
    for w in c_value.weights.iter_mut() { *w *= ratio; }
    c_value.biases[0] = (vn.sigma * c_value.biases[0] + vn.mu - new_mu) / new_sig;

    vn.mu = new_mu;
    vn.nu = new_nu;
    vn.sigma = new_sig;
}
```

**Denormalisation on the inference paths** — `forward_critic` and `forward_critic_batch` return `σ·(raw linear output) + μ`:

```rust
// Wrap the existing return of forward_critic:
let raw = value_out[0];  // existing computation
let denorm = self.value_norm.sigma * raw + self.value_norm.mu;
denorm
```

**Value-loss gradient seed adjustment** — the critic is now regressing to the *normalised* return `ỹ = (ret - μ)/σ`. Modify `update.rs:211`:

```rust
let ret_norm   = (ret - vn.mu) / vn.sigma;
let value_raw  = c_out[s];   // raw linear head output = ỹ prediction
let value_err  = value_raw - ret_norm;
// … unchanged Huber math on value_err; unchanged grad_values[s] seed …
```

Bootstrap values for GAE in `ppo_collect_rewards_all_cars_system` must use the **denormalised** path (return `σ·raw + μ`) so returns stay in reward units when computing advantages. Then PopArt normalises them again internally at the head. This preserves the GAE maths exactly.

### AdamW interaction

`repository fact` — the critic uses AdamW with decoupled decay λ=3e-4 (`src/brain/ppo/model.rs:192`, `optim.rs:72–74`). With PopArt in place, the decay is doing **less** of the "cap weight growth to cap saturation" work because the upstream demand on pre-activation magnitudes shrinks dramatically. `project inference` — after PopArt stabilises, decay could be dropped to 1e-4 or removed entirely, but this is a second-order experiment, not a required change.

## Falsification Section — When None Of This Matters

PopArt and running-return reward scaling both assume the failure mode is **target magnitude**. If any of the following conditions were true, fixing target scale would not lower saturation:

1. **Capacity-shape failure.** If fc2 saturation were driven by the critic needing to represent a non-smooth value function on the existing 128-dim hidden code (e.g. sharp cliffs near corners), normalising the target would not give the network more capacity to curve. Signature: saturation remaining ~40%+ **after** targets become `~N(0,1)`. Remedy: widen critic further, add a third hidden layer, or switch to residual connections.
2. **Gradient pathology.** If saturation were caused by gradient explosion through the Huber → `tanh` path (huge gradients only at rare crashes), target normalisation reduces amplitude but does not change gradient direction. Signature: saturation drops modestly after PopArt but `c_fc2` grad L2 remains order-of-magnitude larger than `c_fc1` grad L2 (`run_1776543971.md:295–296` shows 0.47 vs 0.16 — already 3× heavier on fc1 than fc2; gradient-pathology hypothesis currently looks weak).
3. **Observation distribution shift.** If the 43-dim observation itself drifts non-stationarily as the policy improves (e.g. lookahead features only sample certain curvature values once the car can reach fast corners), the critic faces a second moving target that neither technique normalises. Remedy: running observation normalisation (Andrychowicz recommends).
4. **Reward normalisation redundancy (Liu 2023 contrarian argument).** If the value function were "properly learning" (Liu's precondition), reward whitening could be replaced by LR annealing + value coefficient tuning. In NeuroDrive it is *not* properly learning (fc2 saturation 68.5%, crash-value gap too small to disambiguate danger) — so the precondition fails and the redundancy argument does not apply here. This is a reason the Liu paper does not veto the recommendation.
5. **Entertainment-first reward interaction.** `context/notes/reward-and-entertainment.md` mandates no crash penalty and no survival bonus. PopArt does not change this — it only rescales targets, not reward shape. If a future experiment re-introduces even a small crash penalty, PopArt still works (it is scale-invariant by construction).

## Open Uncertainties And Validation Needs

- Exact `returns` distribution during training is not logged. Add one line to `PreparedUpdate` creation to record mean/std/min/max, then re-verify the "6.7× growth" claim post-implementation.
- Beta (EMA decay) for PopArt's running stats. `aluscher/torchbeastpopart` uses 3e-4 per-step; SB3 `RunningMeanStd` uses exact Welford (not EMA). Recommendation: start with β = 1e-4 per update (not per tick — PPO updates are much rarer than environment steps) and iterate based on whether saturation reduction is stable.
- Whether to apply PopArt only to `c_value` (recommended) or jointly to fc2 (more aggressive, larger impl cost, reduces saturation further but risks wiping useful hidden representations each rescale). Recommend starting at `c_value` only.
- Rust-side numerical safety: AdamW on `c_value` must use the **rescaled** weights after each POP step, which means the Adam moment buffers `m_weights`, `v_weights` for that layer become stale. Standard practice (per PopArt paper, per torchbeastpopart) is to leave moments unchanged — the model output is preserved, and moments retarget within a few updates. Flag this as a known quirk to verify experimentally.

## Relationship To Existing Context

- **Supersedes:** the target-normalisation gap implicitly left open by `context/references/ppo-network-and-training-optimisation.md` (which fixed critic width and weight decay but did not address target scale). That paper marked the prior intervention as "implemented" but its own diagnostic (fc2 saturation should be *reduced* by those fixes) has now regressed to 68.5%.
- **Complements:** `context/references/ppo-critic-architecture.md` — currently only a scaffold — should cite this paper once it is populated.
- **Does not modify:** `context/references/reward-structure-design.md`. Reward **shape** (velocity projection + centreline proximity, no crash penalty) is unchanged by either recommendation. Reward **scale** is only touched by technique #2 (running-return reward scaling), and even there the change is a divisive scalar per tick, not a structural change to the signal.
- **Does not modify:** `context/notes/reward-and-entertainment.md`. The entertainment-first constraint is preserved under both techniques.

## Recommendation for NeuroDrive

Adopt **PopArt on the `c_value` layer** as the primary fix. It is the technique most tightly matched to the observed failure mode (hidden-layer saturation driven by unbounded targets on a linear value head), it is implementable inside the existing `src/brain/common/` primitives without any new dependencies, and its POP step provides a theoretical guarantee (output preservation under scale change) that plain reward scaling does not.

If the full PopArt implementation is not worth the ~60 LoC in this session, adopt **running-return reward scaling** as a near-equivalent quick fix. It addresses the same mechanism (shrink target magnitudes at the source) with lower implementation cost but without the output-preservation guarantee.

Do **not** add `clip_range_vf`-style value-function loss clipping — two independent large-scale ablations (Engstrom 2020, Andrychowicz 2021) find it neutral or harmful, and SB3's own documentation flags that it "depends on reward scaling" that does not yet exist here.

Before either fix ships, add analytics instrumentation for the `PreparedUpdate.returns` distribution (mean/std/min/max per update). Without this, post-fix diagnosis cannot discriminate "targets are now normalised" from "saturation happened to drop for some other reason."

## External Research Trail

**Summary:** 9 WebSearch calls, 12 WebFetch calls across 5 source classes (foundational paper, official documentation, strong reference implementation, peer-reviewed evaluation, contrasting source). Full tables below.

**Key URLs consulted (full list in "Sources consulted" below):**

- [DeepMind PopArt blog](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)
- [ICLR 37-details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [SB3 vec_normalize.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py)
- [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [aluscher/torchbeastpopart popart.py](https://raw.githubusercontent.com/aluscher/torchbeastpopart/master/torchbeast/core/popart.py)
- [opendilab/PPOxFamily popart.py](https://raw.githubusercontent.com/opendilab/PPOxFamily/main/chapter4_reward/popart.py)
- [Liu 2023 "embarrassing redundancy"](https://liujch1998.github.io/2023/04/16/ppo-norm.html) — contrasting
- [Vitalab summary of Engstrom "Implementation Matters"](https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html)
- [dans.world bounded-output networks](https://dans.world/Bounded-output-networks/)
- [NeurIPS 2016 PopArt PDF](https://proceedings.neurips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf) — PDF extraction failed
- [arXiv 1602.07714](https://arxiv.org/pdf/1602.07714) — PDF extraction failed
- [SB3 issue #1165](https://github.com/DLR-RM/stable-baselines3/issues/1165) — running-return design rationale

**Key quoted passage to anchor the primary claim** (full set under "Quoted passages" below):

> "We prevent this from happening by updating the network in the opposite direction whenever we update the statistics, this can be done exactly." — [DeepMind PopArt blog](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)

### Searches run

| # | Query | Tool | Rationale | Sources surfaced |
|---|---|---|---|---|
| 1 | `PopArt "preserving outputs precisely" adaptive rescaling targets Hasselt 2016 paper equations` | WebSearch | Foundational paper for technique #1. | NeurIPS PDF, DeepMind blog, Hado van Hasselt blog, xlnwel notes, DanielTakeshi paper notes |
| 2 | `stable-baselines3 VecNormalize reward normalization running mean std PPO` | WebSearch | Reference implementation for technique #2. | SB3 source code, SB3 docs, Araffin blog, hill-a issue tracker |
| 3 | `PPO value function clipping clip_range_vf stable-baselines3 implementation detail` | WebSearch | Coverage for technique #3 and its mechanics. | SB3 PPO docs across many versions |
| 4 | `PPO reward scaling vs return normalization "engstrom" "what matters" implementation details` | WebSearch | Load-bearing-ness evidence. | ICLR 37-details blog, Engstrom paper, Liu "embarrassing redundancy" blog, Coholich bag-of-tricks |
| 5 | `PPO reward normalization bug off-by-one "discounted returns" criticism running variance` | WebSearch | Contrasting-source hunt on the running-return quirk. | SB3 issue #1165, Liu blog, Coholich blog, Ray RLlib discuss threads |
| 6 | `PopArt implementation CleanRL IMPALA github pytorch value normalization` | WebSearch | Reference implementations beyond the paper. | aluscher/torchbeastpopart, steffenvan/IMPALA-PopArt, opendilab/PPOxFamily, CleanRL |
| 7 | `PopArt equations "W_new" "b_new" "sigma/sigma_new" output layer rescale preserve` | WebSearch | Chase the literal rescaling equations. | xlnwel blog, opendilab PPOxFamily code, arXiv 1602.07714 |
| 8 | `tanh saturation neural network unbounded targets value function regression value head` | WebSearch | Saturation mechanism evidence for the mechanistic argument. | dans.world bounded-output article, ResearchGate saturation measurement paper |
| 9 | `PopArt "value head" "no activation" linear output layer why` | WebSearch | Verify architectural placement recommendation. | (no hits — noted in body) |

### Sources consulted

| URL | Tool | Source class | Quoted in artefact? |
|---|---|---|---|
| https://proceedings.neurips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf | WebFetch | foundational paper (PDF binary — extraction failed) | No (recovered via reference implementations) |
| https://arxiv.org/pdf/1602.07714 | WebFetch | foundational paper (PDF binary — extraction failed) | No (same reason) |
| https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/ | WebFetch | official documentation (author-team blog) | Yes — `[DM-ART-POP]` |
| https://hadovanhasselt.com/2016/08/17/learning-values-across-many-orders-of-magnitude/ | WebFetch | author blog | Partial (content-sparse; no equations) |
| https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | WebFetch | peer-reviewed evaluation (ICLR blog track) | Yes — `[ICLR37]` |
| https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py | WebFetch | strong reference implementation | Yes — `[SB3-CODE]` |
| https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html | WebFetch | official documentation | Yes — `[SB3-VF]` |
| https://github.com/DLR-RM/stable-baselines3/issues/1165 | WebFetch | production write-up / maintainer commentary | Yes (closing context) |
| https://raw.githubusercontent.com/aluscher/torchbeastpopart/master/torchbeast/core/popart.py | WebFetch | strong reference implementation | Yes — `[AL-CODE]` |
| https://raw.githubusercontent.com/opendilab/PPOxFamily/main/chapter4_reward/popart.py | WebFetch | strong reference implementation | Yes — `[OD-CODE]` |
| https://liujch1998.github.io/2023/04/16/ppo-norm.html | WebFetch | **contrasting source** (limits reward-norm recommendation) | Yes — `[LIU]` |
| https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html | WebFetch | peer-reviewed evaluation (Engstrom summary) | Yes — `[VITA]` |

Source classes covered: **foundational paper** (attempted), **official documentation** (DeepMind blog, SB3 docs), **strong reference implementation** (SB3 vec_normalize, torchbeastpopart, PPOxFamily), **peer-reviewed evaluation** (ICLR 37-details, Engstrom summary), **contrasting source** (Liu redundancy argument, Andrychowicz VF-clip negative finding).

### Quoted passages

- **[DM-ART-POP]** — source: [DeepMind PopArt blog](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)
  > "It then uses these statistics to normalise the targets before they are used to update the network's weights."
  > "We prevent this from happening by updating the network in the opposite direction whenever we update the statistics, this can be done exactly."
  > "There are often differences in the reward scales our reinforcement learning agents use to judge success, leading them to focus on tasks where the reward is arbitrarily higher."

- **[AL-CODE]** — source: [aluscher/torchbeastpopart popart.py](https://raw.githubusercontent.com/aluscher/torchbeastpopart/master/torchbeast/core/popart.py)
  > ```python
  > mu = vs.sum((0, 1)) / n
  > nu = torch.sum(vs**2, (0, 1)) / n
  > sigma = torch.sqrt(nu - mu**2)
  > self.mu = (1 - self.beta) * self.mu + self.beta * mu
  > self.weight.data = (self.weight.t() * oldsigma / self.sigma).t()
  > self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma
  > ```

- **[OD-CODE]** — source: [opendilab/PPOxFamily popart.py](https://raw.githubusercontent.com/opendilab/PPOxFamily/main/chapter4_reward/popart.py)
  > "`W' = (W^T * old_σ / new_σ)^T`"
  > "`b' = (old_σ * b + old_μ - new_μ) / new_σ`"
  > "These transformations preserve unnormalized outputs while adapting to changing value distributions, addressing the 'multi-magnitude reward problem' in reinforcement learning."

- **[SB3-CODE]** — source: [SB3 vec_normalize.py L273–299](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py)
  > ```python
  > self.returns = self.returns * self.gamma + reward
  > self.ret_rms.update(self.returns)
  > ...
  > reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon),
  >                  -self.clip_reward, self.clip_reward)
  > ```
  > Rewards are divided by std only (not centered); clipped to ±10.0 by default.

- **[ICLR37]** — source: [ICLR blog track, "The 37 Implementation Details of PPO"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  > "PPO normalizes the advantages by subtracting their mean and dividing them by their standard deviation. In particular, *this normalization happens at the minibatch level instead of the whole batch level!*"
  > "`L^V = max[(V_θ_t - V_targ)², (clip(V_θ_t, V_θ_t-1 - ε, V_θ_t-1 + ε) - V_targ)²]`"
  > "Engstrom, Ilyas, et al., (2020) find no evidence that the value function loss clipping helps with the performance. Andrychowicz, et al. (2021) suggest value function loss clipping even hurts performance."
  > "the rewards are divided by the standard deviation of a rolling discounted sum of the rewards (without subtracting and re-adding the mean)."

- **[SB3-VF]** — source: [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
  > "`clip_range_vf`: Clipping parameter for the value function, it can be a function of the current progress remaining (from 1 to 0)."
  > "This clipping depends on the reward scaling."

- **[LIU]** (contrasting) — source: [Liu 2023 "embarrassing redundancy"](https://liujch1998.github.io/2023/04/16/ppo-norm.html)
  > "For the policy loss, the scaling factor between A_t and Ã_t is wiped out by the advantage whitening trick. For the value loss, this scaling factor can be absorbed into the value loss coefficient α."
  > "the effect of reward whitening can be emulated by properly learning the value function and adjusting the hyperparameters."

- **[VITA]** — source: [Vitalab summary of Engstrom et al. "Implementation Matters"](https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html)
  > "Much of the performance of PPO over TRPO comes from code-level optimization and not the original paper's main selling points."

- **[DANS]** — source: [dans.world bounded-output networks](https://dans.world/Bounded-output-networks/)
  > "tanh(a) ∈ (-1, 1) and never reaches -1 and 1 exactly … the network is forced to output extremely large/small inputs so that tanh gets as close to -1 or 1 as possible. This can cause activations in layers to explode during training."

## Pre-Completion Obligation Audit

| Obligation (from SKILL.md External Research Floor) | Evidence |
|---|---|
| ≥3 distinct WebSearch calls | **9 WebSearch calls**, logged with exact query strings above. |
| ≥3 distinct WebFetch calls against primary sources | **12 WebFetch calls**: SB3 source, SB3 docs, DeepMind blog, Hasselt blog, ICLR blog track, GitHub issue #1165, torchbeastpopart, PPOxFamily popart.py, Liu blog, vitalab Engstrom summary, two NeurIPS/arXiv PDFs (PDFs failed extraction but attempted). |
| ≥2 source classes | **5 classes**: official documentation, strong reference implementation, peer-reviewed evaluation, contrasting source, author blog. |
| ≥1 direct quote per major source-backed claim | Every row in Research Signal has a passage ID in `[SRC-X]` form attached, and every ID has a verbatim quoted passage block. |
| ≥1 contrasting source | Two: `[LIU]` (reward-whitening redundancy) and `[ICLR37]` quoting Engstrom 2020 + Andrychowicz 2021 on value clipping being neutral-to-harmful. |
| `scripts/init_research_artifact.py` run | Yes — stdout: `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/value-target-normalisation.md` (also created an unused folder scaffold which was removed). |
| `scripts/validate_research_artifact.py` run | Pending — run immediately after this artefact is saved. |
| Code inspection: files read | `src/brain/ppo/buffer.rs` (full), `src/brain/ppo/update.rs` (full), `src/brain/ppo/model.rs` (lines 1–250), `src/brain/common/optim.rs` (full), `src/game/episode.rs` (lines 1–200). |
| `context/` files read | `context/architecture.md`, `context/systems/brain-ppo.md`, `context/references/ppo-network-and-training-optimisation.md` (head), `context/references/ppo-critic-architecture.md` (head — scaffold), `reports/analytics/run_1776543971.md` (head + saturation lines). |

## What I Did Not Do

- **Did not successfully extract the NeurIPS/arXiv PDF equations.** Both PDF fetches returned binary streams rather than readable text. I recovered the mathematically-equivalent equations from two independent reference implementations (`aluscher/torchbeastpopart` and `opendilab/PPOxFamily`) and from the DeepMind blog's plain-English description. Reader who wants the paper's exact equation numbers should open the arXiv PDF directly; I cite both reference implementations as the mechanical source.
- **Did not read Andrychowicz 2021 primary PDF.** The PDF fetch failed. I quoted the ICLR 37-details blog's verbatim summary of Andrychowicz's findings on value clipping and advantage normalisation (which itself cites the paper by section). A reader seeking the primary figures should open arXiv 2006.05990 directly.
- **Did not run a training experiment.** This is research, not code. The recommendation is implementable in the code sketch provided; empirical validation belongs in a follow-up plan file under `context/plans/`.
- **Did not extend the artefact into a full `context/plans/popart-implementation.md`.** That is the appropriate next artefact if Caner decides to proceed; this paper stops at the recommendation and the Rust sketch.
- **Did not investigate observation normalisation.** Flagged as out of scope but surfaced as a possible follow-up if falsification condition #3 holds post-PopArt.
- **Did not add running-target instrumentation to the analytics pipeline.** Listed as priority #3 in the recommendation and belongs in an implementation PR, not in this research paper.
