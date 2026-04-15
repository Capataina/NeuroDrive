# PPO Network Architecture and Training Optimisation

## Scope / Purpose

- Answer the repository-specific question: **what changes to the actor-critic architecture, optimiser, regularisation, and exploration mechanics will fix the critic's inability to predict crashes and the throttle exploration collapse — without adding crash penalties or changing the reward structure?**
- Cover network sizing, optimiser choice (Adam vs AdamW), weight regularisation, layer normalisation, observation normalisation, and exploration maintenance.
- Ground every recommendation in the current verified codebase state and the observed training diagnostics.
- This paper supersedes the network-sizing and regularisation sections of `context/references/ppo-optimisation.md` (which contains stale data from the pre-tanh, pre-batching era).

## Current Project Relevance

The PPO implementation trains 8 cars in a 2D racing environment at 60 Hz. Cars have learned to go fast (mean speed 268 u/s, 100% throttle) but crash at the first corner every time (100% crash rate, 76% overshoot). The reward structure intentionally rewards aggressive speed (velocity-projection reward) with no crash penalty — the entertainment constraint requires cars to drive dangerously, and the penalty for dying is supposed to be the loss of future discounted reward.

This design only works if the critic can accurately predict that "approaching a corner at 400 u/s" leads to low future value. Currently it cannot:

- **Critic fc2 is 40.6% saturated** (tanh outputs stuck at ±1)
- **Critic value at crash: 26.4 vs average: 51.4** — the gap is too small to generate strong crash-avoidance advantages
- **Critic fc2 weight norm: 19.3** — unbounded growth is driving saturation
- **Throttle exploration collapsed to std=0.07** — the policy can't discover that modulating throttle helps

The actor is not the bottleneck. The actor has near-zero saturation, healthy gradient norms, and its steering exploration is fine (std=0.78). The problem is entirely in the critic's capacity and the throttle exploration channel.

## Current State Snapshot

*Originally verified on 30 March 2026. The recommendations from this paper have been **implemented** — the snapshot below reflects the pre-fix state. See `context/systems/brain-ppo.md` for the current verified architecture.*

**What was implemented from this paper's recommendations:**
1. **Critic widened to 2x128** (was 2x64) — recommendation #1 done.
2. **AdamW with weight decay λ=3e-4 on critic** — recommendation #2 done.
3. **Log-std floor raised to -1.0** (was -2.0) — recommendation #3 done.

### Pre-Fix Architecture (stale — for historical context)

```text
Actor:  obs(43) → Linear(43×64) → tanh → Linear(64×64) → tanh → Linear(64×2) → mean
        + learnable log_std[2] (clamped to [-2.0, 0.5])

Critic: obs(43) → Linear(43×64) → tanh → Linear(64×64) → tanh → Linear(64×1) → value
```

Both networks are **separate** (no shared trunk). Weights stored as flat `Vec<f32>`, row-major. Batched forward/backward for training (128 samples per chunk). Single-sample forward for action selection.

### Pre-Fix Optimiser (stale)

- Adam with β₁=0.9, β₂=0.999, ε=1e-5
- Actor LR: 3e-4, Critic LR: 5e-4
- **No weight decay** — weights grow without bound
- Gradient clipping: L2 norm cap at 0.5 (actor and critic separately)
- Bias correction precomputed via `powi` once per step

### Training Diagnostics (from latest 3724-episode run with 8 cars)

| Metric | Value | Health |
|--------|-------|--------|
| Explained variance | 0.58 | Acceptable, rising |
| Clip fraction | 9.0% | Healthy |
| Approx KL | 0.003 | Safe |
| Actor fc1 saturation | 0.0% | Excellent |
| Actor fc2 saturation | 1.9% | Good |
| **Critic fc2 saturation** | **40.6%** | **Critical** |
| Critic fc2 weight norm | 19.3 | Growing unbounded |
| Critic fc2 grad norm | 0.14 | Low (gradients partially blocked by saturation) |
| **Throttle std** | **0.07** | **Collapsed** |
| Steering std | 0.78 | Healthy |
| Value at crash | 26.4 | Should be near 0 |
| Value average | 51.4 | Reasonable |
| Value loss | 10.1 | Elevated |

---

## Research Signal

### 1. Asymmetric Actor-Critic Architecture

> **Source:** Andrychowicz et al. (2021), "What Matters for On-Policy Deep Actor-Critic Methods"; Ota et al. (2021), "Honey, I Shrunk the Actor"

**Finding:** Separate policy and value networks lead to better performance in 4 of 5 continuous control environments tested. The critic typically benefits from larger capacity than the actor. Experiments show that even dramatically asymmetric designs (1-neuron actor, [16,16,1] critic) can perform comparably to symmetric architectures, provided the critic has sufficient modelling capacity.

**Repository implication:** The current symmetric 2×64 architecture gives the critic the same capacity as the actor, but the critic's task is harder — it must predict cumulative discounted future reward across the full state space, while the actor just needs to output 2 numbers. The critic's 40.6% saturation directly confirms it is capacity-starved. **Widening the critic to 2×128 (or even 2×256) while keeping the actor at 2×64 is strongly supported.**

### 2. AdamW vs Adam for RL

> **Source:** Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"; Andrychowicz et al. (2021); "Regularization Matters in Policy Optimization" (OpenReview 2022)

**Finding:** AdamW decouples weight decay from the adaptive learning rate, meaning the regularisation effect is consistent regardless of gradient magnitude. In standard Adam, L2 regularisation is applied through the gradient, which means it interacts with Adam's moment estimates and produces inconsistent decay. This is particularly problematic when weights grow large — exactly the situation in NeuroDrive's critic.

The "Regularization Matters" study found that **conventional regularisation on policy networks can bring large improvements, especially on harder tasks.** However, they also found that **only regularising the policy network is typically best** — regularising the value function was less consistently helpful.

**Repository implication:** The critic fc2 weight norm of 19.3 is the direct cause of the 40.6% saturation. With tanh activation, weights of magnitude ~3+ push inputs into the flat regions (tanh(3) = 0.995). A 64×64 layer with weight norm 19.3 has a mean absolute weight of ~0.038, but the distribution likely has outliers pushing individual neurons into saturation. AdamW with a modest decay coefficient (1e-4 to 1e-3) would continuously pull weights toward zero, preventing the norm from growing into the saturation zone.

**Important caveat:** The literature suggests regularising the policy network more carefully than the value network. For NeuroDrive, the actor weights are healthy (norm ~10, 0-2% saturation), so actor weight decay should be light (1e-4) or zero. **The critic is where weight decay is most urgently needed** — contrary to the general RL finding, because our specific failure mode is critic weight growth causing saturation.

### 3. Layer Normalisation

> **Source:** Ba et al. (2016), "Layer Normalization"; various RL implementations (CleanRL, Stable Baselines3, DI-engine)

**Finding:** Layer normalisation normalises the pre-activation values to have zero mean and unit variance before the activation function. This directly prevents tanh saturation because the inputs to tanh are centred around 0 with bounded magnitude. It is commonly used in actor-critic networks in practice.

**Repository implication:** Layer normalisation before each tanh would eliminate the saturation problem by construction, regardless of weight growth. However, it changes the gradient dynamics and adds implementation complexity (need to implement LayerNorm from scratch with learnable scale/bias parameters and its backward pass). **It is the most powerful anti-saturation tool but has the highest implementation cost for a from-scratch codebase.**

An alternative is **simpler max-norm weight clamping**: after each optimiser step, clamp `weight_l2_norm()` to a maximum (e.g., 15.0). This is trivial to implement and directly caps saturation risk, though it's a cruder tool than layer norm.

### 4. Observation Normalisation

> **Source:** Andrychowicz et al. (2021); Huang et al. (2022), "The 37 Implementation Details of PPO"

**Finding:** Running observation normalisation (subtract running mean, divide by running standard deviation) is "very helpful for performance" according to the large-scale Andrychowicz study. Duan et al. (2016) adopted it as a default, and it has been standard in continuous control PPO implementations since. The normalised observations are typically clipped to [-10, 10] after normalisation.

**Repository implication:** NeuroDrive currently uses **static normalisation** — each observation feature is divided by a hardcoded constant (`speed_norm_max=900`, `lateral_offset_norm_max=75`, etc.) and clamped to [-1, 1]. This is adequate when the normalisation ranges are well-tuned, but it can be suboptimal when feature distributions shift during training (which they do — speed goes from ~100 to ~400 as the policy learns). Running normalisation would adapt automatically. **Medium priority — the static normalisation isn't broken, but running normalisation would be more robust.**

### 5. Exploration Collapse Prevention

> **Source:** Stanford CS224R study "A Critical Study of the Entropy Bonus for Exploration"; DI-engine documentation; Schulman et al. (2017) PPO paper

**Finding:** In continuous action spaces, entropy regularisation alone is often insufficient to prevent exploration collapse. The entropy bonus "merely broadens existing action modes rather than discovering new ones" — it widens policy variance without shifting modal preferences. A minimum standard deviation floor is a common practical solution: implementations like Stable Baselines3 use `log_std_init = 0.0` (σ=1.0 initial) with a learnable log_std that has an implicit floor from the clamp range.

**Repository implication:** The current log_std clamp range is [-2.0, 0.5], giving σ ∈ [0.135, 1.649]. Throttle has collapsed to σ=0.07, which is **below the floor** — this means the issue isn't the floor, it's that log_std isn't actually being clamped because the observed σ=0.07 implies log_std ≈ -2.66. Wait — that's below the -2.0 clamp. Let me verify.

*Code inspection:* In `update.rs`, the clamp is applied after the Adam step: `brain.model.a_log_std[j] = brain.model.a_log_std[j].clamp(-2.0, 0.5)`. At -2.0, σ = exp(-2.0) = 0.135. But the analytics report says throttle std is 0.07. The discrepancy: the **reported throttle std** is the standard deviation of the **applied actions** across a batch, not the policy's σ parameter. The policy might output σ=0.135 but the tanh squashing compresses the distribution further, and the 0.5*(tanh+1) remapping halves the effective range. So σ_policy=0.135 maps to roughly σ_action ≈ 0.07 after squashing. The floor IS being hit — the policy has pushed log_std to its minimum allowed value.

**Raising the floor from -2.0 to -1.0 (σ=0.368, mapping to ~0.18 after squashing) would force meaningful throttle exploration.** A more sophisticated approach is per-action entropy scaling — higher entropy coefficient for throttle than steering — but the floor raise is simpler and more predictable.

### 6. Spectral Normalisation

> **Source:** Bjorck et al. (2021), "Towards Deeper Deep Reinforcement Learning with Spectral Normalization"; Gogianu et al. (2021), "Spectral Normalisation for Deep RL"

**Finding:** Constraining the spectral norm (largest singular value) of weight matrices to ≤ 1.0 prevents any layer from amplifying its input by more than 1× in any direction. This bounds gradient magnitudes, prevents weight explosion, and enables stable training with larger networks. For RL specifically, spectral normalisation on the value network improves estimation quality by preventing the critic from making overconfident, extrapolated predictions.

**Repository implication:** Spectral normalisation would solve the critic weight growth problem at the root — by construction, weights cannot grow to produce saturating inputs. However, implementing spectral normalisation from scratch requires computing the dominant singular value via power iteration, which adds complexity. **High impact but high implementation cost for a from-scratch codebase. AdamW weight decay achieves a similar effect with much less code.**

---

## What Fits This Project Well

| Change | Impact on critic saturation | Impact on exploration | Implementation cost | Risk to learning |
|--------|---------------------------|----------------------|--------------------|-----------------:|
| **Widen critic to 2×128** | High — doubles capacity | None | Small | Low |
| **AdamW on critic** | High — prevents weight growth | None | Small | Low |
| **Raise log_std floor to -1.0** | None | High — prevents throttle collapse | Trivial | Low |
| **Max-norm weight clamping** | Medium — caps weight growth | None | Trivial | Low |
| **Layer normalisation** | Eliminates saturation by construction | None | Medium | Low |
| **Running observation normalisation** | Indirect (better gradients) | Indirect | Medium | Low |

## What Fits This Project Badly

| Change | Why it doesn't fit |
|--------|-------------------|
| Shared actor-critic trunk | The actor works fine. Sharing would risk corrupting healthy actor gradients with the critic's saturation pathology. |
| Dropout | Harmful in RL — creates inconsistent value estimates between training and inference. |
| 3+ hidden layers | Diminishing returns at this problem scale (43-dim input, 2-dim output). Vanishing gradients through multiple tanh layers. |
| Spectral normalisation | Correct solution but too complex to implement from scratch. AdamW achieves 80% of the benefit at 10% of the implementation cost. |
| Curiosity/intrinsic reward | Adds a second reward signal, complicates the reward landscape, and risks incentivising novel-but-useless states. |

## Gap Analysis

| Gap | Severity | Current state | Recommended fix |
|-----|----------|--------------|-----------------|
| Critic capacity | **Critical** | 2×64, 40.6% saturated on fc2 | Widen to 2×128 |
| Critic weight regularisation | **Critical** | No weight decay, norm 19.3 and growing | AdamW with λ=3e-4 on critic |
| Throttle exploration floor | **High** | log_std floor at -2.0 (σ=0.135), effectively collapsed to minimum | Raise to -1.0 (σ=0.368) |
| Actor weight regularisation | Low | Weights healthy (norm ~10, 0-2% saturation) | Optional light decay λ=1e-4 |
| Observation normalisation | Medium | Static clipping works but doesn't adapt | Running normalisation (future improvement) |
| Layer normalisation | Deferred | Not implemented | Consider if AdamW insufficient |

## Recommended Priority Order

### 1. Widen critic to 2×128 — keep actor at 2×64

**Why now:** The critic is the bottleneck. 40.6% saturation means nearly half the neurons contribute no useful gradient. Doubling the hidden dimension gives the critic 4× the parameter count (64²=4096 → 128²=16384 for fc2 alone), which dramatically increases representational capacity. The actor's task is simpler (output 2 numbers) and it's not capacity-starved.

**Implementation:** Change `ActorCritic::new` to accept separate `actor_hidden_dim` and `critic_hidden_dim`. Update scratch buffer allocation for the asymmetric sizes. The actor forward path stays at 64; the critic forward path uses 128. The batched backward paths already handle arbitrary dimensions.

**Risk:** The critic forward pass during action selection (`ppo_act_all_cars_system`) runs once per car per tick (8 calls at 60 Hz). At 128 hidden, each forward pass is ~4× more compute than at 64 — but the profiling data shows action selection is only 1.7ms total for 8 cars, so even a 4× increase would add ~5ms. Manageable within the frame budget.

### 2. Switch critic optimiser to AdamW with λ=3e-4

**Why now:** Weight norm 19.3 and climbing is the proximate cause of the saturation. AdamW with decoupled weight decay will continuously pull weights toward zero, preventing them from reaching the saturation zone. The implementation change is small — add a `weight_decay` parameter to `AdamOptimizer` and subtract `lr * λ * weight` after each step.

**Implementation:** Add `weight_decay: f32` to `AdamOptimizer`. In `step()`, after the Adam update, apply `weight -= lr * weight_decay * weight`. Keep the actor's weight decay at 0 (or very light, 1e-4) since the actor is healthy.

**Risk:** Too much decay could prevent the critic from learning large-magnitude features that are legitimately needed. Start with λ=3e-4 (the same order as the learning rate) and monitor weight norms and explained variance.

### 3. Raise log_std lower bound from -2.0 to -1.0

**Why now:** The throttle std has collapsed to its minimum allowed value. At σ=0.135 (after squashing ≈ 0.07), the policy samples throttle values almost exclusively in [0.85, 1.0]. It can never discover that throttle < 0.5 before corners is beneficial. Raising the floor to -1.0 (σ=0.368, after squashing ≈ 0.18) forces the policy to explore throttle values down to ~0.5, which is enough to discover cornering.

**Implementation:** One-line change in `update.rs`: change `.clamp(-2.0, 0.5)` to `.clamp(-1.0, 0.5)`.

**Risk:** Forced exploration noise may slow convergence on straightaways where full throttle is correct. This is acceptable — the policy still converges to mean ≈ 1.0 on straights, but the nonzero std means it occasionally tries lower throttle, which is exactly what we need for corner discovery.

### 4. Optional: max-norm weight clamping as a safety net

**Why later:** If AdamW is insufficient (e.g., the decay rate is too low or the weights grow in early training before decay kicks in), a hard clamp on per-layer weight norms provides a safety net. After each optimiser step, if `layer.weight_l2_norm() > MAX_NORM`, scale all weights by `MAX_NORM / current_norm`. Set MAX_NORM to ~15.0 (below the current problematic 19.3).

**Implementation:** 5 lines of code after the Adam step, per layer.

### 5. Future: running observation normalisation

**Why later:** The static normalisation isn't broken — it just doesn't adapt. Running normalisation would help if speed distributions shift significantly (which they do, from ~100 to ~400). But this is a second-order improvement compared to fixing the critic's capacity and the exploration collapse. Implement after the critic fix proves effective.

### 6. Future: layer normalisation on critic

**Why later:** If AdamW + wider network still shows saturation, layer normalisation eliminates it by construction. But it requires implementing LayerNorm from scratch (forward: normalise pre-activation to zero mean / unit variance, scale by learnable γ, shift by learnable β; backward: chain rule through the normalisation). Defer unless needed.

## Open Uncertainties and Validation Needs

1. **Will 128 hidden be enough for the critic?** The saturation might reappear at 128 if weight growth is not controlled. AdamW is the primary defence, but monitoring fc2 saturation and weight norms in the first 1000 episodes after the change is essential.

2. **What AdamW decay rate is right?** λ=3e-4 is a starting guess based on the learning rate scale. If weight norms don't decrease, increase to 1e-3. If explained variance drops (critic is being regularised too aggressively), decrease to 1e-4.

3. **Will the log_std floor interfere with learning on straights?** The policy should still converge to mean ≈ 1.0 throttle on straights, but with σ=0.37 it will occasionally sample 0.6 or lower. If this significantly hurts straight-line speed learning, the floor can be lowered to -1.5 (σ=0.22).

4. **Will the critic's improved predictions actually change the policy?** The theory is sound — better crash predictions → larger negative advantages for crash-causing actions → policy learns to avoid them. But this is an empirical question. If the critic learns to predict crashes but the policy still doesn't modulate throttle, the issue may be in the advantage estimation or the gradient signal from the PPO objective.

## Relationship To Existing Context

- **Supersedes (partially):** `context/references/ppo-optimisation.md` sections on network sizing and regularisation. That paper was written before the tanh switch, before batching, before 8 cars, and before the critic saturation was diagnosed. Its hyperparameter tables are stale.
- **Complements:** `context/notes/reward-and-entertainment.md` — this paper's recommendations operate within the entertainment constraint (no crash penalty, no reward changes).
- **Read alongside:** `context/plans/ppo-optimisation.md` — remaining Phase 2–3 items (observation normalisation, LR annealing) remain valid future work compatible with these recommendations.

---

*Sources:*
- [Andrychowicz et al. (2021), "What Matters for On-Policy Deep Actor-Critic Methods"](https://openreview.net/pdf?id=nIAxjsniDzg)
- [Ota et al. (2021), "Honey, I Shrunk the Actor"](https://arxiv.org/pdf/2102.11893)
- [Huang et al. (2022), "The 37 Implementation Details of PPO"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization" (AdamW)](https://arxiv.org/abs/1711.05101)
- ["Regularization Matters in Policy Optimization"](https://openreview.net/forum?id=yr1mzrH3IC)
- [Bjorck et al. (2021), "Towards Deeper Deep RL with Spectral Normalization"](https://proceedings.neurips.cc/paper/2021/file/4588e674d3f0faf985047d4c3f13ed0d-Paper.pdf)
- [Stanford CS224R, "A Critical Study of the Entropy Bonus for Exploration"](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_report%20(4)12.pdf)
- [Stable Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable Baselines3 custom policy networks](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
