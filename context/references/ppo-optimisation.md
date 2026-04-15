# PPO Optimisation for NeuroDrive

## Scope / Purpose

- Answer the repository-specific question: **what changes to the PPO implementation, hyperparameters, reward shaping, and runtime architecture would most likely enable full track completion?**
- Cover both *learning performance* (can the policy learn to navigate corners?) and *computational performance* (can we get more training throughput per wall-clock second?).
- Ground every recommendation in verified code inspection of the current implementation.
- This paper supersedes the A2C-era recommendations in `context/references/a2c-for-neurodrive.md` for anything PPO-specific, while that paper remains the canonical reference for A2C's role as a baseline validation tool.

## Current Project Relevance

**This reference was written during the pre-tanh, pre-batching, A2C-era implementation.** Many of its hyperparameter tables, gap analyses, and diagnostic snapshots are now stale. Specifically:
- **Activation**: ReLU has been replaced with tanh (eliminating the dead neuron problem).
- **Architecture**: Now asymmetric actor-critic (actor 2x64, critic 2x128) with AdamW weight decay on the critic.
- **Observations**: 43 dimensions (was 23), with velocity decomposition, speed delta, 12-point lookahead, previous actions.
- **Reward**: Velocity projection + centreline proximity, crash penalty 0.0 (was speed-weighted best-only progress + crash penalty -5).
- **Training**: 8 cars vectorised (was 3), amortised PPO updates at 64 samples/tick (was 128), orthogonal init, per-minibatch advantage normalisation with sample shuffling.

The durable research findings (sections on activation functions, GAE, implementation details, reward shaping principles) remain valuable. The "Current State Snapshot" and "Gap Analysis" sections reflect the **pre-overhaul** codebase and should not be trusted for current implementation truth.

See `context/references/ppo-network-and-training-optimisation.md` for the current PPO architecture analysis, and `context/systems/brain-ppo.md` for verified implementation state.

## Current State Snapshot

Verified by direct code inspection of `src/brain/a2c/`, `src/agent/`, `src/game/`, and supporting modules.

### PPO Hyperparameters (verified)

| Parameter | Current value | Source |
|-----------|--------------|--------|
| Hidden dimension | 64 | `model.rs:37` — `ActorCritic::new` |
| Hidden layers (actor) | 2 × 64, ReLU | `model.rs:38-40` |
| Hidden layers (critic) | 2 × 64, ReLU | `model.rs:42-44` |
| Actor LR | 3e-4 | `model.rs:46` |
| Critic LR | 5e-4 | `model.rs:47` |
| Activation | Tanh | `model.rs` — `Tanh::new()` throughout |
| Weight init | Orthogonal (√2 hidden, 0.01× policy head, 1.0× value head) | `mlp.rs` — `orthogonal_init` |
| Policy output init | Orthogonal with 0.01× scale | `model.rs:40` |
| a_log_std init | `[0.0, 0.0]` (σ = 1.0) | `model.rs:55` |
| a_log_std clamp | `[-2.0, 0.5]` (σ ∈ [0.135, 1.649]) | `update.rs:268` |
| γ (discount) | 0.99 | `mod.rs:52` |
| GAE λ | 0.95 | `mod.rs:53` |
| Rollout horizon (max_steps) | 512 | `mod.rs:54` |
| Min update threshold | 128 | `mod.rs:55` |
| PPO epochs | 4 | `mod.rs:56` |
| Clip ε | 0.2 | `mod.rs:57` |
| Samples per tick | 128 | `mod.rs:58` |
| Entropy coefficient | 0.01 | `update.rs:11` |
| Value loss | Huber (δ=1.0) | `update.rs:8` |
| Grad clip (actor) | 0.5 L2 norm | `update.rs:9` |
| Grad clip (critic) | 0.5 L2 norm | `update.rs:10` |
| Adam β₁/β₂ | 0.9 / 0.999 | `optim.rs:36-37` |
| Adam ε | 1e-5 | `optim.rs:38` |
| LR annealing | None — fixed LR | verified: no schedule code exists |
| Observation normalisation | None — static clipping only | `observation.rs:186-215` |
| Reward/return normalisation | None | verified: no normalisation code exists |
| Value function clipping | None | verified: no VF clip in `update.rs` |
| Advantage normalisation | Per-minibatch (per-chunk) with sample shuffling | `buffer.rs:127-136` |
| Minibatch shuffling | Fisher-Yates shuffle per epoch | `update.rs:113` |
| Orthogonal init | Yes — √2/0.01/1.0 scale | `mlp.rs` |

### Environment and Action Space (verified)

| Parameter | Current value | Source |
|-----------|--------------|--------|
| Car rotation_speed | 4.0 rad/s | `car.rs:57` |
| Car thrust | 750.0 | `car.rs:58` |
| Car drag | 0.985 | `car.rs:59` |
| Steering range | `[-1, 1]` via `tanh` squashing | `mod.rs:153-158` |
| Throttle range | `[0, 1]` via `0.5*(tanh+1)` | `mod.rs:158` |
| Action smoothing | Disabled by default | `action.rs:60` |
| Vectorised cars | 3 (default) | `car.rs:34` |
| Progress reward scale | 100.0 (speed-weighted) | `episode.rs:53` |
| Time penalty | -0.005 / tick | `episode.rs:54` |
| Heading-speed penalty scale | 0.02 | `episode.rs:55` |
| Crash penalty | -5.0 | `episode.rs:58` |
| Lap bonus | 100.0 | `episode.rs:59` |
| Timeout | 30 s | `episode.rs:49` |
| Observation dim | 23 (11 rays + 4 scalars + 4×2 lookahead) | `observation.rs:17-18` |

### Observation Feature Layout (verified)

```text
Index   Feature                  Normalisation           Range
 0-10   Ray distances (11)       /375.0, clamp [0,1]     [0, 1]
   11   Speed                    /900.0, clamp [0,1]     [0, 1]
   12   Signed lateral offset    /75.0, clamp [-1,1]     [-1, 1]
   13   Heading error            /π, clamp [-1,1]        [-1, 1]
   14   Angular velocity         /8.0, clamp [-1,1]      [-1, 1]
15-22   Lookahead (4 samples)    heading: /π             [-1, 1]
        heading_delta, curvature curvature: /0.05        [-1, 1]
```

---

## Research Signal

This section synthesises findings from the PPO literature and maps them to this repository's specific situation.

### Critical Implementation Gaps

These are deviations from research-backed best practice that are most likely to explain the current learning failure.

| Detail | Research-backed practice | NeuroDrive current state | Severity | Evidence source |
|--------|------------------------|-------------------------|----------|-----------------|
| **Activation function** | Tanh consistently outperforms ReLU in on-policy continuous control | ReLU with 34–57% dead neurons | **Critical** | Andrychowicz et al. 2020; "37 Details" #13, #26 |
| **Advantage normalisation scope** | Per-minibatch normalisation, not global batch | Global batch-level only, no minibatch shuffle | **High** | "37 Details" #6, #7 |
| **Observation normalisation** | Running mean/variance normalisation | Static fixed-range clipping only | **High** | "37 Details" #28, #29; Engstrom et al. 2020 |
| **Weight initialisation** | Orthogonal with √2 scale (hidden), 0.01 scale (policy head), 1.0 (value head) | Glorot uniform everywhere (no output scaling) | **High** | "37 Details" #2 |
| **LR annealing** | Linear decay to zero over training | Fixed LR throughout | **Moderate** | "37 Details" #4; empirical RL scheduling research |
| **Reward/return scaling** | Divide rewards by running σ of discounted returns | No reward normalisation | **Moderate** | "37 Details" #30, #31 |
| **Adam ε** | 1e-5 (PPO standard) | 1e-8 (PyTorch default) | **Low-Moderate** | "37 Details" #3 |
| **Policy head init scale** | 0.01× to produce near-zero initial mean | Full Glorot scale on `a_mean` layer | **Moderate** | "37 Details" #2; Andrychowicz et al. 2020 |

### The Dead ReLU Problem — Why This Matters Here

The dead neuron rates observed in NeuroDrive reports (34–57%) are not merely cosmetic. They represent a severe capacity loss:

```text
Layer           Dead ReLU %   Effective neurons (of 64)
─────────────────────────────────────────────────────────
actor_fc1       34–42%        37–42
actor_fc2       28–51%        31–46
critic_fc1      39–45%        35–39
critic_fc2      44–57%        28–36
```

With 2×64 hidden layers, losing 35–57% of neurons means the actor's effective capacity may be as low as a 2×31 network — far too small to represent the nonlinear steering policies needed for cornering. This directly explains the "insufficient steering" failure mode: the policy literally does not have enough active parameters to learn distinct steering behaviours for different track segments.

**Why ReLU causes this in on-policy RL (source-backed):**

ReLU neurons die when large gradient updates push weights so that the pre-activation is persistently negative. In supervised learning this is manageable because training data is stationary. In on-policy RL, the data distribution shifts constantly as the policy improves, creating repeated large gradient shocks that kill neurons progressively. Tanh avoids this entirely because it always has non-zero gradient for finite inputs. The Andrychowicz et al. (2020) large-scale study found tanh outperformed ReLU across their continuous-control experiments. The "37 Implementation Details" blog confirms that the reference PPO implementation uses tanh for continuous control.

**Project inference:** Switching from ReLU to tanh is the single highest-impact change for NeuroDrive. It directly addresses the observed capacity starvation and is the most likely unblock for corner-learning.

### Reward Shaping and Corner Behaviour

The current reward structure is reasonable for straight-line driving but has properties that actively work against learning to corner:

**Problem 1 — Progress reward only counts new-best progress:**

```rust
let progress_gain = (progress.fraction - previous_best_progress).max(0.0);
```

This means once the car reaches ~13% progress, it gets *zero* progress reward on every subsequent attempt that doesn't beat the best. Early in training, most episodes crash at the same point, producing a long sequence of zero-progress-reward episodes. The policy receives almost no gradient signal about *how* it crashed differently — only that it crashed. This creates a learning plateau exactly at the first hard corner.

**Problem 2 — No explicit cornering incentive:**

The heading-speed penalty (`-0.02 × |heading_error/π| × |speed/900|`) is small relative to the progress reward scale (140.0) and only penalises misalignment at speed. It doesn't reward *correct* alignment through corners. Research on autonomous racing (e.g., the Nature 2025 study on reward design) shows that explicit speed-steering coupling penalties and centerline-proximity rewards materially improve cornering:

- **Centerline reward:** `1 - (d_c / d_max)²` quadratic decay with distance
- **Speed-steering penalty:** multiplier reduction (0.5–0.9×) when steering angle exceeds thresholds at speed

**Problem 3 — Crash penalty magnitude:**

The crash penalty is -5.0. With `progress_reward_scale = 140.0`, reaching 13% progress gives `0.13 × 140 = 18.2` total progress reward over the episode. The crash penalty is only 27% of this. The ratio shrinks further for longer episodes with higher progress. This means the policy has little incentive to avoid crashing — crashing is almost free relative to the progress signal.

**Project inference:** The reward function should be revised to:
1. Reward per-tick progress, not just new-best progress
2. Add a centerline proximity term
3. Increase the crash penalty or scale it with progress
4. Consider a speed-curvature coupling penalty

### Network Capacity

| | Current | "37 Details" default | Andrychowicz recommendation |
|---|---|---|---|
| Hidden layers | 2 | 2 | 2 |
| Hidden size | 64 | 64 | 64 (sufficient for most tasks) |
| Separate actor/critic | Yes | Yes (continuous) | Yes |

The network architecture (2×64, separate actor/critic) is within the standard range. **The problem is not capacity per se, but effective capacity lost to dead ReLUs.** With tanh, the current 2×64 architecture should be sufficient for this task.

---

## Gap Analysis

> **Note:** Many gaps identified below have been closed. See the status tables in `context/plans/ppo-optimisation.md` and `context/plans/reward-and-spawn-overhaul.md` for current implementation status.

### Priority-ordered gaps between current implementation and research-backed PPO

```text
Priority   Gap                          Impact    Effort   Risk
─────────────────────────────────────────────────────────────────────
 P0        ReLU → Tanh                  Critical  Low      Very low
 P0        Reward: per-tick progress     Critical  Low      Low
 P1        Orthogonal init + output      High      Low      Very low
           head scaling
 P1        Minibatch shuffling +         High      Low      Very low
           per-minibatch adv norm
 P1        Reward: centerline term       High      Low      Low
 P2        Observation running norm      High      Medium   Low
 P2        LR linear annealing           Moderate  Low      Very low
 P2        Adam ε → 1e-5                 Low-Mod   Trivial  None
 P2        Reward: crash penalty scale   Moderate  Low      Low
 P3        Reward/return normalisation   Moderate  Medium   Low
 P3        Value function clipping       Low-Mod   Low      Low
 P3        More vectorised envs          Moderate  Low      Low
           (3 → 8–16)
 P4        Orthogonal init (full)        Low       Medium   Low
```

### Severity assessment

```text
                    ┌─────────────────────────────────────┐
                    │         LEARNING BOTTLENECK          │
                    │                                     │
 Dead ReLUs ────────┤  Policy cannot represent corner     │
 (34-57%)           │  steering. Effective actor capacity  │
                    │  is ~2×35, not 2×64.                │
                    │                                     │
 Reward plateau ────┤  Zero gradient signal once car      │
 (best-only         │  reaches same crash point.          │
  progress)         │  Policy stagnates at 11-15%.        │
                    │                                     │
 No minibatch ──────┤  Advantage signal quality reduced   │
 normalisation      │  by batch-wide averaging.           │
                    └─────────────────────────────────────┘
```

---

## Recommended Priority Order

### P0 — Unblock corner learning (do these first, together)

#### 1. Switch ReLU → Tanh

**What:** Replace `Relu` with a `Tanh` activation in all four hidden layers (actor and critic).

**Why now:** This is the single most impactful change. It immediately fixes the dead neuron problem (tanh cannot die), restores full network capacity, and aligns with the reference PPO implementation for continuous control. The Andrychowicz et al. large-scale study, the "37 Implementation Details" reference implementation, and Stable Baselines3 all use tanh for continuous-control PPO.

**Implementation cost:** Very low — modify `Relu` to a `Tanh` struct with `forward(x) = x.tanh()` and `backward(x, g) = g * (1 - x.tanh()²)`. Replace `Relu::new()` calls in `model.rs`.

**Risk:** Near zero. Tanh is the standard choice. The only concern is vanishing gradients in very deep networks, but 2 layers is not deep.

#### 2. Change progress reward from best-only to per-tick

**What:** Replace `progress_gain = (progress.fraction - previous_best_progress).max(0.0)` with `progress_gain = (progress.fraction - previous_progress_fraction).max(0.0)` — reward any forward movement within the episode, not just exceeding the all-time best.

**Why now:** The current best-only scheme creates a learning plateau at the first hard corner. Once most episodes crash at the same point, the policy gets zero progress signal. Per-tick progress reward provides dense gradient signal throughout every episode, including through corners where the car is making incremental progress before crashing.

**Implementation cost:** Very low — one line change in `episode.rs`.

**Risk:** Low. Per-tick progress is the standard approach in racing RL. The only risk is that the car might learn to circle slowly near spawn, but the time penalty and heading-speed penalty discourage this.

### P1 — Strengthen training signal quality

#### 3. Orthogonal initialisation with output head scaling

**What:** Replace Glorot uniform with orthogonal init (√2 scale for hidden layers). Scale the actor mean output layer by 0.01× (so initial policy outputs are near-zero, producing roughly uniform actions). Scale the critic value output layer by 1.0×.

**Why now:** The "37 Details" documents this as PPO's standard init. The small policy-head scale is important: it makes the initial policy produce nearly uniform random actions, which is better for exploration than Glorot's default scale which can produce biased initial outputs.

**Implementation cost:** Low — add an `orthogonal_init` function and adjust `ActorCritic::new`.

#### 4. Minibatch shuffling and per-minibatch advantage normalisation

**What:** In `ppo_process_chunk`, shuffle sample indices at the start of each epoch. Normalise advantages within each chunk/minibatch rather than globally.

**Why now:** Global advantage normalisation reduces signal quality. If most samples have similar advantages, normalisation amplifies noise. Per-minibatch normalisation is the standard PPO approach and ensures each gradient update sees meaningfully diverse advantage magnitudes.

**Implementation cost:** Low — add a shuffled index array to `PreparedUpdate`, index through it, and compute advantage mean/std per chunk.

#### 5. Centerline proximity reward

**What:** Add a small per-tick reward for staying near the centerline: `centerline_reward = centerline_coef * (1.0 - (distance / max_distance).min(1.0).powi(2))`.

**Why now:** This provides a continuous signal that helps the policy learn track-following behaviour independent of forward progress. It rewards the car for taking racing lines through corners rather than just maximising straight-line speed.

**Implementation cost:** Low — add one term in `episode_loop_system`, add config parameters.

### P2 — Improve optimisation stability

#### 6. Running observation normalisation

**What:** Track running mean and variance of each observation feature across all ticks. Normalise observations as `(obs - running_mean) / (running_std + 1e-8)`, then clip to `[-10, 10]`.

**Why now:** The current static clipping maps features to fixed ranges that may not match the actual distribution seen during training. Running normalisation adapts as the policy explores different parts of the track.

**Implementation cost:** Medium — add a `RunningStats` resource, update it in the observation system, apply normalisation in `build_observation_vector_system`.

#### 7. Linear LR annealing

**What:** Linearly decay both actor and critic learning rates from their initial values to zero over a configured total number of updates or timesteps.

**Why now:** LR annealing is one of the "37 Details" and prevents policy oscillation in later training. Without it, the policy can overshoot once it's near a good solution.

**Implementation cost:** Low — add a `total_timesteps` config, track progress, scale LR each update.

#### 8. Adam ε → 1e-5

**What:** Change Adam epsilon from 1e-8 to 1e-5.

**Why now:** The PPO reference implementation uses 1e-5. The larger epsilon provides slightly more numerical stability in the Adam denominator, which can matter when gradients are very small (as they often are after advantage normalisation).

**Implementation cost:** Trivial — one constant change.

#### 9. Increase crash penalty

**What:** Increase crash penalty from -5.0 to -10.0 or -15.0 (or scale it with achieved progress).

**Why now:** At current settings, the crash penalty is only ~27% of a typical episode's progress reward. The policy has little incentive to avoid crashing. A stronger penalty makes the cost of crashing more salient relative to the progress signal.

**Implementation cost:** Trivial — one constant change.

### P3 — Polish and scale

#### 10. Reward/return normalisation

**What:** Divide rewards by the running standard deviation of discounted returns (without subtracting mean).

**Why now:** This automatically scales the value function's target range, preventing value loss from dominating when reward magnitudes change as the policy improves.

#### 11. More vectorised environments

**What:** Increase `num_envs` from 3 to 8–16.

**Why now:** More environments produce more diverse transitions per rollout, improving gradient quality and reducing per-episode variance. The current 3-car setup is below typical PPO configurations (8–32 environments is common).

**Implementation cost:** Low — change one config value. May need to verify track has space for more spawn positions.

---

## Computational Performance Improvements

### Current throughput bottlenecks (verified by code inspection)

| Bottleneck | Location | Cause | Impact |
|-----------|----------|-------|--------|
| Per-sample forward pass | `update.rs:121` | Each sample does a full forward pass individually during PPO updates | High — O(batch × epochs) individual forward calls |
| Vec allocations per forward | `mlp.rs:26-34` | Every `Linear::forward` allocates a new `Vec<f32>` output | Moderate — thousands of small allocations per update |
| Input cache cloning | `mlp.rs:25` | Every `forward` call clones the input slice to `Vec` | Moderate |
| Raycast per-step marching | `observation.rs:233` | 11 rays × binary-search refinement per tick per car | Moderate at higher car counts |
| Buffer `clone()` for frozen buffer | `update.rs:78` | Entire rollout buffer cloned when preparing PPO update | One-time per update, moderate |

### Recommended throughput improvements

#### Short-term (low effort, meaningful gain)

1. **Pre-allocate forward pass buffers** — Replace `Vec<f32>` allocations in `Linear::forward` and `Relu::forward` with reusable buffers stored on the struct. This eliminates thousands of small heap allocations per PPO update.

2. **Batch forward pass** — Instead of calling `forward` once per sample in `ppo_process_chunk`, batch all samples in the current chunk into a single matrix multiplication. This is the highest-leverage computational improvement: it replaces N individual dot-product loops with one tiled matrix operation.

3. **Swap frozen buffer clone with swap** — Instead of `buffer.clone()` followed by `buffer.clear()`, swap the buffer contents into `PreparedUpdate` and reset the live buffer. Avoids copying ~500 state vectors.

#### Medium-term (moderate effort, significant gain)

4. **SIMD-accelerated matrix operations** — The inner loops in `Linear::forward` and `backward` are simple dot products that can benefit from explicit SIMD intrinsics or a lightweight linear algebra crate. For 64×23 and 64×64 matrices, even manual `f32x4` vectorisation yields 2–4× speedup.

5. **Increase car count with compute budget** — With the above optimisations, the simulation can support 8–16 cars without frame drops, producing more transitions per wall-clock second.

---

## What Not To Overbuild

- **Do not increase network size** before fixing activations. 2×64 with tanh is sufficient for this task. Larger networks create more parameters to train with the same limited sample budget.
- **Do not add recurrence (LSTM/GRU)** for the PPO baseline. The observation already contains lookahead features that provide the temporal context needed for anticipating turns.
- **Do not implement PPG, DAAC, or other advanced policy-gradient variants** at this stage. The current failure is not algorithmic — it is an implementation-detail and reward-design failure.
- **Do not build a full replay system** for what is still a baseline validator. On-policy PPO with proper implementation details should be sufficient.
- **Do not spend time on headless mode or checkpointing** before the core learning problem is solved. Those are experiment-discipline features that matter after the agent can complete laps.

---

## Open Uncertainties and Validation Needs

1. **Track geometry at the crash point** — The crash hotspot at sector 3 (10–15% progress) may have an unusually sharp corner. If the track demands more turning authority than `rotation_speed = 4.0 rad/s` can physically provide at the speeds the car reaches, no amount of PPO tuning will help. **Validation:** test whether a perfect controller (manually tuned or heuristic) can navigate the corner at typical learned speeds.

2. **Tanh impact magnitude** — While tanh is strongly favoured by the literature, the magnitude of improvement in this specific codebase is unknown until tested. **Validation:** run a direct comparison: same hyperparameters, ReLU vs tanh, measure dead neuron rate and max progress after N episodes.

3. **Per-tick vs best-only progress reward interaction with GAE** — Per-tick progress reward changes the reward density significantly, which may require adjusting `gamma` or `gae_lambda`. Denser rewards can reduce the effective horizon needed. **Validation:** monitor value loss and explained variance after the switch.

4. **Observation normalisation sensitivity** — Running normalisation can be destabilising in the first few hundred ticks when statistics are unreliable. **Validation:** use a warmup period (e.g., 1000 ticks) before enabling normalisation, or initialise with the current static scaling ranges.

5. **Effective steering authority** — The physics model applies `heading += -steering * rotation_speed * dt`. At 60 Hz with `rotation_speed = 4.0`, the maximum heading change per tick is `4.0 / 60 ≈ 0.067 rad ≈ 3.8°`. At high speed with drag 0.985, the car covers significant distance per tick. **Validation:** compute whether this turning rate is geometrically sufficient for the track's corner radii at observed speeds.

---

## Relationship To Existing Context

- **Extends:** `context/references/a2c-for-neurodrive.md` — that paper identified activation choice, observation normalisation, and LR annealing as credible upgrade candidates. This paper provides the concrete implementation guidance and priority ordering.
- **Reads from:** `context/architecture.md` — used for structural understanding and subsystem boundaries.
- **Relevant system docs:** `context/systems/brain-a2c.md`, `context/systems/agent-interface.md`, `context/systems/environment.md`.

---

## Source List

### Foundational

- Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (arXiv 2015): https://arxiv.org/abs/1506.02438

### Implementation practice

- Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" (ICLR Blog Track 2022): https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO" (ICLR 2020): https://arxiv.org/abs/2005.12729
- Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study" (ICLR 2021): https://arxiv.org/abs/2006.05990

### Domain-specific (autonomous racing)

- Bollack et al., "Reward design and hyperparameter tuning for generalizable deep reinforcement learning agents in autonomous racing" (Nature Scientific Reports, 2025): https://www.nature.com/articles/s41598-025-27702-6
- NotAnyMike, "Solving CarRacing with PPO": https://notanymike.github.io/Solving-CarRacing/

### Activation functions and representation

- Ota et al., "Latent Assistance Networks: Rediscovering Hyperbolic Tangents in RL" (arXiv 2024): https://arxiv.org/abs/2406.09079

### Reference implementations

- Stable Baselines3 PPO defaults: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- CleanRL PPO continuous: https://docs.cleanrl.dev/rl-algorithms/ppo/
