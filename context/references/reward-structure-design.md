# Reward Structure Design for Continuous-Control Racing with PPO

## Scope / Purpose

- Answer the repository-specific question: **what reward structure would best enable aggressive, skilful driving behaviour in NeuroDrive without creating conservative local optima or reward-hacking pathologies?**
- Survey how the strongest racing RL implementations design their reward signals, extracting implementation-level detail — not just paper titles.
- Ground every recommendation in the current reward implementation (verified post-session changes on 2026-03-27).
- This paper complements `context/references/ppo-optimisation.md` which covers PPO implementation details. This paper focuses exclusively on the reward signal.

## Current Project Relevance

The reward structure is the project's most active pain point. In the most recent training run (827 episodes, 3 cars):

- The policy converged to **near-constant full brake** (throttle mean -0.94, std 0.17)
- The final episode: **Timeout, 0.1% progress, 1800 ticks, mean speed 0.2** — the car sat still for 30 seconds
- The policy discovered that the time penalty (-0.005/tick = -9.0 over timeout) was cheaper than crash penalties (-5.0 per crash, multiple crashes per session)
- Crash penalty has since been set to 0.0, but the reward structure needs deeper analysis

**The core tension:** the reward must make aggressive driving the only path to positive returns, while providing enough shaping signal that the policy can learn *how* to drive aggressively rather than just *that* it should.

## Current Reward Implementation (Verified)

Post-session changes as of 2026-03-27, verified in `src/game/episode.rs`:

```text
Per-tick reward:
  progress_reward = progress_gain × (speed / 200.0) × 100.0
    where progress_gain = (current_progress_fraction - best_progress_fraction).max(0.0)
    and progress_fraction = distance_driven / total_track_length
  time_penalty = -0.005

Terminal reward:
  crash_penalty = 0.0  (just changed from -5.0)

Episode ends: crash or 30s timeout
Progress metric: cumulative forward arc-length from spawn (no finish line, no laps)
```

### Current Reward Properties

| Property | Current state | Consequence |
|----------|--------------|-------------|
| Progress signal density | **Best-only** — rewards new personal bests, zero for repeated progress | Creates learning plateaus when most episodes crash at the same point |
| Speed coupling | Speed-weighted progress (speed / 200.0 multiplier) | Good — rewards going fast, but only when also making new-best progress |
| Crash cost | 0.0 explicit + episode termination (lost future earning) | Relies entirely on value function to learn opportunity cost |
| Time penalty | -0.005/tick = -0.3/s | Weak — 30s of stalling costs only -9.0. May be insufficient to prevent passivity |
| Backward movement | Clamped to zero — no penalty for going backward | Car can oscillate without cost |
| Centreline proximity | Not rewarded | No incentive to take good lines through corners |
| Action smoothness | Not measured | Jerky steering/throttle switching is free |
| Heading alignment | Not rewarded (was removed in reward simplification) | No direct signal for corner-entry alignment |

---

## What The Research Says

### The Velocity Projection Paradigm

The strongest finding across racing RL literature is that **velocity projection onto the track tangent** outperforms both raw speed rewards and pure progress rewards. The mechanism:

```text
v_progress = velocity · tangent_at_closest_point
```

This single scalar captures:
- **Going fast** (high magnitude)
- **Going in the right direction** (positive projection)
- **Not going sideways** (sideways velocity contributes nothing)
- **Not going backward** (negative projection = negative reward)

Evans et al. ("Reward Signal Design for Autonomous Racing", 2021) systematically compared reward signals on F1/10th racing and found that **Cross-Track-Heading (CTH) error + velocity projection** produced the fastest lap times. The formula:

```text
r = v_norm - λ_d × |d_cross| - λ_θ × |θ_error|
```

Where `v_norm = speed / max_speed`, `d_cross` is lateral distance from reference line, and `θ_error` is heading error relative to the reference tangent.

**Project inference:** This is directly applicable. We already compute `speed`, `signed_lateral_offset`, and `heading_error` in the observation system. The velocity projection reward would replace our speed-weighted best-only progress with a per-tick dense signal that naturally rewards both speed and direction.

### Dense Per-Tick Progress vs Best-Only Progress

Every competitive racing RL implementation rewards **per-tick forward progress**, not best-only progress:

| Implementation | Progress signal | Citation |
|---|---|---|
| OpenAI CarRacing-v2 | +1000/N per new tile visited | OpenAI Gym |
| AWS DeepRacer | Per-step reward based on position/speed | AWS docs |
| Evans et al. F1/10th | Per-step velocity projection | arXiv 2021 |
| Gran Turismo Sophy | Per-step composite (speed + position) | Nature 2022 |
| Bollack et al. | Per-step velocity along centreline | Nature Sci. Rep. 2025 |

**Source-backed finding:** Best-only progress creates a well-documented plateau problem. Once the agent reaches the same crash point repeatedly, it receives zero progress signal on those episodes. Per-tick progress provides gradient signal throughout every episode, including the critical corner-approach phase where the car needs to learn speed management.

**Repository fact:** Our current implementation uses best-only progress (`progress_gain = (progress_fraction - best_progress_fraction).max(0.0)`). This is the single largest deviation from research practice and the most likely contributor to learning stagnation.

### Crash Penalties: The Evidence Is Nuanced

The literature reveals a surprising consensus:

**Explicit crash penalties can create conservative local optima** — exactly what we observed (the car learned to sit still because avoiding crashes was cheaper than earning progress). However, *no* crash penalty relies entirely on the value function learning the opportunity cost of early termination, which takes time.

The resolution from recent research (2025 Nature study, Evans et al.):

> The crash signal should come from the **termination itself** (done=true zeroes future value), not from an explicit penalty term. The key is that the **per-tick reward must be strongly positive for good driving**, so that episode termination is inherently costly.

Gran Turismo Sophy used crash penalties but in a context with much larger positive rewards per step, so the ratio was different. For a small MLP with PPO, the simpler approach is:

```text
crash_penalty = 0.0
Make per-tick reward large enough that dying = losing real money
```

**Project inference:** Setting crash penalty to 0.0 (as we just did) is correct. But it only works if the per-tick reward is dense and strongly positive during good driving. With best-only progress, there are long stretches of zero reward where termination has no opportunity cost. Switching to per-tick progress is essential for the zero-crash-penalty approach to work.

### Creative Reward Terms Worth Considering

#### 1. Velocity Projection (High Priority)

Replace speed-weighted best-only progress with:

```text
v_along_track = dot(velocity, track_tangent_at_car_position)
progress_reward = (v_along_track / speed_reference) × scale
```

This is dense (fires every tick), directional (rewards only forward velocity), and naturally speed-coupled. It subsumes both the progress reward and the removed heading-speed penalty into one clean term.

**Advantage over current:** doesn't require beating a personal best, fires every tick, naturally penalises sideways drift, negative when going backward.

#### 2. Centreline Proximity (Medium Priority)

```text
centreline_reward = c_coef × (1.0 - (|d_cross| / d_max)²)
```

Provides continuous signal for track-following. Quadratic falloff means the penalty increases sharply as the car approaches the edge. Most racing RL papers include this at a small coefficient.

**Important nuance:** some papers (Evans et al.) found that centreline proximity alone can produce overly conservative behaviour — cars that follow the centreline perfectly but slowly. It should be a secondary term, not the primary reward.

#### 3. Action Smoothness Penalty (Low-Medium Priority)

```text
jerk_penalty = -j_coef × (|steering - prev_steering| + |throttle - prev_throttle|)
```

Research on miniature car racing (I-RAS, 2023) found that jerk penalties reduced vehicle jerk by 73% without significantly affecting lap times. This prevents the jerky steering/throttle oscillation that PPO policies often develop.

**Project inference:** Worth adding later if the trained policy produces jittery controls, but not a priority while the car hasn't learned basic cornering.

#### 4. Survival Bonus (Low Priority, Potentially Harmful)

A small per-tick bonus for staying alive. Research is mixed:
- Can help in very sparse reward settings
- Can create the exact "do nothing" pathology we already experienced
- **Not recommended** when dense progress reward exists — it's redundant

#### 5. Curvature-Adaptive Scaling (Advanced, Deferred)

Scale the reward reference trajectory using the **minimum curvature path** rather than the centreline. The 2025 Nature study found this produced faster lap times because the minimum curvature path naturally approximates the racing line.

**Project inference:** Interesting for later but requires computing the minimum curvature path, which is a non-trivial optimisation problem. The centreline is a reasonable starting reference.

---

## Reward Structure Comparison Matrix

```text
                        Dense    Directional  Anti-       Anti-      Complexity
                        signal   (rewards     conservative stalling
                                  correct
                                  heading)
────────────────────────────────────────────────────────────────────────────────
Current (best-only      No ✗     Partially    No ✗        Weakly     Low
 speed-weighted)                 (speed only)             (time pen)

Per-tick arc-length     Yes ✓    No ✗         Partially   Yes ✓      Low
 progress                        (any dir)

Velocity projection     Yes ✓    Yes ✓        Yes ✓       Yes ✓      Low
 onto tangent                                 (naturally
                                               fast)

CTH (velocity +         Yes ✓    Yes ✓        Yes ✓       Yes ✓      Medium
 cross-track +
 heading error)

Full CTH + centreline   Yes ✓    Yes ✓        Yes ✓       Yes ✓      Medium
 + smoothness
────────────────────────────────────────────────────────────────────────────────
```

---

## Recommended Reward Structure for NeuroDrive

Based on the research synthesis, grounded in our specific constraints (2×64 MLP, PPO with GAE, 2D top-down, centreline with arc-length projection):

### Primary Reward: Velocity Projection

```text
v_along = dot(car.velocity, progress.tangent)
progress_reward = (v_along / speed_reference) × progress_scale
```

**Why this over per-tick arc-length progress:**
- Arc-length progress `forward_delta` gives the same reward whether the car is going 100 u/s or 800 u/s in the right direction. Speed weighting helps but it's a multiplication after the fact.
- Velocity projection is inherently speed-coupled: faster = more reward per tick, but only if heading is correct. A car going 800 u/s at 45° to the track gets `800 × cos(45°) = 566` projected, not the full 800.
- Velocity projection can be **negative** when going backward. Arc-length delta is clamped to zero. The negative signal is valuable — it tells the policy "you are actively making things worse."
- Every research source that compared the two found velocity projection produced faster, more aggressive driving.

### Secondary Reward: Centreline Proximity

```text
d_norm = (|signed_lateral_offset| / lateral_max).min(1.0)
centreline_reward = centreline_coef × (1.0 - d_norm²)
```

Small coefficient (e.g., 0.1–0.5). This provides a gentle signal for track-following without dominating the speed incentive. The quadratic falloff means the penalty ramps sharply near the edge, providing a "wall proximity warning" that's smoother than waiting for raycast distances to shrink.

### Time Penalty: Keep but May Need Adjustment

```text
time_penalty = -0.005 per tick
```

With velocity projection as the primary reward, the time penalty becomes less critical because standing still already earns zero progress reward (velocity projection of a stationary car is zero). But it prevents literal stalling and provides a small pressure to move. Keep it, but it shouldn't need to be large.

### Crash Penalty: Zero

```text
crash_penalty = 0.0
```

Episode termination is the crash cost. With dense velocity-projection reward, every tick alive and driving fast is earning. The value function learns that crashing throws away 20+ seconds of earning potential.

### Proposed Configuration

```text
reward_per_tick = velocity_projection_reward + centreline_reward + time_penalty

where:
  velocity_projection_reward = (dot(velocity, tangent) / 200.0) × scale
  centreline_reward = 0.3 × (1.0 - (|lateral_offset| / 50.0)².min(1.0))
  time_penalty = -0.005

terminal:
  crash_penalty = 0.0
  episode ends on crash or 30s timeout
```

### Scaling Considerations

The velocity projection reward needs careful scaling relative to the time penalty. At terminal velocity (833 u/s) going straight:

```text
velocity_projection = 833 / 200 × scale
```

If `scale = 1.0`, that's 4.17 per tick. The time penalty is -0.005. The ratio is 834:1 in favour of progress. That seems reasonable — the car earns massively for driving fast, bleeds slightly for existing. At zero speed: 0.0 progress + (-0.005) time = -0.005/tick.

The centreline reward at the centreline: `0.3 × 1.0 = 0.3` per tick. At the edge (50 units offset): `0.3 × 0.0 = 0.0`. This is a gentle shaping signal, not a dominant term.

---

## What Not To Overbuild

- **Do not add a minimum curvature reference path.** The centreline is a good enough reference for current learning. Minimum curvature paths are an optimisation for squeezing out faster lap times after the car can already complete laps.
- **Do not add jerk/smoothness penalties yet.** Solve cornering first. Add smoothness later if the trained policy is jittery.
- **Do not add survival bonuses.** They create the exact "sit still" pathology we just escaped.
- **Do not add curriculum-based reward scaling.** The current task is simple enough that a static reward structure should work. Curriculum adds complexity without evidence it's needed here.
- **Do not attempt reward normalisation as a first step.** Get the reward *shape* right first, then consider normalisation if value function training is unstable.

---

## Implementation Priority Order

| Priority | Change | Rationale | Effort |
|----------|--------|-----------|--------|
| **P0** | Replace best-only progress with per-tick velocity projection | Fixes the plateau problem, provides dense directional signal every tick | Low — ~20 lines in `episode.rs` |
| **P0** | Keep crash penalty at 0.0 | Already done. Works because velocity projection makes every alive tick valuable | Done |
| **P1** | Add centreline proximity term | Gentle track-following signal that helps corner entry | Low — ~5 lines in `episode.rs` |
| **P2** | Tune scale factors after observing initial training | The velocity projection scale, centreline coefficient, and time penalty may need adjustment based on actual reward magnitudes | Trivial |
| **P3** | Consider action smoothness penalty | Only if trained policy is jittery | Low |

---

## Open Uncertainties

1. **Velocity projection scale.** The right magnitude depends on how it interacts with GAE and the value function. Too large and value estimates become unstable; too small and the signal drowns in noise. The existing `speed_reward_reference = 200.0` and `progress_reward_scale = 100.0` provide a starting point. **Validation:** monitor explained variance and value loss in early training.

2. **Centreline coefficient.** 0.3 is a guess based on the ratio to the velocity projection reward. It should be small enough not to dominate but large enough to be detected by the policy. **Validation:** if the car hugs the centreline but goes slowly, reduce it. If the car drives fast but wide through corners, increase it.

3. **Negative velocity projection.** When the car is going backward, the reward is negative. Combined with the time penalty, this means backward movement is doubly punished. This is probably correct (going backward is worse than standing still), but worth monitoring for pathological behaviour like the car learning to avoid ever going backward at the cost of not turning.

4. **Interaction with braking.** The car now has a `[-1, 1]` throttle with braking. Velocity projection naturally rewards the right braking behaviour: brake before a corner (temporary zero/low reward) to maintain high reward through the corner. But early in training the policy may not discover this temporal trade-off. **Validation:** monitor throttle distribution — if it collapses to always-positive or always-negative, the reward balance needs adjustment.

---

## Relationship to Existing Context

- **Extends:** `context/references/ppo-optimisation.md` — that paper identified reward shaping as a P0/P1 gap. This paper provides the specific design.
- **Supersedes:** the reward-related recommendations in `ppo-optimisation.md` section "Reward Shaping and Corner Behaviour." The centreline proximity recommendation there is preserved; the best-only → per-tick recommendation is expanded to velocity projection.
- **Reads from:** `context/architecture.md` for subsystem boundaries, `src/game/episode.rs` and `src/agent/observation.rs` for implementation verification.

---

## Source List

### Foundational Racing RL

- Evans et al., "Reward Signal Design for Autonomous Racing" (arXiv 2021): Systematic comparison of reward signals on F1/10th platform. CTH + minimum curvature path produced fastest laps.
- Bollack et al., "Reward design and hyperparameter tuning for generalizable deep reinforcement learning agents in autonomous racing" (Nature Scientific Reports, 2025): Velocity-based rewards with centreline coupling outperformed position-only rewards across 21 unseen tracks.

### Production Racing Systems

- Wurman et al., "Outracing champion Gran Turismo drivers with deep reinforcement learning" (Nature 2022): Multi-component reward including speed, collisions, track excursions, sportsmanship.
- AWS DeepRacer documentation: Per-step reward with centreline proximity, speed incentives, steering smoothness.

### Reward Design Theory

- Ng et al., "Policy invariance under reward transformations" (ICML 1999): Potential-based reward shaping preserves optimal policy. Theoretical foundation for why shaping terms are safe.
- OpenAI CarRacing-v2: Dense tile-visiting reward + time penalty. Simple but effective baseline.

### Action Smoothness

- Ogata et al., "Image-based Regularization for Action Smoothness in Autonomous Miniature Racing Car" (arXiv 2023): Jerk penalty reduced vehicle jerk by 73%.

### Crash Penalty Analysis

- Multiple sources converge: explicit crash penalties create conservative optima in on-policy methods. Episode termination + dense positive reward is the preferred approach in recent work (2024–2025 publications).
