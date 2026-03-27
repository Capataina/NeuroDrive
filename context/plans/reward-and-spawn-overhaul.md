# Plan — Reward Simplification and Random Spawn Overhaul

## Status — Group A+B Implemented

| Item | Status |
|------|--------|
| A.1: Reward simplification (speed-weighted progress) | **Done** |
| A.2: Random spawn positions | **Done** |
| A.3: Adam ε → 1e-5 | **Done** |
| B.1: Minibatch shuffling + per-minibatch adv norm | **Done** |
| B.2: Orthogonal init + output head scaling | **Done** |
| C.1: Increase car count (3 → 8) | Not started |
| C.2: Running observation normalisation | Not started |
| C.3: Linear LR annealing | Not started |

Cars are confirmed learning — turning left and right when needed, reaching 100% track position (though this metric is inflated by random spawns). The next priority is the finish-line removal (see `context/plans/finish-line-removal.md`) and analytics rework to report distance-from-spawn honestly.

---

## Goal

Restructure the reward system and spawn logic so the PPO policy learns to drive the entire track at speed rather than optimising for survival on the first 15%. The current approach fails because: (a) too many reward terms create exploitable local optima, (b) the crash penalty discourages the aggressive driving that produces learning, and (c) all cars spawn at the same point so the policy never sees 85% of the track.

## Success Criteria

- Cars regularly reach 30%+ progress within 300 episodes
- No throttle or steering collapse (action std remains above 0.1)
- Crash distribution spans at least 6 of the 20 progress sectors
- At least one episode reaches 50%+ progress within 500 episodes
- The reward decomposition table shows speed-weighted progress as the dominant term

---

## Analysis: What Goes Together, What Conflicts

### Cross-reference with PPO Optimisation Plan

The existing `context/plans/ppo-optimisation.md` contains Phases 0–3 of PPO implementation improvements. Several items interact with this plan:

| PPO Plan Item | Status | Interaction with this plan |
|---|---|---|
| Phase 0.1: ReLU → Tanh | **Done** | Prerequisite satisfied. 0% saturation confirmed. |
| Phase 0.2: Per-tick progress | **Done** | Being replaced by speed-weighted progress here. |
| Phase 1.1: Orthogonal init | Not started | **Compatible.** Implement alongside — better initial exploration helps random spawns. |
| Phase 1.2: Minibatch shuffle + per-minibatch adv norm | Not started | **Strongly recommended with random spawns.** Random spawns create more diverse transitions in the buffer. Minibatch shuffling ensures each gradient update sees a mix of track sections rather than clusters from the same spawn region. Without shuffling, a minibatch might contain only samples from one car's local section, producing biased gradients. |
| Phase 1.3: Centreline proximity reward | **Done, now being removed.** Exploitable — cars farm it by sitting still. |
| Phase 2.1: Running observation normalisation | Not started | **Important with random spawns.** Cars at different track positions produce different observation distributions (tighter rays in corners, wider on straights). Running normalisation adapts to the mixed distribution. Without it, the static scaling may clip important signals from unfamiliar track sections. However, adds complexity — defer to a follow-up group unless throttle collapse recurs. |
| Phase 2.2: Linear LR annealing | Not started | **Compatible.** Independent of reward/spawn changes. Can add later. |
| Phase 2.3: Adam ε → 1e-5 | Not started | **Trivial, do it.** No interaction, one line. |
| Phase 3.3: More vectorised envs (3 → 8–16) | Not started | **Strong synergy with random spawns.** More cars = more simultaneous track coverage. At 3 cars with random spawns, each rollout samples ~3 track sections. At 8 cars, it samples ~8 sections — far better gradient diversity. But this requires verifying frame budget. Recommend testing at 3 first, then scaling up. |

### Noise Risk: Random Spawns with Only 3 Cars

**The concern:** With 3 cars spawning at random positions each episode, the rollout buffer contains transitions from wildly different track sections. This could create noisy gradients because the value function has to predict returns for many different starting contexts with limited data from each.

**Assessment:** This is a real risk, but it's mitigated by several factors:

1. **Per-minibatch advantage normalisation (Phase 1.2)** directly addresses this. Each minibatch's advantages are normalised within themselves, so a batch of "easy straight" transitions and "hard corner" transitions don't dilute each other's signal. **This should be implemented together with random spawns.**

2. **The observation space already encodes local geometry.** The value function doesn't need to learn "position on track → value". It learns "this ray pattern + this curvature = this expected return". Two similar corners at different track positions produce similar observations and should have similar values.

3. **3 cars × many episodes = reasonable coverage.** Over 100 episodes, the 2 ghost cars sample ~200 random positions. That's roughly 1 sample per 0.5% of track — sufficient for initial learning.

4. **Car 0 always starts at spawn** — this provides a stable reference trajectory in every rollout, grounding the value function with consistent data.

**Verdict:** Random spawns at 3 cars are viable *if* we also implement minibatch shuffling and per-minibatch advantage normalisation. Without that, the noise risk is higher and we should consider scaling to 6–8 cars first.

### Reward Simplification: What to Keep, What to Drop

| Current term | Keep / Drop | Reasoning |
|---|---|---|
| Per-tick progress × scale | **Replace** with speed-weighted progress | Core signal, but needs speed gating to prevent slow-crawl strategies |
| Centreline proximity | **Drop** | Farmable by sitting still. Speed-weighted progress makes it redundant — you can't earn progress reward without following the track |
| Time penalty (-0.005/tick) | **Keep** | Prevents crash-reset farming. Without it, "floor it → crash at 5% → reset → repeat" might pay more than surviving to 15% at moderate speed |
| Heading-speed penalty | **Drop** | Redundant. Speed-weighted progress already penalises misalignment implicitly — a misaligned car makes less progress per tick |
| Crash penalty | **Keep at -5.0** | Small tiebreaker. Dying is cheap, not learning is expensive |
| Progress bonus (episode-end) | **Drop** | Added complexity. Speed-weighted progress already rewards better runs with higher cumulative reward. The bonus created critic instability. |
| Lap bonus (100.0) | **Keep** | Big milestone signal. Not load-bearing until cars can complete laps |

### Speed-Weighted Progress: Design Considerations

The formula: `progress_delta × (speed / reference_speed) × scale`

**Choosing `reference_speed`:** This is the speed at which the multiplier equals 1.0. Above it, the multiplier exceeds 1.0 — the car earns bonus reward for being fast. Below it, the multiplier is fractional.

- Current observed mean speeds: 110–280 world units/s depending on track section
- Current physics: `thrust = 750`, `drag = 0.985` → terminal velocity ≈ 750 × 0.985 / (1 - 0.985) ≈ high, but practical speeds top out at ~350–400
- **Recommendation:** `reference_speed = 200.0`. This means a car going 200 u/s gets 1× multiplier, a car at 400 u/s gets 2×, a car at 100 u/s gets 0.5×. This rewards speed proportionally without creating a cliff.

**Choosing `scale`:** With per-tick progress and speed multiplier, the per-tick reward is:
- Progress delta per tick at ~200 speed: roughly `200 * dt / track_length ≈ 200/60 / total_length` as a fraction
- With scale 100 and speed multiplier 1.0, a car going 200 u/s earns roughly `(200/60 / track_length) × 1.0 × 100` per tick
- This needs to produce cumulative episode returns in a reasonable range (5–50 total)
- **Recommendation:** Start with `scale = 100.0`, observe returns, adjust if needed

**Edge case — reversing:** `progress_delta.max(0.0)` already handles this. Going backward earns zero.

**Edge case — spinning in place:** Speed is nonzero (angular velocity) but progress is zero. Reward is zero. Good.

**Edge case — driving sideways off track:** Progress might briefly increase but the car will crash. The small crash penalty handles this, and the car doesn't earn much because progress delta is small when the path is inefficient. Acceptable.

---

## Implementation Plan

### Group A — Core Changes (implement together)

These are the minimum viable set. They must be implemented together because the reward simplification alone won't break the 15% ceiling (the car still only sees 15% of the track), and random spawns alone won't help if the reward still encourages sitting still.

#### A.1 Reward Simplification

**Files:** `src/game/episode.rs`

**Changes:**

1. Replace `EpisodeConfig` reward fields:
   - Remove: `centerline_reward_coef`, `centerline_reward_max_dist`, `progress_bonus_scale`, `heading_speed_penalty_scale`, `speed_norm_max_for_penalty`
   - Add: `speed_reward_reference: f32` (default 200.0)
   - Change: `progress_reward_scale` → 100.0, `crash_penalty` → -5.0

2. Simplify `episode_loop_system` reward computation:
   ```
   progress_delta = (progress.fraction - previous_tick_fraction).max(0.0)
   speed_multiplier = car.velocity.length() / config.speed_reward_reference
   progress_reward = progress_delta * speed_multiplier * config.progress_reward_scale
   time_penalty = config.time_penalty_per_tick  (flat -0.005, no heading component)
   terminal_reward = crash_penalty or lap_bonus (one-off)
   tick_reward = progress_reward + time_penalty + terminal_reward
   ```

3. Remove from `EpisodeState`:
   - `current_tick_centerline_reward`, `current_centerline_reward_sum`, `last_episode_centerline_reward_sum`
   - `current_progress_bonus_sum`, `last_episode_progress_bonus_sum`

4. Update analytics: remove centreline and progress bonus columns from `EpisodeRecord`, `ChunkMetrics`, and the markdown exporter. Add a `speed_multiplier_mean` field instead so we can track how fast the cars are going relative to reference.

**Effort:** Medium — touching EpisodeConfig, EpisodeState, episode_loop_system, analytics models, chunking, markdown exporter. But mostly deletions and simplifications.

**Risk:** Low. Simpler reward = fewer surprises. The main risk is the `reference_speed` tuning, which is easy to adjust after one test run.

#### A.2 Random Spawn Positions

**Files:** `src/game/plugin.rs`, `src/game/episode.rs`, `src/game/car.rs`

**Changes:**

1. Add a `RandomSpawnConfig` resource:
   ```rust
   pub struct RandomSpawnConfig {
       pub main_car_index: u32,  // always spawns at canonical position (0)
   }
   ```

2. In `setup_game()` (plugin.rs):
   - Car 0: spawns at `track.spawn_position` with `track.spawn_rotation` (unchanged)
   - Cars 1–N: spawn at a random position along the centreline
     - Sample random `s` in `[0, total_length)` using a seeded RNG
     - Get position via `centerline.point_at_s(s)`
     - Get heading via `centerline.tangent_at_s(s)` → `tangent.y.atan2(tangent.x)`
     - Create `SpawnConfig` with this random position and heading

3. In `episode_loop_system`, on episode reset for ghost cars:
   - Instead of always resetting to `spawn_config` (which is the initial spawn), generate a new random position
   - This requires access to the `Track` (already available) and a source of randomness
   - Option: store a shared `SpawnRng` resource, or pass the brain's RNG
   - On reset: sample new `s`, update the car's `SpawnConfig` component, then reset to it

4. Car 0 always resets to the canonical spawn (preserving the stable reference).

**Effort:** Medium. The initial spawn in `setup_game()` is straightforward. The per-reset re-randomisation requires modifying the episode reset flow to sample a new position before calling `reset_car_to_spawn`.

**Risk:** Moderate — see noise analysis above. Mitigated by Group B.1 (minibatch shuffling).

#### A.3 Adam ε → 1e-5

**Files:** `src/brain/common/optim.rs`

**Change:** `epsilon: 1e-8` → `epsilon: 1e-5`

**Effort:** Trivial. One constant.

**Risk:** None.

### Group B — Noise Mitigation (implement with or immediately after Group A)

These items directly reduce the gradient noise that random spawns introduce. Group B.1 is **strongly recommended** alongside Group A. Group B.2 is optional but helpful.

#### B.1 Minibatch Shuffling + Per-Minibatch Advantage Normalisation

**Files:** `src/brain/a2c/update.rs`, `src/brain/a2c/buffer.rs`

This is Phase 1.2 from the PPO optimisation plan. With random spawns, the buffer contains transitions from different track sections clustered by car. Without shuffling, a chunk might contain only "straight" samples or only "corner" samples, producing biased gradient updates.

**Steps:** (same as PPO plan Phase 1.2)

1. Add `shuffled_indices: Vec<usize>` to `PreparedUpdate`
2. Shuffle at epoch start using brain RNG
3. Index through shuffled indices in `ppo_process_chunk`
4. Move advantage normalisation from global (in `compute_gae_per_env`) to per-chunk

**Effort:** Low-medium. Well-scoped change within `update.rs` and `buffer.rs`.

**Risk:** Very low. This is standard PPO practice.

#### B.2 Orthogonal Initialisation with Output Head Scaling

**Files:** `src/brain/common/mlp.rs`, `src/brain/common/math.rs`, `src/brain/a2c/model.rs`

This is Phase 1.1 from the PPO optimisation plan. Better initialisation means the initial policy is more exploratory (0.01× policy head), which helps random spawns because the car tries diverse actions from diverse starting points rather than defaulting to a biased initial policy.

**Steps:** (same as PPO plan Phase 1.1)

**Effort:** Low-medium. Add orthogonal init function, modify `ActorCritic::new`.

**Risk:** Very low.

### Group C — Scale Up (implement after Groups A+B are tested)

Only pursue once the reward and spawn changes are validated.

#### C.1 Increase Car Count (3 → 8)

**Files:** `src/game/car.rs`

**Change:** `num_envs: 3` → `num_envs: 8`

With random spawns, more cars = better track coverage per rollout. 8 cars with random spawns means ~7 random track sections per rollout step, plus car 0 at the canonical start. This dramatically improves gradient diversity.

**Prerequisite:** Verify frame budget. Run the simulation at 8 cars and check for sustained 60 FPS. If it stutters, the performance plan needs to come first.

**Effort:** Trivial config change, but may need spawn position validation.

**Risk:** Performance. 8 cars = 8× raycasting, 8× physics, 8× observations per tick. The PPO update cost doesn't change (buffer size is capped at `max_steps`).

#### C.2 Running Observation Normalisation

Phase 2.1 from the PPO plan. More important with random spawns because cars at different track positions produce different observation distributions. Defer unless the policy shows signs of observation-range issues (clamped features, poor learning in specific track sections).

#### C.3 Linear LR Annealing

Phase 2.2 from the PPO plan. Independent of reward/spawn changes. Add when the car is learning consistently and we want to stabilise late-stage training.

### Group D — Things We Should NOT Implement

| Item | Why not |
|---|---|
| Centreline proximity reward | Exploitable. Already demonstrated. Speed-weighted progress makes it redundant. |
| Progress bonus (episode-end) | Added critic instability. The cumulative speed-weighted progress already rewards longer, faster runs. |
| Heading-speed penalty | Redundant with speed-weighted progress. Misaligned cars make less progress. |
| Crash penalty > -5.0 | Discourages the aggressive driving that produces learning. |
| Progress % as observation input | Encourages track memorisation over geometry generalisation. |
| Velocity-based time bonus | Redundant with speed-weighted progress. Would double-count speed. |
| Pareto multi-objective rewards | Overengineered for this stage. Two clean terms are better than five competing objectives. |

---

## Implementation Order

```text
Group A (core — implement together, test as one batch)
  ├── A.1  Reward simplification (speed-weighted progress)
  ├── A.2  Random spawn positions
  ├── A.3  Adam ε → 1e-5
  └── ── CHECKPOINT: 300+ episode test run ──
         Verify: no action collapse, crashes distributed across sectors,
         max progress > 20%, speed_multiplier_mean > 0.5

Group B (noise mitigation — implement together, test)
  ├── B.1  Minibatch shuffling + per-minibatch adv norm
  ├── B.2  Orthogonal init + output head scaling
  └── ── CHECKPOINT: 300+ episode test run ──
         Verify: smoother explained variance, no regression in later chunks

Group C (scale — implement individually, test each)
  ├── C.1  Increase car count to 8
  ├── C.2  Running observation normalisation (if needed)
  └── C.3  Linear LR annealing (when learning is consistent)
```

## Tuning Parameters to Watch

| Parameter | Starting value | Adjust if... |
|---|---|---|
| `progress_reward_scale` | 100.0 | Returns too small (<5) or too large (>50) per episode |
| `speed_reward_reference` | 200.0 | Cars consistently above 300 (lower it) or below 100 (raise it) |
| `crash_penalty` | -5.0 | Cars don't care about crashing at all (increase to -8) or are too cautious (decrease to -3) |
| `time_penalty_per_tick` | -0.005 | Crash-reset loops dominate (increase to -0.01) or episodes are too short (decrease) |
| `num_envs` | 3 → 8 | Frame drops at 8 (stay at 6), or learning is noisy at 3 (increase sooner) |

## Reference

- PPO optimisation plan: `context/plans/ppo-optimisation.md`
- PPO research paper: `context/references/ppo-optimisation.md`
- Last run report: `reports/run_1774573102.md` (throttle collapse, centreline farming, 80% crash rate in final chunk)
- Architecture: `context/architecture.md`
