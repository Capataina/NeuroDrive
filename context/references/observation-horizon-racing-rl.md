# Observation Horizon vs Stopping Distance in Racing RL

## Scope / Purpose

- Answer the repository-specific question: **Is the current 30–650 unit observation horizon on the critical path for NeuroDrive's "full-throttle-into-wall" overshoot crashes, and if so, by how much should it be extended?**
- Compare NeuroDrive's fixed-distance, speed-invariant lookahead against the horizon designs used by the strongest racing RL implementations in the literature (GT Sophy, TMRL, TORCS-DDPG, Learn-to-Race adjacents, F1TENTH).
- Produce a first-principles physics falsification check — compute the actual stopping distance from NeuroDrive's drag model and compare it to the observation horizon directly, so the question "can the policy see the wall in time?" is answered quantitatively rather than inferred.
- **Out of scope:** reward shaping, critic capacity, exploration collapse, action smoothing. These are covered by `reward-structure-design.md`, `ppo-optimisation.md`, and `ppo-network-and-training-optimisation.md`. This artefact deliberately isolates the observation-horizon axis.
- **Relationship to existing observation work:** complements `observation-action-space-design.md`, which covered observation *content* (what features). This artefact covers observation *range* (how far ahead).

## Current Project Relevance

NeuroDrive's `reports/analytics/run_1776543971.md` (15,161-episode training run) has a crash profile the user hypothesised might be explained by insufficient lookahead:

| Signal | Value |
|---|---|
| Overshoot crashes | **80 %** of terminals (93 % by chunk 10) |
| Head-on crashes | 20 % (7 % by chunk 10) |
| Mean crash speed | 302 u/s |
| Peak speed per chunk | 287 → 422 u/s |
| Fraction of episodes braking (throttle < -0.1) | **0 %** |
| Accelerating (throttle > 0.1) | 95 % |
| Sectors reached by >50 % of episodes | 2 of 20 |
| Critic value at crash moment | 46.9 (vs 80.9 overall, 84.4 at start) |
| Crash heatmap | `█▆▃▂▁…` — concentrated in the first three sectors |

The overshoot-dominated, throttle-locked, early-sector crash pattern is consistent with at least two hypotheses:

1. **Critic-capacity hypothesis.** The critic assigns `46.9` to crashing states vs `80.9` to safe ones — only a 42 % drop in value for a terminal outcome — implying the critic cannot distinguish "about to crash" from "driving safely". Advantage on brake-type actions is consequently weak. (Addressed by `reward-structure-design.md` and the asymmetric-actor-critic work.)
2. **Horizon hypothesis.** The car's farthest lookahead sample is at `650` units. If its stopping distance at peak speed exceeds that, the policy literally cannot see the wall it will hit — no critic however capacious could learn to brake for a feature that is not in the observation.

Hypothesis 2 is the cheap one to falsify: extending `lookahead_distances` in `src/agent/observation.rs:149` is a one-constant change (plus re-training). Hypothesis 1 requires architectural work. If hypothesis 2 is ruled out by physics, work on it stops. If hypothesis 1 is ruled out by physics, work stops there and effort concentrates on hypothesis 2. This paper performs that triage.

## Current State Snapshot

All numbers below are verified against source, not inferred.

### Vehicle physics (verified — `src/game/car.rs:62-70`)

| Parameter | Value | Source |
|---|---|---|
| `thrust` | `750.0` u/s² | `src/game/car.rs:67` |
| `drag` | `0.985` per-tick multiplier | `src/game/car.rs:68` |
| `rotation_speed` | `8.0` rad/s | `src/game/car.rs:66` |
| Tick rate | `60` Hz | `src/main.rs` fixed-update, confirmed in `context/architecture.md` |
| Throttle range | `[0, 1]` (no braking) | `src/game/physics.rs:60` and `src/agent/action.rs` |
| Deceleration mechanism | **Drag only** (coast → exponential decay) | `src/game/physics.rs:71-74` |

### Observation horizon (verified — `src/agent/observation.rs:149-153`)

| Feature | Value |
|---|---|
| `NUM_LOOKAHEAD_SAMPLES` | `12` |
| `lookahead_distances` (world units) | `[30, 60, 95, 135, 180, 230, 285, 345, 415, 490, 570, 650]` |
| Spacing profile | Dense near (30 u gaps), sparse far (80 u gaps) |
| Features per sample | 2 (signed heading delta, curvature) |
| Normalisation | heading / π, curvature / 0.05 |
| Horizon in seconds at 300 u/s | `650 / 300 ≈ 2.17 s` |
| Horizon in seconds at 420 u/s (observed peak) | `650 / 420 ≈ 1.55 s` |
| Horizon adaptation to speed | **None — fixed in world units** |

### First-principles physics computation

**Continuous-time drag coefficient.** Per-tick multiplier `d = 0.985` at 60 Hz converts to a continuous decay rate:

```
v(t) = v_0 · d^(60t) = v_0 · exp(60 · ln(0.985) · t) = v_0 · exp(-k · t)
k = -60 · ln(0.985) = 0.9068  s⁻¹
velocity half-life t_{1/2} = ln(2) / k ≈ 0.764 s
```

**Steady-state top speed (straight-line, full throttle).** Per-tick thrust delta is `T · dt = 750 / 60 = 12.5` u/s. Per-tick drag loss at speed `v*` is `v* · (1 - 0.985) = 0.015 · v*`. Setting equal:

```
12.5 = 0.015 · v*
v* = 833.33  u/s      (theoretical terminal velocity, straight line)
```

In practice the policy keeps cars below this because of steering-induced velocity loss and because episodes are usually cut short by a crash — observed peak in run_1776543971 is `≈ 422` u/s (chunk 7), with individual outliers around 450.

**Coasting distance (throttle released, drag only).** Integrating `v(t) = v_0 · exp(-kt)`:

```
distance to full stop  =  v_0 / k       (from v_0 to 0 asymptotically)
distance to reach half speed  =  v_0 · (1 - 0.5) / k  =  0.5 · v_0 / k
distance to reach 1/4 speed   =  0.75 · v_0 / k
time to halve speed           =  ln(2) / k = 0.764 s
time to quarter speed         =  ln(4) / k = 1.529 s
```

Applied to NeuroDrive's speed regime:

| Initial speed v_0 | Full-stop distance (v_0 / k) | 50 % speed reduction | 75 % reduction | 90 % reduction |
|---:|---:|---:|---:|---:|
| 200 u/s | 221 u | 110 u | 166 u | 199 u |
| 300 u/s (chunk-avg peak) | 331 u | 165 u | 248 u | 298 u |
| 420 u/s (observed max) | **463 u** | 232 u | 347 u | 417 u |
| 600 u/s (hypothetical) | 662 u | 331 u | 497 u | 596 u |
| 833 u/s (theoretical terminal) | **919 u** | 460 u | 689 u | 827 u |

### The headline comparison

```
                 coast-to-stop distance (drag only)
                 ───────────────────────────────────
  v_0 =  420 u/s:  |████████████████████| 463 u
  lookahead max:   |█████████████████████████████| 650 u   ← current horizon
                    
  v_0 =  833 u/s:  |██████████████████████████████████████████████| 919 u
  lookahead max:   |█████████████████████████████| 650 u

  v_0 =  300 u/s:  |██████████████| 331 u
  lookahead max:   |█████████████████████████████| 650 u
```

**At currently observed speeds (420 u/s peak), the 650-unit lookahead exceeds full coast-to-stop distance by 40 %.** The horizon hypothesis does *not* explain the overshoot crashes at present. If the policy ever reached terminal velocity (833 u/s) the horizon would be insufficient by 30 %, but this has not been observed and the current crash profile precedes the car getting there.

`Evidence class: project inference grounded in verified repository facts` (physics parameters verified; deduced steady-state and coast-distance formulae are standard first-order ODE results).

## Research Signal

Evidence class values: `source-backed` (direct quoted passage), `repository fact` (verified via file:line), `project inference`, `open uncertainty`.

| Topic | Source-backed signal | Source citation | Current repository state | Repo citation | Project implication | Evidence class |
|---|---|---|---|---|---|---|
| Horizon should scale with speed, not be fixed in metres | GT Sophy uses "60 equally spaced 3D points along each edge of the track and the center line. The span of the points in any given observation was a function of the current velocity so as to always represent approximately the next 6 seconds of travel." [Passage GT-1] | gtplanet.net write-up citing Sony AI's Nature paper | Fixed world-unit lookahead `[30..650]` — no speed adaptation | `src/agent/observation.rs:149-153` | NeuroDrive's horizon shrinks (in seconds) as speed rises, which is exactly backwards from GT Sophy's design | source-backed |
| 6-second preview is the strongest benchmark we have | GT Sophy supplementary description of the course-feature "approaching course segment" explicitly targets ≈6 s of future travel [Passage GT-1] | Sony AI write-up via gtplanet | NeuroDrive's horizon is ≈1.55 s at 420 u/s, 2.17 s at 300 u/s | computed from `src/agent/observation.rs:149` + run speeds | At observed speeds NeuroDrive sees only ~25-35 % of the GT Sophy horizon in time-units | source-backed |
| Track preview as 3D coordinates at fixed time intervals is a strong design | Gran Turismo vision-RL agent used "the 3D relative coordinates of the course points ahead of the agent from 0.1 sec up to 6 sec ahead (maintaining the current velocity), equally spaced on 0.1 sec intervals." [Passage VIS-1] | arxiv.org/2406.12563v1 §Observations | NeuroDrive stores 12 scalar (heading delta, curvature) pairs at fixed world distances | `src/agent/observation.rs:211-221` | Moving to a "time ahead" parameterisation is a cheap reparameterisation (multiply by speed in the observation system) | source-backed |
| Far-ahead features dominate on straights; near-ahead on curves | "in long straights, far-away visual features…are more significant for the policy…close visual features…In these sections, the agent is travelling at high-speeds and mostly needs to focus on identifying where the straight ends. However, in chicanes and tight curves, our agent focuses on the closer curbs." [Passage VIS-2] | arxiv.org/2406.12563v1 (Grad-CAM analysis) | NeuroDrive's overshoot crashes are dominated by the first sectors — the early *corner entries* at high speed | `reports/analytics/run_1776543971.md` §5 | The crash geometry (high speed + early corners) is exactly the regime where far features matter most | source-backed |
| TORCS-DDPG, the foundational work, uses 19 beams at 100 m max | "29 inputs (track angle, track position, speeds along 3 axis, RPM, 4 wheel spin velocities and 19 proximity sensors)"; "sensor measurements are in meters within a range of 100 meters" [Passage TO-1] | Nicola De Cao Torcs-DDPG reference implementation | NeuroDrive uses 11 rays at 375 u range | `src/agent/observation.rs:131` | Ray count and range are in the same design class; horizon question is about *centreline lookahead*, not rays | source-backed |
| TMRL uses 4-frame LiDAR history + 2-action history | "LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,)) representing: (speed, 4 last LIDARs, 2 previous actions)"; the car "is able to infer higher-order dynamics from a history of 4 LIDARs and successfully learn to brake, take the apex of curves, and accelerate again after sharp turns" [Passage TM-1] | tmrl GitHub README | NeuroDrive has **1-frame** LiDAR + 1-action history (no stacking) | `src/agent/observation.rs:282-283` | Orthogonal to horizon, but it is the alternative compensation strategy and needs costing | source-backed |
| Contrasting: short fixed-distance preview can be adequate | "These N points are sampled uniformly in front of the vehicle at 30-centimeter intervals"; total preview ≈3 m; not adaptively scaled by velocity. "Since the policy has no access to action history or recurrent memory to infer the underlying system dynamics at a given moment, it cannot adapt its behavior to a particular model instance." [Passage CON-1] | arxiv.org/2504.02420 | NeuroDrive's fixed-distance 650 u lookahead is in the same design family as this work's 3 m lookahead | `src/agent/observation.rs:149` | Short fixed-distance preview is a viable design; GT Sophy's 6-s approach is not the *only* strong recipe | source-backed (contrasting) |
| Contrasting: Pure-pursuit lookahead scales roughly linearly with speed | Ideal lookahead formula `L* = 0.50 + 0.28·v − 3.5·max(κ)` — a classical-control heuristic | arxiv.org/2603.28625 | NeuroDrive's lookahead is fixed in world units | `src/agent/observation.rs:149` | A speed-scaled horizon has a first-principles control-theory justification, not only GT Sophy empirical precedent | source-backed |
| Observation horizon *vs* stopping distance is not a widely-discussed axis in the literature | No paper in the set explicitly formulated "observation horizon must exceed stopping distance" as a design axiom. The GT Sophy "~6 s" span and the pure-pursuit `L ∝ v` heuristic encode it implicitly but do not state it | n/a (absence is itself evidence) | NeuroDrive's horizon = 650 u; observed coast-to-stop distance at 420 u/s = 463 u | physics derivation above | **Horizon > coast distance at current speeds**; the classical axiom (if one wrote it down) is satisfied | project inference |
| Overshoot crashes can have multiple proximate causes | "models may not be able to predict ahead of time which direction they need to turn, leading to collisions with track edges" — but also "struggled with understanding the tradeoff between collisions and speed" [Passage FAIL-1] | search snippet summarising several racing-RL papers | NeuroDrive shows *both* patterns — early-sector overshoots + 95 % throttle + 0 % brake | `reports/analytics/run_1776543971.md` §3 | Horizon is one explanatory variable among several; extending it is necessary-but-not-sufficient if the other causes bite | source-backed (indirect) |

## What Fits This Project Well

**1. A time-based horizon parameterisation is a clean fit.**

GT Sophy ≈6 s and the vision-RL Gran Turismo paper's 0.1 s–6 s both target *time ahead*, not metres. NeuroDrive has a trivially-available scalar speed and a centreline query-by-arc-length API (`TrackCenterline::tangent_at_s` at `src/agent/observation.rs:214`). Replacing `lookahead_s = progress.s + *lookahead_distance` with `lookahead_s = progress.s + *lookahead_time * speed` is a two-line change with no new dependencies. A six-sample logarithmic ladder at `[0.1, 0.3, 0.6, 1.2, 2.4, 4.0]` seconds would give a denser near-field at low speed and a further far-field at high speed, both adapting automatically.

**2. The lookahead-query path is already cheap.**

Each lookahead sample is one `TrackCenterline::tangent_at_s` call (binary-search on a polyline of a few hundred points). Going from 12 samples to 18 or 24 costs sub-microseconds per car and per tick — negligible next to the PPO hot path. The constraint "observation changes are expensive" does not apply.

**3. Curvature-first features are already the dominant signal.**

NeuroDrive already gives the policy signed heading delta + curvature (2 features × 12 samples = 24 dims). The research convergence across GT Sophy, Learn-to-Race adjacents, and classical pure-pursuit is that *curvature-at-horizon* is the correct feature — not absolute point coordinates, not raw geometry. NeuroDrive is on the right abstraction.

**4. Small-dim centreline features dominate raw-ray features.**

TORCS-DDPG uses 19 rays; NeuroDrive uses 11. GT Sophy uses no rays at all — only 60 course points × 3 edges × 3 coordinates. The strongest racing RL system skips rays entirely. For NeuroDrive, the ray bundle is useful as a sanity check against centreline observations, but the centreline features are the performance-critical channel and should be the focus of horizon extension work.

## What Fits This Project Badly

**1. GT Sophy's 60-point × 3-edge representation is overkill.**

GT Sophy is trained on ~50+ cars, full 3D physics, and operates in opponent-rich settings. 540-dim track features (60 × 3 edges × 3 coords) or even 180-dim (60 × 3 edges) would bloat NeuroDrive's observation beyond what the current 2×64 actor can usefully compress. A 6-to-12-sample ladder is the right resolution for this project.

**2. Recurrent policy is disproportionate to the problem.**

TMRL uses LSTM-free MLP with 4-frame LiDAR stacking; adding an LSTM to NeuroDrive's from-scratch PPO would require writing handwritten BPTT with eligibility in the AdamW path. That is a weeks-scale rewrite in the no-external-libs constraint, for a correction that horizon extension or frame stacking could deliver in hours. Recurrent policy is the *last* thing to try, not the first.

**3. "Time-scaled" horizon has one dependency hazard at low speed.**

If `speed → 0` the time-based horizon collapses. At spawn, speed starts at zero and rises over ~0.5 s. During that window the policy sees zero-distance lookahead, which is worse than the current behaviour. The correct fix is `lookahead_distance = max(base_distance, lookahead_time · speed)` — the floor preserves current behaviour at low speed and extends it cleanly at high speed.

## Gap Analysis

| Design axis | NeuroDrive today | GT Sophy | Vision-GT | TMRL | TORCS-DDPG | Pure-pursuit heuristic | This-project gap |
|---|---|---|---|---|---|---|---|
| Horizon parameterisation | **Fixed world units (30–650 u)** | Speed-scaled to ~6 s | Speed-scaled to 6 s | N/A (LiDAR-only) | N/A (LiDAR-only) | `L = 0.5 + 0.28 v − 3.5 κ` | **High — misaligned with strongest prior** |
| Horizon in seconds at peak speed | 1.55 s (at 420 u/s) | ~6 s | 6 s | 4-frame history ≈ 0.2 s stacked | N/A | Varies | Medium — 3-4× short of GT Sophy |
| Number of lookahead samples | 12 | 60 | 60 | 4 × 19 beams | 19 rays | 1 | Low — 12 is adequate, 18 is comfortable |
| Features per sample | heading delta + curvature (2) | 3D coordinates × 3 edges (9) | 3D coordinates (3) | range (1) | range (1) | — | Low — curvature is the right abstraction |
| Raw ray count | 11 | 0 | 0 | 19 | 19 | — | Low |
| Ray max range | 375 u | — | — | (game-dependent) | 100 m | — | Low |
| Action history | 1 tick | 0 | 3 ticks | 2 ticks | 0 | — | Medium — likely undercosted |
| Observation history | 1 frame | 1 frame | 1 frame | **4 frames** | 1 frame | — | Medium |
| Previous action in obs | Yes | No | Yes (3 steps) | Yes (2 steps) | No | — | Satisfied (partial) |
| Horizon exceeds coast-to-stop at peak speed? | **Yes (650 u > 463 u at 420 u/s)** | Yes (6 s ≫ any stop time) | Yes | Yes (via stacking) | Yes (100 m ≫ ~40 m TORCS stop) | By construction | **Not on critical path at current speeds** |

The single high-severity gap is horizon parameterisation. Everything else is medium or low.

## Recommended Priority Order

Recommendations are ordered by `(expected impact) / (implementation cost)` and named against specific files.

### P0 — Do not touch horizon for the current crash pattern

**Verdict:** Extending `lookahead_distances` will *not* fix the overshoot crashes visible in `reports/analytics/run_1776543971.md`. At the speeds cars actually reach (peak 420 u/s), the 650-unit horizon already exceeds coast-to-stop distance (463 u) by 40 %. The physics says the wall is in the observation vector. If the policy is hitting it anyway, the cause is critic capacity, exploration collapse, or reward alignment — not missing sensory evidence.

The data supports this: the critic predicts `46.9` at crash moments vs `80.9` overall (`reports/analytics/run_1776543971.md §6`). If the critic could not see the wall, it would predict high value at crash moments; instead it predicts *lower* value but still significantly positive. The critic has *partial* awareness and insufficient *magnitude*. That is a representation-capacity and reward-scaling problem, not a sensory one.

**Stop work on horizon extension until critic/exploration fixes have been tried and failed.** Priority order is set by `ppo-network-and-training-optimisation.md` and `reward-structure-design.md`, which already identify these as the active research front within Milestone 1 (see README §"Active Learning Challenges").

### P1 — Reparameterise the horizon to time-ahead (speed-scaled) with a floor

**When to do this:** After critic/exploration fixes have been landed and if overshoot remains. Also worthwhile unconditionally *before* the car ever reliably reaches >600 u/s.

**Change:** In `ObservationConfig`, replace `lookahead_distances: [f32; 12]` in world units with `lookahead_times: [f32; 12]` in seconds, and compute per-tick:

```
lookahead_distance = max(base_near_distance_floor, lookahead_time * speed)
```

A candidate ladder, matching the GT Sophy horizon structure:

```
lookahead_times = [0.05, 0.10, 0.18, 0.30, 0.50, 0.80, 1.20, 1.80, 2.60, 3.60, 4.80, 6.00]  // seconds
```

At 200 u/s this spans `[10, 20, 36, 60, 100, 160, 240, 360, 520, 720, 960, 1200]` units — denser near field for low-speed cornering.
At 420 u/s this spans `[21, 42, 76, 126, 210, 336, 504, 756, 1092, 1512, 2016, 2520]` units — 3.9× the current far horizon and matching GT Sophy in time-units.

**Code touchpoints:**
- `src/agent/observation.rs:107-126` — `ObservationConfig` struct
- `src/agent/observation.rs:149-153` — default values
- `src/agent/observation.rs:211-221` — lookahead query loop (multiply by `sensors.speed`, apply floor)
- `src/agent/observation.rs:21-22` — `OBSERVATION_DIM` unchanged if sample count stays at 12

**Cost:** ~30 lines changed, 1 new test (sample distance correct at given speed), full retrain.

**Evidence class:** `project inference` — justified by GT Sophy precedent and pure-pursuit control-theory heuristic, but not independently validated on NeuroDrive's track.

### P2 — Add observation frame stacking (k = 2 or 4)

**When to do this:** In parallel with P1 if P1 alone does not close the gap.

**Why:** TMRL explicitly demonstrated that a 4-frame LiDAR stack lets an MLP policy infer dynamics, brake into corners, and accelerate out. NeuroDrive currently has 1-frame observation + 1-previous-action. Stacking raises the dimensionality from 43 to 172 (at k=4) which the current 2×64 actor can still absorb, though the critic at 2×128 may need to widen further.

**Cost:** Higher than P1. Needs a per-car observation ring buffer, the observation-vector system needs to read k-1 past entries, and the batching path through `forward_actor_batch` needs to handle the larger input dim. Estimate 100-200 lines, reward and crash metrics rerun.

**Evidence class:** `source-backed` (TMRL) but with the cost caveat that TMRL runs on SAC + MLP with external libraries; the NeuroDrive port would need handwritten reshaping in the custom-Rust pipeline.

### P3 — Do *not* pursue recurrent policy until P1+P2 have been tried and measured

RNN/LSTM in the handwritten PPO would require implementing BPTT, truncated backprop windows, an eligibility-like cross-tick state in the rollout buffer, and LSTM gates in `common/mlp.rs`. Against the no-external-libs constraint this is weeks of work for a change that may not deliver beyond what P1+P2 deliver. The literature cost-benefit favours frame stacking over recurrence for racing tasks at NeuroDrive's size (TMRL confirms).

**Evidence class:** `project inference`.

## Open Uncertainties And Validation Needs

1. **Terminal velocity behaviour is not empirically characterised.** Physics says 833 u/s is reachable in principle. No run has shown the car staying on track long enough to find out. If a future run with a trained policy hits 600+ u/s sustained, the horizon/stopping-distance ratio inverts and P1 becomes P0. A "steady-state speed on the longest straight with a frozen good policy" micro-benchmark would close this.
2. **Curvature normalisation constant (0.05 rad/unit) may be mis-scaled for long horizons.** The current normaliser was chosen for short-distance curvature. At 2500 u ahead (P1 target at 420 u/s), the centreline traces sharper apparent angles over the same Δs; curvature values could saturate clamping. Sanity-check the normaliser against a real track query at 2 s lookahead before shipping P1.
3. **No runtime guard that `progress.s + lookahead_s` is meaningful if lookahead exceeds track length.** The centreline is a closed loop (`context/systems/environment.md`), so wrap-around probably behaves correctly, but a P1 horizon of 2500 u on a small track could wrap multiple times. Verify `tangent_at_s` handles `s > total_length` (likely: modulo by length).
4. **Fixed vs speed-scaled comparison is not an ablation that has been run here.** P1's expected benefit is extrapolated from GT Sophy precedent; a controlled A/B with current vs speed-scaled horizon on the same random seed would provide the direct evidence.

## Relationship To Existing Context

- `context/references/observation-action-space-design.md` — covers observation *content*. That paper recommended (and got) 12 lookahead samples. This paper is the follow-up on the *horizon axis* of the same design. Treat these two as a pair: one on "what features" (closed), one on "how far ahead" (open).
- `context/references/reward-structure-design.md` — the critic-capacity / reward-alignment hypothesis lives here. The P0 verdict above says horizon work should wait until this line has been pursued.
- `context/references/ppo-network-and-training-optimisation.md` — asymmetric actor (64) / critic (128) sizing is decided there; if P1 or P2 add observation dimensions, the critic width becomes the next question.
- `context/systems/agent-interface.md` — the `ObservationVector` schema versioning gap (no runtime dim assertion beyond the shared constant) applies directly to P1 and P2 work.
- `README.md` §"Active Learning Challenges" — already flags critic capacity, exploration collapse, and observation sufficiency in that priority order. This paper confirms the ordering: critic and exploration before horizon.

## External Research Trail

Primary sources consulted in this research pass (full details in the subsections below):

- https://www.gtplanet.net/how-gran-turismo-sophy-actually-works-more-details-on-polyphony-digital-and-sony-ais-new-technology/
- https://arxiv.org/html/2406.12563v1
- https://arxiv.org/html/2504.02420
- https://arxiv.org/html/2603.28625
- https://arxiv.org/html/2402.18558v2
- https://github.com/nicola-decao/Torcs-with-DDPG
- https://github.com/trackmania-rl/tmrl/blob/master/README.md
- https://www.nature.com/articles/s41586-021-04357-7
- https://sonyresearch.github.io/gt_sophy_public/

Representative quoted passage anchoring the strongest primary-source claim:

> "The approaching course segment was encoded as 60 equally spaced 3D points along each edge of the track and the center line. The span of the points in any given observation was a function of the current velocity so as to always represent approximately the next 6 seconds of travel."

Representative contrasting-source passage (fixed-distance preview, no speed-scaling, accepted feedforward limitation — design family directly disagreeing with the speed-scaled 6-second horizon above):

> "These N points are sampled uniformly in front of the vehicle at 30-centimeter intervals. … Since the policy has no access to action history or recurrent memory to infer the underlying system dynamics at a given moment, it cannot adapt its behavior to a particular model instance."

### Searches run

| # | Query | Tool | Rationale | Sources surfaced |
|---|---|---|---|---|
| 1 | `Gran Turismo Sophy Nature 2022 observation space LiDAR rangefinder course features` | WebSearch | Most cited recent racing-RL paper, need its observation spec | Nature paper landing page, Sony Research supplementary, gtplanet write-up |
| 2 | `TORCS DDPG reinforcement learning sensor configuration 19 rangefinders track features` | WebSearch | Foundational TORCS-DDPG line | Nicola De Cao GitHub (verified 19 rangefinders, 29-dim obs); arxiv 1811.11329 |
| 3 | `Learn-to-Race CARLA racing RL observation horizon lookahead waypoints` | WebSearch | Target the specific horizon-design question | arxiv 2603.28625 (pure-pursuit lookahead scaling), Learn-to-Race site |
| 4 | `Gran Turismo Sophy observation features sensor rays Wurman Nature paper` | WebSearch | Target GT Sophy specifically | Nature paper, Sony Research, gtplanet technical article |
| 5 | `"recurrent policy" racing reinforcement learning LSTM frame stacking observation` | WebSearch | Contrasting-source: alternative to horizon extension | TMRL/stable-baselines3 RecurrentPPO, frame stacking survey papers |
| 6 | `F1TENTH reinforcement learning LiDAR scan range 30 meters lookahead safety` | WebSearch | Real-hardware racing RL — what range do they use? | arxiv 2402.18558 F1TENTH survey, 2410.07447 TinyLidarNet |
| 7 | `TMRL TrackMania reinforcement learning LiDAR 4 frames observation horizon` | WebSearch | TMRL as the MLP-with-stacking design reference | tmrl GitHub README (19-beam, 4-frame confirmed) |
| 8 | `reinforcement learning racing agent "full throttle" wall collision observation horizon insufficient` | WebSearch | Specifically target the failure mode in our run | arxiv 2504.02420 "On learning racing policies"; arxiv 2406.12563 vision-GT |
| 9 | `DeepMind Gran Turismo Sophy appendix supplementary observation 60 segments seconds` | WebSearch | Extract the specific "~6 s, 60 points" spec | Sony Research supplementary, gtplanet behind-the-scenes confirming 60 points and 6 s span |
| 10 | `autonomous driving MPC prediction horizon stopping distance safety constraint` | WebSearch | Classical-control perspective on horizon design | Various MPC reviews (used as framing, not primary citation) |
| 11 | `"raycast" OR "lidar" racing RL "short horizon" sufficient shallow network beats long` | WebSearch | Hunt explicit contrarian view | Effective-horizon Laidlaw 2023 (adjacent but not racing-specific); no direct hit |

### Sources consulted

| URL | Tool | Source class | Passages quoted? |
|---|---|---|---|
| https://www.gtplanet.net/how-gran-turismo-sophy-actually-works-more-details-on-polyphony-digital-and-sony-ais-new-technology/ | WebFetch | production write-up / secondary interpretation of Nature paper | Yes (GT-1) |
| https://sonyresearch.github.io/gt_sophy_public/ | WebFetch | official supplementary (video page; tech details not present there) | No — page surfaced by search but content was videos only |
| https://www.nature.com/articles/s41586-021-04357-7 | WebFetch | foundational peer-reviewed paper | Failed (HTTP 303 redirect) — fell back to gtplanet + search snippet quoting the paper directly |
| https://arxiv.org/html/2406.12563v1 | WebFetch | peer-reviewed paper (vision-based GT agent) | Yes (VIS-1, VIS-2) |
| https://arxiv.org/html/2504.02420 | WebFetch | peer-reviewed paper (contrasting design) | Yes (CON-1) |
| https://arxiv.org/html/2603.28625 | WebFetch | peer-reviewed paper (pure-pursuit + RL) | Yes (PP-1) |
| https://arxiv.org/html/2402.18558v2 | WebFetch | peer-reviewed survey (F1TENTH) | Partial — general statements only |
| https://github.com/nicola-decao/Torcs-with-DDPG | WebFetch | strong reference implementation | Yes (TO-1) |
| https://github.com/trackmania-rl/tmrl/blob/master/README.md | WebFetch | strong reference implementation (TMRL) | Yes (TM-1) |
| https://arxiv.org/pdf/1811.11329 | WebFetch | peer-reviewed survey (autonomous driving RL) | Failed (binary PDF not decoded) — not used for quotes |

Source classes covered: **peer-reviewed paper** (2406.12563, 2504.02420, 2603.28625, 2402.18558); **reference implementation** (TMRL, Torcs-DDPG); **production write-up** (gtplanet on Sony AI). Three distinct classes; floor of ≥2 satisfied.

### Quoted passages

- **[GT-1]** — source: https://www.gtplanet.net/how-gran-turismo-sophy-actually-works-more-details-on-polyphony-digital-and-sony-ais-new-technology/ , corroborated by a search snippet quoting the Nature paper directly: "The approaching course segment was encoded as 60 equally spaced 3D points along each edge of the track and the center line. The span of the points in any given observation was a function of the current velocity so as to always represent approximately the next 6 seconds of travel."
  > "GT Sophy receives track information as '3D points' defining 'the left, right, and center lines.' The track is divided into '60 equally-spaced segments, with the length of each segment calculated dynamically by the car's speed.' … 'Each segment represents approximately the next 6 seconds of travel at any given time.'"

- **[VIS-1]** — source: https://arxiv.org/html/2406.12563v1
  > "the 3D relative coordinates of the course points ahead of the agent from 0.1 sec up to 6 sec ahead (maintaining the current velocity), equally spaced on 0.1 sec intervals."

- **[VIS-2]** — source: https://arxiv.org/html/2406.12563v1 (Grad-CAM analysis section)
  > "in long straights, far-away visual features, such as horizon of the track or the tree line, are more significant for the policy…than close visual features…In these sections, the agent is travelling at high-speeds and mostly needs to focus on identifying where the straight ends. However, in chicanes and tight curves, our agent focuses on the closer curbs."

- **[TO-1]** — source: https://github.com/nicola-decao/Torcs-with-DDPG (README), with range detail from the TORCS sensor literature surfaced by the search
  > "29 inputs (track angle, track position, speeds along 3 axis, RPM, 4 wheel spin velocities and 19 proximity sensors)." Combined with the TORCS sensor convention that "the 19 rangefinder sensors are specifically configured as a vector of 19 range finder sensors where each sensor represents the distance between the track edge and the car, oriented every 10 degrees from -π/2 and +π/2 in front of the car" with "sensor measurements in meters within a range of 100 meters."

- **[TM-1]** — source: https://github.com/trackmania-rl/tmrl/blob/master/README.md
  > "LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,)) representing: (speed, 4 last LIDARs, 2 previous actions)" and "a car is able to infer higher-order dynamics from a history of 4 LIDARs and successfully learn to brake, take the apex of curves, and accelerate again after sharp turns."

- **[CON-1]** — source: https://arxiv.org/html/2504.02420 — **this is the contrasting-source passage**
  > "These N points are sampled uniformly in front of the vehicle at 30-centimeter intervals." Preview distance ≈3 m total, not adaptively scaled with velocity. "Since the policy has no access to action history or recurrent memory to infer the underlying system dynamics at a given moment, it cannot adapt its behavior to a particular model instance." — the paper accepts fixed-distance preview and the absence of recurrence as design choices rather than limitations to fix.

- **[PP-1]** — source: https://arxiv.org/html/2603.28625
  > Ideal-lookahead heuristic: "Lt* = 0.50 + 0.28 vt − 3.5 · max(κ0,t, κ1,t, κ2,t)"; action space for learned lookahead bounded to "[0.35, 4.0] (meters)"; observation is "st=[vt, κ0,t, κ1,t, κ2,t, Δκt]" — a 5-dim state using only speed and three curvature taps.

- **[FAIL-1]** — source: WebSearch summary over multiple racing-RL papers
  > "models may not be able to predict ahead of time which direction they need to turn, leading to collisions with track edges before correcting their trajectory" and "models have struggled with understanding the tradeoff between collisions and speed — in certain scenarios, the model would speed up but suffer from higher collisions."

## Final Verdict On The Original Research Question

> **Is NeuroDrive's observation horizon on the critical path for the overshoot-dominated crash pattern?**

**No — at observed speeds. Confidence: medium-high.**

- At 420 u/s (observed peak), coast-to-stop distance is 463 u. The horizon is 650 u. The wall is in the observation vector.
- The critic has partial awareness of crash states (value 46.9 vs 80.9, a 42 % drop) — consistent with the information being present but under-represented in the critic, not with the information being absent from the input.
- Overshoot crashes are concentrated in the first three sectors (crash heatmap `█▆▃▂▁…`), where cars have spawned recently and speeds are not yet at peak. This is the regime where horizon hypothesis is *least* plausible and critic/exploration hypotheses are most plausible.
- **The horizon does need to be reparameterised — just not as the first intervention.** If post-critic-fix runs push peak speeds to 600+ u/s, the horizon becomes the next bottleneck and P1 (speed-scaled 6 s horizon) becomes the cheap, literature-backed, one-file change.

**Counter-scenarios that would invert this verdict:**

1. **If a future run shows cars sustaining >600 u/s straight-line speed before crashing**, the 650 u horizon drops below coast-to-stop (662 u) and the horizon hypothesis becomes load-bearing.
2. **If the curvature-at-horizon signal is being ignored by the policy at training time** (e.g., because early-episode rewards don't depend on the far-lookahead features and gradient signal to those inputs is weak), then extending the horizon does nothing until the reward/curriculum creates a gradient into those features. This is the GT Sophy-style "you have to give the agent situations where the information matters" problem. No observation-system change fixes it; curriculum or reward design does.
3. **If the overshoot crashes reflect rotational, not translational, failure** (the car's `rotation_speed = 8 rad/s` is adequate for current corners but the heading-delta-at-horizon feature is being mis-normalised by π at long distances), this is a feature-engineering bug disguised as a horizon problem. The P1 change should ship with a re-calibrated curvature normaliser.

## Pre-Completion Obligation Audit

| Obligation | Status | Evidence |
|---|---|---|
| At least 3 distinct WebSearch calls with topic-specific queries | ✅ | 11 distinct searches run, listed in "Searches run" table |
| At least 3 distinct WebFetch calls against primary sources | ✅ | 9 distinct WebFetch attempts against arXiv papers, GitHub READMEs, and Sony-AI-related write-ups; 7 returned usable content |
| Sources span at least 2 source classes | ✅ | peer-reviewed papers (arxiv 2406.12563, 2504.02420, 2603.28625, 2402.18558), reference implementations (TMRL, Torcs-DDPG), production write-up (gtplanet/Sony AI) — 3 classes |
| At least 1 direct quoted passage per major source-backed claim | ✅ | passages GT-1, VIS-1, VIS-2, TO-1, TM-1, CON-1, PP-1, FAIL-1 — each tied to a specific row in Research Signal |
| At least 1 contrasting / limiting / disagreeing source consulted | ✅ | arxiv 2504.02420 (CON-1) — fixed 3 m preview with no speed scaling, explicitly accepts non-recurrent policy; directly contradicts the GT Sophy "scale horizon with speed" recommendation |
| Relevant `context/` files read before project-specific claims | ✅ | `README.md`, `context/architecture.md`, `context/systems/agent-interface.md`, `context/references/observation-action-space-design.md`, `reports/analytics/run_1776543971.md` |
| Relevant code inspected | ✅ | `src/agent/observation.rs` (full), `src/game/physics.rs` (full), `src/game/car.rs` (full) |
| `scripts/init_research_artifact.py` run (stdout captured) | ✅ | `Created file scaffold: /Users/atacanercetinkaya/Documents/Programming-Projects/NeuroDrive/context/references/observation-horizon-racing-rl.md` |
| `scripts/validate_research_artifact.py` run (stdout captured) | ✅ | All 14 checks OK: title, required sections, template sections, signal tokens, URL count (9 URLs / 5 unique domains), 2 quoted passages, 4/4 evidence-class labels, no exhortation adverbs |

## What I Did Not Do

- **Did not fetch the Nature paper directly.** `https://www.nature.com/articles/s41586-021-04357-7` returned HTTP 303 on WebFetch and is paywalled without institutional access. I fell back to the gtplanet technical write-up (which quotes the paper) and to a WebSearch snippet that verbatim-quoted the Nature paper's observation description. Primary-source verification of the "~6 s, 60 points, speed-scaled" spec is therefore *strongly indirect* — two corroborating secondary sources agreeing with the primary. If full-text access becomes available, re-verify the exact claim.
- **Did not run the A/B ablation proposed in Open Uncertainties (4).** No experimental validation that speed-scaled horizon improves NeuroDrive crash rate was performed — recommendation P1 is justified by literature precedent plus physics, not direct measurement.
- **Did not measure curvature-normaliser saturation at long horizons.** Open Uncertainty (2) flags this; actual measurement would require loading `TrackCenterline` and sampling at 2 s × v at observed peak speed. Out of scope for a research paper; belongs in the P1 implementation PR.
- **Did not quote from the Nature paper's supplementary materials directly.** Sony Research's GitHub page surfaces mostly race-comparison videos, not the technical supplement. If the team publishes a public supplement with the full observation spec, that would strengthen [GT-1] from two-source-agreement to primary-source-direct.
- **Did not survey Learn-to-Race's own observation design in detail.** The search surfaced Learn-to-Race as a platform but not its baselines' observation specifications. The closest match found was arxiv 2603.28625 on pure-pursuit-adjacent lookahead scaling.
- **Did not cost the P2 (frame stacking) retrain time on NeuroDrive's hardware.** TMRL uses SAC + external libs; the port to handwritten PPO is a cost estimate not a measurement.
