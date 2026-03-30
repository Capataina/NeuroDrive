# Plan — Analytics Expansion and Deep Diagnostics

## Goal

Transform the analytics system from a basic episode-level summary into a comprehensive diagnostic tool that answers **why** the car behaves the way it does, **what** it has learned, **what** it's failing to learn, and **what** to change next. Every section should end with an auto-generated takeaway sentence that interprets the data for the reader.

## Design Principle

Every new metric should answer a specific diagnostic question. The question is listed alongside each metric so the purpose is always clear.

---

## Layer 1 — New Tick-Level Capture (TickTraceRecord)

These fields require capturing data that exists in the runtime but is not currently stored in the trace.

### New fields

| Field | Type | Source | Diagnostic question |
|---|---|---|---|
| `position_x` | f32 | `transform.translation.x` | Where did the car actually go? |
| `position_y` | f32 | `transform.translation.y` | Where did the car actually go? |
| `v_forward` | f32 | `sensors.v_forward` | How much speed is useful vs wasted? |
| `v_lateral` | f32 | `sensors.v_lateral` | Is the car sliding? |
| `speed_delta` | f32 | `sensors.speed_delta` | Is it accelerating or decelerating? |
| `drift_angle_deg` | f32 | computed: `atan2(v_lateral, v_forward).to_degrees()` | How much is the car drifting? |
| `min_ray_distance` | f32 | `sensors.ray_distances.min()` | How close to death is the car? |
| `velocity_projection` | f32 | `dot(velocity, tangent)` | How much reward-relevant speed? |
| `centreline_reward` | f32 | from episode system | How much is the centreline term contributing? |
| `value_prediction` | f32 | from brain forward pass | What does the critic think this state is worth? |
| `policy_steering_mean` | f32 | from brain action distribution | What did the policy intend for steering? |
| `policy_steering_std` | f32 | from brain action distribution | How confident is the policy about steering? |
| `policy_throttle_mean` | f32 | from brain action distribution | What did the policy intend for throttle? |
| `policy_throttle_std` | f32 | from brain action distribution | How confident is the policy about throttle? |
| `previous_steering` | f32 | `action_state.applied.steering` (previous tick) | Action continuity — detect jitter |
| `previous_throttle` | f32 | `action_state.applied.throttle` (previous tick) | Action continuity — detect jitter |

### Implementation notes

**Position capture:** Straightforward — read from `transform.translation` in `capture_episode_tick_trace_system`.

**Velocity decomposition, speed_delta, drift_angle:** Read from the new `SensorReadings` fields. Drift angle is computed as `atan2(v_lateral, v_forward).abs().to_degrees()`.

**min_ray_distance:** `sensors.ray_distances.iter().copied().fold(f32::MAX, f32::min)`.

**velocity_projection and centreline_reward:** These are computed in `episode_loop_system` but not currently exposed beyond the tick reward. Two options:
- Option A: Store them on `EpisodeState` as `current_tick_velocity_projection` and `current_tick_centreline_reward` — the trace system can read them.
- Option B: Recompute in the trace system from velocity and tangent.
- **Recommended: Option A** — avoids duplicating the computation and ensures exact consistency with the reward.

**Value prediction and policy distribution:** This is the most significant capture change. Currently the brain's `a2c_act_all_cars_system` computes the value estimate and the action distribution (mean, std) but only writes the sampled action to `ActionState`. To capture these:
- Add fields to `ActionState` or a new per-car `PolicyOutput` component: `value_prediction: f32`, `steering_mean: f32`, `steering_std: f32`, `throttle_mean: f32`, `throttle_std: f32`.
- Write them in `a2c_act_all_cars_system` alongside the action.
- Read them in the trace capture system.
- This is the highest-effort change in the entire plan but also the highest-value — it makes the critic and policy visible.

**Previous actions:** Already on `SensorReadings` after the observation expansion.

---

## Layer 2 — New Episode-Level Aggregates (EpisodeRecord)

These are derived from the tick trace when an episode completes. Computed in the episode tracker or trace metrics system.

### Speed and momentum

| Field | Type | Computation | Diagnostic question |
|---|---|---|---|
| `mean_speed` | f32 | mean of tick speeds | How fast was this episode overall? |
| `peak_speed` | f32 | max tick speed | What's the fastest the car reached? |
| `mean_v_forward` | f32 | mean of v_forward values | How much speed was in the right direction? |
| `mean_v_lateral_abs` | f32 | mean of |v_lateral| | How much was the car sliding? |
| `mean_velocity_projection` | f32 | mean of dot(velocity, tangent) | Core efficiency: how much speed was useful? |
| `mean_drift_angle_deg` | f32 | mean of |drift_angle| | Average drift level |
| `peak_drift_angle_deg` | f32 | max |drift_angle| | Most extreme drift moment |

### Action behaviour

| Field | Type | Computation | Diagnostic question |
|---|---|---|---|
| `braking_fraction` | f32 | fraction of ticks with throttle < -0.1 | Is the car using brakes? |
| `acceleration_fraction` | f32 | fraction of ticks with throttle > 0.1 | Is the car driving forward? |
| `coasting_fraction` | f32 | fraction of ticks with |throttle| <= 0.1 | Is the car indecisive? |
| `mean_action_change` | f32 | mean of |steer_t - steer_{t-1}| + |throttle_t - throttle_{t-1}| | Action jitter/smoothness |
| `mean_policy_steering_std` | f32 | mean of policy steering std | How confident about steering? |
| `mean_policy_throttle_std` | f32 | mean of policy throttle std | How confident about throttle? |

### Crash forensics

| Field | Type | Computation | Diagnostic question |
|---|---|---|---|
| `crash_v_forward` | Option<f32> | v_forward on crash tick | Was the car going forward into the wall? |
| `crash_v_lateral` | Option<f32> | v_lateral on crash tick | Was the car sliding into the wall? |
| `crash_drift_angle_deg` | Option<f32> | drift angle on crash tick | Was this a drift crash? |
| `crash_heading_error_deg` | Option<f32> | heading error on crash tick | Was the car pointed wrong? |
| `crash_min_ray` | Option<f32> | min ray on crash tick | How close was the nearest wall? |
| `crash_speed` | Option<f32> | **already exists** | — |

### Value function

| Field | Type | Computation | Diagnostic question |
|---|---|---|---|
| `mean_value_prediction` | Option<f32> | mean of value predictions | What did the critic expect from this episode? |
| `value_at_crash` | Option<f32> | value prediction on crash tick | Did the critic see the crash coming? |
| `value_at_start` | Option<f32> | value prediction on first tick | What's the critic's initial assessment? |
| `value_range` | Option<f32> | max - min value over episode | How much does the critic's view change? |

### Efficiency and exploration

| Field | Type | Computation | Diagnostic question |
|---|---|---|---|
| `reward_per_second` | f32 | total_reward / (ticks / 60.0) | Earning rate normalised for time |
| `furthest_sector` | u32 | max sector_index reached | How far along the track did the car get? |
| `wall_proximity_fraction` | f32 | fraction of ticks with min_ray < 30.0 | How much time near walls? |
| `distance_driven` | f32 | **already exists** | — |

---

## Layer 3 — New Chunk Metrics (ChunkMetrics)

Every episode-level metric that has a meaningful trend should be averaged per chunk. These enable sparklines and trend tables across training.

### New ChunkMetrics fields

```text
Speed & momentum:
  avg_speed, avg_peak_speed, avg_velocity_projection

Action behaviour:
  avg_braking_fraction, avg_acceleration_fraction, avg_coasting_fraction
  avg_action_change, avg_policy_steering_std, avg_policy_throttle_std

Crash forensics:
  avg_crash_speed, avg_crash_drift_angle, slide_crash_fraction

Efficiency:
  avg_distance_driven, avg_reward_per_second, avg_life_seconds
  avg_furthest_sector, avg_wall_proximity_fraction

Value function:
  avg_mean_value, avg_value_at_crash
```

---

## Layer 4 — Markdown Report Sections

### Section structure (10 sections total)

```text
1. Run Summary                    (existing — enriched)
2. Is the Policy Learning?        (existing — enriched)
3. Action Behaviour               (NEW)
4. Speed and Momentum             (NEW)
5. Crash Forensics                (existing section 5 — major expansion)
6. What Does the Car Think?       (NEW — value function analysis)
7. Track Coverage and Exploration (NEW)
8. Driving Quality                (NEW)
9. Training Health                (existing section 6 — enriched)
10. Trajectory Snapshots          (existing section 7 — enriched)
```

### Section 1 — Run Summary (enriched)

**Add to summary table:**
- Total distance driven (sum across all episodes)
- Mean episode duration (seconds)
- Mean reward per second (efficiency)
- Track coverage: highest sector reached by any car

**Takeaway generation:** Auto-generate one sentence like:
> "Cars are averaging X.X seconds alive, reaching sector Y, with Z% of time spent braking."

### Section 2 — Is the Policy Learning? (enriched)

**Add sparklines:**
- Distance driven (the new primary progress metric)
- Mean speed per episode
- Life duration (seconds per episode)
- Reward per second (efficiency over time)

**Add to chunk table:**
- Avg distance driven column
- Avg speed column
- Avg life seconds column

**Takeaway generation:**
> "Distance driven is [rising/flat/falling]. Speed is [rising/flat/falling]. The car is [learning to drive further and faster / stagnating / regressing]."

### Section 3 — Action Behaviour (NEW)

**Throttle distribution bar chart:**
```text
Braking  (<-0.1)  ████████░░░░░░░░░░░░  38%
Coasting (±0.1)   ███░░░░░░░░░░░░░░░░░  12%
Throttle (>0.1)   ██████████░░░░░░░░░░  50%
```

**Steering distribution bar chart:**
```text
Hard left  (<-0.5)  ██████░░░░░░░░░░░░░░  22%
Gentle left         ████░░░░░░░░░░░░░░░░  15%
Straight   (±0.1)   ████████░░░░░░░░░░░░  35%
Gentle right        ████░░░░░░░░░░░░░░░░  16%
Hard right (>0.5)   ███░░░░░░░░░░░░░░░░░  12%
```

**Sparklines:**
- Braking fraction over chunks
- Action jitter (mean_action_change) over chunks
- Policy confidence (mean std) over chunks

**Chunk table:**
| Chunk | Brake % | Coast % | Accel % | Jitter | Steer Conf | Throttle Conf |

**Takeaway generation:**
> "The car spends X% of time braking. Action jitter is [high/low/decreasing]. Policy confidence is [increasing/flat] — the car is [exploring / converging on a strategy / stuck with random actions]."

### Section 4 — Speed and Momentum (NEW)

**Speed profile sparkline** over episodes.

**Velocity projection sparkline** — the "useful speed" trend.

**Drift angle sparkline** — is the car learning to drift or grip?

**Speed distribution bar chart:**
```text
Stopped    (<50)    ████████████░░░░░░░░  45%
Slow       (50-200) ██████░░░░░░░░░░░░░░  25%
Medium     (200-500)████░░░░░░░░░░░░░░░░  18%
Fast       (500-800)██░░░░░░░░░░░░░░░░░░   9%
Terminal   (>800)   ░░░░░░░░░░░░░░░░░░░░   3%
```

**Chunk table:**
| Chunk | Avg Speed | Peak Speed | Avg V-proj | Avg Drift° | Distance |

**Takeaway generation:**
> "Mean speed is X u/s ([rising/flat]). The car converts Y% of its speed into forward progress (velocity projection / speed ratio). Drift angle averages Z° — [minimal drift / moderate sliding / heavy sliding]."

### Section 5 — Crash Forensics (major expansion)

**Crash type classification:**
Classify every crash into one of:
- **Slide crash:** |v_lateral| > |v_forward| at crash — the car slid sideways into the wall
- **Head-on crash:** v_forward > |v_lateral| and heading_error < 30° — driving straight into a wall
- **Overshoot crash:** v_forward > |v_lateral| and heading_error > 30° — missed the corner, went wide
- **Spin crash:** drift_angle > 60° — car was spinning
- **Stall crash:** speed < 50 at crash — car was barely moving

**Crash type distribution:**
```text
Slide      ████████████████░░░░  62%
Overshoot  ██████░░░░░░░░░░░░░░  24%
Head-on    ██░░░░░░░░░░░░░░░░░░   8%
Spin       █░░░░░░░░░░░░░░░░░░░   4%
Stall      ░░░░░░░░░░░░░░░░░░░░   2%
```

**Crash speed histogram:**
```text
0-100    ████░░░░░░░░░░░░░░░░  15%
100-200  ██████░░░░░░░░░░░░░░  25%
200-400  ████████████░░░░░░░░  45%
400-600  ███░░░░░░░░░░░░░░░░░  12%
600+     █░░░░░░░░░░░░░░░░░░░   3%
```

**Crash sector heatmap** (existing, kept).

**Crash table:**
| Chunk | Crashes | Avg Crash Speed | Slide % | Overshoot % | Head-on % | Avg Heading Err |

**Takeaway generation:**
> "X% of crashes are slides (car momentum carries it sideways into walls). Mean crash speed is Y u/s. Most crashes happen in sector Z. The car [is/is not] learning to avoid the dominant crash type."

### Section 6 — What Does the Car Think? (NEW)

This section makes the value function and policy visible.

**Value prediction sparkline** — mean value over episodes. Rising = critic is learning that driving is valuable.

**Value at crash vs value on straight:**
| Situation | Mean value | Count |
|---|---|---|
| Corner approach (curvature > threshold) | X.XX | N |
| Straight driving (curvature ≈ 0) | X.XX | N |
| At crash moment | X.XX | N |

**Critic accuracy check:**
- Mean value at crash: if high, the critic doesn't predict crashes (bad)
- Mean value at timeout: should be near zero (no future reward left)
- Value variance: if near zero everywhere, the critic hasn't learned anything useful

**Policy confidence evolution:**
- Mean steering std over chunks (lower = more confident)
- Mean throttle std over chunks

**Takeaway generation:**
> "The critic predicts value X.XX at crash (should be near 0 if it predicts crashes). Policy confidence is [increasing/flat]. The critic [is/is not] distinguishing dangerous states from safe ones."

### Section 7 — Track Coverage and Exploration (NEW)

**Sector reach heatmap:**
```text
Reached by >50% of episodes: S01 S02 S03
Reached by 10-50%:           S04 S05
Reached by <10%:             S06 S07
Never reached:               S08-S20
```

**Furthest sector sparkline** over chunks — is the exploration frontier advancing?

**Sector time distribution** (from trace data):
```text
S01 | ████████████████████  72% of all ticks
S02 | ██████░░░░░░░░░░░░░░  18%
S03 | ██░░░░░░░░░░░░░░░░░░   7%
S04 | ░░░░░░░░░░░░░░░░░░░░   3%
```

**Takeaway generation:**
> "The car reaches sector X on average (Y% of track). Z% of training time is spent in sector 1 — [the car is stuck at the first corner / good early-track coverage]. Exploration frontier is [advancing / stalled]."

### Section 8 — Driving Quality (NEW)

**Centreline distance sparkline** — is the car learning to follow the track?

**Wall proximity sparkline** — % of time near walls.

**Smoothness sparkline** — action jitter trend.

**Reward per second sparkline** — earning efficiency over training.

**Driving quality table:**
| Chunk | CL Distance | Wall Prox % | Smoothness | Reward/s | Drift° |

**Corner vs straight comparison:**
| Context | Mean Speed | Mean Drift° | Mean Throttle | Mean Steer |

**Takeaway generation:**
> "Centreline distance is [improving/flat] at X.X units. The car spends Y% of time near walls. Driving smoothness is [improving/degrading]. Reward efficiency is Z.ZZ per second."

### Section 9 — Training Health (enriched existing)

**Add sparklines:**
- Throttle mean over PPO updates
- Throttle std over PPO updates (is the policy exploring or converging?)
- Steering mean over PPO updates

**Add to latest update table:**
- Clamped action fraction (how often does the policy hit the action bounds?)

Everything else stays.

### Section 10 — Trajectory Snapshots (enriched existing)

**Add columns:**
- Mean drift angle
- Braking fraction
- Mean velocity projection
- Value at start / value at crash

**Add a new "worst crash" selection** alongside latest and best progress — the crash with the highest speed. This shows the most spectacular failure.

---

## Implementation Order

### Phase 1 — Expose brain internals (prerequisite for value/policy capture)

1. Add `PolicyOutput` component or extend `ActionState` with value prediction and policy distribution fields.
2. Write these in `a2c_act_all_cars_system`.
3. Verify the trace system can read them.

### Phase 2 — Expand tick trace capture

1. Add all new fields to `TickTraceRecord`.
2. Add `current_tick_velocity_projection` and `current_tick_centreline_reward` to `EpisodeState`.
3. Update `capture_episode_tick_trace_system` to populate the new fields.
4. Update the test helpers that construct `TickTraceRecord` instances.

### Phase 3 — Expand episode-level aggregates

1. Add all new fields to `EpisodeRecord`.
2. Write a `compute_episode_aggregates` function that derives them from the tick trace.
3. Call it in `episode_tracker_system` when folding a completed episode.
4. Update test helpers that construct `EpisodeRecord` instances.

### Phase 4 — Expand chunk metrics

1. Add all new fields to `ChunkMetrics`.
2. Update `calculate_chunks` to compute them.

### Phase 5 — Crash classification system

1. Write a `classify_crash` function that takes the crash tick's data and returns one of: Slide, HeadOn, Overshoot, Spin, Stall.
2. Store the classification on `EpisodeRecord` as `crash_type: Option<String>`.
3. Add `crash_type_counts` to `ChunkMetrics`.

### Phase 6 — Markdown report expansion

1. Rewrite the markdown exporter to 10 sections.
2. Implement takeaway generation for each section.
3. Add all new sparklines, bar charts, distribution charts, and tables.
4. Ensure all sections degrade gracefully when data is missing (e.g., no traces captured).

### Phase 7 — Validation

1. `cargo check` and `cargo test`.
2. Run the game briefly, export a report, verify all sections render correctly.
3. Verify takeaway sentences make sense.

---

## Takeaway Generation Approach

Every section ends with an auto-generated takeaway — a plain English sentence that interprets the data. The approach:

1. Compute the key metrics for the section.
2. Apply threshold-based classification (e.g., braking_fraction < 0.05 → "not using brakes").
3. Combine classifications into a template sentence.
4. The sentence should tell the reader **what to do next**, not just what happened.

Example templates:

```text
Action: "The car spends {brake_pct}% braking, {accel_pct}% accelerating.
        Action jitter is {jitter_level}. {confidence_statement}"

Speed:  "Mean speed is {speed} u/s ({speed_trend}). Velocity projection
        efficiency is {vp_ratio}%. {drift_statement}"

Crash:  "{dominant_crash_type} crashes dominate at {pct}%. Mean crash speed
        is {speed} u/s. {recommendation}"

Value:  "Critic predicts {val_at_crash} at crash moments ({assessment}).
        {confidence_trend_statement}"
```

The recommendation component uses simple rules:
- If slide crashes dominate → "Consider whether the car needs better lateral awareness"
- If crash speed is very high → "The car is not learning to brake before corners"
- If braking fraction is near zero → "The car has not discovered the brake yet"
- If value at crash is high → "The critic is failing to predict danger"
- If policy std is stuck high → "The policy has not converged — still exploring"
- If furthest sector isn't advancing → "The exploration frontier is stalled"

---

## Status

**Implemented** — all 7 phases completed by 2026-03-27. Tick-level capture, episode-level aggregates, chunk metrics, crash classification, the full 10-section markdown report with takeaway generation, and validation are all live. This file is retained as historical context for the design rationale and diagnostic question mapping.
