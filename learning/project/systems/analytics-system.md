# The Analytics System

## What This File Covers

NeuroDrive's analytics system captures, processes, and exports run data so that learning behaviour and driving quality can be inspected after a training session. This file explains what is captured, how the capture pipeline works, what derived metrics are produced, and how the final reports are generated.

**Status:** Current implementation.

## Prerequisites

- `project/architecture/module-boundaries.md` — analytics is read-only downstream
- `project/architecture/fixed-tick-pipeline.md` — tick-level capture timing
- `project/systems/environment-system.md` — EpisodeState and reward decomposition
- `project/systems/a2c-brain.md` — A2cTrainingStats that analytics records

---

## The Design Principle: Observe, Don't Mutate

Analytics is a pure observer. It reads state from `game/`, `agent/`, and `brain/`, but it **never writes back** to any of them.

This rule keeps the analytics subsystem safe: a bug in a metrics computation cannot corrupt the reward computation, the episode lifecycle, or the learning loop. The environment and brain systems are unaffected by anything analytics does.

The consequence is that analytics is downstream of all runtime truth. It captures what has already happened; it does not determine what happens.

---

## The Three-Layer Architecture

```
Layer 1: Trackers     ← per-tick accumulation during FixedUpdate
Layer 2: Metrics      ← derived diagnostics computed from episode summaries
Layer 3: Exporters    ← JSON and Markdown serialisation on exit
```

### Layer 1: Trackers (src/analytics/trackers/)

Three trackers accumulate data during the fixed-tick loop:

#### EpisodeActionAccumulator

Collects `ActionState.applied` every tick within the current episode. On episode completion, computes and stores per-episode action statistics:

| Stat | Description |
|---|---|
| `steering_mean` | Mean applied steering over the episode |
| `steering_std` | Standard deviation of applied steering |
| `throttle_mean` | Mean applied throttle |
| `throttle_std` | Standard deviation of applied throttle |

High `steering_std` relative to `steering_mean` suggests oscillatory or unstable steering. Very low `throttle_mean` suggests the agent is coasting excessively.

#### EpisodeTraceAccumulator

Records one `TickTraceRecord` per fixed tick. This is the most granular capture layer and produces the largest data volume. Each record stores:

| Field | Source |
|---|---|
| `progress_fraction` | `TrackProgress` |
| `speed` | `Car.velocity` |
| `centreline_distance` | `TrackProgress.distance` |
| `signed_lateral_offset` | `SensorReadings` |
| `heading_error` | `SensorReadings` |
| `applied_steering` | `ActionState.applied` |
| `applied_throttle` | `ActionState.applied` |
| Reward decomposition | `EpisodeState` breakdown |
| Terminal flags | `EpisodeState.current_tick_end_reason` |
| Ray distances | `SensorReadings` |
| Lookahead features | `SensorReadings` |
| `critic_value` | From `A2cBrain` when AI mode is active |

The trace accumulator runs in `SimSet::Measurement` after observation rebuild but before `a2c_collect_reward_system`, so each tick record captures the observation and reward for that tick together.

#### EpisodeTracker

Not a tick-level accumulator — this tracker folds completed episode summaries into a persistent `Vec<EpisodeRecord>`. It runs in Bevy's `Update` schedule (not `FixedUpdate`) and picks up completed episode snapshots produced in the previous fixed tick.

Each `EpisodeRecord` combines:
- the episode summary (steps, return, best progress, end reason)
- action statistics from `EpisodeActionAccumulator`
- the `EpisodeTrace` (the full tick-by-tick record)
- A2C update stats when an update occurred during or at the end of the episode

---

### Layer 2: Metrics (src/analytics/metrics/)

After all episodes are collected, the metrics layer derives higher-level diagnostics:

#### Chunked Trends

Episode data is split into contiguous chunks (e.g. 50-episode windows). For each chunk, statistics are computed: mean return, crash rate, mean best-progress. These trend summaries show whether learning is progressing, stalling, or regressing over time.

#### Input-Learning Summaries

Diagnose whether the policy has learned to modulate inputs effectively:
- Is throttle correlated with progress? (good = full throttle on straights)
- Is steering variance low on straights and high in corners? (good = purposeful steering)
- Is steering oscillation decreasing over training? (good = smoother control)

#### Turn-Execution Diagnostics

Identify whether specific track regions (corners vs straights) are sources of failure:
- Speed profile through corners
- Lateral offset through turns
- Frequency of crashes at specific track positions

#### Critic Diagnostics

Evaluate the value function's quality:
- **Explained variance:** `1 - Var(returns - V(s)) / Var(returns)` — a value near 1.0 means the critic is explaining most of the return variation, near 0 means the critic is useless, negative means it is worse than predicting the mean
- Value prediction error over time
- Whether the critic tracks the learning curve of the policy

#### Narrative Bullets

Short human-readable diagnostic strings generated from the metrics:
- "Policy is improving: mean return increased 23% across the last 100 episodes"
- "High crash rate at corner 3 — heading error peaks before that section"
- "Throttle mean is low (0.31); agent may be too conservative"

These are heuristic and should not be treated as ground truth, but they are useful for quick triage after a run.

---

### Layer 3: Exporters (src/analytics/exporters/)

On app exit, two reports are written under `reports/`:

#### JSON Report: run_<timestamp>.json

A machine-readable record of everything captured. Contains:
- All `EpisodeRecord` entries (full episode summaries, traces, A2C stats)
- Derived metrics
- Trend chunking

The JSON report is useful for programmatic analysis — graphing learning curves, loading traces into Python for visualisation, or comparing multiple runs.

#### Markdown Report: run_<timestamp>.md

A human-readable summary report. Contains:
- Run summary statistics (total episodes, best return, crash rate, final moving averages)
- Chunked trend table
- Narrative diagnostic bullets
- Recent episode list

The Markdown report is suitable for quick post-run inspection without any tooling.

---

## The Export Trigger

The analytics export is triggered by Bevy's `AppExit` message in the `Last` schedule. When the user closes the window:

1. `a2c_flush_on_exit_system` runs (partial rollout update if needed)
2. `analytics_export_system` runs — serialises all accumulated data to JSON and Markdown

**Known limitation:** Export is exit-triggered only. If the app crashes or is forcefully killed, all accumulated data is lost. There is no periodic checkpoint export, no crash-safe save, and no mid-run export capability.

---

## Data Flow Summary

```
FixedUpdate (every tick):
  capture_episode_action_stats_system
      └── EpisodeActionAccumulator.append(ActionState.applied)

  capture_episode_tick_trace_system
      └── EpisodeTraceAccumulator.append(TickTraceRecord)

  snapshot_completed_episode_trace_system
  snapshot_completed_episode_action_stats_system
      └── (when episode ends) finalise EpisodeTrace and push snapshot

Update (every frame):
  episode_tracker_system
      └── EpisodeTracker.fold(completed_snapshots)

Last (on exit):
  export_to_json(EpisodeTracker) → reports/run_<timestamp>.json
  export_to_markdown(EpisodeTracker) → reports/run_<timestamp>.md
```

---

## What the Reports Tell You

When interpreting analytics output, here are the key diagnostic questions:

**Is the agent learning?**
- Check `mean_return` trend. Is it increasing over chunks?
- Check `best_progress_fraction` trend. Is the agent consistently reaching further around the track?

**Is learning stable?**
- Check the variance of `mean_return` within chunks. High variance = unstable
- Check `explained_variance` in critic diagnostics. Low or negative = the value function is not helping

**Where is the agent failing?**
- Check `crash_rate` by episode. Is it improving?
- Check turn-execution diagnostics. Which corners have high lateral error?

**Is the policy exploring adequately?**
- Check `action_std` trend. Should remain reasonably high early; should gradually decrease as the policy specialises
- Check `entropy` in A2C update stats. Should not collapse to zero too quickly

**Is the HUD assessment accurate?**
- The HUD's recent-quarter assessment ("Improving/Mixed/Regressing/Warm-up") should roughly match the analytics trend data. If they disagree, the HUD assessment window may be too short to capture the signal.

---

## Known Gaps

| Gap | Impact |
|---|---|
| No RNG seed in reports | Cannot reproduce a specific run |
| No config snapshot | Cannot compare runs with different hyperparameters |
| No git revision | Cannot trace which code version produced a report |
| No mid-run export | Data lost on crash |
| No multi-run comparison | No tooling to overlay two exported reports |
| No explicit "every episode is recorded exactly once" test | Edge cases in terminal paths could cause double-counting or missed episodes |

---

## Related Files

- `project/systems/debug-runtime.md` — the live HUD is the runtime complement to analytics export
- `project/architecture/fixed-tick-pipeline.md` — capture timing relative to reward and observation systems
- `project/architecture/module-boundaries.md` — analytics as read-only downstream consumer
