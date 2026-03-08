# NeuroDrive's Analytics Pipeline

The analytics pipeline captures, organises, and exports training data so that learning progress can be understood without re-running experiments. It operates as a three-phase data flow: **capture** (per-tick accumulation during simulation), **track** (per-episode and per-update recording at boundaries), and **export** (JSON + Markdown report generation on shutdown). Each phase is tied to a specific Bevy schedule stage, ensuring data integrity and timing correctness.

> **Prerequisites:** [A2C Algorithm](../../concepts/core/reinforcement_learning/a2c_algorithm.md), [Handwritten Neural Network](handwritten_neural_network.md)

**Code location:** `src/analytics/`

---

## Three-Phase Data Flow

### Phase 1 — Capture (FixedUpdate)

During each physics tick in Bevy's `FixedUpdate` schedule, the analytics system accumulates raw action data from the agent. Steering and throttle values produced by the policy network (after action smoothing) are fed into a running accumulator.

The accumulation uses **Welford's online algorithm** for numerically stable computation of mean and variance. Rather than storing every individual action value (which would consume unbounded memory over long episodes), the accumulator maintains a running count, mean, and M2 (sum of squared deviations from the current mean). At any point, the current mean and standard deviation can be extracted without storing the raw samples.

This two-phase action capture design — accumulate during the episode, snapshot at the boundary — means action statistics are always available for the current episode without post-hoc computation.

### Phase 2 — Track (Update)

At episode boundaries (crash, timeout, or lap completion), the analytics system creates a structured `EpisodeRecord` capturing everything notable about the episode that just ended. Similarly, after each A2C parameter update, an `A2cUpdateRecord` captures the optimiser's health metrics.

These records are appended to in-memory vectors. No disk I/O occurs during training — the overhead of file writes during tight learning loops could introduce timing jitter that interferes with the deterministic simulation.

**Deduplication guards** prevent the same episode or update from being recorded twice. Each record carries an ID (episode number or update step), and before appending, the system checks that the new ID exceeds the last recorded ID. This guard is necessary because Bevy systems can run multiple times per frame under certain scheduling conditions, and the episode-boundary detection logic could fire redundantly.

### Phase 3 — Export (Last)

When the application exits (Bevy's `Last` schedule stage or explicit shutdown), the pipeline exports all accumulated data in two formats:

1. **JSON file:** The complete raw data — all episode records and update records — serialised as structured JSON. This is the machine-readable format for programmatic analysis.
2. **Markdown report:** A human-readable training summary with formatted tables, statistics, and diagnostic sections. This is the format a developer reads to understand what happened during a training run.

---

## EpisodeRecord

Each `EpisodeRecord` captures the following fields:

- **episode_id:** Sequential integer identifying the episode.
- **best_progress:** The furthest progress fraction reached during the episode (0.0 to 1.0, where 1.0 is a full lap).
- **total_return:** Sum of all rewards received during the episode, including the terminal reward.
- **pre_terminal_return:** Sum of rewards *excluding* the terminal reward. This separation matters because terminal rewards (crash penalties, lap bonuses) can dominate the total and obscure the shaping signal.
- **Reward decomposition:** Individual totals for each reward component — progress reward, speed bonus, alignment bonus, crash penalty, lap bonus, etc. This decomposition makes it possible to diagnose which reward channels are driving behaviour.
- **ticks:** Number of physics ticks the episode lasted.
- **crashes:** Whether the episode ended in a crash.
- **end_reason:** Enum describing why the episode terminated (crash, timeout, lap completion).
- **crash_position:** If the episode ended in a crash, the world-space coordinates of the crash location. Used for spatial crash analysis.
- **Action summary:** Mean and standard deviation of steering and throttle over the episode, computed from the Welford accumulator snapshot.

---

## A2cUpdateRecord

Each `A2cUpdateRecord` captures optimiser health after a parameter update:

- **Policy loss:** The actor's loss value (negative expected advantage).
- **Value loss:** The critic's mean squared error against computed returns.
- **Entropy:** The Gaussian policy's entropy, measuring exploration breadth. Declining entropy indicates the policy is becoming more deterministic.
- **Explained variance:** `1 - Var(returns - values) / Var(returns)`. A value near 1.0 means the critic accurately predicts returns; near 0.0 means the critic is no better than predicting the mean; negative means the critic is actively harmful.
- **Action statistics:** Mean and standard deviation of actions sampled during the update batch.
- **Layer health:** Per-layer gradient norms and parameter norms for both actor and critic networks. These detect gradient explosion (norms growing unboundedly) and gradient vanishing (norms collapsing to near-zero).

---

## Chunked Metrics

For long training runs, per-episode data is too granular to read directly. The analytics system groups episodes into fixed-size **chunks** (windows of consecutive episodes) and computes summary statistics per chunk:

- **Mean, standard deviation, median, p90:** Computed for total return, best progress, and episode length within each chunk.
- **Crash rate:** Fraction of episodes in the chunk that ended in a crash.
- **Lap rate:** Fraction of episodes in the chunk that completed a full lap.

These chunked metrics provide the trend information needed to assess whether learning is progressing. A decreasing crash rate across chunks indicates the agent is learning to stay on track. An increasing mean progress indicates the agent is travelling further. An increasing lap rate is the ultimate success metric.

---

## Markdown Report Sections

The exported Markdown report contains the following sections:

### Performance Summary

A table of chunked statistics showing mean return, best progress, crash rate, and lap rate per chunk. This is the first thing to read when evaluating a training run.

### Reward Dynamics

Breakdown of reward components over time, showing whether the agent's improvement is driven by progress rewards, alignment bonuses, or reduced crash penalties. This reveals whether the reward shaping is working as intended.

### Crash Locations

Spatial binning of crash positions, identifying which parts of the track are most dangerous. If crashes cluster at a specific corner, it suggests the observation system may not provide sufficient lookahead for that geometry, or the reward function does not incentivise early braking.

### Progress Over Time

A chart (rendered as a text table) showing best progress fraction per episode chunk, illustrating the learning curve.

### A2C Health Tables

Per-update tables of policy loss, value loss, entropy, explained variance, and gradient norms. These are the diagnostics for detecting training pathologies: entropy collapse (premature convergence), exploding gradients, or a critic that fails to learn.

---

## Design Considerations

The analytics pipeline is intentionally conservative about when it performs I/O. All data accumulates in memory during training, and disk writes happen only at shutdown. This avoids introducing file-system latency into the training loop, which could affect the deterministic timing of Bevy's `FixedUpdate` schedule.

The Welford accumulation pattern for action statistics is preferred over naive mean/variance computation because it is numerically stable for large sample counts. Naive computation (sum-then-divide) accumulates floating-point error as the sum grows, which can produce negative variance estimates. Welford's algorithm avoids this by maintaining a running deviation from the current mean.

The deduplication guards are a defensive measure against a real failure mode: Bevy's scheduling can cause systems to execute more frequently than expected during frame spikes. Without deduplication, the same episode could be recorded multiple times, corrupting crash rate and return statistics.

> **See also:** [Progress and Lap System](progress_and_lap_system.md) for how progress and crash positions are computed, [Handwritten Neural Network](handwritten_neural_network.md) for the layer structures whose health is monitored, [Architecture Decisions](../architecture_decisions.md) for the overall system design philosophy.
