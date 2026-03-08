# System — Analytics Export and Run Summaries

## Scope / Purpose

- Persist episode-level results so training behaviour can be inspected after a run rather than only through the live HUD.
- Provide a minimal analysis path that converts raw episode completions into machine-readable and human-readable reports.
- Define a durable analytics taxonomy so performance health, learning health, network health, and exploration health do not get conflated.

## Current Implemented System

- An `AnalyticsPlugin` is wired into the application and initialises an `EpisodeTracker` resource (`src/analytics/plugin.rs`).
- Episode tracking is implemented by reading `EpisodeState` and recording one `EpisodeRecord` per completed episode (`src/analytics/trackers/episode.rs`).
- The episode tracker now also accumulates applied-action statistics across each episode, including steering/throttle mean and standard deviation (`src/analytics/trackers/episode.rs`).
- Each stored episode record now captures episode id, best progress, total return, pre-terminal return, reward decomposition terms including progress reward and time penalty, tick count, crash count, end reason, lap-complete flag, crash position when applicable, and per-episode steering/throttle summary statistics (`src/analytics/trackers/episode.rs`).
- Chunked summary metrics now compute progress mean/std/median/p90, reward mean/std, reward-decomposition means, average ticks, crash rate, lap-completion rate, and chunked steering/throttle summaries (`src/analytics/metrics/chunking.rs`).
- The analytics layer now also stores `A2cUpdateRecord` snapshots containing policy entropy, value loss, explained variance, action-distribution statistics, clamped-action fraction, and per-layer weight/gradient/dead-ReLU health (`src/analytics/trackers/episode.rs`, `src/brain/a2c/mod.rs`, `src/brain/a2c/update.rs`).
- JSON export now serialises the full run report, including both episode records and A2C update records (`src/analytics/exporters/json.rs`).
- Markdown export now includes performance summaries, reward-dynamics summaries including time-penalty terms, crash-location summaries, chunked behaviour summaries, chunked reward-decomposition tables, recent A2C update metrics, and latest layer-health tables (`src/analytics/exporters/markdown.rs`).
- Export triggering on shutdown is now compile-correct and runs from `Last` using Bevy 0.18 `AppExit` messages (`src/analytics/plugin.rs`).
- Episode action statistics are now snapshotted on the fixed tick after episode finalisation, avoiding contamination from the next episode before `Update` tracking runs (`src/analytics/plugin.rs`, `src/analytics/trackers/episode.rs`).

## Implemented Outputs / Artifacts (if applicable)

- `EpisodeTracker` resource containing run-level episode and A2C update records (`src/analytics/trackers/episode.rs`).
- Intended output files under `reports/run_<unix_timestamp>.json` and `reports/run_<unix_timestamp>.md` (`src/analytics/plugin.rs`).

## In Progress / Partially Implemented

- Export triggering is implemented as an on-exit path only; there is no periodic checkpointing, manual export command, or crash-safe flush path.
- The tracked schema now includes first-wave learning-health signals, but it still lacks run metadata such as seed, config snapshot, git revision, and track identity.
- Crash location reporting is currently coarse and summary-oriented rather than a full heatmap representation.
- The current reports are still descriptive rather than diagnostic-by-default; they expose health metrics but do not yet flag failure conditions automatically.
- The latest reward-dynamics reports exposed a live environment bug rather than a reporting bug: `progress_reward_sum` is currently zero across episodes because the best-progress reward term is being computed after best-progress state is already advanced.

## Planned / Missing / To Be Changed

- There is no experiment/run metadata such as seed, config values, git revision, active mode, or track identity in the exported files.
- There is no directory/index policy for multiple runs beyond timestamped filenames.
- There is no validation ensuring episode records are recorded exactly once across all end-reason paths.
- There is no offline comparison tooling beyond one Markdown summary report.
- The current implementation covers only the first high-value slice of analytics; it still lacks richer visitation-diversity metrics, TD-error distributions, run-to-run comparison tooling, and track-sector diagnostics.
- A task-specific driving diagnostics layer is still desirable because the same reward can arise from very different driving pathologies.

## Notes / Design Considerations (optional)

- Proposed analytics taxonomy for future implementation:
  - Performance health: episode return distribution, progress distribution, time-to-crash distribution, lap completion rate, crash locations, progress-at-crash, stagnation rate, and recovery-after-failure trends.
  - Learning health: policy loss, value loss, entropy, advantage mean/std, return versus critic prediction, explained variance, TD error distribution, policy drift/KL, gradient norms, and update-to-parameter norm ratios.
  - Network health: per-layer weight and bias mean/std/min/max, weight norms, activation mean/std, dead-ReLU fraction, per-neuron activation frequency, dominant-neuron concentration, output saturation, and NaN/Inf detection.
  - Exploration and policy-distribution health: steering/throttle mean and std, fraction of clamped actions, steering sign balance, throttle-on percentage, action smoothness, action autocorrelation, and state-visitation diversity.
  - Driving diagnostics: heading-error distribution, centreline-distance distribution, speed distribution, speed-at-crash, steering oscillation, off-track entry angle, crash sensor profile, and sector-based performance.
- These categories should stay separate in both naming and storage. A high reward with collapsed entropy, negative explained variance, or high dead-unit fraction should be treated as suspicious rather than successful.
- The first recommended expansion is a minimal high-value set rather than a huge dashboard:
  - return mean/std over rolling windows
  - progress mean/std/median/p90
  - lap completion rate
  - crash count plus crash location summary
  - policy entropy
  - value loss
  - explained variance
  - gradient norm per layer
  - weight norm per layer
  - dead-ReLU fraction per hidden layer
  - steering/throttle mean and std
  - fraction of clamped actions
- Storage cadence should be explicit:
  - per-episode for behavioural outcomes
  - per-update for A2C optimisation and network-health signals
  - per-run summaries for Markdown reports and experiment comparison
- Both raw and summarised data are needed. Raw data captures instability and spikes; summaries capture trends. One without the other is misleading.
- Failure signatures should be treated as first-class analytics outputs, not ad hoc interpretation:
  - entropy collapsing too early suggests premature convergence
  - negative or near-zero explained variance suggests critic failure
  - dead-unit fraction becoming large suggests representation collapse risk
  - gradient spikes suggest unstable updates
  - flat median progress with rare high-max episodes suggests lucky outliers rather than stable learning
  - low visitation diversity with fast apparent convergence suggests policy collapse rather than mastery
- A useful recent example of analytics value in this repository: the reward-decomposition tables showed `progress_reward_sum == 0.0` and `time_penalty_sum` carrying the full pre-terminal return, which revealed that the reward implementation itself was wrong even though the tracker/exporters were behaving consistently.

- This subsystem should stay focused on data capture and export orchestration, not on deriving environment truth.
- Tracker logic should remain append-only and idempotent per episode; duplicate records would invalidate chunk metrics quickly.
- Exporters should consume stable tracker records and avoid querying live game state directly.

## Discarded / Obsolete / No Longer Relevant

- The previous context folder had no canonical analytics document because this subsystem did not exist; that is now obsolete.
