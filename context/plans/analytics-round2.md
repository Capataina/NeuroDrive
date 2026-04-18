# Analytics Enhancements for Round-2 Measurement

## Purpose

Add the diagnostic signals needed to discriminate "critic target-scaling fix worked" from "fix failed" in the post-round-1-changes training run. The existing analytics (sections 1–10 of the Markdown report) are strong on aggregate outcomes (reward, speed, crash distribution) but weak on the specific signals round 2 research needs:

- Did critic saturation actually fall?
- Did the critic start distinguishing dangerous states from safe ones?
- Did anticipatory throttle-off emerge before crashes?
- Is the fleet converging, or is one car still carrying the average?
- Are PopArt µ/σ tracking returns as returns grow?
- Is target-KL early stop firing (= actor LR is too hot for the new reward scale)?

This plan ships the **analytics first**, in its own commit, so the post-change run captures the new diagnostics from episode 1. Shipping analytics alongside the fixes would mean losing the pre-change baseline on the new fields; shipping analytics later would mean the first post-change run misses them entirely.

## What to add

### 1. Pre-crash forensics (new markdown section)

For every crash episode, a 30-tick window (0.5 s at 60 Hz) before the terminal tick. Data comes from the existing `TickTraceRecord` vec — nothing new captured, just analysed differently on export.

For each crash window, compute:

- Mean throttle in `[t-30, t-10]` vs `[t-10, t]` — did the policy release throttle as the wall approached?
- Throttle-release latency: first tick where throttle dropped below 0.5 before crash (`None` if never).
- Min ray distance trajectory: did the rays show the wall approaching?
- Critic value trajectory: did the value drop before the crash?
- Distance-to-wall at t (= min ray at crash).

Aggregate distributions across all crashes — histograms for throttle-release latency, distance-to-wall at crash, value-drop magnitude.

**Markdown output**: new section `## 11. Pre-Crash Forensics` with:

- Throttle-release latency distribution (ASCII bar histogram or sparkline over crashes).
- Distance-to-wall at crash (median, p10, p90).
- "Critic value dropped before crash" % — tick where value fell by > 20% below episode mean, vs crash tick.
- Top-level takeaway: "anticipation emerging" vs "reactive-only crashes".

### 2. Layer health timeseries (new markdown section)

Existing `PpoUpdateRecord.layer_health` has a snapshot per update. Already stored; just need to render the **evolution** rather than only the latest.

Per layer:

- Weight L2 sparkline across all updates.
- Gradient L2 sparkline across all updates.
- Saturation % sparkline across all updates.

For `c_fc2` specifically — the headline diagnostic — render a larger inline chart, not just sparkline.

**Markdown output**: new section `## 12. Layer Health Over Training` with:

- Saturation timeseries for each activated layer (sparkline per layer).
- Weight-norm timeseries (sparkline per layer).
- Gradient-norm timeseries.
- Takeaway auto-generated: "saturation is {falling / flat / rising}; {c_fc2 specific: …}".

### 3. PopArt µ/σ tracker (new markdown section)

PopArt state needs to be logged per update so the report can show whether the normaliser is tracking returns as they grow.

Per update, capture:

- `value_norm.mu` and `value_norm.sigma` after the update.
- Batch mean/std of `returns` for that update (before EMA blend).
- Implied coverage: `(batch_mean - mu) / sigma` — how far the current batch is from the running stats in std units.

**Markdown output**: new section `## 13. Value Target Scale Tracker (PopArt)` with:

- µ sparkline + σ sparkline.
- Per-update return-batch-mean vs PopArt µ (two aligned sparklines for visual comparison).
- Takeaway: "PopArt is tracking returns" vs "lag visible — β may be too low".

### 4. Critic prediction-error distribution (new markdown section)

Per update, bucket `value - return` errors into a histogram. Already have the data — just need to aggregate it during `ppo_finish_epoch` and export.

Actually this is cheaper: per chunk, bucket `(predicted_value - actual_return) / sigma_of_returns` (standardised residual) into 7 bins: < -2σ, -2σ to -1σ, -1σ to -0.5σ, -0.5σ to +0.5σ, +0.5σ to +1σ, +1σ to +2σ, > +2σ.

**Markdown output**: new section `## 14. Critic Prediction Quality` with:

- Standardised residual distribution (ASCII bar histogram).
- Per-chunk explained variance (already aggregated; just expose per-chunk rather than latest).
- Per-sector value prediction vs actual return (if fidelity allows).
- Takeaway: "critic is well-calibrated" vs "critic is biased {low/high}" vs "critic is undercapacity".

### 5. Per-car fleet variance (new markdown section)

Aggregate `EpisodeRecord` by `env_id`. For each of the 8 cars:

- Max progress ever reached.
- Mean life (s).
- Crash type distribution.
- Mean reward per episode.

**Markdown output**: new section `## 15. Fleet Variance` with:

- Per-car table (8 rows × 4 columns).
- Max-progress distribution (quick bar chart across cars).
- Takeaway: "{all cars converging / one car leading / fleet diverged}".

### 6. Target-KL early-stop tracker (addition to existing section)

If target-KL early stop is enabled (it is in the plan), track per-update:

- Did this update early-stop? (bool)
- How many epochs actually ran?

Add to section 9 (Training Health) a new row: "Epochs completed avg" and "Early stops" count.

### 7. Return distribution per update (addition to existing analytics)

For each `PpoUpdateRecord`, store min/mean/max/std of returns seen by that update. This is essentially free to compute during `ppo_finish_epoch` and provides a direct answer to "are returns growing?" that round 2 research needs.

Add to section 9: a "Return scale" sparkline showing return-std over updates.

## Data capture changes

New fields on existing records:

- `PpoUpdateRecord`:
  - `return_min: f32`, `return_mean: f32`, `return_max: f32`, `return_std: f32`
  - `value_norm_mu: f32`, `value_norm_sigma: f32`
  - `epochs_completed: u32` (distinct from the configured `ppo_epochs`)
  - `early_stopped: bool`
- `EpisodeRecord` — no new fields; per-car aggregation is done at export time.
- `TickTraceRecord` — no new fields; pre-crash analysis reads existing fields.

Critic prediction residual bucketing is computed at **export time** from existing data, not stored.

## Implementation surfaces

Files to modify:

- `src/analytics/models.rs` — extend `PpoUpdateRecord`.
- `src/analytics/trackers/episode.rs` — populate new `PpoUpdateRecord` fields from stats.
- `src/brain/ppo/update.rs` — compute `return_min/mean/max/std` once during `ppo_finish_epoch`; read `value_norm` state (once PopArt is in) and pass to stats. For analytics-first commit, the PopArt fields are populated with `0.0, 1.0` defaults until PopArt lands.
- `src/brain/ppo/mod.rs` — extend `PpoTrainingStats` with the same new fields; wire epochs-completed / early-stopped into stats (dependency on target-KL landing; for analytics-first commit, default to `ppo_epochs` / false).
- `src/analytics/exporters/markdown.rs` — add sections 11–15.
- `src/analytics/metrics/` — new helper module for pre-crash window analysis, standardised-residual bucketing.

## Scope bounds

What's **out of scope**:

- Per-car trajectory replay (the analytics TUI plan covers this separately).
- Sector-level value prediction mapping (too much data; not directly useful for round 2).
- Action-jitter-by-context breakdown (nice-to-have, not on critical path).
- Saturation distribution within a layer (per-neuron) — only per-layer aggregate.

## Success criterion

Plan is complete when:

- [x] `cargo check` clean, `cargo test` passes
- [x] A dry-run export (< 10 episodes) produces a valid Markdown report with all five new sections rendering
- [x] No regression in the existing 10 sections' output
- [x] `PpoUpdateRecord` changes are backwards-compatible for JSON deserialisation (existing JSON exports can still be read; missing fields get default values)

## Commit plan

Single commit titled `analytics: round-2 diagnostics (pre-crash forensics, layer-health timeseries, PopArt tracker, prediction-error distribution, fleet variance)`.
