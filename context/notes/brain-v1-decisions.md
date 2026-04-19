# Brain-Inspired v1 — Implementation Decision Log

Captures the choices made while implementing Milestone 6 (brain-inspired v1).
This file is a companion to `brain-v1-design.md` (which holds the v1 design
rationale) and `baseline-to-brain-inspired.md` (which holds the transition
framing). When this v1 gets revised, refer to this note to understand what
was deliberate vs what was accidental.

## Summary

Six staged commits (`6237aa7` → `c64ce9b`) on branch `master`:

| Stage | Commit | Scope | Tests |
|---|---|---|---|
| S1 | `6237aa7` | Plumbing: `AgentMode` → per-car `Controller` markers + `TrainerLayout`, new `src/brain/inspired/` module, forward pass, no learning | 3 |
| S2 | `e4ee766` | Three-factor plasticity with eligibility traces; raw-reward modulator | 4 |
| S3 | `31625d0` | Homeostasis: synaptic scaling + intrinsic excitability | 3 |
| S4 | `ccd1599` | Structural plasticity: utility, replacement, plateau-triggered neurogenesis, prune/sprout | 6 |
| S5 | `f71d09d` | Analytics integration: `BrainUpdateRecord` through to JSON + three markdown sections | 2 |
| S6 | `c64ce9b` | Side-by-side mode: controller-tagged `EpisodeRecord`, Fleet Comparison markdown section, palette distinction | 3 |

Final test count: 101 unit + 21 brain + 6 gemm + 5 ppo = **133 green** across default, `force-scalar`, and release builds.

## Decisions Made During Implementation

Each decision names the fork, the choice taken, and the alternative that
was rejected. Ordering roughly follows the stage sequence.

### D1 — Global `AgentMode` is replaced, not extended

**Chosen:** Remove `AgentMode` enum entirely. Introduce three ZST marker
components (`PpoCar`, `BrainCar`, `KeyboardCar`) and a `Controller` enum on
each car. Systems filter via `With<PpoCar>` query filters.

**Rejected:** Add a fourth `AgentMode::BrainInspired` variant, keep the
global toggle.

**Why:** The global toggle cannot express side-by-side comparison (8 PPO +
8 brain in the same sim). ZST markers give compile-time query isolation —
cross-controller contamination is impossible by construction rather than
by convention.

### D2 — `TrainerLayout` carries the fleet shape; `num_envs` stays synced

**Chosen:** `TrainerLayout::{Keyboard, AllPpo{count}, AllBrain{count}, SideBySide{ppo, brain}}` as a field on `TrainerConfig`. `num_envs` stays as a field synced from `layout.total_cars()`.

**Rejected:** Remove `num_envs` entirely and expose only `layout.total_cars()`.

**Why:** `num_envs` is referenced in four places (game/plugin spawn loop, analytics context, analytics plugin metadata, debug leaderboard). Keeping it synced lets S1 ship without touching any of those call sites; `TrainerConfig::set_layout` updates both atomically.

### D3 — F4 toggle despawns + respawns instead of swapping markers

**Chosen:** The F4 handler (`cycle_trainer_layout_system`) despawns every `Car` entity, resets PPO and brain state, and spawns a fresh fleet per the new layout.

**Rejected:** Mutate controller markers on existing entities in-place.

**Why:** Marker swaps would leak state — existing eligibility traces, partial rollout buffers, activation histories — into the next run. Respawning is the cheap clean slate.

### D4 — Keyboard layout spawns 1 car, not 8

**Chosen:** `TrainerLayout::Keyboard` emits exactly one car with a `KeyboardCar` marker.

**Rejected:** Spawn all 8 cars but mark only car 0 as keyboard.

**Why:** With only one car responsive, the others just sit with default zero actions — ghosts on the track. In practice nobody trains in keyboard mode, and the one-car layout matches the "I'm driving" mental model.

### D5 — Side-by-side defaults: 8 + 8 = 16 cars

**Chosen:** `SideBySide { ppo: 8, brain: 8 }` in the F4 cycle and as the `TrainerLayout::next()` transition from `AllBrain`.

**Rejected:** 4 + 4 = 8 (match single-layout data rate); 12 + 12 = 24 (more data per learner).

**Why:** The M4 performance overhaul gives ~95% frame-budget headroom at 8 cars. 16 cars lands at ~9% budget, well under the 16.67 ms frame ceiling. Each learner gets the same 8-car data rate as in its single-layout run, so comparison is apples-to-apples.

### D6 — One shared graph, per-car eligibility traces + activations

**Chosen:** `BrainGraph` is a single resource. Synapses carry `eligibility: Vec<f32>` indexed by car. `NeuronActivations` is a per-car Component. Per-tick plasticity accumulates Δw from 8 embodiments into shared weights.

**Rejected:** 8 independent brains (loses "one brain, one lifetime" framing); summing eligibility across cars (loses per-car credit assignment).

**Why:** Matches the README's explicit "one persistent brain" framing and the biological picture. Per-car eligibility keeps reward credit localised to the right embodiment; shared weights provide the 8× data rate.

### D7 — `PolicyOutput` fields are repurposed, not split

**Chosen:** Same per-car `PolicyOutput` component for PPO and brain. `value_prediction` carries the per-car modulator M in brain mode; `*_mean` hold raw output-neuron activations; `*_std` = 0.0. Analytics uses controller markers to discriminate semantics.

**Rejected:** New `BrainOutput` component alongside `PolicyOutput`.

**Why:** Zero disruption to analytics trace capture. The semantic drift is documented in `src/brain/types.rs`. If it ever becomes misleading, splitting is a one-commit refactor.

### D8 — Slot-stable graph storage with free-lists

**Chosen:** `BrainGraph.neurons: Vec<Neuron>` with `alive: bool` on each. Dead slots sit in `free_neuron_slots: Vec<NeuronId>` and are recycled by neurogenesis. Same pattern for synapses.

**Rejected:** Compact the `Vec` on every deletion (invalidates `NeuronId`s); use a `HashMap<NeuronId, Neuron>` (worse cache locality).

**Why:** Structural plasticity would thrash a compacted `Vec` — every neuron death requires O(n) shift and every stored `NeuronId` downstream requires remapping. Slot-stable storage keeps `NeuronId` eternally valid and converts "dead neuron" to an O(1) flag flip.

### D9 — Forward pass reads `prev`, writes `curr`

**Chosen:** At the start of each tick, `prev ← curr`. Input neurons are set from the observation directly (no tanh). Non-input neurons compute `z = bias + Σ prev[source] × weight` and write `curr = tanh(z)`.

**Rejected:** Topological sort + within-tick propagation (would forbid cyclic connections); iterative settling (expensive, unclear convergence).

**Why:** One-step propagation makes cyclic connections trivially well-defined, forward-pass order-independent, and biologically defensible (real neurons have non-zero integration time constants). See `brain-v1-design.md` §Forward Pass.

### D10 — Pre/post for eligibility: `pre = prev[source]`, `post = curr[target]`

**Chosen:** At plasticity time (after forward pass), `prev` holds tick t-1 activations and `curr` holds tick t activations. Eligibility uses `pre_i = prev[source]` and `post_j = curr[target]`.

**Rejected:** `pre = curr[source]` (would correlate simultaneous activations rather than the causal "source fires, then target fires" pattern).

**Why:** The biological interpretation of three-factor plasticity requires causal (pre-before-post) correlation. This ordering gives STDP-like semantics without sub-tick scheduling.

### D11 — Sum per-car weight updates rather than average

**Chosen:** `sum_per_car_updates = true` by default. Eight cars' Δw contributions sum into shared weights each tick.

**Rejected:** Average across cars (safer but slower data rate).

**Why:** Matches the "8× data into one brain" intent of the shared-graph choice. `sum_per_car_updates = false` is exposed as a config flag for ablation if summing destabilises.

### D12 — Config dials flagged RESEARCH-ANCHORED vs TUNE

Every dial in `BrainInspiredConfig` is explicitly annotated. RESEARCH-ANCHORED dials (λ = 0.992, η_u = 0.99, maturity = 1000, etc.) are grounded in specific research papers or biological constants. TUNE dials (η, ρ, structural cadence, plateau window, etc.) are explicit tuning surface without firm guidance — starting values chosen for plausibility, expected to be swept empirically.

### D13 — Three `enable_*` flags for ablation without recompiling

`enable_plasticity`, `enable_homeostasis`, `enable_structural` all default true. Turning any off at runtime lets us isolate "does this mechanism matter" questions without touching the code.

### D14 — Intrinsic homeostat runs every tick; synaptic scaling on cadence

**Chosen:** `update_intrinsic_homeostat` every tick (mean-rate EMA, age advancement, bias nudge). `apply_synaptic_scaling` every `structural_cadence` ticks (default 128).

**Rejected:** Both on cadence (bias correction is slow per-tick; batching it would mean `age_ticks` lags). Both every tick (synaptic scaling is a non-trivial scan).

**Why:** Per-tick mean-rate tracking + age advancement needs per-tick frequency to stay coherent with `maturity_ticks` (otherwise CBP's maturity gate is meaningless). Synaptic scaling is a whole-graph scan — running it every 128 ticks is plenty given biological scaling time constants are on hours.

### D15 — Replacement zeroes outgoing weights; resamples incoming

**Chosen:** CBP Rank 1 protocol: zero the dead neuron's outgoing weights (behaviour-preserving at the moment of replacement), resample its incoming weights from the Gaussian init distribution, zero eligibility across all cars.

**Rejected:** Re-initialise bidirectionally (would cause immediate downstream shock); leave dead neuron silent for K ticks before rewiring (added complexity without biological justification).

**Why:** Outgoing-zero is the invariant that makes replacement non-disruptive. Plasticity on downstream edges will rebuild useful structure once the resampled incoming weights find signal.

### D16 — Plateau window is cleared after neurogenesis triggers

**Chosen:** When `detect_plateau` returns true and we grow a neuron, clear `brain.reward_window`. Next plateau must fill afresh.

**Rejected:** Leave the window in place (would re-trigger neurogenesis every cadence pass until reward changed enough).

**Why:** Without clearing, a long genuine plateau would fire neurogenesis every 128 ticks until the brain exploded. Clearing gates growth to once-per-plateau-regime.

### D17 — `BrainUpdateRecord` written on structural cadence

**Chosen:** One analytics record per 128 ticks, not one per episode. Records are timestamped by `tick_start..tick_end` range and carry per-window counts (replacements, neurogenesis, prune, sprout) which are reset after each flush.

**Rejected:** One record per completed episode (analytics granularity would be too coarse to see structural dynamics).

**Why:** Structural events are rare at the tick level but frequent over a run. Per-cadence records let sparklines show trajectory; per-window counts (rather than cumulative) let downstream readers see event density directly.

### D18 — `brain_records` uses `#[serde(default)]` for back-compat

**Chosen:** `EpisodeTracker.brain_records` and `CompactRunExport.brain_records` both carry `#[serde(default)]`. Pre-M6 JSON files with no such field deserialise cleanly with empty `brain_records`.

**Rejected:** Versioned schema; migration scripts.

**Why:** Runs before M6 were recorded without brain data. Default-empty preserves existing analytics tooling.

### D19 — Markdown section 19 auto-detects side-by-side from `controller` tags

**Chosen:** Fleet Comparison section renders when `tracker.episodes` contains at least one "Ppo" and at least one "Brain" controller entry.

**Rejected:** Read `run_metadata.layout` and render only when "SideBySide".

**Why:** The controller-tag detection also works for mixed runs that span multiple F4 cycles (e.g., someone ran AllPpo for 500 episodes, then cycled to SideBySide for another 1000). Layout-based gating would miss those.

### D20 — No explicit cross-contamination tests

**Chosen:** Invariant is enforced structurally: `With<PpoCar>` filter means the PPO rollout buffer's `env_ids` column is populated from a query that excludes brain cars. Rust's type system makes cross-contamination impossible.

**Rejected:** Write tests that run a brief SideBySide simulation, iterate the PPO buffer, and assert no brain env_ids appear.

**Why:** Such a test would verify the compiler is doing its job rather than catch a real failure mode. The palette distinctness test + TrainerLayout sum test are the places the partitioning logic could actually go wrong; both are covered.

### D21 — Default launch layout is AllBrain, not AllPpo

**Chosen:** `TrainerLayout::default()` returns `AllBrain { count: 8 }`. First boot puts the brain-inspired learner on screen.

**Rejected:** Keep `AllPpo` as default (matches pre-M6 behaviour — most recent working run).

**Why:** User-directed. The project's stated thesis is the brain-inspired learner; PPO is the diagnostic baseline. The default launch should reflect the thesis. PPO-only and side-by-side layouts are one and two F4 presses away.

### D22 — F4 cycle excludes Keyboard

**Chosen:** F4 cycles through the three learning modes only: `AllBrain → SideBySide → AllPpo → AllBrain`. Keyboard layout exists and can be set programmatically, but the next press from Keyboard drops back into the cycle at `AllBrain`.

**Rejected:** Four-way cycle including Keyboard.

**Why:** Keyboard is a manual-intervention escape hatch for cases where the learners misbehave and the user needs to take the wheel. It is not a routine step in the learning-mode rotation, and putting it in the cycle means users have to tap past it to get back to learners. If keyboard ever needs a routine binding, a dedicated key (e.g. `F7`) is cleaner than cramming it into F4's cycle.

### D23 — Reports get layout slug in filename; PPO sections suppressed in brain-only runs

**Chosen:**
- Markdown report filename: `run_<timestamp>_<slug>.md` where slug is one of `brain` / `side` / `ppo` / `keyboard`. JSON companion files follow the same pattern.
- Sections 9 (Training Health), 12 (Layer Health), 13 (Value Target Scale), 14 (Critic Prediction Quality) — all PPO-centric — skip **entirely** (no header at all) when `tracker.ppo_updates.is_empty()`. No more "No PPO updates recorded" stubs in brain-only reports.
- Brain sections 16, 17, 18 already skip when `tracker.brain_records.is_empty()`.
- Fleet Comparison section 19 skips unless the run had at least one PPO-tagged and one brain-tagged episode.

**Rejected:** Keep placeholder stubs for PPO sections in brain-only reports; name files by timestamp only.

**Why:** User-directed. A brain-only run should produce a brain-only report — the file's name, content, and verdicts should all reflect what actually ran. The slug also lets `ls reports/analytics/` communicate which runs to look at for which comparison.

## Tests by Stage

21 brain-pipeline tests total:

| Stage | Tests |
|---|---|
| S1 | seed_graph_has_correct_io_counts · forward_pass_is_deterministic_with_fixed_seed · forward_pass_output_is_in_action_range |
| S2 | eligibility_trace_decays_to_zero_with_m_zero · weight_update_magnitude_scales_with_eta · plasticity_preserves_no_nan_no_inf_over_10k_ticks · terminal_episode_zeros_eligibility |
| S3 | synaptic_scaling_brings_sum_to_target · intrinsic_homeostat_moves_bias_toward_target_band · homeostasis_idempotent_at_steady_state |
| S4 | replacement_selects_lowest_utility_mature_neurons · replacement_zeros_outgoing_weights · replacement_preserves_graph_connectivity_invariant · plateau_detector_triggers_on_flat_reward_window · neurogenesis_grows_neuron_count · utility_tick_updates_toward_contribution |
| S5 | brain_update_record_serializes_and_deserializes · compact_run_export_skips_brain_records_when_absent |
| S6 | trainer_layout_total_cars_is_sum_of_controllers · trainer_layout_cycle_visits_all_four_variants · warm_and_cool_palettes_are_visually_distinct |

## What Still Needs Doing (Not in M6 Scope)

- **Context drift pass.** `context/architecture.md` and `context/systems/*`
  still describe the pre-M6 global-`AgentMode` world in several places.
  Recommend running `upkeep-context` as a dedicated pass when convenient.
- **HUD column split in side-by-side.** S6 added the analytics column split
  only at the report level. The live HUD still shows single-column PPO
  stats. Not gated on M6 acceptance; minor polish.
- **First real SideBySide training run** to validate the M6 success bar:
  brain fleet reward trend rising over ~2000 episodes, directional bias
  signature observable, at least one replacement + one neurogenesis event.
  Deferred because it requires actual wall-clock training time.
- **Tuning sweep over η, ρ, structural cadence.** The TUNE dials need
  empirical data before they can be pinned.

## Failure-Mode Discipline (Per the M6 Plan)

Named here so the next session doesn't miss them:

1. **Biology-first under pressure.** If the brain does not learn on the
   first SideBySide run, resist the pull to import PPO's critic as the
   modulator (Option D). That is explicitly M8, not a v1 rescue. Diagnose
   in biology terms first.
2. **Procedure vs outcome.** "Weights change" is not "the brain learned."
   The acceptance bar is directional bias + survival improvement in
   side-by-side comparison, not internal dynamics alone.
3. **Honest negative result is a valid M6 ship.** Shipping v1 with "visible
   plasticity signature, no loop completion yet" is the plan's stated
   success bar. The project's contribution is the falsification, not a
   success narrative.
