# System — Brain-Inspired Learner (v1)

## Scope / Purpose

- Implement the project's stated-intent learner: a sparse directed graph of rate-coded tanh neurons trained by local plasticity, no backprop, no external ML libraries.
- Run additively alongside PPO rather than replacing it — PPO stays permanently live as the diagnostic baseline.
- Support side-by-side comparison (PPO fleet + brain fleet in the same simulation) so brain learning is measured against a known-working learner under identical conditions.
- Deliver the substrate of the biology-first arc (M6–M9 plus the Long-Term Plan).

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/brain/inspired/mod.rs` | `BrainInspiredPlugin`, `BrainBrain` resource, `BrainRunningStats`, `BrainUpdateRecord`, `BrainTrainingStats`, the per-car `NeuronActivations` Component re-export, the per-tick `brain_act_all_cars_system` and per-tick `brain_learn_all_cars_system`, the `build_brain_update_record` cadence snapshot helper | Environment truth, observation production, PPO machinery |
| `src/brain/inspired/config.rs` | `BrainInspiredConfig` — every dial in one place with `RESEARCH-ANCHORED` or `TUNE` annotations, plus three `enable_*` ablation flags | Learning-rule mechanics |
| `src/brain/inspired/graph.rs` | `NeuronId` / `SynapseId`, `NeuronRole` enum (`Input(u8)` / `Hidden` / `Output(u8)`), `Neuron`, `Synapse`, `BrainGraph` with slot-stable storage + free-lists; seed-graph construction | Learning dynamics, homeostasis, structural plasticity |
| `src/brain/inspired/forward.rs` | `NeuronActivations` per-car Component, pure `forward_tick` function (prev-tick reads, one-step propagation) | Eligibility, weight updates, structural change |
| `src/brain/inspired/plasticity.rs` | `CarLearnSample<'a>`, `PlasticitySample` diagnostic struct, `apply_plasticity_tick`, `sample_plasticity_health` | Forward pass, graph structure |
| `src/brain/inspired/homeostasis.rs` | `apply_synaptic_scaling`, `update_intrinsic_homeostat` | Plasticity, forward pass |
| `src/brain/inspired/structural.rs` | `update_utility_tick`, `detect_plateau`, `grow_hidden_neuron`, `replace_low_utility`, `prune_synapses`, `sprout_synapses` | Plasticity, homeostasis |
| `src/brain/types.rs` | `Controller` enum + `PpoCar` / `BrainCar` / `KeyboardCar` ZST marker components (shared across PPO and brain-inspired); `PolicyOutput` with controller-dependent field semantics | Algorithm logic |
| `src/brain/plugin.rs` | `cycle_trainer_layout_system` — F4 handler that despawns + respawns cars on layout change and resets both learners' state | Per-tick learner work |

## Current Implemented Reality

### The v1 Learning Rule

Three-factor plasticity with per-car eligibility traces, applied every tick to every live synapse:

```text
e_ij[c] ← λ · e_ij[c] + pre_i · post_j        (eligibility update, per car c)
Δw_ij  += η · M_c · e_ij[c]                   (weight-update contribution, per car)
w_ij   += Σ_c Δw_ij                           (shared weights, 8 updates per tick summed)
```

- `λ` is the eligibility decay per tick (default 0.992, research-anchored: τ_e ≈ 2s at 60Hz, matching γ=0.995's ~3.3s credit horizon).
- `η` is the synaptic learning rate (default 1e-3, `TUNE` — research does not pin a value).
- `M_c` is car c's raw per-tick reward from `EpisodeState.tick.reward` — Option C from the seven-paper research round. No critic in v1.
- `pre_i = NeuronActivations.prev[source]` (previous-tick source activation — the buffer rotation at the start of `forward_tick` makes this "t-1's activation").
- `post_j = NeuronActivations.curr[target]` (this-tick target activation).
- On episode terminal, that car's eligibility is zeroed across every synapse so stale correlations do not bleed across resets.

The `sum_per_car_updates` config flag defaults `true` (8× data rate into one shared brain). Setting `false` averages per-car Δw contributions — safety fallback if summing destabilises.

### Graph Topology

- **Sparse directed graph, not layered.** One shared `BrainGraph` resource on `BrainBrain`; all cars marked `BrainCar` read from and contribute updates to this single graph.
- **Slot-stable storage.** `Vec<Neuron>` and `Vec<Synapse>` with `alive: bool` flags. Dead slots accumulate in `free_neuron_slots` / `free_synapse_slots` and are recycled by neurogenesis / sprouting. `NeuronId` / `SynapseId` are stable for the lifetime of a node or edge — structural events mutate the flag, not the Vec geometry.
- **Seed graph** (built on `BrainGraph::seed` at brain construction / F4 reset):
  - 43 `NeuronRole::Input(i)` neurons bound to observation dimensions.
  - 15 `NeuronRole::Hidden` neurons (config `initial_hidden_neurons`).
  - 2 `NeuronRole::Output(i)` neurons bound to steering (idx 0) and throttle (idx 1).
  - Roughly 10% density of allowable edges (input→hidden, hidden→hidden excluding self-loops, hidden→output, input→output; output→anything forbidden; input-target forbidden).
  - Weights drawn from `Normal(0, initial_weight_sigma)` where σ defaults to 0.1.
- **Cyclic connections allowed.** Correctness is preserved by the one-step propagation rule: each tick reads `prev` (t-1 activations) and writes `curr` — order-independent within a tick.

### Forward Pass (per car, per tick)

`forward_tick(graph, &mut activations, observation) -> (steering, throttle)`:

1. Rotate: `prev ← curr`.
2. Input neurons: `curr[input_id] = observation[dim_idx]` directly (no tanh — inputs carry the raw normalised observation vector).
3. Every live non-input neuron: `z = bias + Σ_incoming (prev[source] · weight)`, then `curr = tanh(z)`.
4. Read outputs: `steering = curr[output_neurons[0]]` (already in [-1, 1] from tanh); `throttle = 0.5 * (curr[output_neurons[1]] + 1.0)` (remapped to [0, 1] to match the PPO action-space contract).

`NeuronActivations::ensure_sized` lazily grows `prev` / `curr` when the graph grows (S4 structural plasticity can increase `graph.neurons.len()` between ticks).

### Homeostasis

Two biological mechanisms, running on different cadences:

**Intrinsic excitability (per tick)** — `update_intrinsic_homeostat`:
- For each live non-input neuron j: `mean_rate_j ← (1-α) · mean_rate_j + α · mean_c(|curr[j][c]|)` with α = 0.01.
- If `mean_rate_j < lo_band`: `bias_j += intrinsic_bias_rate` (default 1e-4).
- If `mean_rate_j > hi_band`: `bias_j -= intrinsic_bias_rate`.
- `age_ticks` advances by 1 here (not in the structural pass) so CBP's maturity gate stays honest.
- Input neurons are skipped — adjusting their bias would corrupt the I/O contract.

**Synaptic scaling (every `structural_cadence` ticks, default 128)** — `apply_synaptic_scaling`:
- For each live non-input neuron j: `s = Σ_incoming |w_ij|`.
- `factor = clamp(1 + rate · (target - s) / target, 0.5, 2.0)` with default `target = 2.0`, `rate = 0.05`.
- Multiply all incoming weights by factor.
- The clamp prevents catastrophic single-pass corrections; repeated passes walk weights back toward target smoothly.

### Structural Plasticity

Continual-backprop-style utility tracking, plus plateau-triggered growth, plus synapse prune/sprout.

**Per-tick utility EMA** — `update_utility_tick`:
```text
u_i ← η_u · u_i + (1 − η_u) · mean_c(|h_i[c]|) · Σ_outgoing |w|
```
Default `η_u = 0.99` (research-anchored, CBP §Rank 1 [CBP-UTIL]). Captures how much a neuron's activation actually influences the network (activation × downstream weight magnitude averaged across cars).

**On the structural cadence (every 128 ticks)** four operations in sequence:

1. **`replace_low_utility`** — picks the `ρ · hidden_count` lowest-utility hidden neurons with `age_ticks ≥ maturity_ticks` (default `ρ = 5e-4`, `maturity_ticks = 1000`). For each:
   - Zero all outgoing weights (behaviour-preserving at the moment of replacement).
   - Resample incoming weights from `Normal(0, initial_weight_sigma)`.
   - Zero eligibility across all cars on both outgoing and incoming synapses.
   - Reset `utility = 0`, `age_ticks = 0`, `mean_rate = 0`, `bias = 0`.

2. **`detect_plateau` + `grow_hidden_neuron`** — if `reward_window` (rolling episode returns) shows mean-of-first-half ≈ mean-of-second-half within `plateau_threshold` (default 0.02), allocate a new hidden neuron wired to ~10 random incoming sources and ~10 random outgoing targets (no self-loops, no input-targets, no output-sources). Clear the window afterwards so the next plateau is measured fresh.

3. **`prune_synapses`** — any live synapse with `|weight| < prune_weight_threshold` (default 0.01) is marked dead; weight + eligibility zeroed; source/target adjacency lists updated; slot returned to `free_synapse_slots`.

4. **`sprout_synapses`** — with probability `sprout_probability` (default 0.10), sample `sprout_candidates_per_event` random (source, target) pairs. For each pair that is unconnected (no live synapse already joins them) and legal (not self-loop, target not input, source not output), create a new synapse with `Normal(0, σ)` weight. Slot reused from `free_synapse_slots` when available.

### Integration with the Runtime

**System scheduling** — `BrainInspiredPlugin` registers:
- `brain_act_all_cars_system` in `SimSet::Input`, after `keyboard_action_input_system` and before `action_smoothing_system` (same placement as `ppo_act_all_cars_system`).
- `brain_learn_all_cars_system` in `SimSet::Measurement`, after `episode_loop_system` and after `build_observation_vector_system` (same placement as `ppo_collect_rewards_all_cars_system`).

Both systems filter cars via `(With<Car>, With<BrainCar>)` query filters. In any layout where no car carries `BrainCar`, the systems iterate zero cars and exit without doing work — there is no `enabled: bool` gate; the marker components are the gate.

**Per-car Component contract:**

| Component | Purpose | Written by | Read by |
|---|---|---|---|
| `NeuronActivations` | Per-car `prev` / `curr` activation buffers (lazy-sized) | `forward_tick` | Next tick's forward pass, plasticity |
| `BrainCar` (ZST marker) | "Brain drives this car" | Car spawn in `spawn_cars_for_layout` | All brain-inspired query filters + analytics |
| `Controller::Brain` | Source-of-truth controller identity | Car spawn | Analytics (episode tagging) |
| `PolicyOutput` | Analytics/HUD surface, semantically repurposed: `value_prediction = per-car modulator M`, `*_mean = raw output-neuron activations`, `*_std = 0.0` | `brain_act_all_cars_system` + `brain_learn_all_cars_system` (M write) | Analytics trace capture, HUD |
| `ActionState.desired` | Written from forward-pass outputs, clamped | `brain_act_all_cars_system` | `action_smoothing_system` then physics |

### Learn-System Structure

`brain_learn_all_cars_system` uses **two disjoint Bevy queries on the same entity set**:

- `read_query`: `(&EnvInstanceId, &EpisodeState, &NeuronActivations)` — supplies the per-car activation slices for plasticity.
- `write_query`: `(&EnvInstanceId, &mut PolicyOutput)` — surfaces per-car M into analytics.

Bevy allows this because the component sets are disjoint. `CarLearnSample`s built from `read_query` are dropped before the write pass so nothing aliases.

The system's internal order (once per tick):
1. Collect samples + push terminal-episode returns onto `brain.reward_window` (cap 1024).
2. Plasticity — `apply_plasticity_tick` into shared `BrainGraph`.
3. Homeostasis — `update_intrinsic_homeostat` every tick; `apply_synaptic_scaling` on cadence; also compute `saturation_fraction` diagnostic.
4. Structural plasticity — `update_utility_tick` every tick; `replace_low_utility` → `detect_plateau`/`grow_hidden_neuron` → `prune_synapses` → `sprout_synapses` on cadence. Uses field-level destructuring (`let BrainBrain { graph, rng, stats: brain_stats, reward_window, .. } = &mut *brain;`) to borrow `graph`, `rng`, `stats`, `reward_window` simultaneously without the aliasing borrow-check error.
5. Expose per-car M via `PolicyOutput.value_prediction`.
6. Refresh plasticity-health scan (`mean_abs_weight`, `mean_abs_eligibility`, `dead_neuron_fraction`).
7. On cadence: build a `BrainUpdateRecord` and push to `BrainTrainingStats.history`; reset per-window counters (`replacement_events`, `neurogenesis_events`, `prune_events`, `sprout_events`, `plasticity_updates`) so the next record captures fresh counts.

### BrainBrain Resource

Mirrors `PpoBrain`'s shape for consistency:

```rust
pub struct BrainBrain {
    pub graph: BrainGraph,
    pub config: BrainInspiredConfig,
    pub rng: StdRng,                  // seeded from config.rng_seed or rand::rng()
    pub tick_counter: u64,            // advances in brain_act_all_cars_system
    pub stats: BrainRunningStats,     // per-window counters, refreshed each tick
    pub reward_window: VecDeque<f32>, // plateau detection (cap 1024)
}
```

`Default::default()` seeds a graph sized for the worst-case default (16 cars — side-by-side uses 8 brain cars, but the per-synapse eligibility Vec length is the number of cars the graph was built for, and the learn system silently skips cars whose index exceeds the eligibility Vec length).

`BrainBrain::reset_to_seed(num_cars)` rebuilds the graph, reseeds the RNG, clears `tick_counter`, `stats`, and `reward_window`. Called by `cycle_trainer_layout_system` whenever F4 changes the layout.

### Side-by-Side Mode

`TrainerLayout::SideBySide { ppo: 8, brain: 8 }` puts 16 cars on the track — 8 PPO (warm palette via `car_colour_warm`) and 8 brain (cool palette via `car_colour_cool`). The partitioning is enforced structurally:

- PPO systems query with `(With<Car>, With<PpoCar>)` — only see PPO cars.
- Brain-inspired systems query with `(With<Car>, With<BrainCar>)` — only see brain cars.
- PPO rollout buffer's `env_ids` column is populated from its query — contains no brain env_ids by construction. No runtime check needed.
- Brain `reward_window` is populated from the brain read_query — contains no PPO returns by construction.

There is no shared state between the two learners other than the environment (track, physics, collision, episode logic) which is layer-neutral.

### Hyperparameters and Ablation Flags

All dials in `BrainInspiredConfig`. Key entries:

| Dial | Default | Status |
|---|---|---|
| `obs_dim` / `action_dim` | 43 / 2 | RESEARCH-ANCHORED (contract) |
| `initial_hidden_neurons` | 15 | RESEARCH-ANCHORED |
| `initial_edge_density` | 0.10 | RESEARCH-ANCHORED |
| `initial_weight_sigma` | 0.1 | RESEARCH-ANCHORED |
| `lambda` (eligibility decay) | 0.992 | RESEARCH-ANCHORED (τ_e ≈ 2s) |
| `eta` (synaptic LR) | 1e-3 | TUNE |
| `sum_per_car_updates` | `true` | TUNE |
| `eta_utility` | 0.99 | RESEARCH-ANCHORED (CBP Rank 1) |
| `maturity_ticks` | 1000 | RESEARCH-ANCHORED (CBP) |
| `replace_fraction` | 5e-4 | TUNE (mid of CBP range) |
| `structural_cadence` | 128 | TUNE |
| `plateau_episode_window` | 50 | TUNE |
| `plateau_threshold` | 0.02 | TUNE |
| `prune_weight_threshold` | 0.01 | TUNE |
| `sprout_probability` | 0.10 | TUNE |
| `sprout_candidates_per_event` | 8 | TUNE |
| `synaptic_scaling_target` | 2.0 | TUNE |
| `synaptic_scaling_rate` | 0.05 | TUNE |
| `intrinsic_rate_band` | (0.10, 0.60) | TUNE |
| `intrinsic_bias_rate` | 1e-4 | TUNE |
| `enable_plasticity` | `true` | ablation flag |
| `enable_homeostasis` | `true` | ablation flag |
| `enable_structural` | `true` | ablation flag |
| `rng_seed` | `None` | determinism hook |

## Key Interfaces / Data Flow

| Interface | Producer | Consumer | Notes |
|-----------|----------|----------|-------|
| `ObservationVector` (43-dim) | agent | `brain_act_all_cars_system` | Same contract PPO consumes; any dim change desynchronises both learners |
| `ActionState.desired` | `brain_act_all_cars_system` | smoothing → physics | Same control boundary as PPO and keyboard |
| `PolicyOutput` | act + learn systems | analytics trace capture, HUD | Field semantics differ from PPO — `value_prediction` is modulator M, not critic estimate |
| `EpisodeState.tick.reward` | game | `brain_learn_all_cars_system` | Raw per-tick reward = modulator M |
| `EpisodeState.tick.end_reason` | game | `brain_learn_all_cars_system` | Terminal → zero per-car eligibility |
| `EpisodeState.last.return_sum` | game | `brain_learn_all_cars_system` | Pushed to reward_window on terminal, drives plateau detector |
| `BrainTrainingStats.history` | `brain_learn_all_cars_system` (cadence flush) | analytics `episode_tracker_system` | Copied into `EpisodeTracker.brain_records` for markdown export |

```text
Brain-mode tick lifecycle (8 embodiments of one graph):
  observation_t (per car) → brain_act_all_cars_system → forward_tick per car
    → ActionState.desired + PolicyOutput.*_mean (per car)
  → smoothing → physics → environment step (all cars)
  → episode_loop_system computes reward_t, end_reason_t (per car)
  → observation_t+1 rebuilt (post-reset if terminal)
  → brain_learn_all_cars_system:
    - push terminal-episode returns to reward_window
    - apply plasticity across all cars (accumulate Δw into shared weights)
    - intrinsic homeostat (per tick); on cadence: synaptic scaling + structural ops
    - write per-car M to PolicyOutput.value_prediction
    - on cadence: flush BrainUpdateRecord into BrainTrainingStats.history
```

## Implemented Outputs / Artifacts

- **Runtime resources:** `BrainBrain`, `BrainTrainingStats`.
- **Runtime components (per car):** `NeuronActivations`, `BrainCar` (ZST marker), `Controller` (enum Component — shared with PPO/Keyboard).
- **Serialisable records:** `BrainUpdateRecord` (snake-cased fields serialise cleanly via serde; carried through JSON and markdown exports).
- **Tests:** 21 integration tests in `tests/brain_inspired_pipeline.rs`:
  - S1: `seed_graph_has_correct_io_counts`, `forward_pass_is_deterministic_with_fixed_seed`, `forward_pass_output_is_in_action_range`.
  - S2: `eligibility_trace_decays_to_zero_with_m_zero`, `weight_update_magnitude_scales_with_eta`, `plasticity_preserves_no_nan_no_inf_over_10k_ticks`, `terminal_episode_zeros_eligibility`.
  - S3: `synaptic_scaling_brings_sum_to_target`, `intrinsic_homeostat_moves_bias_toward_target_band`, `homeostasis_idempotent_at_steady_state`.
  - S4: `replacement_selects_lowest_utility_mature_neurons`, `replacement_zeros_outgoing_weights`, `replacement_preserves_graph_connectivity_invariant`, `plateau_detector_triggers_on_flat_reward_window`, `neurogenesis_grows_neuron_count`, `utility_tick_updates_toward_contribution`.
  - S5: `brain_update_record_serializes_and_deserializes`, `compact_run_export_skips_brain_records_when_absent`.
  - S6: `trainer_layout_total_cars_is_sum_of_controllers`, `trainer_layout_cycle_visits_three_learning_modes`, `warm_and_cool_palettes_are_visually_distinct`.

## Known Issues / Active Risks

- **Acceptance bar not yet validated on a real run.** "Visible learning relative to PPO — brain fleet reward trend rising over ~2000 episodes, directional bias signature, ≥1 replacement and ≥1 neurogenesis event" has not been observed yet. The infrastructure + tests are green; the empirical claim is pending.
- **Per-car eligibility vector sizing is static.** `Synapse.eligibility` is sized once at `BrainGraph::seed` / `reset_to_seed` time from the `num_cars` argument. If the layout changes from 8-car brain to 16-car side-by-side at runtime, `reset_to_seed(16)` re-seeds the graph anyway (new run starts fresh), so this is fine by construction; the guard is the silent skip in `apply_plasticity_tick` when `car >= syn.eligibility.len()`. If a future change keeps the graph across F4 transitions, this guard becomes a silent correctness hazard.
- **Shared-weight contention is unmeasured.** The design assumes 8 concurrent per-car Δw contributions summed into shared weights are stable. Synaptic scaling is the backstop. If it fires every cadence to contain weight growth, the assumption is wrong — consider switching `sum_per_car_updates = false` (average instead).
- **Slot-recycling apoptosis ≠ biology.** Real biology does not reuse dead-neuron slots outside restricted neurogenesis regions. Documented as an acknowledged simplification in README "Known Biological Simplifications". Not a bug; a scope choice.
- **No save/load, no headless.** Same as PPO — becomes more pressing as the first real brain-mode training run is attempted. Without save/load, every F4 cycle or app exit loses all learning.
- **`value_prediction` semantic drift.** `PolicyOutput.value_prediction` means "critic estimate in reward units" for PPO cars and "per-car modulator M (raw tick reward)" for brain cars. Analytics uses the `PpoCar` / `BrainCar` markers to discriminate, but the field name is misleading in brain context. Splitting into separate `BrainOutput` was considered and deferred to keep the analytics pipeline unchanged (decisions note D7).

## Partial / In Progress

- M6 is shipped as a complete substrate, but the project's acceptance bar is empirical (see Risks). The infrastructure is ready for training runs.
- Brain-mode runs produce reports with sections 1–8, 10, 11, 15, 16, 17, 18 (PPO-centric sections 9, 12, 13, 14 suppressed when no PPO updates recorded). Filenames land as `reports/analytics/run_<ts>_brain.md`.
- `PolicyOutput` field repurposing works but reads as slightly odd in brain-only reports — mild semantic drift.

## Planned / Missing / Likely Changes

- **M7 — brain visualisation** (next milestone). Real-time 2D graph render of the brain via Bevy gizmos, F5 toggle, per-car view. Graph-first topology was chosen partly to enable this — the renderer has a direct data source.
- **M8 — plastic value predictor (Option B).** If raw-reward modulator M proves insufficient, the next step is a plasticity-trained sub-graph that learns a TD-error modulator. Deferred until measurement motivates it.
- **M9 — multi-neuromodulator channels.** Dopamine (already present as the reward channel), novelty, salience.
- **Save/load** for the brain graph — same maturity gap as PPO. Becomes urgent if a training run produces a brain we do not want to lose.
- **HUD column split** in side-by-side — analytics column split is done at report level; live HUD still shows single-column PPO stats. Minor polish, flagged in decisions note.

## Durable Notes / Discarded Approaches

- **One brain, eight embodiments (decisions note D6).** All brain cars share a single graph; per-car state lives in `NeuronActivations` (Component) and the eligibility traces on each `Synapse`. Rejected: 8 independent graphs. Reason: matches the README's "one persistent brain" framing; gives 8× data rate into one learner.
- **Slot-stable graph storage with free-lists (D8).** Dead neurons/synapses stay in their Vec slots with `alive = false`. Rejected: compact Vec on deletion (invalidates IDs); HashMap<NeuronId, Neuron> (worse cache locality). Reason: structural plasticity would thrash either alternative.
- **One-step propagation with previous-tick reads (D9).** Each tick reads `prev` (t-1 activations) and writes `curr` — order-independent, cyclic-safe. Rejected: topological sort + within-tick propagation (forbids cycles), iterative settling (expensive). Reason: biologically plausible (real neurons have non-zero integration time constants) and mathematically clean.
- **`pre = prev[source]`, `post = curr[target]` for eligibility (D10).** Biological "source fires, then target fires" causal interpretation requires this ordering. Rejected: `pre = curr[source]` (simultaneous, not causal). Would replicate without the STDP-like semantics.
- **Sum per-car weight updates by default (D11).** 8× data rate into shared weights. Configurable to average via `sum_per_car_updates = false` for safety. Synaptic scaling is the backstop; if it fires every tick, the assumption is wrong and we should switch.
- **Config dials annotated RESEARCH-ANCHORED or TUNE (D12).** Every hyperparameter is explicit about whether research pinned it or it needs empirical tuning. Saves arguing about starting values and surfaces the tuning surface clearly.
- **Three ablation flags (D13) — `enable_plasticity`, `enable_homeostasis`, `enable_structural`.** All default true. Individual off-switches let ablations run without recompiling — "does homeostasis matter" is a one-line config edit.
- **Slot-recycling apoptosis protocol (D15).** Replacement zeroes outgoing weights (behaviour-preserving at the moment of replacement), resamples incoming from init distribution, zeroes per-car eligibility. Rejected: re-init bidirectionally (downstream shock). The outgoing-zero invariant is what makes replacement non-disruptive.
- **Plateau window cleared after neurogenesis fires (D16).** Without clearing, a genuine plateau would re-trigger neurogenesis every cadence pass until the brain explodes. Clearing gates growth to once-per-plateau-regime.
- **Biology-first discipline under pressure.** If the first real brain-mode training run shows no learning, the pull will be toward importing PPO's critic as the modulator (Option D). Resist — that is M8 by design. An honest negative result is a valid v1 ship; the project's contribution is the falsification, not a success narrative. See `notes/biology-first-principle.md` for the full framing.

## Obsolete / No Longer Relevant

- Any reference to `AgentMode::{Keyboard, Ai}` or the global `AgentMode` Resource is obsolete. The two-state global enum was replaced in S1 by `Controller` (per-car enum Component) + three ZST marker components (`PpoCar`, `BrainCar`, `KeyboardCar`) — see `notes/brain-v1-decisions.md` D1.
- Any reference to `TrainerConfig.num_envs` as the source of truth for fleet size is half-obsolete — `num_envs` is kept as a synced convenience field, but the source of truth is now `TrainerConfig.layout.total_cars()`.
- Any claim that PPO is the only live learner is obsolete — the brain-inspired learner is live; layouts `AllBrain`, `SideBySide`, and `AllPpo` all produce running simulations.
- Any claim that `AgentMode::Keyboard` is in the F4 cycle is obsolete — F4 cycles `AllBrain → SideBySide → AllPpo → AllBrain` only; Keyboard is an escape-hatch layout reachable only programmatically (see D22).
