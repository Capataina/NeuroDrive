# Codebase Conventions

Recurring patterns that appear in three or more locations across the codebase and are **not** enforced by `rustfmt`, `clippy`, or the type system. A new contributor (human or agent) would not discover these from the code alone — they are de facto standards that should be maintained for consistency.

Captured during the 2026-04-18 upkeep pass after source-level scans; refreshed 2026-04-19 after the round-2 critic target-scaling work landed (PopArt + observation normaliser + target-KL early stop).

## Current Understanding

### 1. Per-car runtime state lives in Components, not Resources

Anything that holds per-car state is a Bevy `Component` attached to the car entity, not a `Resource` keyed by `EnvInstanceId`. Verified in:

- `ActionState` — `src/agent/action.rs`
- `EpisodeState`, `EpisodeMovingAverages` — `src/game/episode.rs`
- `SensorReadings`, `ObservationVector` — `src/agent/observation.rs`
- `TrackProgress` — `src/game/progress.rs`
- `PolicyOutput` — `src/brain/types.rs`
- `Car`, `SpawnConfig`, `CarColour`, `EnvInstanceId`, `Collided` — `src/game/car.rs`, `src/game/collision.rs`

Shared state that is not per-car (e.g., `TrainerRolloutBuffer`, `PpoBrain`, `PpoTrainingStats`, `TrainerLiveRanking`, `EpisodeTracker`, `FrameTimings`) is a `Resource`. The boundary is simple: if 8 cars can have 8 values of it, it is a Component; if there is one shared instance across all cars, it is a Resource.

The historical pattern where `ActionState` or `EpisodeState` was a singleton Resource has been fully removed.

### 2. Shared RNG seeding pattern: `StdRng::from_rng(&mut rand::rng())`

Two places seed a local `StdRng` the same way:

- `SpawnRng` — `src/game/car.rs:49`
- `PpoBrain.rng` — `src/brain/ppo/mod.rs:99`

Both use `rand::rng()` (thread-local, uncontrollable) to seed, then `from_rng` to obtain a deterministic `StdRng` for the session. This is the current determinism weakness documented in `systems/determinism.md` — the sessions are deterministic internally given their derived state, but the initial thread-local state is not user-controllable. If user-seeded determinism is ever implemented, both sites must be updated together.

### 3. Configuration structs follow the `Config { ... }` pattern with `Default` + Bevy `Resource`

All runtime configuration is a top-level `struct` ending in `Config`, with a `Default` impl providing canonical values, registered as a Bevy `Resource`:

- `EpisodeConfig` — `src/game/episode.rs` (reward coefficients, timeout)
- `TrainerConfig` — `src/game/car.rs` (car count, alpha values)
- `ObservationConfig` — `src/agent/observation.rs` (ray layout, lookahead)
- `PpoConfig` — `src/brain/ppo/mod.rs` (all PPO hyperparameters in one place — Phase 4 consolidation)
- `AnalyticsConfig` — `src/analytics/models.rs` (full_trace_export)
- `ProfilingConfig` — `src/profiling/config.rs` (duration, ring buffer size)

Hyperparameters should be added to the relevant `*Config` struct, not hard-coded inside systems. The Phase 4 code-health audit specifically consolidated four separate PPO hyperparameter sites into `PpoConfig`.

### 4. No Bevy events — the runtime uses the resource-component-query pattern

A project-wide grep for `EventWriter|EventReader` returns zero matches inside `src/`. All inter-system communication uses either:

- a shared `Resource` (e.g., `TrainerRolloutBuffer`, `PpoTrainingStats`, `TrainerLiveRanking`),
- a per-car `Component` read downstream in the same or later `SimSet` (e.g., `Collided`, `ActionState`, `EpisodeState.current_tick_*`),
- direct query reads during the same fixed tick.

The only Bevy event used anywhere is `AppExit` (implicit, Bevy-owned, consumed by the `Last` schedule exit systems in `analytics::plugin` and `profiling::mod`). Any new cross-system signal should follow the existing pattern unless there is a compelling reason to introduce events.

### 5. `debug_assert!` for invariants that would bloat release builds

`debug_assert!` is used in hot paths where the invariant is structural and failing it indicates a bug, not a runtime condition:

- `src/brain/ppo/mod.rs:336` — rollout buffer alignment
- `src/maps/centerline.rs:182` — "centreline must have at least two points"

Active plan work has flagged that release-build coverage is weaker than it should be (see `systems/brain-ppo.md` — "All rollout buffer alignment is checked by `debug_assert!` only — not active in release builds"). Prefer `assert!` when a corruption would silently propagate into training data, `debug_assert!` when the invariant is structural and the surrounding code already upholds it by construction.

### 6. Feature gating: `#[cfg(feature = "profiling")]`, zero cost when off

Only one feature flag exists: `profiling`. Every profiling touch point uses `#[cfg(feature = "profiling")]` at the module and plugin-registration level. No runtime branch checks the feature. This is the convention to follow if more feature-gated subsystems are added — compile the whole subsystem out rather than paying a runtime toggle cost.

### 7. Export directories follow `reports/{json,<category>}/<category>/` + retention

Both analytics and profiling exporters write to two parallel trees:

- `reports/json/analytics/` + `reports/analytics/`
- `reports/json/performance/` + `reports/performance/`

Both call `analytics::exporters::cleanup::enforce_retention(dir, 3)` to cap directory sizes. Any new export subsystem should reuse `enforce_retention` rather than implementing its own cleanup logic, and follow the same `reports/json/<x>/` + `reports/<x>/` pair convention.

### 8. Run-context snapshots use the shared `RunContext` struct

`analytics::exporters::context::RunContext` is the canonical snapshot of run configuration (car count, reward coefficients, PPO hyperparameters, observation layout, timeout). It is captured once at export time and included in both JSON and Markdown reports. The profiling exporter imports it too. Any new long-form exporter should use the same struct rather than inventing a parallel snapshot schema — keeping the metadata layer unified is what keeps historical reports comparable across analytics and profiling.

### 9. System registration goes through `Plugin::build`, no direct `app.add_systems` in `main.rs`

`main.rs` only calls `add_plugins`; individual system registration happens inside each subsystem's `Plugin::build`. Systems are placed in a `SimSet` for fixed-tick work, `Update` for per-frame work, `Startup`/`PostStartup` for init, and `Last` for exit-time cleanup. Any new runtime system must follow this placement or the `SimSet` ordering contract breaks silently.

## Rationale

These conventions collectively enable:

- **Multi-car iteration** without special-casing car 0 (Component-first design).
- **Deterministic ordering** that PPO depends on for reward-observation alignment (no events, strict `SimSet` placement).
- **Single canonical configuration home** per subsystem (`*Config` pattern), which made the Phase 4 `PpoConfig` consolidation possible.
- **Zero-cost feature gating** for profiling, keeping the release binary small.
- **Shared export infrastructure** (`RunContext`, `enforce_retention`), preventing parallel evolution of run metadata formats.

### 10. Normalisation state lives on the Brain or its own Resource, never on the Model

Running statistics that change during training — `ValueNorm { mu, sigma }` on `PpoBrain` for PopArt, `ObservationNormalizer` as a standalone Bevy `Resource` — are kept outside the `ActorCritic` network. The network struct stays focused on weights, gradients, and scratch; the running stats sit alongside it. Denormalisation then happens at well-defined boundary call sites (`forward_critic` bootstrap, the value read in `ppo_act_all_cars_system`, the exit flush) rather than baked into the forward pass.

Reason: normalisation state has a different lifecycle than network parameters (never reset between episodes, updated on different cadences, serialised differently if we ever save models) and blending it into the network struct creates coupling that makes ablations painful.

### 11. Disable flags for every normaliser, for ablations

Each training-time normaliser exposes an explicit off-switch:

- `PpoConfig.popart_enabled: bool` — when false, `ValueNorm` stays at `(µ=0, σ=1)` and the POP rescale is skipped; the training pipeline is numerically equivalent to pre-PopArt.
- `ObservationNormalizer.enabled: bool` — when false, the normaliser is an identity pass-through regardless of warmup state.
- `PpoConfig.target_kl: Option<f32>` — `None` disables the target-KL early-stop guardrail entirely.

Advantage normalisation is the exception — it is inherent to `ppo_process_chunk` and has no disable flag. Any future training-time transform should follow the same pattern so ablation experiments are one-line config edits, not refactors. See `notes/normalisation-layers.md` for the full picture of how the three normalisations compose.

## What Was Tried

- **Earlier singleton `ActionState` / `EpisodeState` as Resources** — removed as part of the multi-car vectorised trainer work. Reverting would re-introduce first-car shims throughout analytics.
- **Per-tick unsafe raw-pointer aliasing in `update.rs`** — fully removed in 2026-04-18 by splitting `ActorCritic` scratch into sibling `BatchIo` (inputs + gradient seeds) and `BatchScratch` (forward/backward intermediates) fields. Rust's disjoint-field borrow inference now accepts `&mut self.scratch` and `&self.batch_io.*` simultaneously without any raw-pointer aliasing. All `unsafe` blocks in `src/brain/common/gemm_*.rs` are FFI entry points into Apple Accelerate or `matrixmultiply`, not a Rust-internal workaround.
- **Per-feature runtime toggles** — never used; profiling is compile-time gated only.

## Guiding Principles

- When in doubt, favour Component over Resource for per-car state.
- Consolidate hyperparameters into the relevant `*Config` struct; do not scatter them across system bodies.
- Do not introduce Bevy events without a specific reason — the existing resource-component-query pattern handles all current inter-system signalling cleanly.
- Any export subsystem reuses `RunContext` and `enforce_retention` from `analytics::exporters`.
