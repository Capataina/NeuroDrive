# System — Agent Interface

## Scope / Purpose

- Own the stable controller-facing boundary between the environment and any brain implementation.
- Expose one action contract (`CarAction` via `ActionState`) and one observation contract (`ObservationVector`) regardless of whether control is manual or learned.
- This boundary is intentionally stable so that brains, replay systems, and analytics can be swapped or extended without changing the environment.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/agent/action.rs` | `CarAction`, `ActionState` (desired/applied), `ActionSmoothing`, keyboard input system, smoothing system | Policy logic, reward computation, physics mutation |
| `src/agent/observation.rs` | `SensorReadings`, `ObservationVector`, `ObservationConfig`, ray constants, all sensor update + normalisation logic | Environment state production, centreline derivation |
| `src/agent/plugin.rs` | Fixed-tick scheduling for both action and observation systems | SimSet definition (owned by `sim`) |

## Current Implemented Reality

### Action Contract

- `CarAction` is the stable control surface:
  - `steering`: clamped to `[-1, 1]` (left negative, right positive)
  - `throttle`: clamped to `[0, 1]` (0 = coast, 1 = full throttle). Deceleration happens only through drag.
- `ActionState` is a **per-car Component** separating `desired` from `applied`:
  - Controllers write `desired` once per fixed tick.
  - `action_smoothing_system` iterates all cars and copies or low-pass filters `desired → applied`.
  - Physics and analytics consume `applied` only.
- `ActionSmoothing` exists as a **global Resource** but defaults to **disabled** (`enabled: false`, time_constant=0.12s). The smoothing config is shared; the smoothing state (previous `applied`) lives in each car's `ActionState` component.
- **Keyboard control** is mode-gated: `keyboard_action_input_system` exits immediately unless `AgentMode` is `Keyboard`. In multi-car mode, keyboard controls **`EnvInstanceId(0)` only**. Controls: A/D steer, W throttle.

### Observation Contract

The observation vector has **43 dimensions** (`OBSERVATION_DIM = 43`):

| Feature group | Count | Range | Source |
|---------------|-------|-------|--------|
| Ray distances (normalised by 375.0 max range) | 11 | [0, 1] | Grid-based raycasts at fixed angular offsets |
| Forward velocity (normalised) | 1 | [-1, 1] | Car velocity projected onto forward axis |
| Lateral velocity (normalised) | 1 | [-1, 1] | Car velocity projected onto right axis |
| Signed lateral offset (normalised by 75.0) | 1 | [-1, 1] | Distance from centreline, left-positive |
| Signed heading error (normalised by π) | 1 | [-1, 1] | Angle between car forward and centreline tangent |
| Angular velocity (normalised by 8.0 rad/s) | 1 | [-1, 1] | Estimated yaw rate from heading delta / dt |
| Speed delta (normalised) | 1 | [-1, 1] | Frame-to-frame speed change (acceleration/deceleration signal) |
| Lookahead heading deltas (×12, normalised by π) | 12 | [-1, 1] | Car-forward vs centreline tangent at 12 distances ahead |
| Lookahead curvatures (×12, normalised by 0.05) | 12 | [-1, 1] | Turn rate at lookahead points |
| Previous steering | 1 | [-1, 1] | Last tick's applied steering fed back as observation |
| Previous throttle | 1 | [0, 1] | Last tick's applied throttle fed back as observation |

**Layout:** `[rays(11), v_forward, v_lateral, lateral_offset, heading_error, angular_velocity, speed_delta, lookahead(12×2), prev_steering, prev_throttle]`

- `SensorReadings` is a **per-car component** holding raw world-unit values plus debug rendering data (ray hits, ray directions). Includes `v_forward`, `v_lateral`, `speed_delta`, `previous_speed`, `previous_steering`, `previous_throttle`.
- `ObservationVector` is a **per-car component** holding the normalised fixed-size feature array.
- **Running observation normaliser (round-2, 2026-04-19):** an `ObservationNormalizer` Resource applies per-dim Welford-algorithm online mean/variance to the assembled observation vector in `build_observation_vector_system`. During a `warmup_samples=1000` phase the normaliser is a pass-through while accumulating stats; after warmup it centres and scales each dim by `(x − μ) / σ` and clips to `[-10, 10]` (SB3 `VecNormalize.clip_obs` convention). Stats persist across episodes (the full training run is one distribution). When `enabled=false` the normaliser is an identity pass-through regardless of warmup state. See `context/references/ppo-critic-architecture.md` (Andrychowicz et al. 2021 recommend observation normalisation as the single most-cited PPO implementation detail).
- Ray layout: 11 rays at angles [-150°, -90°, -60°, -35°, -15°, 0°, +15°, +35°, +60°, +90°, +150°] relative to car forward.
- Raycast implementation: grid-march with 3.0 unit steps, binary-search refinement (8 iterations) at road boundary.
- Lookahead: 12 samples at [30, 60, 95, 135, 180, 230, 285, 345, 415, 490, 570, 650] world units. Dense near-field (steering precision), sparser far-field (anticipation at speed). At 300 u/s the farthest point gives ~2.2s warning.

### Scheduling

- **Input systems** run in `SimSet::Input`:
  - `keyboard_action_input_system` → `action_smoothing_system` (chained).
  - PPO act system inserts between keyboard and smoothing via `.after()` / `.before()`.
- **Observation systems** run in `SimSet::Measurement`:
  - `update_sensor_readings_system` runs after both `update_track_progress_system` and `episode_loop_system` — so post-reset observations represent the reset spawn state rather than stale crash state.
  - `build_observation_vector_system` runs after sensor update.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer(s) | Notes |
|-----------|----------|-------------|-------|
| `ActionState.desired` | Keyboard or brain systems | Smoothing system | Single desired action per fixed tick |
| `ActionState.applied` | Smoothing system | Physics, analytics, debug | Authoritative executed control |
| `SensorReadings` | Observation systems | Debug overlays, HUD helpers | Raw world-derived measurements including v_forward, v_lateral, speed_delta, previous actions |
| `ObservationVector` | Observation systems | PPO brain and future controllers | Fixed-size normalised policy input (dim 43) |
| `ObservationConfig` | Resource defaults | Observation systems and debug overlays | Centralises ray and lookahead configuration |

## Implemented Outputs / Artifacts

- **Runtime resources:** `ActionSmoothing`, `ObservationConfig`
- **Runtime components (per car):** `ActionState`, `SensorReadings`, `ObservationVector`
- **Tests:** Signed lateral offset sign convention test in `observation.rs`

## Known Issues / Active Risks

- **Observation schema versioning** does not exist — will matter once snapshots or offline replay depend on observation compatibility.
- The ray layout is **manually enumerated** rather than generated from a higher-level spread specification.
- No dedicated runtime assertion that observation producer and all consumers remain **dimension-aligned** beyond shared constant use.

## Partial / In Progress

- The observation contract has shifted from "manual-debug aid" to **live learning input**, raising the cost of accidental drift.
- Centreline-relative features are becoming the primary representation, but the current input is still a hybrid of geometry features plus a full ray bundle.

## Planned / Missing / Likely Changes

- A more explicit centreline-first observation hierarchy is a likely next step if PPO continues to underperform on turn anticipation.
- A reduced ray bundle remains a plausible experiment, but only after geometry-derived features are measured cleanly.
- Input-health validation would be useful: saturation detection, dead-ray detection, feature distribution drift.
- `ActionState` has been migrated to a per-car Component as part of the vectorised trainer work.

## Durable Notes / Discarded Approaches

- `TrackProgress` is **intentionally excluded** from `ObservationVector`. The design exposes geometry-relative features but avoids leaking privileged completion/progress state directly to the policy input.
- The controller boundary is intentionally stable across keyboard and AI paths. That keeps replay, analytics, and future brains from coupling to a specific control implementation.
- **Scalar speed was replaced by v_forward + v_lateral decomposition** to give the policy richer information about the car's motion state (e.g. detecting drift). The speed_delta observation provides an acceleration/deceleration signal without requiring the policy to differentiate speed over time.
- **Previous actions are fed back as observations** to give the policy awareness of its own recent control outputs, enabling smoother action sequences and reducing jitter.
## Obsolete / No Longer Relevant

- Older context treating the observation system as preparatory scaffolding only is obsolete — it is already part of the live control path.
- Any reference to 23-dimensional or 27-dimensional observations is obsolete — the observation vector is now 43 dimensions.
- Any reference to scalar speed as an observation feature is obsolete — replaced by v_forward + v_lateral.
- Any reference to throttle range [-1,1] or braking via negative throttle is obsolete — throttle is [0,1], braking was reverted because the policy converged to "mostly brake" as a safe local optimum.
- Any reference to 4 lookahead samples or distances [50, 100, 175, 260] is obsolete — now 12 samples spanning to 650 units.
