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
  - `throttle`: clamped to `[0, 1]` (0 = coast, 1 = full throttle)
- `ActionState` is a **per-car Component** separating `desired` from `applied`:
  - Controllers write `desired` once per fixed tick.
  - `action_smoothing_system` iterates all cars and copies or low-pass filters `desired → applied`.
  - Physics and analytics consume `applied` only.
- `ActionSmoothing` exists as a **global Resource** but defaults to **disabled** (`enabled: false`, time_constant=0.12s). The smoothing config is shared; the smoothing state (previous `applied`) lives in each car's `ActionState` component.
- **Keyboard control** is mode-gated: `keyboard_action_input_system` exits immediately unless `AgentMode` is `Keyboard`. In multi-car mode, keyboard controls **`EnvInstanceId(0)` only**. Controls: A/D steer, W throttle.

### Observation Contract

The observation vector has **23 dimensions** (`OBSERVATION_DIM = 23`):

| Feature group | Count | Range | Source |
|---------------|-------|-------|--------|
| Ray distances (normalised by 375.0 max range) | 11 | [0, 1] | Grid-based raycasts at fixed angular offsets |
| Speed (normalised by 900.0) | 1 | [0, 1] | Car velocity magnitude |
| Signed lateral offset (normalised by 75.0) | 1 | [-1, 1] | Distance from centreline, left-positive |
| Signed heading error (normalised by π) | 1 | [-1, 1] | Angle between car forward and centreline tangent |
| Angular velocity (normalised by 8.0 rad/s) | 1 | [-1, 1] | Estimated yaw rate from heading delta / dt |
| Lookahead heading deltas (×4, normalised by π) | 4 | [-1, 1] | Car-forward vs centreline tangent at 50/100/175/260 units ahead |
| Lookahead curvatures (×4, normalised by 0.05) | 4 | [-1, 1] | Turn rate at lookahead points |

- `SensorReadings` is a **per-car component** holding raw world-unit values plus debug rendering data (ray hits, ray directions).
- `ObservationVector` is a **per-car component** holding the normalised fixed-size feature array.
- Ray layout: 11 rays at angles [-150°, -90°, -60°, -35°, -15°, 0°, +15°, +35°, +60°, +90°, +150°] relative to car forward.
- Raycast implementation: grid-march with 3.0 unit steps, binary-search refinement (8 iterations) at road boundary.
- Lookahead distances: [50, 100, 175, 260] world units along the centreline from current projection.

### Scheduling

- **Input systems** run in `SimSet::Input`:
  - `keyboard_action_input_system` → `action_smoothing_system` (chained).
  - A2C act system inserts between keyboard and smoothing via `.after()` / `.before()`.
- **Observation systems** run in `SimSet::Measurement`:
  - `update_sensor_readings_system` runs after both `update_track_progress_system` and `episode_loop_system` — so post-reset observations represent the reset spawn state rather than stale crash state.
  - `build_observation_vector_system` runs after sensor update.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer(s) | Notes |
|-----------|----------|-------------|-------|
| `ActionState.desired` | Keyboard or brain systems | Smoothing system | Single desired action per fixed tick |
| `ActionState.applied` | Smoothing system | Physics, analytics, debug | Authoritative executed control |
| `SensorReadings` | Observation systems | Debug overlays, HUD helpers | Raw world-derived measurements |
| `ObservationVector` | Observation systems | A2C brain and future controllers | Fixed-size normalised policy input |
| `ObservationConfig` | Resource defaults | Observation systems and debug overlays | Centralises ray and lookahead configuration |

## Implemented Outputs / Artifacts

- **Runtime resources:** `ActionSmoothing`, `ObservationConfig`
- **Runtime components (per car):** `ActionState`, `SensorReadings`, `ObservationVector`
- **Tests:** Signed lateral offset sign convention test in `observation.rs`

## Known Issues / Active Risks

- **Observation schema versioning** does not exist — will matter once snapshots or offline replay depend on observation compatibility.
- The ray layout is **manually enumerated** rather than generated from a higher-level spread specification.
- No dedicated runtime assertion that observation producer and all consumers remain **dimension-aligned** beyond shared constant use.
- **No brake channel** — throttle is currently coast-or-accelerate only.

## Partial / In Progress

- The observation contract has shifted from "manual-debug aid" to **live learning input**, raising the cost of accidental drift.
- Centreline-relative features are becoming the primary representation, but the current input is still a hybrid of geometry features plus a full ray bundle.

## Planned / Missing / Likely Changes

- A more explicit centreline-first observation hierarchy is a likely next step if A2C continues to underperform on turn anticipation.
- A reduced ray bundle remains a plausible experiment, but only after geometry-derived features are measured cleanly.
- Input-health validation would be useful: saturation detection, dead-ray detection, feature distribution drift.
- `ActionState` has been migrated to a per-car Component as part of the vectorised trainer work.

## Durable Notes / Discarded Approaches

- `TrackProgress` is **intentionally excluded** from `ObservationVector`. The design exposes geometry-relative features but avoids leaking privileged completion/progress state directly to the policy input.
- The controller boundary is intentionally stable across keyboard and AI paths. That keeps replay, analytics, and future brains from coupling to a specific control implementation.

## Obsolete / No Longer Relevant

- Older context treating the observation system as preparatory scaffolding only is obsolete — it is already part of the live control path.
