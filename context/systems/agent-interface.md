# System — Agent Interface

## Scope / Purpose

- Own the stable controller-facing boundary between the environment and any brain implementation.
- Expose one action contract (`CarAction` via `ActionState`) and one observation contract (`ObservationVector`) regardless of whether control is manual or learned.

## Boundaries / Ownership

- Owns action semantics in `src/agent/action.rs`.
- Owns observation production and normalisation in `src/agent/observation.rs`.
- Owns fixed-tick scheduling for both in `src/agent/plugin.rs`.
- Does not own environment truth such as progress rewards or episode boundaries; it only reads that truth.
- Does not own policy logic; brains consume `ObservationVector` and write `ActionState.desired`.

## Current Implemented Reality

- `CarAction` is the stable control surface with steering clamped to `[-1, 1]` and throttle clamped to `[0, 1]`.
- `ActionState` separates `desired` from `applied`, which allows optional smoothing without changing the controller-facing contract.
- `ActionSmoothing` exists but defaults to disabled; when disabled, applied action is a direct copy of desired action.
- Keyboard control is mode-gated: `keyboard_action_input_system` exits unless `AgentMode` is `Keyboard`.
- Observations are built from:
  - `11` ray distances,
  - speed,
  - signed lateral offset from the centreline,
  - signed heading error,
  - angular velocity,
  - `4` lookahead samples, each with heading-delta and curvature features.
- `OBSERVATION_DIM` is therefore `23`, and the A2C model reads that constant directly.
- Observation rebuild runs in `SimSet::Measurement` after both progress update and episode finalisation so post-reset observations represent the reset spawn state rather than stale crash state.

## Key Interfaces / Data Flow

| Interface | Producer | Consumer | Notes |
|---|---|---|---|
| `ActionState.desired` | keyboard or brain systems | smoothing system | single desired action per fixed tick |
| `ActionState.applied` | smoothing system | physics, analytics, debug | authoritative executed control |
| `SensorReadings` | observation systems | debug overlays, HUD helpers | raw world-derived measurements |
| `ObservationVector` | observation systems | A2C brain and future controllers | fixed-size normalised policy input |
| `ObservationConfig` | resource defaults | observation systems and debug overlays | centralises ray and lookahead configuration |

- Fixed-tick ordering inside `AgentPlugin` is deliberate:
  - input systems run in `SimSet::Input`,
  - observation systems run in `SimSet::Measurement`,
  - sensor update chains before vector build.

## Implemented Outputs / Artifacts

- Runtime resources:
  - `ActionState`
  - `ActionSmoothing`
  - `ObservationConfig`
- Runtime components attached to the car:
  - `SensorReadings`
  - `ObservationVector`
- A small unit test verifies the sign convention for lateral offset.

## Known Issues / Active Risks

- Observation schema versioning does not exist yet, which will matter once snapshots or offline replay depend on observation compatibility.
- The ray layout is still manually enumerated rather than generated from a higher-level spread specification.
- There is no dedicated runtime assertion that the observation producer and all consumers remain dimension-aligned beyond shared constant use.
- The action interface has no brake channel; throttle is currently coast-or-accelerate only.

## Partial / In Progress

- The observation contract has shifted from “manual-debug aid” to live learning input, which raises the cost of accidental drift.
- Centreline-relative features are becoming the primary representation, but the current input is still a hybrid of geometry features plus a full ray bundle.

## Planned / Missing / Likely Changes

- A more explicit centreline-first observation hierarchy is a likely next step if A2C continues to underperform on turn anticipation.
- A reduced ray bundle remains a plausible experiment, but only after current geometry-derived features are measured cleanly.
- A brake action channel may be added later as a separate action-space change.
- More explicit input-health validation would be useful:
  - saturation detection,
  - dead-ray detection,
  - feature distribution drift over a run.

## Durable Notes / Discarded Approaches

- `TrackProgress` is intentionally excluded from `ObservationVector`. The current design exposes geometry-relative features but avoids leaking privileged completion/progress state directly to the policy input.
- The controller boundary is intentionally stable across keyboard and AI paths. That keeps replay, analytics, and future brains from coupling to a specific control implementation.

## Obsolete / No Longer Relevant

- Older context that treated the observation system as preparatory scaffolding only is obsolete; it is already part of the live control path.
