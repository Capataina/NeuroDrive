# System — Determinism

## Scope / Purpose

- Capture the parts of the repository that support repeatable behaviour and replay-friendly reasoning.
- Document where determinism is strong today and where it still breaks down.

## Boundaries / Ownership

- `src/sim/sets.rs` owns the shared fixed-update ordering contract.
- `src/main.rs` owns the fixed 60 Hz timestep configuration.
- `src/game/physics.rs` owns the pure deterministic car stepper and the only direct action-to-motion mutation point.
- This topic is cross-cutting, but its canonical home is justified because ordering and repeatability now affect the environment, brain, analytics, and debug layers.

## Current Implemented Reality

- The simulation runs on a fixed `60 Hz` timestep using `Time<Fixed>::from_hz(60.0)`.
- Core simulation sets are explicitly chained as `Input -> Physics -> Collision -> Measurement`.
- `step_car_dynamics()` is a pure helper used by runtime physics and a deterministic replay unit test.
- Track construction is hard-coded and contains no runtime randomness.
- The controller boundary through `CarAction` and `ActionState` keeps physics insulated from controller implementation details.
- Current test coverage includes a deterministic replay test proving identical trajectories for identical seeded action streams at the pure dynamics level.

## Key Interfaces / Data Flow

| Determinism surface | Current status | Notes |
|---|---|---|
| Fixed timestep | strong | central runtime invariant |
| Sim set ordering | strong | explicit chain in `GamePlugin` |
| Pure car dynamics | strong | covered by unit test |
| Track construction | strong | hard-coded layout, no RNG |
| A2C action sampling | weak | uses ad hoc RNG creation |
| Analytics export filenames | weak | timestamp-based, naturally non-deterministic |
| Full ECS replay | missing | no end-to-end action/observation/reward replay harness |

## Implemented Outputs / Artifacts

- Unit test:
  - deterministic replay of `step_car_dynamics()`
- Structural contract:
  - `SimSet` ordering shared across runtime systems

## Known Issues / Active Risks

- Determinism largely stops at the environment core. The A2C path is not meaningfully reproducible yet because RNG ownership is not centralised.
- Analytics filenames depend on wall-clock time, which is fine for storage but not deterministic.
- There is no ECS-level replay or action log that would let the full runtime be re-run and compared.

## Partial / In Progress

- More subsystems now depend on deterministic ordering than older docs captured:
  - A2C rollout alignment,
  - analytics trace capture timing,
  - reset/observation ordering,
  - HUD episode summaries.

## Planned / Missing / Likely Changes

- A single owned RNG/seed strategy is the clearest missing prerequisite for reproducible AI runs.
- Full replay would likely need recorded streams for:
  - actions,
  - observations,
  - rewards,
  - episode endings,
  - and possibly update summaries.
- If headless training is added, determinism expectations should be rechecked at the whole-app level rather than only in pure helpers.

## Durable Notes / Discarded Approaches

- Determinism should continue to be treated as layered:
  - pure maths/physics determinism,
  - ECS schedule determinism,
  - controller/analytics determinism.
- The pure-physics replay test is useful but insufficient as proof of end-to-end reproducibility.

## Obsolete / No Longer Relevant

- Any older note that treats replay as only an environment concern is obsolete; controller training and analytics timing now depend on deterministic ordering too.
