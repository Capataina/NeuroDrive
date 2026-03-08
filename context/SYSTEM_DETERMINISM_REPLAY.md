# System — Determinism and Replay

## Scope / Purpose

- Keep the environment reproducible under fixed timestep and fixed action streams.
- Provide enough replay and determinism infrastructure to debug environment drift and future learning regressions.

## Current Implemented System

- The simulation runs on a fixed 60 Hz timestep with explicit `SimSet` ordering (`src/main.rs`, `src/sim/sets.rs`, `src/game/plugin.rs`, `src/agent/plugin.rs`).
- Track layout is hard-coded and contains no runtime RNG (`src/maps/monaco.rs::build_tiles`).
- The action boundary is stable through `CarAction` and `ActionState`, which gives a deterministic control surface to physics (`src/agent/action.rs`).
- Car dynamics are factored into a pure `step_car_dynamics()` function separate from ECS system wiring (`src/game/physics.rs`).
- A deterministic replay unit test exists for the pure physics stepper and verifies identical trajectories for identical seeded action streams (`src/game/physics.rs`).

## Implemented Outputs / Artifacts (if applicable)

- Unit test `deterministic_replay_same_seed_same_actions_identical_trajectory` in `src/game/physics.rs`.
- Fixed-order simulation pipeline through `SimSet::{Input, Physics, Collision, Measurement}` (`src/sim/sets.rs`).

## In Progress / Partially Implemented

- The repository now contains more deterministic-sensitive systems than the old docs captured: A2C action production, episode reward collection, and analytics tracking all depend on stable fixed-tick ordering.
- Determinism is therefore still strongest in the environment core and much weaker in the newly added brain/analytics layers.

## Planned / Missing / To Be Changed

- There is still no ECS/world-level action recording and replay harness.
- There is still no serialised replay format for observations, actions, rewards, or episode endings.
- There are no end-to-end determinism assertions for progress, collision timing, episode summaries, analytics exports, or A2C rollout contents.
- The broader runtime now compiles and exports analytics, but deterministic validation still does not extend to analytics contents or A2C rollout/replay behaviour.

## Notes / Design Considerations (optional)

- Determinism should continue to be treated as a layered property:
  - pure dynamics determinism,
  - fixed-schedule ECS determinism,
  - controller/analytics determinism.
- The handwritten A2C path introduces random action sampling, so reproducibility will require an explicit RNG ownership strategy rather than ad hoc `rand::rng()` calls.

## Discarded / Obsolete / No Longer Relevant

- The older context that positioned replay as only a future environment concern is incomplete now that controller training and analytics export have entered the runtime path.
