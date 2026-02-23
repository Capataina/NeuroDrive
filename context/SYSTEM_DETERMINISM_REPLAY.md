# System — Determinism and Replay

## Scope / Purpose

- Make the simulation deterministic under a fixed seed and fixed action stream.
- Provide a deterministic replay test to catch physics/measurement drift and make debugging reproducible.

## Current Implemented System

- Track layout is fixed and hard-coded (no runtime RNG) (`src/maps/monaco.rs::build_tiles`).
- Fixed-timestep simulation is configured at 60 Hz and core sim systems run in `FixedUpdate` with explicit pipeline ordering (`src/main.rs`, `src/sim/sets.rs`, `src/game/plugin.rs`).
- A stable action interface is latched on the fixed tick (`src/agent/action.rs::ActionState`, `src/agent/action.rs::keyboard_action_input_system`).
- A deterministic replay unit test exists for the pure car dynamics stepper: same seed + same action stream produces an identical final trajectory (`src/game/physics.rs` test `deterministic_replay_same_seed_same_actions_identical_trajectory`).
- There is still no full ECS action recording/replay harness in `src/`.

## Implemented Outputs / Artifacts (if applicable)

- Unit test coverage for deterministic replay of dynamics stepping in `src/game/physics.rs`.

## In Progress / Partially Implemented

- None tracked in `context/` yet.

## Planned / Missing / To Be Changed

- Action recording is missing (no serialisation of actions over time exists in `src/`).
- Action replay is missing (no mode that feeds recorded actions back into the sim exists in `src/`).
- End-to-end deterministic replay assertion at the ECS/world level is missing (current test targets pure dynamics only).

## Notes / Design Considerations (optional)

- Determinism should be scoped explicitly (platform, build mode, floating point behaviour, and Bevy scheduling all influence what “identical” means).

## Discarded / Obsolete / No Longer Relevant

- None tracked in `context/` yet.
