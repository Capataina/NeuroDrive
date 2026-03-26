# System — Determinism

## Scope / Purpose

- Document the parts of the repository that support repeatable behaviour and replay-friendly reasoning.
- Track where determinism is strong today and where it still breaks down.
- This is a cross-cutting topic with its own canonical file because ordering and repeatability now affect the environment, brain, analytics, and debug layers.

## Boundaries / Ownership

| Owner | Owns |
|-------|------|
| `src/sim/sets.rs` | Shared fixed-update ordering contract (`SimSet` enum) |
| `src/main.rs` | Fixed 60 Hz timestep configuration |
| `src/game/physics.rs` | Pure deterministic car stepper (`step_car_dynamics`) and the only action-to-motion mutation point |
| `src/maps/monaco.rs` | Hard-coded track construction (no runtime randomness) |

## Current Implemented Reality

### Determinism Surface Map

| Surface | Status | Evidence |
|---------|--------|----------|
| Fixed timestep (60 Hz) | **Strong** | `Time::<Fixed>::from_hz(60.0)` in `main.rs` |
| SimSet ordering | **Strong** | Explicit `Input → Physics → Collision → Measurement` chain in `GamePlugin` |
| Pure car dynamics | **Strong** | `step_car_dynamics()` is pure; covered by deterministic replay unit test |
| Track construction | **Strong** | Hard-coded tile layout in `monaco.rs`, no RNG |
| Controller boundary | **Strong** | `CarAction`/`ActionState` insulates physics from controller implementation |
| Centreline projection | **Strong** | Purely geometric, no RNG |
| Observation production | **Strong** | Grid raycasts and math are deterministic given car state and track |
| A2C action sampling | **Improved** | Uses a seeded `StdRng` stored in `A2cBrain` — deterministic given the same seed |
| A2C model initialisation | **Weak** | Uses `rand::rng()` once at startup for init, then seeds a `StdRng` for runtime — init seed not yet user-controllable |
| Analytics export filenames | **Weak** | Timestamp-based, naturally non-deterministic (acceptable) |
| Full ECS replay | **Missing** | No end-to-end action/observation/reward replay harness |

### What Is Currently Reproducible

Given the same compiled binary and identical fixed-tick action streams, the environment produces **bitwise-identical** car trajectories, collision events, progress measurements, and rewards. This is verified by the deterministic replay test.

### What Is Not Reproducible

The A2C path has **improved** determinism but is not fully controlled:
1. **Model initialisation** — weights still depend on thread-local RNG state at startup (not user-seeded).
2. **Action sampling** — now uses a seeded `StdRng` stored in `A2cBrain`, derived from the init RNG. Given the same init state, all subsequent policy sampling is deterministic.

Two runs of the same binary will still produce different A2C behaviour because the init seed is not user-controllable, but within a run, the policy sampling is now reproducible from the seeded state.

## Key Interfaces / Data Flow

The ordering contract is the most critical determinism surface:

```text
Input (actions chosen)
  → Physics (state mutated)
    → Collision (off-road detected)
      → Measurement (progress, rewards, observations, analytics, A2C reward)
```

Any violation of this ordering could produce:
- rewards computed from pre-physics state,
- observations reflecting pre-reset crash state,
- analytics capturing stale data,
- A2C collecting misaligned reward/observation pairs.

## Implemented Outputs / Artifacts

- **Unit test:** `deterministic_replay_same_seed_same_actions_identical_trajectory` in `src/game/physics.rs` — 1200 steps with LCG-generated actions, verifying bitwise position/velocity/heading match.
- **Structural contract:** `SimSet` ordering shared across all runtime plugins.

## Known Issues / Active Risks

- Determinism largely stops at the environment core. The **A2C path is not meaningfully reproducible** because RNG is not centralised.
- There is **no ECS-level replay** or action log that would let the full runtime be re-run and compared.
- Analytics filenames depend on wall-clock time (fine for storage, not deterministic).

## Partial / In Progress

- More subsystems now depend on deterministic ordering than originally intended:
  - A2C rollout alignment
  - Analytics trace capture timing
  - Reset/observation ordering
  - HUD episode summaries

## Planned / Missing / Likely Changes

- A **user-controllable init seed** is the remaining prerequisite for fully reproducible AI runs (the runtime RNG is now seeded, but the init seed itself is not configurable).
- Full replay would need recorded streams for actions, observations, rewards, episode endings, and possibly update summaries.
- If headless training is added, determinism expectations should be rechecked at the **whole-app level**.

## Durable Notes / Discarded Approaches

- Determinism should be treated as **layered**:
  1. Pure maths/physics determinism — strong today.
  2. ECS schedule determinism — strong via explicit set ordering.
  3. Controller/analytics determinism — weak due to uncontrolled RNG.
- The pure-physics replay test is useful but **insufficient as proof of end-to-end reproducibility**.

## Obsolete / No Longer Relevant

- Any note treating replay as only an environment concern is obsolete — controller training and analytics timing now depend on deterministic ordering too.
