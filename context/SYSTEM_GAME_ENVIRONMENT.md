# System — Game Environment (Car + Track + Collision + Episode Boundaries)

## Scope / Purpose

- Define the deterministic environment the controller acts in, separate from any specific learning algorithm.
- Keep the environment responsible for motion, collisions, resets, progress, and episode boundaries, while leaving observation building and policy logic outside this subsystem.

## Current Implemented System

- A single `Track` entity is spawned from a tile-grid definition and carries the driveable surface, spawn pose, and derived closed centreline (`src/maps/track.rs`, `src/maps/monaco.rs`).
- The track is a 14x9 Sepang-inspired closed loop built from `TilePart` connectivity rather than free-form spline geometry (`src/maps/monaco.rs`, `src/maps/parts/mod.rs`).
- Grid-derived rendering exists for road surfaces, straight walls, curved corner walls, and a visual finish-line stripe (`src/maps/grid.rs`, `src/maps/monaco.rs::render_finish_line`).
- The car is a single Bevy entity with deterministic velocity/drag physics on the fixed tick and is spawned with attached progress and observation-related components (`src/game/car.rs`, `src/game/physics.rs`).
- Off-track detection checks the rotated car rectangle corners against `TrackGrid::is_road_at()` and emits a `CollisionEvent` as soon as any corner leaves the driveable area (`src/game/collision.rs`).
- Crash handling resets the car to the track spawn pose and zeroes velocity without despawn/respawn (`src/game/collision.rs::handle_collision_system`).
- Episode lifecycle management is implemented in fixed update with crash, timeout, and lap-complete termination paths (`src/game/episode.rs::episode_loop_system`).
- Reward accumulation is already live in the environment loop through positive gain in episode-best progress, a small per-tick time penalty, crash penalty, and lap bonus, and those terms are tracked separately in episode state for downstream analytics (`src/game/episode.rs`).

## Implemented Outputs / Artifacts (if applicable)

- Runtime `Track` component carrying the tile grid, spawn pose, and centreline (`src/maps/track.rs`).
- Runtime `Car` component plus sprite and attached progress/observation components (`src/game/car.rs::spawn_car`).
- Collision message type `CollisionEvent` used between detection and resolution (`src/game/collision.rs`).
- Episode resources: `EpisodeConfig`, `EpisodeState`, and `EpisodeMovingAverages` (`src/game/episode.rs`).

## In Progress / Partially Implemented

- The episode subsystem now exposes per-tick reward via `EpisodeState.current_tick_reward` to support the A2C rollout path, but that integration has not yet been validated end to end.
- The episode subsystem now also exposes `current_tick_end_reason` so downstream systems can distinguish the true terminal step from sticky episode-summary state (`src/game/episode.rs`).
- Reward decomposition now exists at the episode-state level, but the current best-progress reward path is still incorrect: `current_best_progress_fraction` is updated before reward-gain calculation, so the positive progress reward term is currently zeroed out in practice (`src/game/episode.rs`).
- Crash resets happen in `collision.rs`, while non-crash episode resets happen in `episode.rs`; the split is functional but increases reset-path duplication and verification burden.

## Planned / Missing / To Be Changed

- Headless or fast-simulation mode is still missing; the app currently assumes a windowed Bevy runtime from `src/main.rs`.
- Environment-level tests remain thin beyond the pure physics replay test; there is still no ECS-level regression harness for collisions, lap wraps, or episode transitions.
- The finish-line stripe remains visual only; lap completion still relies on progress-wrap thresholds rather than explicit line crossing.
- The reward-ordering bug in `episode_loop_system()` must be corrected so positive best-progress gains actually contribute to return; current reports show `progress_reward_sum == 0.0` across episodes.

## Notes / Design Considerations (optional)

- The environment currently mixes “world truth” and “learning support” in `episode.rs` because reward accumulation lives there; this is acceptable for now but should remain clearly separated from policy update code.
- The reward design no longer uses signed net progress delta as the main signal; it now rewards only new best progress within the episode so backtracking does not erase earlier gains.
- `TrackGrid::is_road_at()` is the authoritative driveable-area query, so collision behaviour and raycast behaviour share the same geometric truth.
- The fixed tick and explicit `SimSet` ordering remain essential invariants for determinism and future replay.

## Discarded / Obsolete / No Longer Relevant

- Older architecture assumptions that the repository only contained environment code are no longer relevant; brain and analytics subsystems now exist even though they are still partial.
