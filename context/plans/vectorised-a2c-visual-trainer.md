# Implement Now — Vectorised A2C Visual Trainer

## Header

- [x] Status: **Stages 1–3 complete.** Per-car state, multi-car spawn, vectorised rollout with per-env GAE, trainer ranking with hysteresis, unique car colours, live leaderboard. Stages 4–5 (analytics + HUD overhaul, cleanup) are superseded by separate plans: `analytics-overhaul-brief.md` and Stage 5 cleanup will happen alongside the PPO upgrade.
- [x] Scope: The runtime now spawns configurable multi-car training (default 3) under one shared A2C policy. Best car is highlighted. Analytics/HUD use temporary shims pending the analytics overhaul.
- [x] Exit rule (partially met): cars spawn, train, terminate, and reset independently under shared policy. Best car is visually highlighted. Per-car analytics and cohort summaries are deferred to the analytics overhaul. Singleton assumptions are removed from game/agent/brain; analytics/HUD shims target first car.

## Implementation Structure

- [ ] Modules / files affected (expected):
  - `src/main.rs`
  - `src/agent/action.rs`
  - `src/agent/observation.rs`
  - `src/agent/plugin.rs`
  - `src/brain/plugin.rs`
  - `src/brain/types.rs`
  - `src/brain/a2c/mod.rs`
  - `src/brain/a2c/buffer.rs`
  - `src/brain/a2c/update.rs`
  - `src/game/car.rs`
  - `src/game/collision.rs`
  - `src/game/episode.rs`
  - `src/game/plugin.rs`
  - `src/game/progress.rs`
  - `src/game/physics.rs`
  - `src/debug/hud.rs`
  - `src/debug/overlays.rs`
  - `src/debug/plugin.rs`
  - `src/analytics/models.rs`
  - `src/analytics/plugin.rs`
  - `src/analytics/trackers/action.rs`
  - `src/analytics/trackers/trace.rs`
  - `src/analytics/trackers/episode.rs`
  - `src/analytics/metrics/*`
  - `src/analytics/exporters/json.rs`
  - `src/analytics/exporters/markdown.rs`
  - `context/systems/brain-a2c.md`
  - `context/systems/analytics.md`
  - `context/systems/debug-runtime.md`
  - `context/systems/environment.md`
- [ ] Responsibility boundaries:
  - `maps` stays singleton unless later evidence demands per-instance tracks.
  - `game` owns per-car environment truth and per-car episode truth.
  - `agent` owns per-car action and observation boundaries, but no longer as singleton resources/components-only assumptions.
  - `brain/a2c` owns the shared policy, the trainer rollout buffer, rollout scheduling, update cadence, and trainer-level selection of "best current car".
  - `analytics` owns both per-car episode facts and trainer-level cohort aggregation.
  - `debug` owns visual differentiation and trainer dashboard surfaces only; it must not define "best" truth independently.
- [ ] Recommended default runtime shape:
  - one shared track entity
  - twenty-five independent car entities
  - one shared policy/value network
  - one trainer resource aggregating transitions across all cars
  - one visible scene containing all cars
  - cars rendered with reduced alpha by default
  - one best-performing current car rendered at full opacity and stronger colour
- [ ] Alternative architecture to keep in reserve:
  - one visible environment plus many headless environments or separate worlds
  - worse for the current goal because the user explicitly wants to see all 25 cars at once
  - better only if visual clutter or ECS scaling makes one-world training too awkward later

## Current Code Reality (Audit Baseline)

This section documents what actually exists in the code so every migration step can reference concrete current state rather than assumptions.

### Systems Already Multi-Car Ready

These systems already use `.iter()` / `.iter_mut()` and will work with 25 cars without query changes:

| System | File:Line | Query |
|--------|-----------|-------|
| `car_physics_system` | `physics.rs:27` | `Query<(&mut Transform, &mut Car)>` via `.iter_mut()` |
| `update_track_progress_system` | `progress.rs:38` | `Query<(&Transform, &mut TrackProgress), With<Car>>` via `.iter_mut()` |
| `update_sensor_readings_system` | `observation.rs:132` | `Query<(&Transform, &mut SensorReadings, &TrackProgress), With<Car>>` via `.iter_mut()` |
| `build_observation_vector_system` | `observation.rs:186` | `Query<(&SensorReadings, &mut ObservationVector)>` via `.iter_mut()` |
| `draw_geometry_overlay_system` | `overlays.rs:55` | `Query<(&Transform, &TrackProgress, &Car), With<Car>>` via `.iter()` |
| `draw_sensor_overlay_system` | `overlays.rs:151` | `Query<(&Transform, &SensorReadings), With<Car>>` via `.iter()` |

**However**, `car_physics_system` reads `Res<ActionState>` — a single global action applied to ALL cars via the loop. This is a critical hidden singleton: with 25 cars, every car would receive the same steering/throttle. This must become per-car before multi-car spawn.

### Systems With Singleton Car Queries (Must Change)

| System | File:Line | Query Pattern | What It Accesses |
|--------|-----------|---------------|------------------|
| `collision_detection_system` | `collision.rs:19` | `car_query.single()` | `&Transform` for corner checks |
| `episode_loop_system` | `episode.rs:175` | `car_query.single_mut()` | `(&mut Transform, &mut Car, &mut TrackProgress)` |
| `a2c_act_system` | `a2c/mod.rs:142` | `obs_query.single()` | `&ObservationVector` |
| `a2c_collect_reward_system` | `a2c/mod.rs:159` | `obs_query.single()` | `&ObservationVector` for bootstrap |
| `a2c_flush_on_exit_system` | `a2c/mod.rs:199` | `obs_query.single()` | `&ObservationVector` for bootstrap |
| `update_driving_hud_stats_system` | `hud.rs:376` | `progress_query.single()` | `&TrackProgress` |
| `update_driving_hud_text_system` | `hud.rs:453` | `car_query.single()` | `(&TrackProgress, &SensorReadings)` |
| `capture_episode_tick_trace_system` | `trackers/trace.rs:73` | `sensor_query.single()` | `&SensorReadings` |

Track queries (`.single()` on `Query<&Track>`) are correct and stay singleton — there is only one track.

### Singleton Resources Storing Car-Specific State (Must Migrate)

| Resource | File | Fields | Migration Target |
|----------|------|--------|-----------------|
| **`EpisodeState`** | `episode.rs:66` | 27 fields: current episode counters, tick rewards, progress accumulators, last-episode summaries | **Component on car entity** |
| **`EpisodeMovingAverages`** | `episode.rs:150` | 3 `VecDeque<f32>` + 3 means (returns, progress, crashes) | **Component on car entity** — each car needs its own rolling window. Trainer-wide aggregates derive from per-car data |
| **`ActionState`** | `action.rs:31` | `desired: CarAction`, `applied: CarAction` | **Component on car entity** |
| **`A2cBrain`** | `a2c/mod.rs:21` | `model: ActorCritic`, `buffer: RolloutBuffer`, hyperparams, `step_counter` | **Split**: model + hyperparams stay as shared `Res<A2cBrain>`, buffer becomes `TrainerRolloutBuffer` (new resource), step counter moves to trainer scope |
| **`A2cTrainingStats`** | `a2c/mod.rs:57` | Update metrics, losses, explained variance, layer health | **Stays as Resource** — trainer-wide, one update produces one stats snapshot |
| **`DrivingHudStats`** | `hud.rs:24` | `deaths`, `best_progress_fraction`, `best_progress_episode` | **Rewrite to trainer scope** — read from `TrainerLiveRanking` |
| **`DrivingHudEpisodeAccumulator`** | `hud.rs:42` | Per-tick centreline/heading accumulation for quarter summaries | **Remove or rewrite** — trainer HUD uses cohort data, not single-car accumulation |
| **`DrivingHudHistory`** | `hud.rs:78` | `VecDeque` of completed episode snapshots for quarter table | **Rewrite to trainer scope** — aggregate from per-car episode history or `TrainerLiveRanking` |
| **`EpisodeActionAccumulator`** | `trackers/action.rs:17` | Per-episode steering/throttle sums for mean/std | **Indexed by `EnvInstanceId`** in a trainer-scope analytics resource, or component on car entity |
| **`EpisodeTraceAccumulator`** | `trackers/trace.rs:17` | Per-tick trace records, completed trace snapshot | **Indexed by `EnvInstanceId`** in a trainer-scope analytics resource, or component on car entity |

### Resources That Stay Global (No Migration Needed)

| Resource | File | Reason |
|----------|------|--------|
| `EpisodeConfig` | `episode.rs:21` | Shared reward/timeout parameters — all cars use the same rules |
| `ObservationConfig` | `observation.rs:85` | Shared ray/normalisation config |
| `ActionSmoothing` | `action.rs:49` | Shared smoothing configuration (the smoothing *state* lives in `ActionState.applied`, which moves to per-car) |
| `AgentMode` | `brain/types.rs:8` | Trainer-wide mode toggle |
| `DebugOverlayState` | `overlays.rs:10` | F1/F2/F3 toggle state |
| `EpisodeTracker` | `analytics/models.rs:141` | Trainer-wide analytics sink — but schemas need `env_id` tagging |

### Critical Hidden Coupling: Physics ← ActionState

`car_physics_system` (`physics.rs:27`) currently reads `Res<ActionState>` and applies `action_state.applied` to every car in its `.iter_mut()` loop. This means if 25 cars exist today, they would all receive identical steering and throttle.

Once `ActionState` becomes a component, `car_physics_system` must read the per-car `ActionState` component from the same entity rather than a global resource. The system already loops over car entities, so the fix is to add `ActionState` to the query tuple:

```
// Before
fn car_physics_system(action_state: Res<ActionState>, mut query: Query<(&mut Transform, &mut Car)>)

// After
fn car_physics_system(time: ..., mut query: Query<(&mut Transform, &mut Car, &ActionState)>)
```

Similarly, `action_smoothing_system` currently mutates `ResMut<ActionState>`. It must become a query-based system iterating over all `(&mut ActionState)` components, using the same global `Res<ActionSmoothing>` config.

### Critical Hidden Coupling: Brain::act() Pushes to Buffer Internally

`A2cBrain::act()` (`a2c/mod.rs:73`) both computes the action AND pushes to `self.buffer` (states, actions, latent_actions, safety_clamp_hits, values). For multi-car, the act loop must:

1. Call `model.forward()` for each car's observation (no buffer push inside act).
2. Push to the trainer-scope `TrainerRolloutBuffer` with the corresponding `env_id`.
3. Write the resulting action to that car's `ActionState` component.

This means the `Brain` trait's `act()` method needs refactoring. The recommended approach is to stop using the trait for the vectorised path and instead have the `a2c_act_all_cars_system` call `model.forward()` directly, sample actions, and push to the trainer buffer explicitly.

### Critical Hidden Coupling: CollisionEvent is a Global Message

`CollisionEvent` (`collision.rs:8`) is a unit struct message — it carries no car identity. The `episode_loop_system` reads collision events and assumes any event means "the" car crashed.

With 25 cars, collision detection must either:
- **Recommended:** Replace the message with a per-car component flag (e.g., `Collided` marker component added/removed each tick), or
- **Alternative:** Make `CollisionEvent` carry the colliding entity: `CollisionEvent(Entity)`.

The component-flag approach is simpler because `episode_loop_system` already queries car entities and can check the flag in the same loop. The message approach requires matching events to entities.

### Camera and Keyboard Behaviour

**Camera:** Currently spawned as `Camera2d::default()` in `setup_game` (`plugin.rs:53`). It does not follow any car — it stays at origin. The track is positioned so the default camera view covers it. With 25 cars on the same track, this should remain unchanged. No camera-follow logic is needed unless the user later requests zooming to the best car.

**Keyboard mode:** `keyboard_action_input_system` (`action.rs:69`) writes to `ResMut<ActionState>`. With `ActionState` as a component, keyboard mode needs a designated "keyboard car" entity. Options:
- **Recommended:** In keyboard mode, keyboard controls the car with `EnvInstanceId(0)` only. All other cars are idle or run a no-op controller.
- **Alternative:** Disable keyboard mode entirely when trainer is active. Simpler but loses manual testing ability.

### RNG Ownership

`A2cBrain::act()` creates a new `rand::rng()` on every call (`a2c/mod.rs:75`). This is non-deterministic and uncontrolled.

For the vectorised trainer:
- **Recommended:** Store a seeded `StdRng` in the `A2cBrain` resource (or a separate trainer RNG resource). Use it for all policy sampling. This enables deterministic training runs.
- Spawn offsets should also derive from this RNG or a separate spawn-seed.

## Function / Type Inventory

### New Types

- `TrainerConfig` (Resource)
  - Fields: `num_envs: usize` (default 3 — start small to validate performance and correctness, raise to 10–25 once stable), `update_horizon: usize` (total transitions before update, default 512), `min_update_transitions: usize`, `ranking_window: usize` (episodes per car for ranking), `ranking_update_cadence_ticks: u32`, `spawn_lateral_spread: f32`, `default_car_alpha: f32`, `best_car_alpha: f32`.
  - Used by: startup, brain, analytics, debug.

- `EnvInstanceId(u32)` (Component)
  - Attached to every car entity at spawn.
  - Stable for the lifetime of the entity.
  - Used for: buffer tagging, analytics grouping, ranking keying, HUD focus.

- `SpawnConfig` (Component)
  - Fields: `position: Vec2`, `rotation: f32`.
  - Attached at spawn, used by reset to return each car to its own assigned start.

- `TrainerRolloutBuffer` (Resource)
  - Fields: `states: Vec<Vec<f32>>`, `actions: Vec<Vec<f32>>`, `latent_actions: Vec<Vec<f32>>`, `safety_clamp_hits: Vec<[bool; 2]>`, `rewards: Vec<f32>`, `values: Vec<f32>`, `dones: Vec<bool>`, `env_ids: Vec<u32>`.
  - Methods: `push_pre_step(env_id, state, action, latent, clamp_hits, value)`, `push_reward(reward, done)`, `clear()`, `len()`, `compute_gae(...)`.
  - GAE must handle `dones` correctly per-transition (the current implementation already does this — each `dones[t]` masks the bootstrap independently).

- `TrainerLiveRanking` (Resource)
  - Fields: `best_env_id: Option<u32>`, `worst_env_id: Option<u32>`, `scores: Vec<(u32, f32)>` (sorted), `last_update_tick: u64`.
  - Ranking score per car: weighted combination of recent best-progress mean and recent return mean from that car's `EpisodeMovingAverages`.
  - Hysteresis: a new car must exceed the current best by a margin (e.g., 5% relative) to take the highlight, preventing flicker.

- `PerCarAnalytics` (Resource)
  - Wraps per-`EnvInstanceId` analytics accumulators: action stats and trace accumulators indexed by `env_id`.
  - Replaces the singleton `EpisodeActionAccumulator` and `EpisodeTraceAccumulator`.

### Migrated Types

- `ActionState` — changes from `Resource` to `Component` on car entities.
  - Fields unchanged: `desired: CarAction`, `applied: CarAction`.

- `EpisodeState` — changes from `Resource` to `Component` on car entities.
  - Fields unchanged (all 27 fields). Each car tracks its own episode counter, rewards, progress, and last-episode summaries independently.
  - The `current_episode` counter becomes per-car (car 3 might be on episode 47 while car 7 is on episode 52).

- `EpisodeMovingAverages` — changes from `Resource` to `Component` on car entities.
  - Fields unchanged. Each car maintains its own rolling window.

### Modified Systems — Concrete Signature Changes

```
// car_physics_system — add ActionState to query, remove Res<ActionState>
fn car_physics_system(
    time: Res<Time<Fixed>>,
    mut query: Query<(&mut Transform, &mut Car, &ActionState)>,
)

// action_smoothing_system — query-based instead of resource-based
fn action_smoothing_system(
    time: Res<Time<Fixed>>,
    smoothing: Res<ActionSmoothing>,
    mut query: Query<&mut ActionState>,
)

// collision_detection_system — iterate all cars, set per-car collision flag
fn collision_detection_system(
    mut car_query: Query<(Entity, &Transform), With<Car>>,
    track_query: Query<&Track>,
    mut commands: Commands,
)
// Adds a `Collided` marker component to the entity if off-road.
// Or: writes to a per-car `CollisionFlag` component.

// episode_loop_system — iterate all cars with per-car episode state
fn episode_loop_system(
    time: Res<Time<Fixed>>,
    config: Res<EpisodeConfig>,
    track_query: Query<&Track>,
    mut car_query: Query<(
        &mut Transform,
        &mut Car,
        &mut TrackProgress,
        &mut EpisodeState,
        &mut EpisodeMovingAverages,
        &SpawnConfig,
        Has<Collided>,              // or &CollisionFlag
    )>,
)
// Loops over all cars. Each car's episode state is self-contained.
// Reset uses SpawnConfig instead of track.spawn_position.

// keyboard_action_input_system — target EnvInstanceId(0) only
fn keyboard_action_input_system(
    mode: Option<Res<AgentMode>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&EnvInstanceId, &mut ActionState)>,
)

// a2c_act_all_cars_system (replaces a2c_act_system)
fn a2c_act_all_cars_system(
    mode: Res<AgentMode>,
    mut brain: ResMut<A2cBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut car_query: Query<(&EnvInstanceId, &ObservationVector, &mut ActionState)>,
)
// Iterates all cars, calls model.forward() for each, samples action,
// pushes pre-step data to TrainerRolloutBuffer with env_id,
// writes action to per-car ActionState.

// a2c_collect_rewards_all_cars_system (replaces a2c_collect_reward_system)
fn a2c_collect_rewards_all_cars_system(
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &ObservationVector, &EpisodeState)>,
    mut brain: ResMut<A2cBrain>,
    mut buffer: ResMut<TrainerRolloutBuffer>,
    mut stats: ResMut<A2cTrainingStats>,
)
// Iterates all cars, pushes per-car reward/done to buffer.
// Triggers update when total buffer size >= horizon.
// Bootstrap: for non-terminal cars, passes their current obs to model.forward()
//   for the bootstrap value. For terminal cars, bootstrap = 0.

// capture_episode_action_stats_system — per-car
fn capture_episode_action_stats_system(
    car_query: Query<(&EnvInstanceId, &EpisodeState, &ActionState)>,
    mut per_car_analytics: ResMut<PerCarAnalytics>,
)

// capture_episode_tick_trace_system — per-car
fn capture_episode_tick_trace_system(
    mode: Res<AgentMode>,
    car_query: Query<(&EnvInstanceId, &EpisodeState, &ActionState, &SensorReadings)>,
    track_query: Query<&Track>,
    observation_config: Res<ObservationConfig>,
    brain: Res<A2cBrain>,
    buffer: Res<TrainerRolloutBuffer>,
    mut per_car_analytics: ResMut<PerCarAnalytics>,
)

// update_driving_hud_stats_system — reads trainer ranking, not single car
fn update_driving_hud_stats_system(
    ranking: Res<TrainerLiveRanking>,
    car_query: Query<(&EnvInstanceId, &TrackProgress, &EpisodeState)>,
    mut hud_stats: ResMut<DrivingHudStats>,  // rewritten for trainer scope
)

// update_driving_hud_text_system — reads best car + trainer summary
fn update_driving_hud_text_system(
    ranking: Res<TrainerLiveRanking>,
    car_query: Query<(&EnvInstanceId, &TrackProgress, &SensorReadings, &EpisodeState)>,
    // ... text queries ...
)
```

### Wiring Summary

```
startup:
  spawn track once → spawn 25 cars with EnvInstanceId(0..24), SpawnConfig, ActionState,
                      EpisodeState, EpisodeMovingAverages as components

fixed tick (SimSet::Input):
  keyboard_action_input_system  → writes ActionState for EnvInstanceId(0) only (keyboard mode)
  a2c_act_all_cars_system       → writes ActionState.desired for all cars (AI mode)
  action_smoothing_system       → smooths desired → applied for all cars

fixed tick (SimSet::Physics):
  car_physics_system            → reads per-car ActionState, mutates transform/velocity
  capture_episode_action_stats  → records per-car steering/throttle stats

fixed tick (SimSet::Collision):
  collision_detection_system    → checks all cars, marks collided entities

fixed tick (SimSet::Measurement):
  update_track_progress_system  → per-car centreline projection (already multi-car)
  episode_loop_system           → per-car reward/terminal/reset using per-car EpisodeState
  update_sensor_readings_system → per-car raycasts (already multi-car)
  build_observation_vector_system → per-car obs normalisation (already multi-car)
  capture_episode_tick_trace    → per-car trace records
  snapshot_completed_traces     → per-car trace snapshots
  snapshot_completed_action_stats → per-car action snapshots
  a2c_collect_rewards_all_cars  → appends per-car rewards to TrainerRolloutBuffer,
                                  triggers shared update at horizon

update:
  episode_tracker_system        → folds per-car completed episodes into EpisodeTracker
  update_trainer_ranking_system → recomputes TrainerLiveRanking from per-car EpisodeMovingAverages
  update_car_visual_roles       → adjusts sprite alpha/colour based on ranking
  debug overlays + HUD          → target best car for detail, show trainer summary

last:
  a2c_flush_on_exit             → flush partial TrainerRolloutBuffer
  on_exit_system                → export JSON + Markdown with per-car + cohort data
```

## Algorithm / System Sections

### 1. Instance-scoped environment state

The first job is to break the singleton assumptions cleanly. Right now the runtime assumes one car, one progress state, one action state, and one episode state. Vectorised A2C cannot be layered on top of those assumptions safely; trying to do so would produce hidden coupling and misaligned analytics.

The recommended default is to keep one visible track entity and to make every car-scoped runtime concept explicitly instance-scoped. That means per-car components for action, progress, observation, and episode truth.

- [ ] Discovery (completed by audit):

  **Car-query singletons requiring `.iter()` migration** (8 systems):

  | System | File:Line | Current Query |
  |--------|-----------|---------------|
  | `collision_detection_system` | `collision.rs:24` | `car_query.single()` |
  | `episode_loop_system` | `episode.rs:187` | `car_query.single_mut()` |
  | `a2c_act_system` | `a2c/mod.rs:152` | `obs_query.single()` |
  | `a2c_collect_reward_system` | `a2c/mod.rs:192` | `obs_query.single()` |
  | `a2c_flush_on_exit_system` | `a2c/mod.rs:222` | `obs_query.single()` |
  | `update_driving_hud_stats_system` | `hud.rs:386` | `progress_query.single()` |
  | `update_driving_hud_text_system` | `hud.rs:469` | `car_query.single()` |
  | `capture_episode_tick_trace_system` | `trackers/trace.rs:94` | `sensor_query.single()` |

  **Singleton Resources that must become per-car** (5 resources):

  | Resource | Current Scope | Target |
  |----------|---------------|--------|
  | `EpisodeState` (27 fields) | `Res` | Component |
  | `EpisodeMovingAverages` | `Res` | Component |
  | `ActionState` | `Res` | Component |
  | `EpisodeActionAccumulator` | `Res` | Indexed in `PerCarAnalytics` |
  | `EpisodeTraceAccumulator` | `Res` | Indexed in `PerCarAnalytics` |

  **Hidden singleton coupling in already-iterating systems** (1 critical):

  | System | File:Line | Problem |
  |--------|-----------|---------|
  | `car_physics_system` | `physics.rs:27` | Reads `Res<ActionState>` then applies same action to ALL cars in `.iter_mut()` loop |

- [ ] Implementation playbook:
  - [ ] Introduce `EnvInstanceId(u32)` as a component.
  - [ ] Introduce `SpawnConfig { position: Vec2, rotation: f32 }` as a component.
  - [ ] Convert `ActionState` from `Resource` to `Component`:
    - Remove `init_resource::<ActionState>()` from `AgentPlugin`.
    - Add `ActionState::default()` to the car spawn bundle.
    - Update `car_physics_system` to read `&ActionState` from the query tuple instead of `Res<ActionState>`.
    - Update `action_smoothing_system` to iterate `Query<&mut ActionState>` instead of `ResMut<ActionState>`.
    - Update `keyboard_action_input_system` to query `(&EnvInstanceId, &mut ActionState)` and only write to `EnvInstanceId(0)`.
  - [ ] Convert `EpisodeState` from `Resource` to `Component`:
    - Remove `init_resource::<EpisodeState>()` from `GamePlugin`.
    - Add `EpisodeState::default()` to the car spawn bundle.
    - Rewrite `episode_loop_system` to iterate all cars with `(&mut Transform, &mut Car, &mut TrackProgress, &mut EpisodeState, &mut EpisodeMovingAverages, &SpawnConfig, ...)`.
    - Each iteration handles reward, terminal check, and reset independently.
    - Use `SpawnConfig` instead of `track.spawn_position` for reset target.
  - [ ] Convert `EpisodeMovingAverages` from `Resource` to `Component`:
    - Remove `init_resource::<EpisodeMovingAverages>()` from `GamePlugin`.
    - Add `EpisodeMovingAverages::default()` to the car spawn bundle.
  - [ ] Refactor `CollisionEvent` to carry per-car identity:
    - **Recommended approach:** Replace the global message with a `Collided` marker component. `collision_detection_system` iterates all cars, adds `Collided` to any car with an off-road corner, removes it from cars that are on-road. `episode_loop_system` checks `Has<Collided>` in its query.
    - This eliminates the message-based coupling entirely and keeps collision truth local to each car entity.
  - [ ] Replace singleton queries with `for` loops over all cars.
  - [ ] Keep one shared `Track` query — track is genuinely singleton.
- [ ] Stop-and-verify checkpoints:
  - [ ] `cargo check` passes with all Resource→Component migrations.
  - [ ] The runtime can spawn 25 cars without panic from `single()` assumptions.
  - [ ] Each car has independent progress, reward, and reset behaviour.
  - [ ] One car crashing does not reset any other car.
  - [ ] `car_physics_system` applies each car's own `ActionState`, not a global one.
- [ ] Invariants / sanity checks:
  - [ ] Every training car has exactly one `EnvInstanceId`.
  - [ ] Every training car has exactly one `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, and `SpawnConfig`.
  - [ ] There are no remaining singleton-car queries in the training path.
  - [ ] No `Res<ActionState>` or `Res<EpisodeState>` remains anywhere in the codebase.
- [ ] Minimal explicit test requirements:
  - [ ] Add at least one ECS-level test or runtime assertion proving that two cars can terminate independently in the same tick range without shared-state corruption.
  - [ ] Add a test that spawns two cars with different `SpawnConfig` values and verifies reset returns each to its own position.

### 2. Multi-car spawn, reset, and visibility model

The visual goal changes the implementation significantly. This plan is not for a mostly headless trainer with one representative viewport. The user wants to see all 25 cars at once, with the best current performer fully coloured and the rest semi-transparent.

The recommended default is one shared track with 25 non-colliding cars because car-to-car collisions are not part of the current environment truth. That keeps the environment semantics stable while making the trainer visually inspectable.

- [ ] Discovery (completed by audit):
  - `spawn_car()` in `car.rs:31` spawns one car with hardcoded red colour `Color::srgb(0.9, 0.2, 0.2)` and z=10.0.
  - `setup_game()` in `plugin.rs:50` calls `spawn_car()` once at `track.spawn_position`.
  - Camera is `Camera2d::default()` — no follow logic, stays at origin. Track is positioned so the default view covers it. This remains correct for 25 cars on the same track.
  - No camera-follow logic exists or is needed.
- [ ] Implementation playbook:
  - [ ] Replace `setup_game` with `spawn_training_cars_system`:
    - Read `TrainerConfig` for `num_envs` and `spawn_lateral_spread`.
    - Compute deterministic spawn offsets: lateral offsets evenly spaced within `[-spawn_lateral_spread, +spawn_lateral_spread]` perpendicular to the track tangent at spawn. No heading jitter — keep starts fair.
    - Example: for 25 cars with spread 50.0, offsets range from -50 to +50 in steps of ~4.17 pixels perpendicular to spawn heading.
    - Each car gets `SpawnConfig { position: offset_position, rotation: track.spawn_rotation }`.
  - [ ] Expand `spawn_car()` signature:
    ```
    pub fn spawn_car(
        commands: &mut Commands,
        env_id: u32,
        spawn_config: SpawnConfig,
        base_alpha: f32,
    )
    ```
    - Attach: `EnvInstanceId(env_id)`, `SpawnConfig`, `ActionState::default()`, `EpisodeState::default()`, `EpisodeMovingAverages::default()`.
    - Set sprite colour with alpha from `base_alpha` (e.g., `Color::srgba(0.9, 0.2, 0.2, 0.3)` for non-best cars).
  - [ ] Add a `RenderRole` component:
    ```
    enum RenderRole { Default, Best }
    ```
    Derived from `TrainerLiveRanking`, updated by `update_car_visual_roles_system`.
  - [ ] Ensure `reset_car_to_spawn` uses `SpawnConfig` fields:
    ```
    fn reset_car_to_spawn(transform: &mut Transform, car: &mut Car, spawn: &SpawnConfig) {
        transform.translation.x = spawn.position.x;
        transform.translation.y = spawn.position.y;
        transform.rotation = Quat::from_rotation_z(spawn.rotation);
        car.velocity = Vec2::ZERO;
    }
    ```
  - [ ] Z-ordering: best car at z=11.0, default cars at z=10.0, so the best car renders on top.
- [ ] Stop-and-verify checkpoints:
  - [ ] All 25 cars are visible at startup, spread perpendicular to the track at the start.
  - [ ] Default opacity is visibly lower for non-best cars.
  - [ ] Best-car highlight updates when rankings change.
  - [ ] Visual role changes do not affect physics or training logic.
  - [ ] After reset, each car returns to its own spawn offset, not the canonical track spawn.
- [ ] Invariants / sanity checks:
  - [ ] Rendering role is derived from `TrainerLiveRanking`, never hand-authored independently in debug code.
  - [ ] Car-to-car overlap must not create gameplay truth because collisions remain track-only.
  - [ ] Spawn staggering must be deterministic given a seed/config.
  - [ ] All spawn offsets are on driveable track surface (verify against `track.grid.is_road_at()`).
- [ ] Minimal explicit test requirements:
  - [ ] Add at least one test or assertion proving per-car reset returns to that car's assigned spawn transform.

### 3. Shared-policy synchronous rollout collection

This is the actual vectorised A2C core. The shared policy must act for all cars each tick, and all per-car transitions must be gathered into one coherent trainer batch before one update is taken. That is the part that makes the trainer "A2C" rather than 25 unrelated single-agent runs.

The recommended default is one shared policy and one shared critic, with a trainer rollout buffer that flattens all transitions but preserves `env_id` tagging so debugging and per-instance analytics remain possible.

- [ ] Discovery (completed by audit):

  **Current act path** (`a2c/mod.rs:73-116`):
  - `Brain::act()` calls `model.forward(&obs.values)` → `(ActionDist, value)`.
  - Samples from Gaussian, applies tanh squashing + throttle rescaling.
  - Pushes states, actions, latent_actions, safety_clamp_hits, values to `self.buffer` internally.
  - Returns `CarAction`.

  **Current collect path** (`a2c/mod.rs:159-197`):
  - Reads `EpisodeState.current_tick_reward` and `current_tick_end_reason`.
  - Pushes reward and done to buffer.
  - Triggers update when `buffer.states.len() >= max_steps` OR (done AND `buffer.states.len() >= min_update_steps`).

  **Current buffer** (`buffer.rs:1-59`):
  - Flat vectors, no env_id tagging.
  - `compute_gae()` iterates in reverse, uses `dones[t]` to mask bootstrapping — this already works correctly for interleaved multi-env transitions as long as dones are set correctly per transition.

  **Current update** (`update.rs:11-248`):
  - Takes `&mut A2cBrain` and `bootstrap_state: Option<&[f32]>`.
  - Single bootstrap value for the entire buffer — this must change for multi-env because different cars may be terminal or non-terminal at the buffer boundary.

  **Key insight on GAE with interleaved envs:**
  The current `compute_gae()` already handles `dones` per-transition correctly. However, it computes a single bootstrap value at the end of the buffer. With interleaved multi-env transitions, the last transition in the buffer might be from env 3, but env 7's last transition (somewhere in the middle) also needs its own bootstrap. Two approaches:
  - **Recommended:** At update time, for each env that has a non-terminal last transition in the buffer, compute a bootstrap value by running `model.forward()` on that env's current observation. Then rewrite the buffer's GAE to handle per-env bootstrap values by looking up the correct bootstrap for each env's final transition.
  - **Alternative:** Use per-env sub-buffers that are concatenated at update time. Simpler GAE but more complex buffer management.

- [ ] Implementation playbook:
  - [ ] Create `TrainerRolloutBuffer` as a new resource:
    ```rust
    pub struct TrainerRolloutBuffer {
        pub states: Vec<Vec<f32>>,
        pub actions: Vec<Vec<f32>>,
        pub latent_actions: Vec<Vec<f32>>,
        pub safety_clamp_hits: Vec<[bool; 2]>,
        pub rewards: Vec<f32>,
        pub values: Vec<f32>,
        pub dones: Vec<bool>,
        pub env_ids: Vec<u32>,
    }
    ```
  - [ ] Refactor `A2cBrain` to no longer own a `RolloutBuffer`:
    - Keep `model: ActorCritic`, hyperparams (`gamma`, `gae_lambda`), and `step_counter`.
    - Remove `buffer` field.
    - Remove `Brain` trait implementation (the vectorised path calls `model.forward()` directly).
  - [ ] Store a seeded `StdRng` in `A2cBrain` for deterministic policy sampling.
  - [ ] Write `a2c_act_all_cars_system`:
    ```
    for (env_id, obs, mut action_state) in car_query.iter_mut() {
        let (dist, value) = brain.model.forward(&obs.values);
        let (car_action, latent, clamp_hits) = sample_and_squash(&dist, &mut brain.rng);
        action_state.desired = car_action;
        buffer.push_pre_step(env_id.0, obs.values.to_vec(), ...);
    }
    brain.step_counter += car_query.iter().count();
    ```
    Note: 25 sequential `model.forward()` calls per tick is fine. The forward pass is a 2×64 MLP (~10K multiplies). Batching would require matrix-level refactoring of the handwritten linear layers for negligible gain at this scale.
  - [ ] Write `a2c_collect_rewards_all_cars_system`:
    ```
    for (env_id, obs, episode_state) in car_query.iter() {
        buffer.push_reward(episode_state.current_tick_reward, done);
    }
    if buffer.len() >= trainer_config.update_horizon {
        // Compute per-env bootstrap values for non-terminal last transitions
        let bootstraps = compute_per_env_bootstraps(&buffer, &car_query, &mut brain.model);
        a2c_update_vectorised(&mut brain, &mut buffer, &mut stats, &bootstraps);
    }
    ```
  - [ ] Rewrite `compute_gae()` for multi-env awareness:
    - At each position `t` in the reversed buffer, if `t` is the last occurrence of `env_ids[t]` in the buffer AND `dones[t]` is false, use the bootstrap value for that env instead of `values[t+1]`.
    - If `dones[t]` is true, mask as before (next_val * 0).
    - Otherwise, use `values[t+1]` as before (the next transition may be from a different env, but that's fine because `dones` already gate the bootstrap).

    **Wait — this needs more careful thought.** In a flat interleaved buffer, `values[t+1]` might be from a completely different env than `values[t]`. The standard vectorised A2C approach handles this by having `dones[t]` mask the connection: if env 3 is at index 5 and env 7 is at index 6, and env 3 did NOT terminate at index 5, the GAE would incorrectly use env 7's value at index 6 as the bootstrap for env 3.

    **Correct approach:** Compute GAE per-env, not across the flat buffer. Group transitions by `env_id`, compute GAE within each group independently, then reassemble advantages/returns in the original flat order for the update. This is the standard approach in vectorised A2C implementations (e.g., Stable Baselines3).

    ```rust
    pub fn compute_gae_per_env(
        &self,
        bootstrap_values: &HashMap<u32, f32>,  // env_id → bootstrap value
        gamma: f32,
        lambda: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0; self.rewards.len()];
        let mut returns = vec![0.0; self.rewards.len()];

        // Group indices by env_id
        let mut env_indices: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, &eid) in self.env_ids.iter().enumerate() {
            env_indices.entry(eid).or_default().push(i);
        }

        // Compute GAE per env
        for (eid, indices) in &env_indices {
            let bootstrap = bootstrap_values.get(eid).copied().unwrap_or(0.0);
            let mut gae = 0.0;
            for &t in indices.iter().rev() {
                let next_val = /* next index for this env, or bootstrap */;
                let mask = if self.dones[t] { 0.0 } else { 1.0 };
                let delta = self.rewards[t] + gamma * next_val * mask - self.values[t];
                gae = delta + gamma * lambda * mask * gae;
                advantages[t] = gae;
                returns[t] = gae + self.values[t];
            }
        }

        // Normalize advantages globally across all envs
        normalize_advantages(&mut advantages);
        (advantages, returns)
    }
    ```
  - [ ] Update horizon semantics:
    - **Recommended:** Update when total transitions across all cars reaches `update_horizon` (default: keep at 512, so with 25 cars this triggers roughly every 20 ticks — much faster updates than singleton mode).
    - Consider raising the horizon to maintain similar update frequency: e.g., `512 * num_envs / baseline_scaling_factor`.
    - The `min_update_steps` check on terminal batches now applies to total buffer size, not per-env.
  - [ ] Continue to clear rollout state on mode switches (`F4`), but now clear `TrainerRolloutBuffer`.
- [ ] Stop-and-verify checkpoints:
  - [ ] Batch sizes are roughly `num_envs × ticks_between_updates`.
  - [ ] The number of rewards always matches the number of stored state/action transitions.
  - [ ] Partial terminal episodes from some cars do not corrupt non-terminal fragments from others.
  - [ ] GAE values are computed per-env with correct bootstrapping — no cross-env value leakage.
- [ ] Invariants / sanity checks:
  - [ ] For every buffer index `i`, all rollout fields share the same `env_id`.
  - [ ] Done masking is applied per transition, not globally per update.
  - [ ] A trainer update never mixes missing reward entries with live state entries.
  - [ ] GAE is computed within env-groups, not across the flat interleaved buffer.
  - [ ] Advantage normalisation is global across all envs in the batch (standard practice).
- [ ] Minimal explicit test requirements:
  - [ ] Unit test for `TrainerRolloutBuffer` alignment with two or more env ids interleaved.
  - [ ] Unit test for per-env GAE with mixed terminal and non-terminal fragments — verify no cross-env value leakage.
  - [ ] Unit test comparing per-env GAE output against the singleton `compute_gae()` when only one env_id is present (regression test).

### 4. Per-car episode logic and ranking model

The old singleton `EpisodeState` currently mixes environment truth, reward decomposition, and last-episode summaries for one car. In the vectorised trainer, `EpisodeState` becomes a component and each car runs its own independent episode loop. The ranking system is a new layer built on top of per-car episode data.

The ranking should not be based on one noisy scalar only. The recommended default is to rank cars by a short rolling performance score built primarily from best progress and return, with explicit tie-breakers and stable hysteresis so the highlight does not flicker every tick.

- [ ] Discovery (completed by audit):

  **EpisodeState field categories:**

  | Category | Fields | Count |
  |----------|--------|-------|
  | Current-tick facts | `current_tick_reward`, `current_tick_progress_reward`, `current_tick_time_penalty`, `current_tick_terminal_reward`, `current_tick_end_reason`, `current_tick_progress_fraction`, `current_tick_progress_s`, `current_tick_centerline_distance`, `current_tick_speed`, `current_tick_heading_error`, `current_tick_forward`, `current_tick_tangent` | 12 |
  | Current-episode accumulators | `current_episode`, `ticks_in_episode`, `previous_progress_fraction`, `lap_armed`, `current_return`, `current_progress_reward_sum`, `current_time_penalty_sum`, `current_terminal_reward_sum`, `current_crash_penalty_sum`, `current_lap_bonus_sum`, `current_best_progress_fraction`, `current_crashes` | 12 |
  | Last-episode summaries | `last_end_reason`, `last_episode_return`, `last_episode_pre_terminal_return`, `last_episode_progress_reward_sum`, `last_episode_time_penalty_sum`, `last_episode_terminal_reward_sum`, `last_episode_crash_penalty_sum`, `last_episode_lap_bonus_sum`, `last_episode_best_progress_fraction`, `last_episode_crashes`, `last_episode_ticks`, `last_episode_crash_position` | 12 |

  All 27 fields (actually 36 by count above — 12+12+12) are per-car and move to the component wholesale. No splitting needed — the struct is already cohesive around one car's episode lifecycle.

  **Safe metrics for live ranking:**
  - `EpisodeMovingAverages.best_progress_mean` — most stable signal
  - `EpisodeMovingAverages.return_mean` — secondary signal
  - `EpisodeMovingAverages.crash_mean` — tie-breaker (lower is better)

- [ ] Implementation playbook:
  - [ ] `EpisodeState` and `EpisodeMovingAverages` become components (covered in Section 1).
  - [ ] `episode_loop_system` iterates all cars. Each iteration is self-contained — the current implementation already takes `episode_state` as a mutable reference and `moving_avg` as a mutable reference; the only change is that these come from the query tuple instead of `ResMut`.
  - [ ] `finalize_episode()` helper stays as-is but takes `&mut EpisodeState` and `&mut EpisodeMovingAverages` directly (already does).
  - [ ] `reset_car_to_spawn()` takes `&SpawnConfig` instead of `&Track`.
  - [ ] Create `TrainerLiveRanking` resource:
    ```rust
    pub struct TrainerLiveRanking {
        pub best_env_id: Option<u32>,
        pub worst_env_id: Option<u32>,
        pub rankings: Vec<(u32, f32)>,  // (env_id, score) sorted descending
        pub last_update_tick: u64,
        pub hysteresis_margin: f32,     // e.g., 0.05 (5% relative improvement to overtake)
    }
    ```
  - [ ] Create `update_trainer_ranking_system` running in `Update`:
    ```
    fn update_trainer_ranking_system(
        trainer_config: Res<TrainerConfig>,
        mut ranking: ResMut<TrainerLiveRanking>,
        car_query: Query<(&EnvInstanceId, &EpisodeMovingAverages)>,
    )
    ```
    - Ranking score = `0.7 * best_progress_mean + 0.3 * return_mean_normalised`.
    - Hysteresis: new best must exceed current best by `hysteresis_margin` to take over.
    - Update at bounded cadence (e.g., every 60 ticks = 1 second) to prevent flicker.
    - Cars with zero completed episodes get a score of -∞ (not eligible for best).
  - [ ] Create `update_car_visual_roles_system` running in `Update` after ranking:
    ```
    fn update_car_visual_roles_system(
        ranking: Res<TrainerLiveRanking>,
        trainer_config: Res<TrainerConfig>,
        mut car_query: Query<(&EnvInstanceId, &mut Sprite, &mut Transform)>,
    )
    ```
    - Best car: full alpha, z=11.0.
    - All others: reduced alpha, z=10.0.
- [ ] Stop-and-verify checkpoints:
  - [ ] The highlighted car remains stable long enough to be visually meaningful (at least 1 second between changes).
  - [ ] Cars can terminate and reset independently without affecting trainer ranking bookkeeping.
  - [ ] Worst-car and percentile groups update correctly as episodes accumulate.
  - [ ] A newly spawned car that hasn't completed any episodes is not ranked as best or worst.
- [ ] Invariants / sanity checks:
  - [ ] Ranking source of truth lives in `TrainerLiveRanking`, not in debug code.
  - [ ] A car that just reset is not automatically considered best or worst without actual episode data.
  - [ ] Live ranking windows and exported analytics windows are documented separately if they differ.
- [ ] Minimal explicit test requirements:
  - [ ] Add at least one deterministic ranking test covering ties, resets, and flicker-prevention behaviour.

### 5. Analytics redesign for cohort summaries

The analytics model must support per-car traces plus trainer-level cohort summaries in the same run. The recommended default is to keep raw per-car records, then derive grouped cohort summaries in the metrics/export layer.

- [ ] Discovery (completed by audit):

  **Current analytics singleton resources:**

  | Resource | File | Singleton Assumption |
  |----------|------|---------------------|
  | `EpisodeActionAccumulator` | `trackers/action.rs:17` | Single `episode_id`, single `steering_sum`/`throttle_sum` |
  | `EpisodeTraceAccumulator` | `trackers/trace.rs:17` | Single `episode_id`, single `ticks: Vec<TickTraceRecord>` |
  | `EpisodeTracker` | `models.rs:141` | Flat `Vec<EpisodeRecord>` with no env_id |

  **Current schemas missing `env_id`:**
  - `EpisodeRecord` — no env_id field
  - `TickTraceRecord` — no env_id field
  - `EpisodeTrace` — no env_id field
  - `A2cUpdateRecord` — no env_id (but this is trainer-wide, so it stays without)

- [ ] Implementation playbook:
  - [ ] Add `env_id: u32` field to `EpisodeRecord`, `TickTraceRecord`, and `EpisodeTrace`.
  - [ ] Create `PerCarAnalytics` resource:
    ```rust
    pub struct PerCarAnalytics {
        pub action_accumulators: HashMap<u32, EpisodeActionAccumulator>,
        pub trace_accumulators: HashMap<u32, EpisodeTraceAccumulator>,
    }
    ```
    Initialise with entries for all `num_envs` at startup.
  - [ ] Remove singleton `EpisodeActionAccumulator` and `EpisodeTraceAccumulator` resources.
  - [ ] Update `capture_episode_action_stats_system` and `capture_episode_tick_trace_system` to iterate all cars and index into `PerCarAnalytics` by `env_id`.
  - [ ] Update `episode_tracker_system` to fold per-car completed episodes into `EpisodeTracker`, now with `env_id` tags on each record.
  - [ ] Add cohort summary types and computation:
    ```rust
    pub struct CohortSummary {
        pub best: PercentileBucket,
        pub top_25: PercentileBucket,
        pub middle_50: PercentileBucket,
        pub bottom_25: PercentileBucket,
        pub worst: PercentileBucket,
        pub overall_mean: f32,
        pub overall_std: f32,
    }

    pub struct PercentileBucket {
        pub mean_progress: f32,
        pub mean_return: f32,
        pub mean_crashes: f32,
        pub mean_centerline_distance: f32,
        pub mean_abs_heading_error: f32,
        pub mean_ticks: f32,
    }
    ```
  - [ ] Compute cohort summaries at export time from `EpisodeTracker.episodes` grouped by `env_id`, then ranked by per-car aggregate performance.
  - [ ] Update Markdown report structure:
    - Section 1: Trainer-wide summary (total episodes, total updates, overall progress curve).
    - Section 2: Cohort breakdown table (best / top-25% / mid-50% / bottom-25% / worst).
    - Section 3: Per-update A2C health (unchanged).
    - Section 4: Per-car detail appendix (optional, toggled by export config).
  - [ ] Update JSON export to include `env_id` on all per-car records and add a `cohort_summary` top-level key.
- [ ] Stop-and-verify checkpoints:
  - [ ] Exported JSON contains stable `env_id` tags for every per-car record.
  - [ ] Markdown report clearly shows requested cohort groupings.
  - [ ] Aggregate means and percentile splits reconcile with raw per-car data.
- [ ] Invariants / sanity checks:
  - [ ] Best/worst reported in analytics uses the same documented metric family as live trainer ranking, or the distinction is made explicit.
  - [ ] Quartile buckets are well-defined even when the number of completed cars/episodes is not divisible cleanly (use floor-based bucketing).
  - [ ] No per-car episode is recorded twice.
- [ ] Minimal explicit test requirements:
  - [ ] Unit tests for cohort bucketing and aggregate statistics.
  - [ ] At least one exporter test or golden-output check for best/worst and percentile sections.

### 6. HUD and overlay redesign for trainer-wide observability

The current HUD is a single-car driving diagnostics panel. In a 25-car trainer, the HUD must answer:

1. How is the trainer doing overall?
2. What is the currently highlighted best car doing right now?

The recommended default is a split HUD: trainer summary panel + focused best-car detail line.

- [ ] Discovery (completed by audit):

  **Current HUD singleton assumptions:**

  | System | File:Line | Singleton Pattern |
  |--------|-----------|-------------------|
  | `update_driving_hud_stats_system` | `hud.rs:386` | `progress_query.single()` on `TrackProgress` |
  | `update_driving_hud_text_system` | `hud.rs:469` | `car_query.single()` on `(TrackProgress, SensorReadings)` |
  | `capture_driving_hud_episode_metrics_system` | `hud.rs:~` | Reads singleton `EpisodeState` |

  **HUD resources that need trainer-scope rewrite:**
  - `DrivingHudStats` — currently tracks single-car deaths and best progress
  - `DrivingHudEpisodeAccumulator` — per-tick accumulation for one car
  - `DrivingHudHistory` — VecDeque of one car's episode snapshots

  **Overlays** already iterate all cars and will automatically draw for all 25. The only change needed is potentially filtering sensor overlays to the best car only (25 sets of raycasts would be visual noise).

- [ ] Implementation playbook:
  - [ ] Rewrite `DrivingHudStats` for trainer scope:
    ```rust
    pub struct DrivingHudStats {
        pub total_episodes: u32,
        pub total_crashes: u32,
        pub best_ever_progress: f32,
        pub best_ever_env_id: u32,
        pub current_best_env_id: Option<u32>,
        pub trainer_mean_progress: f32,
        pub trainer_std_progress: f32,
    }
    ```
  - [ ] Rewrite `DrivingHudEpisodeAccumulator` to read from trainer-wide data rather than accumulating per-tick for one car.
  - [ ] Replace singleton car queries in HUD systems with:
    - Trainer summary: read from `TrainerLiveRanking` + aggregate across all cars' `EpisodeMovingAverages`.
    - Best-car focus: query car entity matching `ranking.best_env_id` for live progress/offset/heading.
  - [ ] HUD text layout:
    ```
    ┌─ Trainer Summary ──────────────────────────────┐
    │ Envs: 25  Episodes: 1247  Updates: 38          │
    │ Best: env#7 (52% progress)  Worst: env#19      │
    │ Mean progress: 34% ± 12%   Crash rate: 0.4     │
    │ A2C: loss=0.12  entropy=1.3  ev=0.67           │
    ├─ Best Car (env#7) ─────────────────────────────┤
    │ Progress: 52%  Offset: 3.2  Heading: -0.04     │
    │ Speed: 210  Reward: +0.8  Life: 340 ticks      │
    └────────────────────────────────────────────────┘
    ```
  - [ ] Sensor overlays: filter to best car only in `draw_sensor_overlay_system` by checking `EnvInstanceId` against `ranking.best_env_id`. Geometry overlays can remain for all cars (they're less cluttered).
  - [ ] Quarter-summary table: adapt to show trainer-wide cohort progress over time windows rather than single-car episode batches.
- [ ] Stop-and-verify checkpoints:
  - [ ] HUD remains readable with 25 cars on screen.
  - [ ] Best-car focus data updates correctly when leadership changes.
  - [ ] Sensor overlay follows the best car only.
  - [ ] Geometry overlays render for all cars without excessive clutter.
- [ ] Invariants / sanity checks:
  - [ ] HUD must not compute its own competing trainer statistics if `TrainerLiveRanking` or analytics resources already own them.
  - [ ] Focus-car overlay target must be derived from `TrainerLiveRanking`.
- [ ] Minimal explicit test requirements:
  - [ ] At least one small test for trainer assessment logic if the current heuristic is rewritten.

### 7. Staged migration plan

This change touches almost every runtime layer. A direct big-bang rewrite is risky. The recommended execution strategy is staged migration where each stage compiles and passes `cargo test` before the next begins.

- [x] **Stage 1 — Per-car components and multi-car spawn** (Sections 1 + 2) — **COMPLETE**
  - [x] Converted `ActionState`, `EpisodeState`, `EpisodeMovingAverages` from Resources to Components.
  - [x] Added `EnvInstanceId`, `SpawnConfig`, `CarColour`, `TrainerConfig` components/resource.
  - [x] Spawned `num_envs` cars (default 3) with deterministic lateral offsets and unique colours.
  - [x] Migrated `car_physics_system` to read per-car `ActionState`.
  - [x] Migrated `action_smoothing_system` to query-based.
  - [x] Replaced `CollisionEvent` with per-car `Collided` marker component.
  - [x] Migrated `episode_loop_system` to iterate all cars with per-car state, reset via `SpawnConfig`.
  - [x] Migrated `keyboard_action_input_system` to target `EnvInstanceId(0)`.
  - [x] Added shims for analytics/HUD systems targeting first car.
  - **Gate passed:** `cargo check` clean, `cargo test` 4/4 passed.

- [x] **Stage 2 — Vectorised A2C rollout** (Section 3) — **COMPLETE**
  - [x] Created `TrainerRolloutBuffer` resource with `env_ids` field and `compute_gae_per_env()`.
  - [x] Refactored `A2cBrain` to remove buffer, added seeded `StdRng`.
  - [x] Wrote `a2c_act_all_cars_system` — loops all cars, samples from shared policy, pushes to trainer buffer.
  - [x] Wrote `a2c_collect_rewards_all_cars_system` — per-car reward/done, per-env bootstrap, shared update.
  - [x] Updated `a2c_update` to accept `TrainerRolloutBuffer` and `HashMap<u32, f32>` bootstraps.
  - [x] Updated `a2c_flush_on_exit_system` for trainer scope.
  - [x] Added 2 unit tests: single-env GAE regression, multi-env GAE isolation.
  - **Gate passed:** `cargo check` clean, `cargo test` 6/6 passed.

- [x] **Stage 3 — Ranking and visual roles** (Section 4) — **COMPLETE**
  - [x] Created `TrainerLiveRanking` with hysteresis (5% margin, 60-tick cadence).
  - [x] Wrote `update_trainer_ranking_system` and `update_car_visual_roles_system`.
  - [x] Each car keeps its unique `CarColour`; ranking adjusts alpha and z-order only.
  - [x] Added live leaderboard HUD panel (top-right, F3-toggled) with colour swatches and live progress.
  - **Gate passed:** `cargo check` clean, `cargo test` 6/6 passed.

- [ ] **Stage 4 — Analytics and HUD** (Sections 5 + 6) — **SUPERSEDED**
  - Deferred to a separate full analytics visual overhaul. See `context/plans/analytics-overhaul-brief.md`.
  - Current state: analytics/HUD use temporary shims targeting first car.

- [ ] **Stage 5 — Cleanup** — **DEFERRED**
  - Dead code remains: old `RolloutBuffer` struct, `Brain` trait.
  - Will be cleaned up alongside the PPO upgrade or analytics overhaul.

- [ ] Stop-and-verify checkpoints:
  - [ ] Each stage compiles before the next begins.
  - [ ] No stale singleton path silently drives production behaviour at the end.
- [ ] Invariants / sanity checks:
  - [ ] Temporary shims (if any) are clearly labelled with `// SHIM: remove in Stage 5` comments.
- [ ] Minimal explicit test requirements:
  - [ ] Run `cargo check` and `cargo test` at each stage boundary.

## Integration Points

- [ ] Where it plugs into the existing pipeline:
  - `GamePlugin` startup must spawn 25 cars instead of one, and no longer register `EpisodeState`/`EpisodeMovingAverages`/`ActionState` as resources.
  - `AgentPlugin` must no longer register `ActionState` as a resource. Systems must process all cars via queries.
  - `BrainPlugin` keeps one shared `AgentMode` and one shared `A2cBrain` (model + hyperparams). A2C systems act over all cars. `TrainerRolloutBuffer` is registered as a new resource.
  - `AnalyticsPlugin` must register `PerCarAnalytics` and aggregate trainer-wide records from per-car data.
  - `DebugPlugin` must present trainer-wide summaries plus best-car focus.
- [ ] Order of execution and lifecycle placement:
  - startup spawns track once and cars 25 times
  - every fixed tick:
    - policy acts for all cars (writes per-car `ActionState.desired`)
    - smoothing applies per car (reads/writes per-car `ActionState`)
    - physics runs per car (reads per-car `ActionState`)
    - collisions run per car (sets per-car collision flag)
    - progress and episode logic run per car (reads/writes per-car `EpisodeState`)
    - observations rebuild per car (already multi-car)
    - analytics capture per car (reads per-car state, writes to `PerCarAnalytics`)
    - trainer reward collector appends per-car rewards and updates policy when batch horizon is met
  - update:
    - episode tracker folds completed per-car episodes
    - trainer ranking refreshes (bounded cadence)
    - visual highlight refreshes
    - HUD refreshes
  - last:
    - partial trainer rollout flushes
    - analytics export writes trainer and per-car summaries
- [ ] Pre-conditions:
  - all singleton-car assumptions are identified (see audit above)
  - ranking metric and analytic cohort semantics are defined before exporter work
  - visual opacity/highlight rules are documented before HUD/overlay rewrites
- [ ] Post-conditions:
  - all training cars share one policy
  - all training cars own independent environment truth as components
  - one trainer batch aggregates transitions across cars with per-env GAE
  - exported analytics contain both raw per-car facts and cohort summaries
  - one best-performing current car is visually obvious at runtime

## Debugging / Verification

- [ ] Required logs, assertions, or inspection steps:
  - log trainer startup with `num_envs` and spawn positions
  - log per-update batch size, env-id distribution, and number of terminal transitions
  - assert rollout alignment across all fields (states.len() == actions.len() == rewards.len() == env_ids.len())
  - assert per-car reset only changes that car's transform (not others)
  - log best-car changes with rank score, env id, and previous best
  - inspect exported analytics for best/worst/quartile consistency
- [ ] Manual inspection steps:
  - visually confirm 25 cars are rendered at startup
  - confirm non-best cars are semi-transparent
  - confirm the best car changes highlight when another car outperforms it
  - confirm multiple cars can crash/reset in the same tick without affecting each other
  - confirm trainer HUD shows mean plus spread, not just one-car stats
  - confirm sensor overlays render only for the best car
  - confirm geometry overlays render for all cars
- [ ] Focused runtime signals to check:
  - update batch size should be ~25× larger than singleton mode (or proportional to horizon setting)
  - explained variance and losses should remain finite
  - quartile progress gaps should be plausible, not all identical
  - best/worst car IDs should not flap every frame without underlying metric changes
  - per-env GAE should produce different advantage values for different envs in the same batch
- [ ] Common failure patterns:
  - leftover `single()` queries panic once more than one car exists
  - `Res<ActionState>` or `Res<EpisodeState>` still used somewhere → compile error after migration (good — catches it early)
  - `car_physics_system` applying same action to all cars (the critical hidden coupling)
  - one car's terminal reward resets or overwrites another car's episode state
  - GAE computed across flat interleaved buffer without per-env grouping → cross-env value leakage
  - rollout buffer `rewards.len()` drifting from `states.len()` due to multi-car push ordering
  - HUD or overlays accidentally following an arbitrary first car instead of the best car
  - analytics bucketing cars by entity order rather than actual performance
  - spawn offsets placing cars off the driveable surface → instant crash on tick 1
  - keyboard mode writing to all cars' `ActionState` instead of just `EnvInstanceId(0)`

## Completion Criteria

- [x] Functional correctness: 3 cars (configurable) run, learn, terminate, and reset independently under one shared A2C policy.
- [x] Visual correctness: all cars visible with unique colours, non-best cars reduced-opacity, best car highlighted with full alpha and z=11.
- [x] Trainer correctness: rollout collection is synchronous and aggregate across cars, with per-env GAE and one coherent update path.
- [ ] Analytics correctness: **deferred** — exports still single-car via shims. Per-car `env_id` tagging and cohort summaries planned in analytics overhaul.
- [x] Integration correctness: no `Res<ActionState>`, `Res<EpisodeState>`, or `Res<EpisodeMovingAverages>` remains. No singleton car queries remain in the training path. Analytics/HUD use labelled shims.
- [x] Tests passing: `cargo check` clean (zero warnings), `cargo test` 6/6 passed, including 2 new per-env GAE tests.
- [x] Context updates completed: all system docs updated to reflect vectorised trainer reality.
- [ ] File removal or archival condition: this plan can be archived once analytics overhaul is complete and Stage 5 cleanup is done. Until then, it serves as reference for the remaining work.
