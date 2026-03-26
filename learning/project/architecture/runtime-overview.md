# Runtime Overview

## Why This Architecture Matters

Understanding NeuroDrive's architecture is not just about navigating the codebase — it is about understanding the design decisions that make the project coherent. The subsystem split is deliberate. The ownership boundaries are enforced. The dependency direction is unidirectional. These are all choices with reasons.

**Status:** Current implementation (Bevy 0.18, Rust).

## Prerequisites

- `concepts/foundations/bevy-ecs-primer.md` — ECS, systems, resources, components, plugins

---

## The Six Subsystems

```text
NeuroDrive/src/
│
├── maps/          ← Track topology (foundational, no runtime deps)
├── game/          ← Car physics, collision, progress, rewards, episodes
├── agent/         ← Observation/action contract (stable interface)
├── brain/         ← Controller implementation (A2C baseline)
├── analytics/     ← Post-run data capture and export
├── debug/         ← Live overlays and HUD
└── sim/           ← Shared schedule ordering contract
```

### Subsystem Responsibilities

| Subsystem | Owns | Does NOT own |
|---|---|---|
| `maps` | Track geometry, centreline, tile semantics, spawn pose | Runtime car state, physics, learning |
| `game` | Car entity, physics, collision, progress, reward, episode lifecycle | Observations, policy logic, analytics export |
| `agent` | Action and observation contract | Physics, reward definition, policy decisions |
| `brain` | Controller mode and the A2C baseline | Reward truth, raw environment state, analytics capture |
| `analytics` | Episode/update capture, derived metrics, export | Simulation truth, training state, reward definitions |
| `debug` | Live overlays, HUD, runtime diagnostics | Simulation truth, training decisions |
| `sim` | Named system set ordering shared across plugins | Any runtime state |

---

## Dependency Direction

The dependency graph is deliberately acyclic:

```
sim        (shared ordering — no deps)
maps       (no runtime deps)
game       (depends on maps, sim)
agent      (depends on game, maps, brain::types for mode)
brain      (depends on agent, game)
analytics  (depends on game, agent, brain — read-only)
debug      (depends on maps, game, agent, brain — read-only)
main       (wires plugin order, owns global config)
```

**Key rule:** lower layers do not import from higher layers. `game` does not depend on `brain`. `agent` only references `brain::types` for the mode enum, not any brain implementation. This keeps the environment independent of whatever controller is plugged in.

---

## The Stable Controller Boundary

The most important design principle in the architecture:

```
                    ┌────────────────┐
                    │   Environment  │
                    │  (maps, game)  │
                    └───────┬────────┘
                            │ EpisodeState (reward, done)
                            │ TrackProgress
                            │ CollisionEvent
                            ▼
                    ┌────────────────┐
                    │     agent/     │
                    │ ObservationVector │
                    │ ActionState    │
                    └───────┬────────┘
                            │ obs → brain
                            │ action ← brain
                            ▼
                    ┌────────────────┐
                    │    brain/      │
                    │ (A2C baseline) │
                    └────────────────┘
```

The `agent/` layer is the stable boundary. It exposes exactly:
- `ObservationVector` — the normalised policy input (23-dim)
- `ActionState` — the desired and applied car control (`CarAction`)

**Why this matters:** Any controller — keyboard, A2C, or the future biological brain — uses the same interface. The environment never needs to know what kind of brain is in control.

---

## The Plugin System

Each subsystem registers as a Bevy plugin:

```rust
app.add_plugins((
    MonacoPlugin,       // maps: builds track, spawns track entity
    GamePlugin,         // game: spawns car, registers physics/episode systems
    AgentPlugin,        // agent: registers observation/action systems
    A2cPlugin,          // brain: registers A2C systems, initialises A2cBrain
    AnalyticsPlugin,    // analytics: registers capture and export systems
    DebugPlugin,        // debug: registers overlays and HUD
));
```

Plugin order matters for resource initialisation but system ordering within `FixedUpdate` is controlled by the `SimSet` contract.

---

## The Single Car Model

The current runtime assumes exactly one car:

- Many queries use `single()` / `single_mut()` which panic if more or fewer than one entity matches
- `EpisodeState` and `ActionState` are global resources (not per-car components)
- Analytics captures per-episode data from the single car

This is a known architectural constraint. The planned vectorised trainer (`context/plans/vectorised-a2c-visual-trainer.md`) will break these singleton assumptions to support 25 concurrent cars.

---

## Subsystem Interaction Summary

```text
Startup:
  MonacoPlugin → spawns Track entity
  GamePlugin   → spawns Camera + Car entity
  All plugins  → initialise resources

FixedUpdate (every 60Hz tick):
  SimSet::Input:
    keyboard_action_input_system (if keyboard mode)
    a2c_act_system               (if AI mode)
    action_smoothing_system

  SimSet::Physics:
    car_physics_system
    capture_episode_action_stats_system

  SimSet::Collision:
    collision_detection_system

  SimSet::Measurement:
    update_track_progress_system
    episode_loop_system          ← reward, terminal, reset
    update_sensor_readings_system
    build_observation_vector_system
    capture_episode_tick_trace_system
    snapshot_completed_episode_*_systems
    a2c_collect_reward_system    ← append reward, maybe update
    update_driving_hud_stats_system
    capture_driving_hud_episode_metrics_system

Update (every frame):
  toggle_agent_mode_system
  episode_tracker_system       (analytics fold)
  debug overlay rendering
  HUD text update

Last (end of frame):
  a2c_flush_on_exit_system
  analytics on-exit export
```

---

## Pressure Points and Future Tension

The current architecture handles the single-car case cleanly. Several pressure points exist for future development:

1. **Singleton assumptions** are the main blocker for vectorised training. Every `single()` query and global resource that stores per-car state needs to be per-entity.

2. **Brain is modular but minimal.** The `Brain` trait in `brain/types.rs` is minimal — it just maps `ObservationVector → CarAction`. A richer pluggable-brain interface will be needed when the biological brain architecture is introduced.

3. **Analytics is exit-triggered.** All data is exported when the app exits. Crash-safe checkpointing is a known gap.

4. **No headless mode.** The runtime requires a window. A headless accelerated training mode would allow much faster experiments.

---

## Related Files

- `project/architecture/fixed-tick-pipeline.md` — the SimSet ordering in detail
- `project/architecture/module-boundaries.md` — ownership and dependency rules
- `project/systems/environment-system.md` — the game/maps layer in depth
- `project/systems/agent-interface.md` — the stable controller boundary
