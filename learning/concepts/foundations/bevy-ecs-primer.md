# Bevy ECS Primer

## Why This Matters Here

NeuroDrive is a Bevy 0.18 application. Bevy uses an **Entity Component System (ECS)** architecture — a fundamentally different way of structuring application state and logic compared to object-oriented design. Without understanding ECS, the NeuroDrive source code looks puzzling: there are no classes with methods that update themselves, no central game loop calling things in sequence, and state is scattered across typed components and resources rather than nested in objects.

This file teaches enough ECS and Bevy to read the NeuroDrive codebase fluently. It is not a comprehensive Bevy tutorial.

## Prerequisites

- Comfortable with any statically typed language
- No prior Bevy or ECS knowledge required

## Notation

| Term | Short definition |
|---|---|
| Entity | A lightweight identifier (an integer) representing a "thing" in the world |
| Component | Data attached to an entity (e.g. `Transform`, `Car`, `TrackProgress`) |
| Resource | Global singleton data shared across the application (e.g. `EpisodeState`, `A2cBrain`) |
| System | A function that runs on a schedule and can read/write components and resources |
| Query | A typed parameter in a system that selects all entities with specific components |
| World | The ECS database — stores all entities, their components, and resources |
| Schedule | An ordered collection of systems that run in a specific phase |
| Plugin | A bundle of systems and resource initialisation registered as a group |

---

## Core Idea

Traditional object-oriented game code looks like this:

```
class Car {
    position: Vec2
    velocity: Vec2

    fn update(dt: f32) { ... }
    fn render() { ... }
}
```

ECS separates data from behaviour:
- **Data** lives in **components** attached to **entities**
- **Behaviour** lives in **systems** that query entities by their component types

The same data can be processed by many different systems without those systems knowing about each other. This is the key to why NeuroDrive's A2C brain, physics system, analytics, and debug overlays can all read the same car state without coupling to each other.

---

## Entities

An entity is just a unique identifier — essentially a number. On its own, it has no meaning. Meaning comes from the components attached to it.

NeuroDrive's car entity has (among others):
- `Transform` — position and rotation
- `Car` — velocity and kinematic parameters
- `TrackProgress` — current centreline projection
- `SensorReadings` — current raycasts and geometry
- `ObservationVector` — the normalised policy input

Adding or removing a component from an entity changes what systems can see it.

---

## Components

A component is a plain data struct tagged with `#[derive(Component)]`.

Example from NeuroDrive:

```rust
#[derive(Component)]
pub struct TrackProgress {
    pub s: f32,
    pub fraction: f32,
    pub closest_point: Vec2,
    pub tangent: Vec2,
    pub distance: f32,
}
```

Components must not contain heavy logic. They hold data; systems act on it.

---

## Resources

A resource is global singleton state accessible by any system. Tagged with `#[derive(Resource)]`.

Examples from NeuroDrive:
- `EpisodeState` — current tick's reward, progress, and end reason
- `A2cBrain` — the actor-critic model and rollout buffer
- `ActionState` — the current desired and applied control action
- `EpisodeConfig` — reward scaling constants

Resources are accessed in systems via `Res<T>` (read-only) or `ResMut<T>` (mutable).

---

## Systems

A system is a Rust function registered with the Bevy app. Bevy calls it automatically based on schedule membership.

A system's parameters declare what it needs:

```rust
pub fn car_physics_system(
    time: Res<Time<Fixed>>,          // read the fixed timestep
    mut car_query: Query<(&mut Transform, &mut Car)>,  // mutate car entities
    action_state: Res<ActionState>,  // read the action
) {
    for (mut transform, mut car) in car_query.iter_mut() {
        // update physics
    }
}
```

Key points:
- `Res<T>` is immutable resource access
- `ResMut<T>` is mutable resource access
- `Query<T>` selects entities that have all the listed components
- Bevy automatically parallelises systems that access disjoint data

---

## Queries

A `Query<T>` selects entities whose component set matches the query's type signature:

```rust
Query<(&Transform, &Car)>          // entities with both Transform and Car
Query<(&mut Transform, &Car)>      // entities where Transform is mutably accessed
Query<(&Car, Without<Keyboard>)>   // entities with Car but without Keyboard
```

The `single()` and `single_mut()` methods are used when exactly one entity is expected:

```rust
let (mut transform, mut car, mut progress) = car_query.single_mut()?;
```

NeuroDrive currently has one car entity, so many queries use `single()`. The planned vectorised trainer will replace these with `for` loops over all car entities.

---

## Schedules and SimSet

Bevy runs systems in named schedules. The most relevant in NeuroDrive are:
- `Startup` — runs once when the app starts
- `FixedUpdate` — runs at a fixed 60 Hz timestep
- `Update` — runs every frame (uncapped frame rate)
- `Last` — runs after Update, last phase per frame

NeuroDrive adds its own ordering abstraction on top of `FixedUpdate`:

```rust
pub enum SimSet {
    Input,
    Physics,
    Collision,
    Measurement,
}
```

These are system sets that enforce ordering within `FixedUpdate`:

```
Input → Physics → Collision → Measurement
```

Every system registered to `FixedUpdate` is placed into one of these sets. The set membership controls when the system runs within each 60 Hz tick. This is crucial: getting the order wrong produces subtle bugs (e.g. reward collected before episode truth is finalised).

See `project/architecture/fixed-tick-pipeline.md` for the full ordering contract and rationale.

---

## Plugins

A plugin is a bundle of systems and resource initialisations. NeuroDrive registers each subsystem as a plugin:

```rust
app.add_plugins((
    MonacoPlugin,
    GamePlugin,
    AgentPlugin,
    A2cPlugin,
    AnalyticsPlugin,
    DebugPlugin,
));
```

Each plugin's `build` method adds its systems and initialises its resources. This keeps subsystem wiring local and avoids one giant `main.rs` that touches everything.

---

## Bevy's Fixed Timestep

NeuroDrive runs the simulation at exactly 60 Hz:

```rust
app.insert_resource(Time::<Fixed>::from_hz(60.0));
```

`FixedUpdate` systems run exactly once per fixed timestep period. If the frame rate is higher than 60 Hz, physics does not run more frequently. If the frame rate drops below 60 Hz, multiple physics updates may run per frame to catch up.

This determinism is foundational: the same sequence of actions produces the same trajectory regardless of frame rate or hardware. All learning systems (A2C act, reward collection) run in `FixedUpdate`, so the training loop is deterministic given fixed actions.

---

## Message Events

Bevy provides a message-passing mechanism for one-shot events. NeuroDrive uses it for:
- `CollisionEvent` — emitted by the collision system, consumed by the episode loop
- `AppExit` — the Bevy engine's built-in exit event, consumed by the analytics flush

Messages are sent and read within the same or successive ticks.

---

## How NeuroDrive Uses ECS Effectively

The ECS model is particularly good for NeuroDrive because:

1. **The brain, physics, and analytics are decoupled.** The A2C brain reads the `ObservationVector` component and writes `ActionState` — it never touches the `Car` transform or `EpisodeState` directly. The environment does not know the brain exists.

2. **Multiple controller modes are transparent to the environment.** Whether control comes from `keyboard_action_input_system` or `a2c_act_system`, the physics system only sees `ActionState.applied`. The control source is irrelevant downstream.

3. **Analytics is a passive consumer.** It reads components and resources without ever mutating training truth. This architectural separation prevents analytics from accidentally influencing training.

4. **Scheduling is explicit.** The `SimSet` ordering is not implicit. Adding a new system requires deliberately placing it in the right set, which makes ordering bugs visible rather than hidden.

---

## Common Misunderstandings

❌ "Systems are like methods on objects"
✅ Systems are standalone functions. They do not belong to any particular entity. They operate on all entities matching their query.

❌ "Resources and components are the same thing"
✅ Resources are global singletons — there is exactly one `A2cBrain` in the world. Components are per-entity — there is one `TrackProgress` per car entity.

❌ "Systems run in the order they are registered"
✅ By default, Bevy may run compatible systems in parallel. The `SimSet` ordering contract and `.before()/.after()` annotations are how NeuroDrive enforces the correct sequence.

---

## Related Files

- `project/architecture/runtime-overview.md` — the full set of NeuroDrive subsystems
- `project/architecture/fixed-tick-pipeline.md` — the SimSet execution chain
- `project/architecture/module-boundaries.md` — which systems own which data
