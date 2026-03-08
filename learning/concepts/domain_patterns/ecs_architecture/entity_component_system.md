# Entity-Component-System Architecture

## 1. What Is This Pattern?

Entity-Component-System (ECS) is an architectural paradigm that separates identity, data, and behaviour into three distinct concepts:

- **Entities** are lightweight identifiers — just an ID, nothing more.
- **Components** are plain data structs attached to entities. No methods, no inheritance, no behaviour.
- **Systems** are functions that operate on sets of components. They contain all the logic.

ECS replaces the deep inheritance hierarchies of traditional object-oriented design with flat composition. Instead of a `RaceCar extends Vehicle extends Entity` class chain, you have an entity with `Position`, `Velocity`, `CarPhysics`, and `Renderable` components — assembled like building blocks.

## 2. When To Use This Pattern

**Good for:**
- Simulations with many entities sharing overlapping but non-identical sets of behaviours
- Performance-critical systems that benefit from cache-friendly data layouts
- Projects requiring modular, independently testable subsystems
- Rapid prototyping where you add/remove capabilities without refactoring class hierarchies

**Not good for:**
- Simple applications with few entity types and minimal shared behaviour
- Systems where deep object identity and encapsulation are critical (banking, access control)
- Codebases where the team has no ECS experience and the learning curve outweighs the benefit

## 3. Core Concept

### Data-Oriented Design

ECS is fundamentally about data-oriented design. Traditional OOP stores data per object: each `Car` instance holds its position, velocity, and rendering data together. ECS stores data per component type: all positions are packed in one array, all velocities in another.

This matters for CPU cache performance. When a physics system iterates over all positions and velocities, it reads contiguous memory — filling cache lines efficiently. In OOP, iterating over car objects would load position, velocity, rendering data, AI state, and everything else into the cache, wasting bandwidth on data the physics system never reads.

### The Three Pieces

**Entity:** Just an integer (or generational index). It holds no data and has no methods. You attach components to it; you query systems by component presence.

**Component:** A struct containing data. Examples: `Position { x: f32, y: f32 }`, `Health { current: i32, max: i32 }`, `Renderable { sprite: Handle<Image> }`. Components are inert — they do nothing on their own.

**System:** A function that queries for entities possessing specific component combinations and operates on them. A physics system queries all entities with `Position` and `Velocity`, then updates positions. A rendering system queries all entities with `Position` and `Renderable`, then draws them.

### Bevy-Specific Concepts

Bevy — the Rust ECS game engine NeuroDrive uses — adds several layers atop the core ECS:

- **Plugins:** Modular registration units. Each plugin registers its components, resources, systems, and sub-plugins. The application is composed by adding plugins.
- **Resources:** Singleton data shared across systems. Unlike components (per-entity), resources exist once globally. Examples: simulation clock, input state, neural network weights.
- **SystemSets:** Named groups used for ordering. You declare "System A runs before set X" and "System B runs in set X" to enforce execution order.
- **Schedules:** Bevy supports multiple schedules: `FixedUpdate` (constant-rate simulation), `Update` (frame-rate rendering), `Last` (cleanup). Each schedule runs its systems independently.

## 4. Key Design Decisions

| Decision | Option A | Option B |
|---|---|---|
| State sharing | Resources (global singletons) | Components (per-entity) |
| System communication | Direct resource mutation (simple) | Events/messages (decoupled) |
| System ordering | Explicit sets/labels (safe) | Implicit / unordered (fast, fragile) |
| Plugin granularity | Many small plugins (modular) | Few large plugins (less boilerplate) |

**Key trade-off:** Modularity vs coupling. Resources are easy to use but create implicit coupling — any system can read/write them. Events provide decoupling but add indirection and buffering complexity. Start with resources for simplicity; introduce events when you need to decouple subsystems that evolve independently.

## 5. Simplified Example Implementation

```rust
// Pseudocode illustrating ECS concepts (not compilable Bevy)

// Components — just data
struct Position { x: f32, y: f32 }
struct Velocity { dx: f32, dy: f32 }
struct Health { current: i32 }

// Resource — global singleton
struct SimClock { tick: u64, dt: f32 }

// System — a function over queries
fn physics_system(clock: Res<SimClock>,
                  mut query: Query<(&mut Position, &Velocity)>) {
    for (mut pos, vel) in &mut query {
        pos.x += vel.dx * clock.dt;
        pos.y += vel.dy * clock.dt;
    }
}

fn damage_system(mut query: Query<&mut Health>,
                 mut events: EventReader<CollisionEvent>) {
    for event in events.read() {
        if let Ok(mut hp) = query.get_mut(event.entity) {
            hp.current -= event.damage;
        }
    }
}

// Plugin — modular registration
struct PhysicsPlugin;
impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimClock>()
           .add_systems(FixedUpdate,
               physics_system.in_set(SimSet::Physics));
    }
}
```

## 6. How NeuroDrive Implements This

**Plugin composition:** The application is built from a chain of plugins, each responsible for one subsystem:

- `MonacoPlugin` — Track geometry, grid, centreline, spawn points
- `AgentPlugin` — Car entity, physics, sensors, actions
- `BrainPlugin` — Neural network, A2C algorithm, rollout buffer
- `AnalyticsPlugin` — Episode tracking, metrics, report generation
- `GamePlugin` — Episode lifecycle, reward computation, reset logic
- `DebugPlugin` — Overlay rendering, telemetry visualisation

Each plugin registers only its own components, resources, and systems. No plugin directly references another plugin's internals — they communicate through shared resources and events.

**SimSet ordering contract:** A `SimSet` enum defines the execution pipeline:

```
Input → Physics → Collision → Measurement → Reward → Brain → Analytics
```

Every system declares its set membership. Bevy enforces that sets execute in the declared order within `FixedUpdate`. This is the backbone of NeuroDrive's determinism — the ordering is explicit, testable, and unchangeable at runtime.

**Resource pattern for shared mutable state:**
- `A2cBrain` — The neural network weights, optimiser state, and rollout buffer. Accessed by the brain system (read/write) and analytics (read).
- `EpisodeState` — Current tick, accumulated reward, best progress, terminal flags. Written by the game system, read by analytics and brain.
- `ActionState` — The current action selected by the brain or human input. Written by the input/brain system, read by the physics system.

**Event pattern for decoupled communication:**
- `CollisionEvent` — Emitted by the collision system when the car hits a wall. Consumed by the episode system to trigger terminal conditions. The collision system does not know about episodes; the episode system does not know about geometry.

**Schedule usage:** All simulation logic runs in `FixedUpdate` (60 Hz). Debug overlays and rendering run in `Update` (variable frame rate). Analytics finalisation runs in `Last` (after all update systems). This separation ensures rendering never affects simulation outcomes.

## 7. Variations

- **Archetype-based ECS (Bevy, flecs):** Groups entities by their component combination (archetype). Queries over a specific archetype read contiguous memory. Very cache-friendly for homogeneous entity sets.
- **Sparse-set ECS (EnTT):** Uses sparse sets for component storage. Faster add/remove operations but potentially less cache-friendly for iteration.
- **Hybrid ECS:** Some engines (Unity DOTS) offer both OOP and ECS, letting developers use ECS for performance-critical paths and OOP elsewhere.
- **Actor model:** Entities are active "actors" that receive messages. More concurrency-friendly but harder to reason about data layout and cache behaviour.

## 8. Common Pitfalls

- **God resources:** Putting too much state in a single resource creates a bottleneck — every system that touches it becomes implicitly ordered. Split resources by access pattern.
- **System ordering bugs:** Two systems that both write to the same component without declared ordering produce non-deterministic results. Always declare ordering when systems share mutable state.
- **Over-decomposition:** Splitting every field into its own component creates excessive query complexity and storage overhead. Group fields that are always accessed together into one component.
- **Ignoring change detection:** Bevy tracks component changes. Systems that run every tick but only need to react to changes waste cycles. Use `Changed<T>` filters.
- **Mixing simulation and rendering state:** If rendering data leaks into simulation components, visual-only changes can affect gameplay. Keep simulation and rendering components strictly separate.

## 9. Projects That Use This Pattern

- **Bevy Engine:** A Rust ECS game engine used by NeuroDrive. Archetype-based, with plugins, schedules, and system sets for modular composition.
- **Unity DOTS (Data-Oriented Technology Stack):** Unity's high-performance ECS framework. Used in production games requiring thousands of entities (e.g., city builders, RTS games).
- **Overwatch (Blizzard):** Uses a custom ECS for networked gameplay. The deterministic simulation and replay system rely on ECS's clean separation of state and logic.

## 10. Glossary

| Term | Definition |
|---|---|
| **Entity** | A lightweight identifier (integer or generational index) with no data or behaviour |
| **Component** | A data-only struct attached to an entity |
| **System** | A function that queries and operates on entities by their component composition |
| **Resource** | A global singleton accessible by any system (Bevy-specific) |
| **Plugin** | A modular unit that registers components, resources, and systems (Bevy-specific) |
| **SystemSet** | A named group used to declare execution ordering between systems (Bevy-specific) |
| **Schedule** | A collection of systems that run together at a specific cadence (FixedUpdate, Update, Last) |
| **Archetype** | The set of component types attached to an entity; entities with the same archetype are stored contiguously |

## 11. Recommended Materials

- **"Bevy Book"** (bevyengine.org/learn/book) — The official Bevy tutorial covering ECS fundamentals, plugins, resources, and system ordering. Start here for Bevy-specific patterns.
- **"Data-Oriented Design"** by Richard Fabian — A comprehensive treatment of why data layout matters for performance, with chapters on ECS and cache-friendly iteration.
- **"Overwatch Gameplay Architecture and Netcode"** (GDC 2017 talk by Timothy Ford) — A production ECS case study showing how ECS enables deterministic simulation, replay, and networked gameplay at scale.
