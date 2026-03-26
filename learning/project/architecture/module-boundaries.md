# Module Boundaries

## Why This Matters

NeuroDrive's architecture enforces strict ownership rules between its subsystems. These rules are not bureaucratic convention — they are the mechanism that keeps the environment independent from the controller, the controller independent from analytics, and the whole system coherent as new learning algorithms are introduced over the project's nine-milestone arc.

Understanding the module boundaries is essential for:
- knowing where to add code for a new feature,
- understanding why certain data flows in one direction only,
- avoiding coupling that would make replacing A2C with a biological brain difficult later.

**Status:** Current implementation. These boundaries reflect the live codebase.

## Prerequisites

- `concepts/foundations/bevy-ecs-primer.md` — plugins, components, resources
- `project/architecture/runtime-overview.md` — the six subsystems and their purposes

---

## The Seven Modules

```text
src/
├── sim/          ← shared schedule ordering, no runtime state
├── maps/         ← static track topology, no runtime deps
├── game/         ← physics, collision, progress, rewards, episode lifecycle
├── agent/        ← observation and action contract
├── brain/        ← controller mode and A2C implementation
├── analytics/    ← capture, metrics, export
└── debug/        ← live overlays, HUD, diagnostics
```

### `sim/`

**Owns:** `SimSet` enum — the named fixed-tick ordering sets shared by all `FixedUpdate` systems.

**Does not own:** any runtime state, any ECS resources, any component data.

**Why it exists as a separate module:** Every plugin that registers `FixedUpdate` systems needs access to `SimSet`. If `SimSet` lived in `game/` or any other module, every downstream module would need to depend on that module just to access ordering constants. `sim/` is a pure schema module — a dependency sink that nobody needs to be insulated from.

---

### `maps/`

**Owns:**
- Track topology: the closed tile path that defines the circuit
- `TrackGrid`: driveable-area query used by collision detection and raycasts
- `TrackCenterline`: closed-loop centreline used for progress measurement and lookahead
- Spawn pose for the car
- Track visual geometry

**Does not own:** runtime car state, physics, learning, episode boundaries, analytics.

**Why it is foundational:** The track is a static structure. It does not change at runtime. All other modules that need spatial information about the track read from the `Track` component that `maps/` creates at startup. No other module writes back to track geometry.

---

### `game/`

**Owns:**
- Car entity lifecycle (spawning, component layout)
- Car physics via `step_car_dynamics()` and `car_physics_system`
- Collision detection: `collision_detection_system` emits `CollisionEvent`
- Progress measurement: `update_track_progress_system` projects car position onto centreline
- Reward accumulation: `episode_loop_system` computes per-tick rewards
- Episode boundaries: crash, timeout, lap completion, car reset
- Moving averages over recent episodes: `EpisodeMovingAverages`

**Does not own:**
- Observation construction (that is `agent/`)
- Policy updates (that is `brain/`)
- Analytics export (that is `analytics/`)
- Visual overlays (that is `debug/`)

**Why this boundary matters:** `game/` defines what is *true* about the world — where the car is, whether it crashed, what reward was earned. This truth must not depend on what controller is active. The same reward and collision logic runs whether a keyboard player or an A2C brain is in control.

---

### `agent/`

**Owns:**
- `CarAction` and `ActionState`: the stable controller-facing action contract
- `ObservationVector`: the 23-dimensional normalised policy input
- `SensorReadings`: raw world-derived measurements (ray distances, centreline-relative geometry)
- Observation production: raycasting, lookahead sampling, feature normalisation
- Action smoothing (currently disabled)

**Does not own:** physics, reward truth, policy logic, analytics export.

**The stable boundary principle:**

```
Environment (maps, game) → agent/ → brain/
```

The `agent/` module is the only thing standing between the raw world state and the controller. Any controller — keyboard, A2C, or future biological brain — consumes exactly `ObservationVector` and writes to `ActionState.desired`. The environment never needs to know which controller is active.

**Why this matters for the long-term project:** NeuroDrive is going to replace A2C with a biologically-inspired system across several milestones. If `brain/` depended directly on `game/` internals, that replacement would require touching every piece of the environment that the brain touched. The `agent/` boundary ensures that the only thing the brain ever reads is a clean, stable, versioned observation vector.

---

### `brain/`

**Owns:**
- `AgentMode`: keyboard vs AI mode switch
- `Brain` trait: the minimal pluggable-controller interface
- A2C implementation: model, rollout buffer, GAE, update
- Handwritten ML primitives: `Linear`, `Relu`, `Adam` in `brain/common/`
- `biological/` placeholder for future local-plasticity implementations

**Does not own:** reward definition, raw physics state, observation production, analytics capture.

**What it reads:**
- `ObservationVector` from `agent/` (policy input)
- `EpisodeState.current_tick_reward` and `current_tick_end_reason` from `game/` (learning signal)

**What it writes:**
- `ActionState.desired` (controller output)

**Why `brain/` depends on `game/` but not vice versa:** The environment must be controller-agnostic. `game/` defines reward truth. `brain/` reads that truth to learn. If `game/` depended on `brain/`, swapping out the learning algorithm would require changing the environment, which would be architecturally disruptive. The dependency arrow points one way: brain depends on game, game does not depend on brain.

---

### `analytics/`

**Owns:**
- Per-tick data capture during `FixedUpdate`
- Episode summary records on terminal steps
- A2C update snapshots
- Derived metrics: trends, diagnostics, narrative bullets
- JSON and Markdown export on app exit

**Does not own:** simulation truth, reward definitions, training decisions.

**Rule:** analytics is a read-only downstream consumer. It reads state produced by `game/`, `agent/`, and `brain/`, but does not write back to any of them.

**Why this read-only rule matters:** If analytics could write to `EpisodeState` or `A2cBrain`, a subtle bug in a metrics computation could corrupt the learning loop without any visible connection to the analytics code. Keeping analytics as a pure observer eliminates that class of bug entirely.

---

### `debug/`

**Owns:**
- World-space visual overlays (centreline, car vectors, sensor rays)
- Runtime HUD (episode stats, moving averages, learning health)
- `F1/F2/F3` overlay toggles
- Recent-quarter run assessment

**Does not own:** simulation truth, training state, analytics export.

**Rule:** same as analytics — read-only downstream consumer. It reads from `maps/`, `game/`, `agent/`, and `brain/` but does not write to any of them.

---

## The Dependency Matrix

```
           sim  maps  game  agent  brain  analytics  debug
sim         —    —     —     —      —       —          —
maps        ✓    —     —     —      —       —          —
game        ✓    ✓     —     —      —       —          —
agent       ✓    ✓     ✓     —      ✓*      —          —
brain       —    —     ✓     ✓      —       —          —
analytics   —    ✓     ✓     ✓      ✓       —          —
debug       —    ✓     ✓     ✓      ✓       —          —
main        ✓    ✓     ✓     ✓      ✓       ✓          ✓
```

`✓*` = `agent/` references `brain::types` for the mode enum only — not any brain implementation.

**Key structural properties:**
- No cycles in the dependency graph.
- `game/` does not depend on `brain/`. This is the most important rule.
- `sim/` is dependency-free. It can safely be imported by anyone.
- `analytics/` and `debug/` are purely downstream.

---

## Violation Examples and Their Consequences

Understanding why the rules exist is best done by considering what happens if they are broken.

### Violation 1: `game/` imports from `brain/`

```rust
// In game/episode.rs — WRONG
use crate::brain::a2c::A2cBrain;
fn episode_loop_system(brain: Res<A2cBrain>, ...) { ... }
```

**Consequence:** The episode system now depends on the A2C brain. Replacing A2C with a biological brain requires changing `episode.rs`. The environment is no longer controller-agnostic. Every new controller experiment requires touching environment code.

### Violation 2: `brain/` directly reads raw physics

```rust
// In brain/a2c/mod.rs — WRONG
use crate::game::car::Car;
fn a2c_act_system(car: Query<&Car>, ...) { ... }
```

**Consequence:** The A2C brain bypasses the observation contract. Raw physics state leaks into the controller. The `ObservationVector` normalisation and feature engineering are skipped. A future biological brain that expects the standardised observation input breaks because it is getting different data.

### Violation 3: `analytics/` writes to `EpisodeState`

```rust
// In analytics/plugin.rs — WRONG
fn some_analytics_system(mut episode: ResMut<EpisodeState>, ...) {
    episode.current_tick_reward += correction;
}
```

**Consequence:** Analytics is no longer a pure observer. Bugs in analytics can now corrupt the learning loop. The source of reward truth is no longer clearly `game/episode.rs` — it is `game/episode.rs` plus whatever analytics decides to add.

---

## Ownership of Key Data Structures

| Structure | Owner | Who reads it |
|---|---|---|
| `Track` | `maps/` | `game/`, `agent/`, `debug/` |
| `TrackGrid` | `maps/` | `game/collision`, `agent/observation` |
| `Car` component | `game/` | `game/physics`, `debug/` |
| `CollisionEvent` | `game/collision` | `game/episode`, `debug/` |
| `TrackProgress` | `game/progress` | `game/episode`, `agent/observation`, `debug/`, `analytics/` |
| `EpisodeState` | `game/episode` | `brain/a2c`, `debug/`, `analytics/` |
| `ActionState` | `agent/action` | `game/physics`, `brain/`, `analytics/`, `debug/` |
| `ObservationVector` | `agent/observation` | `brain/a2c`, `analytics/` |
| `A2cBrain` | `brain/a2c` | (owned and managed by `brain/`) |
| `A2cTrainingStats` | `brain/a2c` | `analytics/`, `debug/` |

---

## How to Locate New Code

When adding a new feature, ask:

1. **Does it define a property of the track or world geometry?** → `maps/`
2. **Does it involve physics, collision, progress, reward, or episode state?** → `game/`
3. **Does it change what the policy observes or what control channels exist?** → `agent/`
4. **Does it implement a controller or learning algorithm?** → `brain/`
5. **Does it capture, summarise, or export run data?** → `analytics/`
6. **Does it provide live visual or textual inspection?** → `debug/`
7. **Does it define system set ordering shared across plugins?** → `sim/`

If a feature spans multiple categories, it should be split: each module owns its portion, and they communicate via the established interfaces.

---

## Related Files

- `project/architecture/runtime-overview.md` — high-level subsystem map
- `project/architecture/fixed-tick-pipeline.md` — the SimSet ordering contract
- `project/systems/environment-system.md` — game/ and maps/ in depth
- `project/systems/agent-interface.md` — the stable controller boundary in depth
