# Deterministic Simulation for RL Environments

## 1. What Is This Pattern?

A deterministic simulation guarantees that given the same initial conditions and the same sequence of inputs, the system produces bitwise-identical outputs every time. No randomness, no timing variation, no platform-dependent divergence — just pure reproducibility.

This is not merely "approximately the same." Determinism means identical floating-point values, identical state transitions, identical trajectories. It is the foundation for debugging, replay, regression testing, and fair experimental comparison in reinforcement learning.

## 2. When To Use This Pattern

**Good for:**
- RL research where you need to compare algorithms on identical rollouts
- Debugging complex agent behaviour by replaying exact trajectories
- Regression testing (detecting when a code change alters simulation outcomes)
- Recording and replaying demonstrations for imitation learning

**Not good for:**
- Systems that intentionally require stochasticity (e.g., domain randomisation where reproducibility is per-seed)
- Real-time multiplayer games where network jitter makes frame-perfect lockstep impractical
- Simulations involving external hardware or sensors with inherent noise

## 3. Core Concept

### Threats to Determinism

Determinism does not happen by default. Several common patterns silently destroy it:

**Floating-point non-associativity:** `(a + b) + c ≠ a + (b + c)` in IEEE 754 arithmetic. If your systems process entities in varying order, the accumulated results differ. Parallel reductions are particularly dangerous.

**Variable timesteps:** If physics advances by `delta_time` and that varies per frame, trajectories diverge. Even tiny variations (16.6ms vs 16.7ms) compound over hundreds of steps.

**Unordered system execution:** If System A writes data that System B reads, but their execution order is unspecified, results are non-deterministic. This is endemic in ECS frameworks without explicit ordering.

**Random number generators:** Any use of randomness must be seeded and sequenced identically. Using the system clock as a seed, or drawing random numbers in data-dependent order, breaks reproducibility.

### Solutions

**Fixed timestep:** Advance the simulation by a constant `dt` every tick, regardless of wall-clock time. NeuroDrive uses `dt = 1/60` (60 Hz). The rendering frame rate and the simulation tick rate are decoupled.

**Explicit system ordering:** Declare the execution order of all systems that share mutable state. In NeuroDrive, the `SimSet` enum defines the pipeline: `Input → Physics → Collision → Measurement → Reward → Brain → Analytics`. Every system declares which set it belongs to.

**Pure-function extraction:** Move core logic (physics integration, reward calculation) into pure functions that take inputs and return outputs with no side effects. These can be tested outside the ECS. NeuroDrive's `step_car_dynamics` is a pure function: `(position, velocity, heading, action, dt) → (new_position, new_velocity, new_heading)`.

**RNG ownership strategy:** Each source of randomness owns its own deterministic RNG, seeded from a master seed. Draw counts are tied to logical events, not wall-clock or system-order.

## 4. Key Design Decisions

| Decision | Option A | Option B |
|---|---|---|
| Timestep | Fixed (deterministic, simple) | Variable (smooth visuals, non-deterministic) |
| System ordering | Explicit enum / labels (safe, verbose) | Implicit (convenient, fragile) |
| Physics extraction | Pure functions (testable, portable) | Inline in ECS systems (less boilerplate) |
| RNG approach | Per-component seeded (isolated) | Global RNG (simple but ordering-sensitive) |

**Key trade-off:** Strictness vs convenience. Full determinism requires discipline at every layer — fixed timesteps, ordered systems, isolated RNGs. Each relaxation makes the code simpler but risks silent divergence. Start strict; relax only when you can prove it does not affect reproducibility.

## 5. Simplified Example Implementation

```python
import struct

class DeterministicSim:
    def __init__(self, seed, dt=1/60):
        self.dt = dt
        self.tick = 0
        self.rng = LCG(seed)
        self.state = {"x": 0.0, "v": 0.0}

    def step(self, action):
        # Pure physics (no side effects, no external state)
        a = action * 5.0 - self.state["v"] * 0.1  # thrust - drag
        self.state["v"] += a * self.dt
        self.state["x"] += self.state["v"] * self.dt
        self.tick += 1

    def snapshot(self):
        xb = struct.pack('d', self.state["x"])
        vb = struct.pack('d', self.state["v"])
        return xb + vb  # bitwise-comparable bytes

class LCG:
    """Linear congruential generator — deterministic, seedable."""
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_f32(self):
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state / 0xFFFFFFFF

# Determinism test
sim_a = DeterministicSim(seed=42)
sim_b = DeterministicSim(seed=42)
rng = LCG(seed=99)

for _ in range(1000):
    action = rng.next_f32()
    sim_a.step(action)

rng_b = LCG(seed=99)
for _ in range(1000):
    action = rng_b.next_f32()
    sim_b.step(action)

assert sim_a.snapshot() == sim_b.snapshot(), "Determinism broken!"
```

## 6. How NeuroDrive Implements This

**Fixed timestep:** Bevy's `FixedUpdate` schedule runs at 60 Hz. All simulation systems execute in `FixedUpdate`, decoupled from the rendering frame rate.

**SimSet ordering contract:** An enum `SimSet` defines the pipeline stages. Each system is assigned to a stage via Bevy's `.in_set()` API. The ordering is enforced at schedule-build time:

`Input → Physics → Collision → Measurement → Reward → Brain → Analytics`

This guarantees that collision detection always runs after physics, measurements always run after collision, and so on.

**Pure-function extraction:** `step_car_dynamics` is a free function that takes the current car state and an action, and returns the next state. It has no knowledge of Bevy, no access to resources, no side effects. This function is called both from the ECS system and from standalone determinism tests.

**Determinism regression test:** NeuroDrive includes a test that:
1. Seeds an LCG with a known value.
2. Generates a sequence of random actions.
3. Runs the simulation for N ticks.
4. Records the final position and velocity.
5. Repeats from the same seed.
6. Asserts bitwise equality of the final state.

This catches any change — compiler update, system reordering, floating-point optimisation flag — that would break reproducibility.

**Layered determinism:** The project tracks determinism at four levels: dynamics (pure function), schedule (system ordering), controller (action generation), and analytics (metric computation). Each layer can be tested independently.

## 7. Variations

- **Lockstep networking:** All clients advance one tick per network round-trip, exchanging only inputs. Requires determinism. Used in RTS games (StarCraft, Age of Empires).
- **Seeded stochastic simulation:** Deterministic given a seed, but intentionally stochastic across seeds. Standard in RL for domain randomisation.
- **Snapshot-and-restore:** Serialise full simulation state for save/load or branch-and-bound search. Complements determinism but does not require it.

## 8. Common Pitfalls

- **Compiler optimisation flags:** `-ffast-math` or Rust's `--emit=llvm-ir` with aggressive FP reordering can change floating-point results across builds. Pin your optimisation flags.
- **HashMap iteration order:** Hash maps do not guarantee iteration order. If you iterate over a HashMap to update state, results may vary between runs. Use `BTreeMap` or a sorted vector.
- **Parallel system execution:** ECS frameworks may parallelise systems within the same stage. If two parallel systems write to overlapping state, results are non-deterministic. Ensure exclusive access or sequential ordering.
- **Logging-induced side effects:** If logging triggers lazy evaluation or changes RNG draw order, it can silently alter simulation results. Keep logging strictly read-only.
- **Platform differences:** x87 FPU (80-bit) vs SSE (64-bit) floating-point produces different results on x86. Force SSE for cross-platform determinism.

## 9. Projects That Use This Pattern

- **DeepMind Lab / dm_env:** Deterministic environments seeded per episode, enabling exact replay of research results.
- **Factorio:** Fully deterministic simulation enabling lockstep multiplayer and replay files. A masterclass in disciplined determinism at industrial scale.
- **PhysX (NVIDIA):** Offers a deterministic mode for robotics sim-to-real transfer, where trajectory reproducibility is critical.

## 10. Glossary

| Term | Definition |
|---|---|
| **Fixed timestep** | Advancing the simulation by a constant time increment regardless of wall-clock time |
| **System ordering** | An explicit declaration of which systems execute before others |
| **Pure function** | A function whose output depends only on its inputs, with no side effects |
| **LCG** | Linear congruential generator — a simple, seedable, deterministic PRNG |
| **Bitwise equality** | Two values are identical at the binary level, not merely "close enough" |
| **Lockstep** | A networking model where all participants advance one tick per synchronisation round |

## 11. Recommended Materials

- **"Fix Your Timestep!"** by Glenn Fiedler (gafferongames.com) — The canonical article on fixed-timestep game loops. Essential reading.
- **"Deterministic Lockstep"** by Glenn Fiedler — Companion article covering deterministic networking. Illustrates why determinism matters beyond single-player.
- **"Floating Point Determinism"** by Bruce Dawson (randomascii.wordpress.com) — Deep dive into IEEE 754 pitfalls, compiler flags, and cross-platform reproducibility.
