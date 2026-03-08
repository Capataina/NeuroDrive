# NeuroDrive's Car Physics Model

NeuroDrive's car physics are implemented as a pure function that takes the current state, control inputs, a timestep, and physical parameters, then returns a new state. This design — extracting physics into a side-effect-free computation — is central to the project's determinism and testability guarantees.

> **Prerequisites:** [Observation System](observation_system.md), [Architecture Decisions](../architecture_decisions.md)

---

## The Pure Physics Function

**Code location:** `src/game/physics.rs`

The core function signature is:

`step_car_dynamics(state, steering, throttle, dt, params) → new_state`

This function receives the car's current position, velocity, and heading, applies the steering and throttle inputs over a time interval `dt`, and returns the updated state. It reads no global state, mutates no external data, and produces no side effects. Given identical inputs, it always produces identical outputs.

This purity matters for two reasons. First, it enables deterministic replay — a recorded sequence of `(steering, throttle)` pairs will reproduce the exact same trajectory when replayed against the same initial conditions. Second, it enables isolated unit testing — the physics can be validated without spawning a Bevy world.

---

## Heading Update

Heading is updated first, before thrust or position:

`heading += -steering × rotation_speed × dt`

The negative sign means positive steering input turns the car clockwise (to the right) when viewed from above, which matches the conventional screen coordinate system where angles increase counter-clockwise. The `rotation_speed` constant is `4.0 radians/second`, meaning full steering lock (`steering = 1.0`) produces roughly 230°/second of rotation — fast enough for responsive cornering but not so fast that the car spins uncontrollably.

---

## Thrust and Velocity

After updating the heading, the forward direction vector is computed as `(cos(heading), sin(heading))`. Thrust is applied along this vector:

`velocity += forward × thrust × throttle × dt`

The `thrust` constant is `750` (in acceleration units). At full throttle (`throttle = 1.0`) and 60 ticks per second (`dt ≈ 0.0167`), this adds approximately 12.5 units/second² to the velocity magnitude along the forward axis. The throttle input is continuous — values between 0 and 1 produce proportional acceleration, and negative throttle provides braking/reverse.

---

## Drag (Velocity Damping)

After applying thrust, drag is applied multiplicatively:

`velocity *= drag`

The `drag` constant is `0.985`, applied once per physics tick. This is *not* a traditional drag force proportional to velocity² — it is exponential decay. At 60 ticks per second, the effective decay factor per second is `0.985^60 ≈ 0.406`, meaning the car loses about 60% of its speed per second when coasting without throttle.

This multiplicative model was chosen for simplicity and numerical stability. It naturally produces a terminal velocity where thrust equals drag losses: at equilibrium, `thrust × throttle × dt = velocity × (1 - drag)`, giving `v_terminal = thrust × throttle × dt / (1 - drag)`.

---

## Position Integration

Position is updated via Euler integration:

`position += velocity × dt`

Euler integration is the simplest possible numerical integration scheme. It introduces small errors compared to higher-order methods (RK4, Verlet), but for the purposes of a tile-grid racing game at 60 Hz, the error is negligible. The car's bounding box is `12 × 6` units, and positional errors from Euler integration at typical speeds are orders of magnitude below a single pixel.

---

## Physical Constants Summary

| Constant | Value | Meaning |
|---|---|---|
| `rotation_speed` | 4.0 rad/s | Steering responsiveness |
| `thrust` | 750 | Acceleration magnitude |
| `drag` | 0.985 | Per-tick velocity retention |
| Car width | 12 units | Bounding box (along forward axis) |
| Car height | 6 units | Bounding box (lateral axis) |

---

## Action Smoothing

**Code location:** `src/agent/action.rs`

Raw neural network outputs are not applied directly to the physics. Instead, they pass through a first-order exponential low-pass filter (exponential moving average):

`smoothed = smoothed + α × (raw - smoothed)`

where `α = 1 - exp(-dt / τ)` and the time constant `τ = 0.12` seconds. This filter serves two purposes:

1. **Physical plausibility:** Real steering and throttle mechanisms have inertia. The filter prevents the car from instantly snapping between full-left and full-right steering on consecutive ticks.
2. **Learning stability:** Without smoothing, the policy can exploit high-frequency oscillations that achieve reward through unphysical behaviour. The filter constrains the effective action space to smooth, physically-meaningful control trajectories.

At 60 Hz, the smoothing coefficient is `α ≈ 0.131`, meaning each tick moves about 13% of the way from the current smoothed value toward the raw output. The time constant of 0.12 seconds means the filter reaches 63% of a step change in roughly 7 ticks and 95% in about 20 ticks.

---

## Deterministic Replay

**Code location:** `src/game/physics.rs` (tests)

Because `step_car_dynamics` is a pure function and all floating-point operations are deterministic on a given platform, the physics are exactly reproducible. This is validated by a deterministic replay test:

1. A linear congruential generator (LCG) produces a fixed pseudo-random sequence of steering and throttle inputs.
2. The physics function is run forward for N steps, recording the final state.
3. The same sequence is replayed from the same initial conditions.
4. The final states are compared for exact bitwise equality.

This test guarantees that physics determinism has not been accidentally broken by code changes. It would detect, for example, the introduction of hash-map iteration (non-deterministic ordering), unintended use of system time, or floating-point expression reordering by the compiler.

Exact determinism is a prerequisite for the training pipeline: the A2C algorithm samples actions and computes returns over a trajectory. If replaying the same actions produced different states, the computed advantages would be inconsistent with the experienced rewards, corrupting the policy gradient.

---

## Why Pure Function Extraction Matters

The decision to extract physics into a pure function — rather than implementing it inline within a Bevy system that directly mutates `Transform` and `Velocity` components — was deliberate. The Bevy system that runs during `FixedUpdate` calls this function and then applies its output to the ECS world. This separation means:

- **Testability:** Physics tests do not require a Bevy `App`, entity spawning, or system scheduling. They are plain unit tests that call a function and check the result.
- **Determinism:** The function's purity guarantees that no hidden state (system ordering, component access patterns, frame timing) can influence the result.
- **Reusability:** The same function can be called by the training rollout, by the replay system, or by a hypothetical headless simulation without any modification.

This pattern — pure computation extracted from ECS side effects — is applied throughout NeuroDrive wherever determinism or testability is required.

> **See also:** [Observation System](observation_system.md) for how the car state feeds into the agent's perception, [Progress and Lap System](progress_and_lap_system.md) for how position maps to track progress.
