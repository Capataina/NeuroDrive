# Fixed-Timestep Simulation

## Why This Matters Here

NeuroDrive is not just a visual game loop. The fixed-timestep model is what makes reward timing, action timing, deterministic physics, and learning updates reason about the same sequence of events.

## Core Idea

In a fixed-timestep simulation, the important state updates happen at a constant rate rather than “whenever the frame arrives”. That gives you:

- stable physics,
- comparable measurements,
- deterministic scheduling assumptions,
- and cleaner reasoning about cause and effect.

## Build-Up

### Step 1: Separate rendering from simulation

Rendering can vary with machine speed. Physics and reward logic should not.

### Step 2: Make state mutation happen on the fixed tick

If steering, velocity, progress, collision, and reward all update on the same fixed cadence, the runtime has one authoritative timeline.

### Step 3: Place dependent systems in an explicit order

Fixed step alone is not enough. You also need ordering. NeuroDrive uses named sets so action happens before physics, physics before collision, and collision before measurement.

## Worked Examples

### Example 1: Action before physics

If the controller writes a new action after physics, the car would always move one tick behind the policy.

### Example 2: Reward after progress measurement

If reward is collected before progress is updated, the learner trains on stale information.

### Example 3: Deterministic replay tests

The pure car stepper can replay the same action stream and produce the same trajectory because the update step is fixed and isolated.

## How This Appears In The Project

- `src/main.rs` inserts `Time<Fixed>::from_hz(60.0)`.
- `src/game/plugin.rs` chains `SimSet::Input -> Physics -> Collision -> Measurement`.
- `src/game/physics.rs` contains the pure `step_car_dynamics()` helper used by the deterministic replay test.

## Common Misunderstandings

❌ “Fixed timestep means the whole app is deterministic.”
✅ It means the simulation core has a strong determinism foundation. RNG and export timestamps can still break full reproducibility.

❌ “Frame rate and simulation rate are the same thing.”
✅ Rendering can fluctuate while the simulation still advances at 60 Hz.

## Terms Used Here

- fixed timestep
- schedule ordering
- deterministic replay
- simulation truth
