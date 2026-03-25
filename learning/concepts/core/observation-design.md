# Observation Design

## Why This Matters Here

NeuroDrive is trying to answer whether the current task is learnable. That depends heavily on what information the controller receives and how that information is scaled.

## Core Idea

An observation vector is the controller's sensory world. Good observation design gives the learner enough information to act, but not so much privileged structure that the task becomes misleadingly easy.

## Build-Up

### Step 1: Pick information the controller could plausibly use

Distances to walls, speed, lane offset, heading error, and upcoming track shape all help driving.

### Step 2: Keep the contract stable

The learner should not have to guess changing input order or dimension. NeuroDrive keeps a fixed-size observation vector with a shared constant.

### Step 3: Normalise features

Different physical units need compatible scales, otherwise one feature can dominate optimisation for accidental numerical reasons.

### Step 4: Avoid direct solution leakage

NeuroDrive does not feed raw `TrackProgress` into the observation vector. That preserves a meaningful control problem rather than handing the agent a near-scoreboard input.

## Worked Examples

### Example 1: Ray distances

Rays tell the car how much free space exists around it, which helps with wall avoidance.

### Example 2: Signed lateral offset

Offset from the centreline tells the controller whether it is drifting left or right of the intended path.

### Example 3: Lookahead curvature

Upcoming curvature helps the learner react before the corner is already under the car, which is important for turn anticipation.

## How This Appears In The Project

- `src/agent/observation.rs` defines `OBSERVATION_DIM`, sensor resources, and observation construction.
- The live observation vector currently contains:
  - 11 ray distances,
  - speed,
  - signed lateral offset,
  - heading error,
  - angular velocity,
  - 4 lookahead heading-delta values,
  - 4 lookahead curvature values.

## Common Misunderstandings

❌ “More inputs always help.”
✅ More inputs can increase noise, coupling, and scaling problems.

❌ “If progress is measurable, it should be a direct observation.”
✅ Not necessarily. In NeuroDrive, progress is environment truth used for reward and analytics, not controller input.

## Terms Used Here

- observation vector
- feature scaling
- observation leakage
- controller boundary
