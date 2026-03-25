# Deterministic Racing Environment

## Why This Matters Here

NeuroDrive is not trying to be a flashy racing game. It is trying to be a controlled learning laboratory where behaviour, reward, and failure modes are inspectable.

## Core Idea

A good learning environment for this project needs:

- continuous control,
- dense but interpretable feedback,
- meaningful failure states,
- enough structure to be learnable,
- and enough determinism to debug.

## Build-Up

### Step 1: Make the driving task non-trivial but not opaque

The car must steer and throttle continuously around a track with collisions and progress measurement.

### Step 2: Measure progress in a stable way

NeuroDrive projects the car onto a centreline rather than using brittle checkpoint jumps.

### Step 3: Keep reward interpretable

Reward comes from progress plus explicit penalties and bonuses, not from a hidden objective.

### Step 4: Make failure inspectable

Debug overlays and analytics let you inspect whether a failure came from perception, control, reward incentives, or training instability.

## Worked Examples

### Example 1: Dense signal without cheating

Centreline-relative reward gives useful feedback every tick without feeding direct progress into the policy input.

### Example 2: Crash as terminal truth

A crash both ends the episode and contributes a one-off penalty, which makes the failure visible in control, reward, and analytics layers.

### Example 3: Turn anticipation

Lookahead features improve anticipatory steering without abandoning the overall “engineered but interpretable” input philosophy.

## How This Appears In The Project

- `src/maps/` owns track construction and centreline geometry.
- `src/game/` owns physics, collision, progress, rewards, and resets.
- `src/agent/observation.rs` converts environment state into controller input.

## Common Misunderstandings

❌ “Dense reward means the task is easy.”
✅ Dense reward only makes learning more diagnosable. It does not guarantee the policy or observation design is good enough.

❌ “Deterministic means realistic.”
✅ Determinism is a debugging property, not a realism claim.

## Terms Used Here

- centreline projection
- dense reward
- terminal state
- environment truth
