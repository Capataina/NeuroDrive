# Materials: Rust, Bevy, And Game-Loop Engineering

## Why This Topic Matters Here

A learner or controller is only as trustworthy as the runtime it sits inside. NeuroDrive relies heavily on:

- fixed-timestep thinking,
- ECS scheduling,
- clean subsystem ownership,
- careful separation of truth creation from truth observation.

## What To Study

Focus on:

- fixed-update simulation patterns,
- ECS scheduling and system ordering,
- deterministic game-loop reasoning,
- resource versus component ownership,
- observability patterns in simulation code.

## Why This Matters More Than It First Appears

Many subtle learning bugs are actually systems bugs:

- stale observation timing,
- incorrect reset ordering,
- wrong reward capture point,
- analytics recorded from the wrong phase.

Better engine-level reasoning often prevents more debugging pain than another round of network tuning.

## Best Time To Use This Guide

Use this materials guide alongside:

- `project/architecture/data-flow-and-schedule.md`
- `project/systems/environment.md`
- `project/systems/debug-runtime.md`
