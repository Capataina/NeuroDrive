# Project Architecture Path

## Who This Path Is For

This path is for learners who want to understand NeuroDrive as a software system — how it is structured, what each subsystem owns, and how the runtime flows from a car receiving sensor input to the policy applying a steering command. It emphasises architecture, ownership boundaries, data flow, and the engineering decisions that produced this structure.

It is the right path if you want to contribute to the codebase, audit the architecture, or understand the design rationale behind the subsystem split.

## What This Path Assumes

- Comfortable reading Rust or a similar systems language
- Some familiarity with event loops, fixed timesteps, or game engine concepts is helpful
- No prior Bevy ECS knowledge required (the path includes a primer)

## What You Will Understand by the End

- How NeuroDrive is divided into subsystems and why the boundaries sit where they do
- What the Bevy ECS model is and how NeuroDrive uses it for scheduling and state management
- What the `SimSet` execution chain is and why ordering matters
- What each subsystem owns, what it reads, and what it must not modify
- How the observation vector is constructed from raw sensor data
- How the action contract keeps controllers decoupled from physics
- How the A2C brain fits into the fixed-tick pipeline
- How analytics captures runtime data without contaminating training truth
- How the debug runtime provides live inspection without becoming a simulation authority
- Where the current architecture has known tensions and future pressure points

## Recommended Sequence

- [ ] `concepts/foundations/bevy-ecs-primer.md`
  - Read this first if you have not used Bevy or ECS before. Understanding systems, components, resources, and schedule sets is essential for everything else in this path.

- [ ] `project/architecture/runtime-overview.md`
  - The top-level map. Read this to understand the full set of subsystems and their primary relationships before diving into any one of them.

- [ ] `project/architecture/fixed-tick-pipeline.md`
  - The `SimSet` execution chain. Understand `Input → Physics → Collision → Measurement` and why each system sits where it does.

- [ ] `project/architecture/module-boundaries.md`
  - Ownership, dependency direction, and the data-flow contracts between subsystems.

- [ ] `project/systems/environment-system.md`
  - The track, car physics, collision detection, centreline progress, reward, and episode lifecycle. This is the foundation everything else depends on.

- [ ] `project/systems/agent-interface.md`
  - The observation and action contracts. How sensors become a normalised vector. Why `ActionState` separates `desired` from `applied`.

- [ ] `project/systems/a2c-brain.md`
  - How the A2C brain plugs into the pipeline via `SimSet::Input` (act) and `SimSet::Measurement` (reward collection). The rollout buffer, GAE, and update path.

- [ ] `project/systems/analytics-system.md`
  - How analytics captures data across the run and exports to JSON and Markdown without modifying training state.

- [ ] `project/systems/debug-runtime.md`
  - F1/F2/F3 overlays, the HUD, and how live diagnostics are separated from offline analytics.

- [ ] `exercises/core/trace-observation-vector.md`
  - Trace through one complete tick's observation construction to verify you can follow the system flow.

## After This Path

From here, proceed to:

- `paths/implementation-first-path.md` — go deeper into the A2C and analytics implementation details
- `paths/reinforcement-learning-path.md` — understand the RL theory the architecture supports
- `project/evolution/from-baseline-to-brain.md` — understand how the architecture needs to change for Milestone 2

## Notes

- The key insight in this path is that `agent/` is a **stable interface layer** between the environment and any controller. It is not just an implementation detail — it is a deliberate abstraction that keeps the RL brain, future biological brain, and keyboard mode all using the same contract.
- The `SimSet` ordering is not arbitrary. Several subtle bugs in earlier development came from systems running in the wrong order (reward collected before episode truth was finalised, observations rebuilt before reset). The current ordering encodes hard-won lessons.
- The `analytics/` subsystem is intentionally passive — it reads runtime truth from other systems and must never mutate training state. Understanding why this boundary exists is as important as understanding what analytics does.
