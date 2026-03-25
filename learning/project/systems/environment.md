# Environment

## Status

Current in the project runtime.

## What This System Does

The environment owns the world the controller is trying to master. In practical repository terms, that means:

- deterministic vehicle motion,
- collision truth,
- progress along the track,
- reward shaping,
- episode boundaries,
- reset behaviour,
- rolling episode summaries.

## Current Implemented Reality

The runtime is still explicitly singleton-car.

That shows up in several ways:

- one car is spawned,
- several queries use `single()` or `single_mut()`,
- episode truth is stored in singleton resources,
- collisions do not carry per-car identity.

This is an important architectural fact because it makes the proposed vectorised trainer a meaningful refactor rather than a small tweak.

## Reward Composition

The current reward is not a black box. It is decomposed into interpretable pieces:

- progress reward based on best-so-far progress gain,
- time penalty,
- heading-speed penalty,
- crash penalty,
- lap bonus.

This is good engineering because the reward can be analysed later in traces and reports instead of being an opaque scalar.

## Episode End Conditions

An episode ends when one of three things happens:

- crash,
- timeout,
- lap completion.

Lap completion is currently wrap-based rather than explicit finish-line crossing. That is acceptable for now, but it is also one of the places where the environment remains slightly heuristic.

## Why The Environment Is Stronger Than It Looks

It would be easy to undersell this subsystem because the world is visually simple. That would be a mistake.

The environment already has:

- a deterministic stepper,
- spatial projection onto a centreline,
- interpretable geometric measurements,
- shaped reward decomposition,
- per-episode moving averages,
- reset sequencing that downstream systems rely on.

That is enough structure for subtle bugs to matter.

## Main Risks

- ECS-level regression coverage is still thin,
- lap completion remains heuristic,
- singleton assumptions block multi-car training,
- headless mode does not exist,
- multiple tracks are not yet a reality.

## What A Good Edit Must Respect

When changing the environment, preserve these invariants:

- environment defines reward and terminal truth,
- policy systems may read that truth but should not redefine it,
- analytics and debug remain consumers,
- reset ordering must stay compatible with post-reset observation correctness.

## Related Files

- `project/architecture/data-flow-and-schedule.md`
- `concepts/domain-patterns/reward-shaping-and-credit-assignment.md`
- `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
