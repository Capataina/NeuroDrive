# Observations, Actions, And Representation

## Why This Matters Here

Most learning failures in control projects are not purely "algorithm" failures. They are often representation failures:

- the policy does not see enough,
- it sees the wrong thing,
- it sees too much privileged truth,
- or its action boundary is awkward and unstable.

NeuroDrive already treats this seriously. The agent interface is one of the repository’s strongest current design boundaries.

## Current Observation Shape

The current observation dimension is `23`.

It contains:

- `11` ray distances,
- `1` speed scalar,
- `1` signed lateral offset,
- `1` heading error,
- `1` angular velocity,
- `4` lookahead heading deltas,
- `4` lookahead curvatures.

This is an intentionally hybrid representation:

- local free-space geometry via rays,
- immediate kinematics via speed and angular velocity,
- centreline-relative pose via offset and heading error,
- near-future path shape via lookahead geometry.

## Why This Is Better Than Rays Alone

A ray-only observation often tells you what is near the car, but not enough about:

- whether you are already misaligned,
- whether the next turn is tightening,
- how far you have drifted from the centre,
- whether your rotation rate is already dangerous.

NeuroDrive’s added centreline features reduce the burden on the policy to infer those facts indirectly.

## Why Progress Is Excluded

One of the most important design choices in the repository is that `TrackProgress` remains environment truth rather than policy input.

Why that matters:

- progress is extremely privileged information,
- feeding it directly risks letting the policy optimise a shortcut of the metric rather than geometry-aware driving,
- keeping it out preserves a cleaner distinction between "what the world knows" and "what the driver senses".

## Current Action Shape

The control boundary is:

- steering in `[-1, 1]`,
- throttle in `[0, 1]`.

There is currently no explicit brake channel.

This matters because every policy design and every extension proposal must respect the stable controller boundary first. It is the contract joining brain, physics, analytics, and debug subsystems.

## Desired Versus Applied Action

`ActionState` separates:

- `desired`
- `applied`

Why that is a good design:

- the controller interface stays stable,
- smoothing can be inserted without changing controller code,
- analytics and physics can inspect what was actually executed,
- the system stays open to replay or alternative controllers later.

## Representation Trade-Offs

The current observation design makes several bets:

### Bet 1: geometry-aware engineered features are acceptable

This repository is not trying to prove that raw pixels can learn to drive. It is trying to build a meaningful, inspectable learning system in a controlled environment.

### Bet 2: centreline-relative features help turn anticipation

This is likely correct for the current task because lookahead curvature and heading-delta features encode future path structure more directly than raw rays.

### Bet 3: a stable interface is worth protecting

Changing observation schema is not a cheap local edit. It changes:

- A2C input dimensionality,
- analytics trace content,
- debug interpretation,
- comparability across runs.

## Common Failure Modes

- adding features without versioning or metadata,
- changing dimensionality silently,
- mixing privileged environment truth into observations,
- forgetting that analytics and debug code also depend on the interface,
- assuming a weak learner means the observation design is necessarily wrong.

## Related Files

- `project/systems/agent-interface.md`
- `project/systems/a2c-baseline.md`
- `project/systems/analytics.md`
- `exercises/project/inspect-the-observation-pipeline.md`
