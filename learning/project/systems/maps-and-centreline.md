# Maps And Centreline

## What This System Does

The maps layer creates the geometry truth that almost every other subsystem depends on.

It owns:

- track topology,
- road occupancy grid,
- centreline derivation,
- spawn pose,
- shared track entity data.

## Why It Matters So Much

NeuroDrive’s environment is not just "a car on a plane". The centreline gives the project a geometric spine:

- progress along the lap,
- tangent direction,
- closest-point projection,
- lookahead samples,
- lane-relative quantities.

Without this layer, the reward, observations, and debug views would all become weaker and more ad hoc.

## Current Runtime Reality

The repository currently ships one hard-coded closed loop. The code and `context/` describe it as Sepang-inspired even though the plugin is named `MonacoPlugin`. The deeper point is that the current runtime has one canonical visible track rather than a library of track variants.

That means:

- overfitting risk is real,
- debugging geometry is manageable,
- experiment diversity is limited.

## Centreline As Shared Infrastructure

The centreline is used by:

- progress measurement,
- heading-error computation,
- lateral offset,
- lookahead heading deltas,
- lookahead curvature,
- debug overlay markers.

This is a good example of one abstraction serving multiple subsystems without becoming overgeneralised.

## Important Boundary

The maps layer creates geometry truth, but it does not decide:

- reward,
- terminal state,
- policy updates,
- analytics interpretation.

That separation should remain intact.

## Related Files

- `project/systems/environment.md`
- `project/systems/agent-interface.md`
- `project/systems/debug-runtime.md`
