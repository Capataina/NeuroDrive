# Analytics

## Status

Current in the project runtime.

## What This System Does

Analytics captures enough runtime truth to inspect behaviour after the app exits.

It separates:

- raw tracking,
- derived metrics,
- export rendering.

That separation is a strong design choice because it prevents report-writing concerns from contaminating truth capture.

## Current Outputs

The current analytics layer exports:

- episode records,
- tick-level traces,
- A2C update records,
- JSON reports,
- Markdown reports.

The trace layer is rich enough to include:

- progress,
- speed,
- centreline distance,
- lateral offset,
- heading error,
- control inputs,
- reward decomposition,
- ray distances,
- lookahead features,
- current critic prediction when AI is active.

## Why This Subsystem Is More Important Than It Looks

In many learning projects, analytics appears late and remains shallow. Here it is already substantial, which matters because:

- reward design can be interrogated,
- turn-execution behaviour can be inspected,
- update health can be tracked,
- failure modes can be described with more nuance than "reward went up/down".

## Current Weaknesses

The biggest missing capability is disciplined experiment metadata. Reports still lack:

- RNG seed,
- config snapshot,
- git revision,
- track identity,
- explicit evaluation/training context.

Without that information, reports are useful for introspection but weaker for rigorous comparison across runs.

## Important Boundary

Analytics must remain downstream of runtime truth. It should summarise and interpret:

- not decide reward,
- not create terminal state,
- not define "best" truth independently of source systems.

## Related Files

- `project/systems/environment.md`
- `project/systems/a2c-baseline.md`
- `project/systems/debug-runtime.md`
- `exercises/project/extend-the-analytics-schema.md`
