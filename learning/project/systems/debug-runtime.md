# Debug Runtime

## Status

Current in the project runtime.

## What This System Does

The debug runtime provides live interpretability while the simulation is running.

It includes:

- world-space overlays,
- a telemetry HUD,
- recent-run quarter summaries,
- a lightweight learning-health line for A2C updates.

## Why This Matters

Analytics tells you what happened after the run. Debug tells you whether the current behaviour looks plausible right now.

That split is healthy:

- debug supports live intervention and intuition,
- analytics supports deeper post-run diagnosis.

## Current Toggle Surface

- `F1` toggles geometry overlays,
- `F2` toggles sensor overlays,
- `F3` toggles the HUD,
- `F4` toggles control mode between keyboard and AI.

These controls make the runtime actively inspectable rather than opaque.

## What The HUD Shows

The current HUD includes:

- progress and geometry-relative state,
- episode and crash information,
- moving averages,
- learning-health information when A2C update stats exist,
- recent quarter summaries with quick run assessment labels.

This is exactly the right level of ambition for a current baseline: more than a toy overlay, less than a full experiment dashboard.

## Current Limitations

- several queries are still singleton-oriented,
- the HUD is summary-heavy rather than deeply drillable,
- policy mean/std visualisation is limited,
- vectorised training would require structural redesign here.

## Related Files

- `project/systems/analytics.md`
- `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
