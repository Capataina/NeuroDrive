# System Cheat Sheet

| Area | Owns | Should Not Own |
|---|---|---|
| `maps` | track geometry, grid, centreline, spawn pose | reward, training, analytics |
| `game` | car physics, collision truth, progress, episode logic | policy updates, report export |
| `agent` | action contract, smoothing, sensors, observations | reward truth, analytics truth |
| `brain` | control mode, A2C baseline, rollout/update logic | environment truth |
| `analytics` | tracking, metrics, export | reward definition, terminal truth |
| `debug` | live overlays and HUD | source-of-truth simulation facts |
