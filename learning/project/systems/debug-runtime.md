# The Debug Runtime

## What This File Covers

NeuroDrive has a live debug layer that provides visual and textual diagnostics during interactive runs. This file explains the three overlay modes, the HUD structure, the quarter-summary assessment system, and the design principles that keep debug observation separated from simulation truth.

**Status:** Current implementation.

## Prerequisites

- `project/architecture/module-boundaries.md` — debug is a read-only downstream consumer
- `project/architecture/fixed-tick-pipeline.md` — debug capture timing
- `project/systems/environment-system.md` — the episode and progress state that debug reads

---

## The Design Principle: See Without Touching

The debug runtime is an observer, not a participant. Like the analytics system, it reads state from `maps/`, `game/`, `agent/`, and `brain/`, but it never writes to any of them.

The consequence is that toggling overlays on and off does not change the simulation. The same fixed-tick pipeline runs identically whether the debug overlay is visible or not. This is important for reproducibility — a debugging session should not alter the trajectory being debugged.

---

## The Three Toggles

`DebugOverlayState` holds three independent boolean flags:

| Toggle | Key | Default | What it enables |
|---|---|---|---|
| Geometry overlay | F1 | **On** | Track centreline, projection point, car vectors, lookahead markers |
| Sensor overlay | F2 | Off | Ray segments and hit points from SensorReadings |
| HUD | F3 | **On** | The full diagnostics panel |

Each toggle is independent — you can have sensors visible without geometry, or the HUD without any world-space overlays.

### Geometry Overlay (F1)

Renders world-space Bevy gizmos for:

- **Centreline polyline:** the closed track centreline as a continuous line
- **Projection point:** the nearest centreline point to the car's current position — shows which centreline location the progress system is tracking
- **Tangent arrow:** the centreline tangent at the projection point — shows the "ideal forward direction"
- **Car forward vector:** the actual car forward direction — visual angle between this and the tangent arrow shows heading error
- **Velocity vector:** the actual velocity direction and magnitude — useful for seeing if the car is sliding or drifting
- **Lookahead preview markers:** points at each of the four lookahead distances along the centreline, with tangent arrows showing expected track direction ahead of the car

This overlay is most useful for understanding why the agent is failing. Common failure patterns visible in the geometry overlay:
- Heading error accumulating before a corner (agent is not turning early enough)
- Car cutting across the centreline on a corner exit (overshoot)
- Projection point jumping (progress measurement instability)

### Sensor Overlay (F2)

Renders the 11 rays cast by the observation system:
- Each ray is drawn as a line segment from the car's position to its hit point (or max range if no hit)
- Hit points are marked

This overlay is useful for understanding what the policy sees. Key things to look for:
- Very short rays on one side = car is close to the wall on that side
- All forward rays long = car is in open space, should accelerate
- Rays blocked in multiple directions = car is in a tight section and must steer carefully

### HUD (F3)

A Bevy UI panel in a fixed position on screen. The HUD is not world-space — it is a 2D overlay drawn at the screen level.

---

## The HUD Structure

The HUD is divided into several sections:

### Current State

Real-time values from the current tick:
- Progress fraction (how far around the lap)
- Signed lane offset
- Centreline gap (raw distance from centreline, unsigned)
- Heading error (angle between car forward and centreline tangent)
- Current speed

### Run State

Accumulated run statistics:
- Total episodes completed
- Total crashes
- Best progress fraction reached in any episode this run

### Moving Averages (from EpisodeMovingAverages)

Rolling means over the last `moving_average_window` episodes:
- Mean episode return
- Mean best-progress fraction
- Crash rate

### Live A2C Learning Line

When AI mode is active and at least one training update has completed, the HUD shows a compact learning-health summary from `A2cTrainingStats`:
- Value loss
- Policy loss
- Entropy
- Explained variance

This gives a live signal of whether learning is progressing or stalling, without waiting for the exported analytics report.

### Recent-Quarter Summaries

This is the most distinctive part of the HUD. The recent episode history is split into four equal quarters, and each quarter is summarised. The assessment logic:

```
q1 = first 25% of recent history (oldest)
q2 = second 25%
q3 = third 25%
q4 = fourth 25% (most recent)

assessment = if q4_mean_return > q1_mean_return and q4_mean_return > q3_mean_return:
                 "Improving"
             elif q4_mean_return < q1_mean_return and q4_mean_return < q3_mean_return:
                 "Regressing"
             elif recent episodes < threshold:
                 "Warm-up"
             else:
                 "Mixed"
```

Each quarter grid cell shows: mean return, crash rate, mean progress for that quarter.

**Important caveats:**
- The assessment window is bounded by `moving_average_window` (default 50 episodes). The quarters represent the most recent 50 episodes, not the full run history.
- The assessment is a heuristic. "Improving" in the most recent window does not mean the agent has mastered the task — it means recent performance is higher than earlier recent performance.
- A more complete performance picture requires the exported analytics report.

A unit test covers the assessment logic: given a set of episode return sequences with clear improvement and regression patterns, the heuristic must correctly classify them.

---

## Capture Timing

The debug runtime touches two Bevy schedules:

### FixedUpdate: Stats Accumulation

`capture_driving_hud_episode_metrics_system` runs at the end of `SimSet::Measurement` to update the HUD's episode accumulator when an episode completes. This ensures the HUD sees the correct episode summary.

### Update: Rendering and Text Refresh

All visual rendering (gizmos, UI text updates) runs in the normal `Update` schedule. This is frame-rate-dependent — the visual quality depends on the render framerate, but the simulation data being displayed is always from the most recent `FixedUpdate` tick.

This two-schedule design means:
- The simulation is deterministic at 60 Hz regardless of frame rate
- Overlays render at whatever frame rate Bevy can achieve
- There is a brief visual lag between a simulation event and when it appears on screen, but this is imperceptible at normal framerates

---

## Interaction with Analytics

The HUD and analytics serve different roles:

| HUD | Analytics |
|---|---|
| Live, frame-rate updates | Post-run export |
| Bounded recent window (50 episodes) | Full run history |
| Heuristic assessment | Full metric derivation |
| Interactive (toggleable) | Offline (file output) |
| For real-time intervention decisions | For post-session diagnosis |

The design intentionally keeps them separate. The HUD answers: "is this run going well right now?" The analytics report answers: "how did this run go in full detail?"

---

## Known Limitations

| Limitation | Notes |
|---|---|
| Single-car assumption | Uses `single()` queries; will break before multi-car trainer |
| No rollout buffer visualisation | Cannot see A2C buffer internals, current horizon fill, update cadence in real time |
| No policy distribution display | Cannot see mean action, std, or critic state space-locally |
| Assessment is heuristic | Not statistically rigorous; can be fooled by short favourable streaks |
| Overlay performance not validated | Under long AI runs, gizmo draw cost has not been profiled |

---

## Related Files

- `project/systems/analytics-system.md` — the offline complement to the live HUD
- `project/architecture/module-boundaries.md` — debug as read-only downstream consumer
- `project/architecture/fixed-tick-pipeline.md` — when debug systems run relative to simulation
