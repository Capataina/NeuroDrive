# System — Debug Runtime

## Scope / Purpose

- Provide live visual and textual inspection tools during manual driving and AI runs.
- Keep runtime diagnostics lightweight and interactive, separate from offline analytics export.
- The debug layer is a **read-only inspector** — it must not become a source of simulation truth.

## Boundaries / Ownership

| Owner | Owns | Does not own |
|-------|------|-------------|
| `src/debug/overlays.rs` | World-space gizmo overlays, `DebugOverlayState`, F1/F2/F3 toggle systems | Environment geometry production |
| `src/debug/hud.rs` | `DrivingHudStats`, `DrivingHudHistory`, quarter summaries, assessment logic, Bevy UI panel | Reward computation, episode truth |
| `src/debug/leaderboard.rs` | Live leaderboard panel with colour-coded per-car ranking, `LeaderboardRoot` | Ranking computation (owned by `brain/ranking.rs`) |
| `src/debug/plugin.rs` | Debug resource setup and scheduling | SimSet definitions |

## Current Implemented Reality

### Toggle Controls

| Key | Controls | Default |
|-----|----------|---------|
| **F1** | Geometry overlay (centreline, tangent, forward, velocity, lookahead) | On |
| **F2** | Sensor overlay (ray segments and hit points) | Off |
| **F3** | Telemetry HUD panel | On |

### Geometry Overlay (F1)

Draws via Bevy gizmos in world space:
- Centreline polyline
- Current closest-point projection marker
- Tangent arrow at projection point
- Car forward vector
- Velocity vector
- Lookahead preview markers and tangent arrows at configured distances

### Sensor Overlay (F2)

Draws ray segments and hit points from `SensorReadings` — 11 rays with their detected road-boundary contacts.

### HUD Panel (F3)

The HUD is a compact **Bevy UI panel** (440px wide, 72% opacity, blue accent palette, positioned top-left):

- **Assessment line:** learning phase status and guidance (blue accent text)
- **Live metrics line:** progress %, life-best %, centreline gap, heading error, lateral offset
- **Run line:** episode count, life seconds, reward, death count, last end reason
- **Run detail line:** best-ever progress with episode number, rolling average progress and return
- **PPO learning line:** update count, explained variance, value loss, entropy, clip %, KL divergence
- **Quarter summary table:** six-column grid (Q, N, Prog, Life, Rwd, C/L/T) — Gap and Heading columns removed for compactness; assessment heuristic still uses them internally

Text lines use condensed labels and 1-decimal precision to prevent wrapping within the 440px panel.

### Quarter Summary Assessment

- The HUD keeps its own recent episode history bounded by `EpisodeConfig.moving_average_window`.
- Splits recent episodes into four chronological quarters.
- Computes per-quarter means for progress and return.
- Classifies the run as **Improving / Mixed / Regressing / Warm-up** based on whether the most recent quarter shows improvement over earlier quarters.
- A unit test covers the assessment heuristic.

### Leaderboard Panel (F3)

A live leaderboard panel in the top-right corner, toggled with F3 alongside the HUD:

- Shows all training cars ranked by `TrainerLiveRanking` score.
- Each row displays: rank number, colour swatch matching the car's unique `CarColour`, car ID, live progress %, average progress %, and a `*` marker for the current best.
- Rows re-sort when rankings change.
- Reads from `TrainerLiveRanking`, `TrackProgress`, and `EpisodeMovingAverages` per car.

### Scheduling

- **Fixed-tick** (`SimSet::Measurement`):
  - `update_driving_hud_stats_system` — tracks best progress and deaths across **all cars**
  - `capture_driving_hud_episode_metrics_system` — folds **all cars'** completed episodes into HUD history
- **Update** (every frame):
  - `debug_overlay_toggle_system` — F1/F2/F3 key handling
  - `draw_geometry_overlay_system` — centreline and car vector gizmos (all cars)
  - `draw_sensor_overlay_system` — ray rendering (all cars)
  - `update_driving_hud_visibility_system` — shows/hides HUD based on F3
  - `update_driving_hud_text_system` — refreshes all HUD text sections (displays **best car** via `TrainerLiveRanking`, falls back to first car)
  - `update_leaderboard_system` — refreshes leaderboard ranking display (all cars)

## Key Interfaces / Data Flow

| Interface | Source | Debug use |
|-----------|--------|----------|
| `Track` and `TrackProgress` | maps/game | Geometry overlay and progress display |
| `SensorReadings` | agent | Sensor overlay and current HUD values |
| `EpisodeState` and `EpisodeMovingAverages` | game (per-car Components) | Run status, current reward, quarter summaries, leaderboard |
| `Collided` marker | game | Death counting in HUD stats |
| `A2cTrainingStats` | brain | Live learning-health line in HUD |
| `TrainerLiveRanking` | brain/ranking | Leaderboard ordering and best-car identification |
| `CarColour` | game/car | Leaderboard swatch colours |

## Implemented Outputs / Artifacts

- **Runtime resources:** `DebugOverlayState`, `DrivingHudStats`, `DrivingHudHistory`, `DrivingHudEpisodeAccumulator`
- **Runtime UI entities:** `DrivingHudRoot`, fixed-column quarter summary grid, `LeaderboardRoot` with per-car rows
- **World-space overlays:** via Bevy gizmos
- **Tests:** HUD assessment unit test covering the "recent quarter is cleaner" improvement heuristic

## Known Issues / Active Risks

- The HUD is informative but still **summary-oriented** — it does not expose full rollout-buffer internals or detailed per-layer update history.
- Overlay performance under very long AI runs with multiple cars has not been explicitly evaluated.
- The assessment heuristic is intentionally lightweight and should not substitute for offline analytics.

## Partial / In Progress

- The debug layer has expanded beyond simple environment overlays and now includes a live learning-health view.
- It remains environment-heavy; direct world-space visualisation of policy mean/std or critic state does not exist yet.

## Planned / Missing / Likely Changes

- Brain-specific live inspection could be added: sampled vs mean action, rollout size, update cadence, richer network-health display.
- A dedicated heading-error glyph or clearer lane-position visualisation may be worthwhile.

## Durable Notes / Discarded Approaches

- Keeping runtime debug **separate from analytics export** is the right split:
  - debug is for live plausibility and intervention,
  - analytics is for longer-horizon diagnosis after a run.

## Obsolete / No Longer Relevant

- Any note describing F3 as future-facing is obsolete — it already controls a substantial diagnostics panel.
