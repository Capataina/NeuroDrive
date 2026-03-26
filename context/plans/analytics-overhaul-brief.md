# Analytics Visual Overhaul — Intent Brief

**Status:** Planned. Blocked on vectorised A2C completion (need multi-car data first).

## Why

Current analytics exports are plain text and tables — functional but hard to read at a glance. Debugging learning behaviour requires spotting patterns across episodes, cars, and track regions. Visual formats make this dramatically faster.

## Direction

Replace text-heavy reports with rich visual outputs:

- **Heat maps** — crash locations on track, track coverage density, reward hotspots
- **Time-series graphs** — progress curves, reward over episodes, loss trajectories
- **Distribution charts** — action distributions (steering/throttle), per-car performance spread
- **Reward decomposition visuals** — stacked charts showing progress reward vs penalties vs bonuses
- **Track overlays** — trajectory paths coloured by speed/reward/heading error
- **ASCII/Unicode art** — compact terminal-friendly visualisations for quick inspection
- **Infographics** — cohort comparisons (best vs worst car behaviour), learning phase summaries

## Open Questions (to resolve during planning)

- Output format: SVG files? HTML report? Markdown with embedded ASCII? All three?
- Which visualisations are most valuable for debugging the biological brain (Milestone 2) vs A2C?
- Per-car vs cohort vs trainer-wide — which views matter most?
- Real-time (in HUD) vs post-run (in exported reports) vs both?

## Sequencing

1. Complete vectorised A2C (stages 1–2)
2. Run training, collect multi-car data
3. Plan the analytics overhaul with real data to inform which visuals matter most
4. Implement
