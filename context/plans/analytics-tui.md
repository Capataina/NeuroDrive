# Idea — Analytics TUI Explorer

## Vision

A terminal-based interactive analytics explorer for NeuroDrive run reports, inspired by the visual quality and interactivity of tools like **btop**, **gotop**, **lazygit**, and **lazydocker**. The TUI would read the most recent Markdown/JSON report from `reports/` and render it as a rich, navigable dashboard with real ASCII charts, colour, and interactive widgets.

## Why This Matters

- The current Markdown reports are informative but static — you have to read them linearly
- A TUI allows real-time exploration: zoom into sectors, compare chunks, inspect individual episodes
- ASCII charts with proper axes and scaling are far more readable than the current sparkline approximations
- Interactivity means you can drill down into crash locations, filter by car, or compare runs
- It doubles as a powerful debugging tool during PPO tuning — run a session, then immediately explore what happened

## Inspirations and Visual Targets

- **btop/gotop:** Multi-panel layout, colour-coded graphs, real-time updating feel
- **lazygit/lazydocker:** Keyboard-navigable panels, context-sensitive detail views, modal dialogs
- **Grafana terminal:** Time-series with proper axes, heatmaps, summary cards

## Potential Features (brainstorm — not committed)

### Dashboard panels
- **Run summary card** — episodes, max progress, crash rate, learning phase
- **Progress time-series** — proper line chart with axes, showing per-episode progress over time
- **Reward decomposition stacked bar** — progress vs time penalty vs crash penalty by chunk
- **Crash heatmap** — 2D track-position heatmap of crash locations (ASCII colour blocks)
- **Sector breakdown table** — speed, steering, gap stats per sector, sortable columns
- **Per-car comparison** — side-by-side sparklines or bars for each car's progress/reward
- **PPO health panel** — entropy, clip %, KL, explained variance as mini time-series
- **Layer health bars** — dead neuron / saturation percentage as coloured bars per layer

### Interactive widgets
- **Episode scrubber** — arrow keys to step through episodes, see trajectory details
- **Chunk zoom** — select a chunk range, all panels filter to that range
- **Run comparison** — load two JSON reports side by side, diff the metrics
- **Failure mode pie chart** — ASCII pie chart of crash classifications

### Navigation
- Tab/Shift-Tab between panels
- Vim-style hjkl or arrow key navigation
- `/` to search episodes by criteria (e.g., "progress > 14%")
- `q` to quit, `r` to reload latest report

## Technical Considerations

- **Rust TUI crate:** `ratatui` (successor to `tui-rs`) — mature, well-documented, supports all the visual primitives needed
- **Data source:** Read from `reports/*.json` (the compact export has all the data needed)
- **Separate binary:** Should be a separate binary target in `Cargo.toml`, not part of the simulation runtime
- **No runtime dependency:** The TUI reads exported reports, never touches the running simulation

## Open Questions

- Should the TUI also support live-streaming from a running simulation (via a socket or shared file)?
- How much of the current Markdown report structure should the TUI mirror vs. reimagine?
- Should it support exporting screenshots (ansi-to-image) for sharing?
- Is `ratatui` the right choice, or should we evaluate alternatives?

## Status

Idea stage. No implementation work started. Revisit after PPO optimisation work is complete and the analytics pipeline is stable.
