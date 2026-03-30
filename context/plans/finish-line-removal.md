# Plan — Finish Line Removal and Distance-From-Spawn Paradigm

## Goal

Remove the finish line concept entirely. Replace absolute track position as the progress metric with cumulative forward arc-length from spawn. This makes the episode model "drive as far as you can, as fast as you can, in 30 seconds" — no special positions, no lap-complete detection, no wrap-point gaming.

## Motivation

With random spawn positions, the current lap-complete detection is actively harmful:
- A car spawning at 90% can "complete a lap" by driving 10% of the track, earning a 100.0 lap bonus
- Analytics report absolute progress (0–100%) which inflates ghost car metrics — a car at 70% position didn't necessarily drive 70% of the track
- The policy can learn that being near the end of the track is disproportionately valuable, creating bad behaviours

The cleaner model: every metre of track is equally valuable. Progress = how far you drove from where you started.

## Changes Required

### Episode System (`src/game/episode.rs`)

1. Remove from `EpisodeConfig`:
   - `lap_arm_fraction`
   - `lap_wrap_from_fraction`
   - `lap_wrap_to_fraction`
   - `lap_bonus`

2. Remove from `EpisodeState`:
   - `lap_armed`

3. Remove `EpisodeEndReason::LapComplete` variant

4. Add to `EpisodeState`:
   - `distance_driven: f32` — cumulative forward arc-length from spawn
   - `spawn_progress_s: f32` — arc-length position where this episode's spawn occurred

5. Modify per-tick progress computation:
   - Compute forward arc-length delta with wrap handling:
     ```
     raw_delta = current_s - previous_s
     if raw_delta < -total_length/2 { raw_delta += total_length }  // wrapped forward
     if raw_delta > total_length/2 { raw_delta -= total_length }   // wrapped backward
     forward_delta = raw_delta.max(0.0)
     ```
   - Accumulate: `distance_driven += forward_delta`
   - Progress fraction becomes: `distance_driven / total_length`

6. Episode ends on crash or timeout only

### Analytics Rework

1. `EpisodeRecord.progress` should report `distance_driven / total_length` rather than absolute track position
2. Add `spawn_position_fraction: f32` to `EpisodeRecord` so analytics can distinguish car 0 from ghost cars
3. Add `distance_driven: f32` to `EpisodeRecord` for honest distance reporting
4. Consider separate car 0 vs ghost car sections in the markdown report
5. Remove lap completion rate from chunking and trend tables
6. "Max progress" becomes "longest distance driven in one life"

### Reward Changes

- Remove lap bonus from terminal reward
- Speed-weighted progress reward already handles everything: `progress_delta * speed_multiplier * scale` where progress_delta is now arc-length-based

## Open Questions

- Should the timeout remain at 30 seconds? Yes — it's the clock they're racing against.
- Should we allow multi-lap driving? Yes — if a car drives past its spawn point, distance keeps accumulating. A car that drives 1.5× the track in 30 seconds is genuinely better.
- How to handle the HUD? It currently shows lap-related stats that would need updating.

## Status

**Implemented** — completed 2026-03-27. The finish line, lap detection, and lap bonus have been fully removed. Episodes end on crash or timeout only. Progress is tracked as cumulative forward arc-length from spawn (`distance_driven`), with wrap-aware delta computation. All cars receive a fresh random spawn position each episode. This file is retained as historical context for the design rationale.
