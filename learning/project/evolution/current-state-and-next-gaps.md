# Current State And Next Gaps

## Current Implemented Reality

- Deterministic fixed-step racing environment is live.
- Stable action and observation boundary is live.
- Handwritten A2C baseline is live.
- Debug overlays and runtime HUD are live.
- Analytics export to JSON and Markdown is live.

## Important Missing Capabilities

- controlled RNG ownership for reproducibility,
- headless or accelerated training mode,
- save/load checkpoints,
- evaluation-only mode,
- richer run metadata,
- broader environment and learner regression coverage.

## Why These Gaps Matter

The current repo is already beyond a toy prototype. The next risks are no longer “can we wire a learner at all?” but:

- can we compare runs honestly,
- can we trust the learning signal,
- can we scale experiments without manual observation only,
- can we keep A2C baseline work from swallowing the long-term biological direction.

## Near-Term Pressure Points

- the proposed vectorised A2C trainer would require breaking singleton-car assumptions,
- the README now reflects that Milestone 1 is partially implemented rather than still purely planned,
- `learning/` and `context/` should continue to distinguish current implementation truth from long-term intent.
