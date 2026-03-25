# Project State And Next Pressure Points

## Current State

NeuroDrive is past the stage of being "mostly intent". The current runtime already includes:

- deterministic fixed-step racing simulation,
- a stable controller interface,
- a handwritten A2C baseline,
- analytics export,
- a meaningful debug HUD.

That means the project is now in a validation-and-transition phase rather than a pre-implementation phase.

## Missing Or Future Pieces

Several obvious next steps remain:

- controlled RNG and stronger reproducibility,
- richer run metadata,
- checkpointing and evaluation mode,
- headless or accelerated training modes,
- vectorised synchronous A2C,
- eventually the actual biological-learning subsystem.

## Why These Gaps Matter

Each missing piece blocks a different kind of confidence:

- without reproducibility, experiments are weak,
- without evaluation mode, learning and measurement are entangled,
- without vectorisation, A2C remains structurally limited,
- without the biological subsystem, the repository has not yet touched its hardest research question.

## Recommended Reading Of The Current Moment

The best current interpretation is:

- the environment and observability layers are ready enough to support more serious experiments,
- the A2C baseline is useful but still immature,
- the next engineering work should strengthen experiment discipline and scaling before the project attempts to replace A2C with a more ambitious learner.

## Related Files

- `project/comparisons/current-baseline-vs-target-biological-system.md`
- `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
