# Why A2C Exists In A Brain-Inspired Project

## Decision Summary

The project currently uses a handwritten A2C baseline because the repository needs a concrete learnability probe before it can responsibly invest in a much harder biological-learning architecture.

## Why This Was The Right Call

The current codebase needed answers to narrower engineering questions first:

- can the environment support learning,
- is the observation interface informative enough,
- is the reward signal usable,
- is the schedule disciplined enough,
- are analytics and debug surfaces capable of diagnosing behaviour.

A2C can answer those questions sooner than a full local-plasticity implementation can.

## Alternatives

### Alternative 1: go straight to biological learning

Why worse right now:

- much larger research and implementation risk,
- harder to debug,
- too many unknowns change at once,
- failure would be harder to attribute.

### Alternative 2: use a mature external ML framework baseline

Why worse for this repository:

- conflicts with the project’s first-principles ethos,
- hides important algorithm and infrastructure details,
- reduces educational transparency.

### Alternative 3: delay autonomous learning until much later

Why worse:

- slows validation of the environment and interface,
- leaves too much architecture speculative,
- delays observation and reward debugging.

## Trade-Off Accepted

The project accepted one major trade-off:

- the live learner is not philosophically aligned with the final vision.

That trade-off is acceptable only if the repository continues to document the distinction clearly and keeps the baseline modular rather than allowing it to become permanent architectural gravity.

## What This Decision Unlocks

- live end-to-end autonomous control,
- measurable training behaviour,
- stress-testing of environment and representation,
- a firmer basis for future biological-learning experiments.

## Related Files

- `project/comparisons/current-baseline-vs-target-biological-system.md`
- `concepts/advanced/a2c-vs-biological-learning.md`
