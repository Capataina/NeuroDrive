# Exercise: Design The Vectorised Trainer Boundaries

## Goal

Show that you understand why the proposed vectorised trainer is a structural refactor rather than a cosmetic feature.

## Starting Point

Read:

- `project/comparisons/singleton-runtime-vs-vectorised-trainer.md`
- `project/systems/environment.md`
- `project/systems/a2c-baseline.md`
- `project/systems/debug-runtime.md`

Then inspect:

- `context/plans/vectorised-a2c-visual-trainer.md`

## Tasks

- identify at least four singleton assumptions that would need to change,
- propose which state should become per-car and which should remain trainer-wide,
- explain one analytics consequence and one debug consequence,
- explain the main risk of trying to patch this in incrementally without first clarifying ownership.

## Expected Outcome

You should finish with a subsystem boundary sketch, not with code.
