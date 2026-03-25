# Analytics And Debugging

## What This System Does

This learning file groups the two observability surfaces because they teach complementary lessons:

- `debug` is the live runtime inspection layer,
- `analytics` is the post-run exported analysis layer.

## Where It Fits

Both systems sit downstream of runtime truth. They should help you understand the car and learner, but they should not invent the environment or training facts they display.

## Key Mechanics

Runtime debug:

- `src/debug/overlays.rs` draws centreline, projection, vectors, lookahead markers, and rays.
- `src/debug/hud.rs` builds the Bevy UI diagnostics panel shown with `F3`.
- The HUD tracks recent episode history in four quarters and shows a lightweight run assessment.

Analytics:

- `src/analytics/trackers/` accumulates per-tick traces, per-episode action summaries, and exported run records.
- `src/analytics/models.rs` defines the canonical schemas.
- `src/analytics/exporters/` writes JSON and Markdown reports under `reports/`.

## Important Trade-Offs

- Debug is live and lightweight; analytics is slower but richer.
- Analytics is currently exit-triggered, so abrupt termination can lose a run.
- Reports are already useful, but still miss experiment metadata such as seed, config snapshot, git revision, and explicit track identity.
- Both layers still assume a single active car in key places, so vectorised training would force broader redesign.

## Learning Links

- Related systems: `learning/project/systems/environment.md`
- Related systems: `learning/project/systems/a2c-baseline.md`
- Related exercise: `learning/exercises/project/reason-about-schedule-order.md`

## Status

Current for this project.
