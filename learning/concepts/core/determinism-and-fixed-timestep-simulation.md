# Determinism And Fixed-Timestep Simulation

## Why This Matters Here

If you cannot trust the step ordering of a learning environment, it becomes difficult to reason about:

- physics correctness,
- reward timing,
- observation freshness,
- training stability,
- analytics interpretation.

NeuroDrive already has a relatively clean fixed-step structure. That is one of the project’s practical strengths.

## Fixed Timestep

The simulation advances at `60 Hz`.

Why this matters:

- the same logical systems run on the same cadence,
- physics receives stable integration intervals,
- reward accumulation and observation updates are easier to reason about,
- deterministic replay becomes more plausible.

## Ordered Simulation Sets

The current environment is explicitly arranged as:

`Input -> Physics -> Collision -> Measurement`

This is not decorative naming. It encodes causal truth:

1. choose action,
2. step the vehicle,
3. detect whether it left the road,
4. measure progress, reward, observations, traces, and update triggers.

## Why Order Matters For Learning

Consider a single tick.

If reward were collected before the environment finalised crash truth, the learner could receive stale reward. If observations were rebuilt before reset, the post-terminal observation might describe the crash frame instead of the reset frame. Small ordering errors here produce subtle training bugs.

That is why the current docs emphasise schedule placement so heavily.

## Layers Of Determinism

Determinism is not all-or-nothing. NeuroDrive currently has:

- **stronger determinism** in fixed-step environment logic, track construction, and pure car dynamics,
- **weaker determinism** in the A2C path because random sampling is not yet controlled by a single explicit seed strategy,
- **missing determinism** at the full experiment level because analytics filenames, run metadata, and reproducible training runs are not yet fully disciplined.

## Worked Practical Interpretation

The environment core can answer:

- "If I apply the same control sequence to the same car dynamics helper, do I get the same trajectory?"

It cannot yet confidently answer:

- "If I rerun the whole A2C training session under the same declared seed, do I get meaningfully comparable behaviour?"

That distinction is important. The repository is not non-deterministic everywhere, but neither is it fully reproducible end to end.

## Why This Matters For Future Work

Several planned directions become much harder without stronger reproducibility:

- checkpoint comparison,
- evaluation mode,
- controlled observation ablations,
- vectorised trainer validation,
- fair comparisons between A2C upgrades and future biological-learning experiments.

## Common Misunderstandings

❌ "Deterministic physics is enough."

Why wrong:
The controller path, RNG ownership, analytics timing, and update conditions also affect what experiment you actually ran.

❌ "Because Bevy is an engine, determinism is out of reach anyway."

Why wrong:
Perfect determinism may be hard, but much stronger experiment discipline is still achievable and worthwhile.

## Related Files

- `project/architecture/data-flow-and-schedule.md`
- `project/systems/environment.md`
- `project/systems/a2c-baseline.md`
- `project/systems/analytics.md`
