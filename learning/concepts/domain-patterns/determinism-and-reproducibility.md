# Determinism and Reproducibility

## Why This Matters Here

A learning system that cannot be reproduced is a learning system whose results cannot be trusted. If the same code and the same initial conditions produce different outcomes on different runs, you cannot tell whether an improvement was caused by your change or by a lucky random seed.

NeuroDrive explicitly treats determinism as a layered concern. Some layers are strong today; others are known gaps. Understanding this structure is important for anyone running or interpreting experiments.

**Status:** Current. Describes the current state of determinism in the codebase, including known weaknesses.

## Prerequisites

- `concepts/foundations/bevy-ecs-primer.md` — the ECS scheduling model

---

## Why Determinism Matters for RL

In a typical software test, determinism means "same input, same output." In RL, it means:

> Given the same random seed, initial weights, and environment configuration, the training trajectory should be identical across runs.

Without this, any claim "my change improved performance by X%" could just mean "my change was run with a luckier seed." Reproducibility is the minimum bar for honest evaluation.

### The Three Layers of Determinism in NeuroDrive

```
Layer 1: Physics determinism
Layer 2: Schedule determinism
Layer 3: Controller determinism
```

---

## Layer 1: Physics Determinism

**Current status: Strong**

The car dynamics are computed by `step_car_dynamics()`, a pure function with no global state. Given identical input (current velocity, applied action, timestep), it always produces identical output.

The simulation runs at a fixed `60 Hz` timestep:

```rust
app.insert_resource(Time::<Fixed>::from_hz(60.0));
```

This means physics is not frame-rate dependent — the car's position after 10 seconds of identical actions is the same regardless of the rendering frame rate.

A unit test in `src/game/physics.rs` explicitly verifies this:
- Run the physics for N steps with a fixed action sequence
- Run the same steps again
- Assert identical trajectory

**What this guarantees:** The physics simulation is deterministic given deterministic actions.

---

## Layer 2: Schedule Determinism

**Current status: Strong**

The `SimSet` ordering contract (`Input → Physics → Collision → Measurement`) is explicitly enforced. Every system in the fixed-tick pipeline runs in a defined order every tick.

This matters because RL correctness depends on ordering:
- Reward must be computed after physics (otherwise the tick's action has not affected the state)
- Observations must be rebuilt after episode resets (otherwise the post-reset observation reflects the pre-reset state)
- A2C reward collection must happen after episode logic (otherwise `done` flags are wrong)

These are not "nice to have" orderings — they are correctness requirements. The schedule is the implementation of this.

---

## Layer 3: Controller Determinism

**Current status: Weak**

The A2C brain samples actions from a Gaussian distribution using a random number generator:

```rust
let mut rng = rand::rng();    // created ad hoc from the OS entropy
let latent = sample_normal(mean, std, &mut rng);
```

This RNG is created fresh each time without a fixed seed. Different runs produce different action samples, even for identical observations and weights. This means:

1. You cannot replay an A2C training run by providing the same initial seed
2. You cannot prove that two runs with different configurations are truly comparable (vs. lucky random differences)
3. Debugging specific failure trajectories is much harder

**Why this is a known gap, not a bug:** The A2C baseline is still at the "learnability validation" stage. Controlled seed management is documented as the highest-priority missing experimental requirement in `context/references/a2c-for-neurodrive.md`.

---

## The Fixed Timestep and Non-Determinism Boundary

There is an important boundary to understand:

```
Deterministic side:          Non-deterministic side:
  physics                      A2C action sampling
  track construction           analytics export filenames (timestamps)
  centreline projection        HUD rendering order (frame-rate dependent)
  collision detection
```

Everything on the left is deterministic given the same action sequence. Everything on the right introduces non-determinism that survives the fixed-timestep guarantee.

---

## Analytics Export Filenames

Analytics reports are written to timestamped files:

```
reports/run_2025-03-15T14:23:00.json
```

This is a small, acceptable non-determinism — file names are never part of any learning calculation. But it means you cannot compare two run files by name across machines; you must compare by content.

---

## Why Controlled Seeds Matter for A2C

An on-policy method like A2C has high sensitivity to the random samples it sees during each rollout. With a fixed seed:

1. **Reproducibility:** the same training run can be reconstructed exactly
2. **Ablation validity:** comparing two configurations with the same seed isolates the effect of the change
3. **Debugging:** a problematic trajectory can be replayed exactly
4. **Regression tests:** a known-good training curve can be used as a baseline

The fix is straightforward in principle: initialise a seeded RNG at startup and thread it through the A2C act path. The challenge in Bevy/ECS is that resources are accessed by reference and the RNG must be mutably accessible in the act system.

---

## Determinism in the Planned Biological Brain

When the project transitions to local plasticity (Milestone 2), determinism becomes even more important:

- Eligibility traces update every tick — small RNG differences could compound over thousands of ticks
- Structural plasticity (Milestone 5) involves probabilistic growth/pruning decisions
- The "one brain, one lifetime" property means errors accumulate; there is no "reset to a checkpoint"

The seed management infrastructure built for A2C validation will be load-bearing for biological learning experiments.

---

## Common Misunderstandings

❌ "The physics test proves the whole system is deterministic"
✅ The physics test proves that `step_car_dynamics()` is deterministic given identical inputs. The full system (with A2C action sampling) is not deterministic at the controller level.

❌ "Fixed timestep guarantees reproducibility"
✅ Fixed timestep guarantees that physics produces identical results for identical actions. It does not constrain what actions are chosen.

❌ "Non-deterministic experiments are useless"
✅ Non-deterministic experiments can show consistent trends across multiple seeds and are widely used in RL research. They just require running multiple seeds and reporting statistics, rather than treating a single run as conclusive.

---

## Related Files

- `project/systems/determinism.md` — `context/` implementation memory for this topic
- `project/systems/a2c-brain.md` — the current controller (non-deterministic)
- `context/references/a2c-for-neurodrive.md` — why seed control is the highest-priority gap
