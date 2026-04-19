# Normalisation Layers in the PPO Stack

NeuroDrive's PPO has **three distinct normalisation layers** after the 2026-04-19 round-2 intervention. They are orthogonal — each fixes a different problem — and it is easy to confuse them. This note exists because they get misattributed in RL debugging discussions.

## The three normalisations

| Layer | What it normalises | Where it lives | What it fixes |
|---|---|---|---|
| **Advantage normalisation (per-minibatch)** | Advantages `A_t = GAE` within each training chunk | `src/brain/ppo/update.rs` `ppo_process_chunk` (`chunk_adv_mean`, `chunk_adv_std` used as `adv = (A − µ)/σ`) | Policy-gradient variance. Actor-side. Has been in place since the earliest PPO landing. |
| **Value-target normalisation (PopArt on `c_value`)** | GAE returns (regression targets for the critic) | `src/brain/ppo/update.rs` `popart_absorb_batch`; state on `PpoBrain.value_norm` | Critic target scale. Prevents the `c_fc2` saturation the round-1 research diagnosed. Ships 2026-04-19. |
| **Observation normalisation (Welford running stats)** | Each of the 43 observation dims, centred + scaled | `src/agent/observation.rs` `ObservationNormalizer`, applied in `build_observation_vector_system` after raw feature assembly | Input-side distribution. Stabilises the critic's first hidden layer. Andrychowicz 2021's #1 recommendation. Ships 2026-04-19. |

## Why these are orthogonal, not redundant

Advantage normalisation acts on the **policy gradient weight**. It reduces the variance of the actor's update by keeping advantages on a consistent scale within a minibatch. It does **nothing for the critic's target distribution** — returns can still grow unbounded even while advantages are normalised.

PopArt acts on the **critic's regression target**. It keeps the critic learning against `~N(0, 1)` targets regardless of return scale. This is what prevents the `c_fc2` tanh saturation that the round-1 research identified as the round-1 training bottleneck. PopArt does nothing for the policy gradient scale and nothing for the input distribution.

Observation normalisation acts on the **input to the networks**. It keeps the observation distribution stable as the policy changes what parts of the state space it visits. This is what Andrychowicz et al. 2021 call out as the single most-cited PPO implementation detail, and it was missing from NeuroDrive until 2026-04-19.

## Common misattribution to avoid

When the critic is failing to distinguish safe from dangerous states, the reflex is to blame the policy-gradient side and reach for advantage normalisation. **It is already in place and is the wrong lever for this symptom.** Target scale is the right lever (→ PopArt). If a future session notices `c_fc2` saturation again, the fix stack to check is:

1. Is PopArt enabled? `brain.config.popart_enabled`.
2. Is `value_norm.sigma` tracking the return scale? Analytics section 13 "Value Target Scale Tracker" surfaces this.
3. Is observation normalisation active? Check `ObservationNormalizer.count > warmup_samples` and `enabled`.

Only if 1–3 are all good should the next debug step be critic architecture (widen, LayerNorm, activation).

## Disable-flags for ablation

Each normalisation has an explicit disable:

- Advantage norm: inherent to the training loop — no flag, cannot be disabled without editing `ppo_process_chunk`.
- PopArt: `PpoConfig.popart_enabled = false`. `ValueNorm` stays at `(µ=0, σ=1)` so all normalisation becomes identity.
- Observation norm: `ObservationNormalizer.enabled = false`. Identity pass-through regardless of warmup state.

This lets a future session run ablations cleanly — flip one switch, re-run, compare analytics.

## References

- `context/references/value-target-normalisation.md` — PopArt derivation and Rust implementation sketch.
- `context/references/ppo-critic-architecture.md` — saturation diagnosis; includes the steelman section explaining when advantage norm is incorrectly blamed for critic issues.
- Commit `3bed996` — the single integrated change that landed PopArt + observation normaliser + γ=0.995 + target-KL early stop together. Its message contains the full per-intervention rationale.
