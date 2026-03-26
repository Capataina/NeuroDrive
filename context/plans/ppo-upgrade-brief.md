# PPO Upgrade — Intent Brief

**Status:** Planned. After vectorised A2C stages are complete.

## Problem

A2C exhibits policy oscillation: cars make progress, then suddenly crash repeatedly after an update, then recover, then crash again. This is a known A2C weakness — each update can change the policy drastically, and on-policy learning discards all old data after each update. With 3 envs and a 512-transition buffer, updates fire roughly every 2.8 seconds and can destabilise the policy.

## Solution

Upgrade the update function from A2C to PPO. PPO clips the policy ratio to `[1-ε, 1+ε]` (typically ε=0.2), preventing any single update from changing the policy too much. This directly addresses the oscillation.

## Scope

Small change — almost entirely within the update path:

- `buffer.rs` — add `old_log_probs: Vec<Vec<f32>>` to `TrainerRolloutBuffer`, pushed at act time
- `mod.rs` — compute and store log-probs when pushing to buffer in `a2c_act_all_cars_system`
- `update.rs` — wrap the forward/backward loop in K epochs (3–10), compute `π_new / π_old` ratio, apply clipping to the surrogate objective

The model, act/collect systems, per-env GAE, analytics, HUD — none of these change.

## Sequencing

1. Complete remaining vectorised A2C stages (3–5)
2. Implement PPO upgrade
3. Test whether oscillation is resolved
4. Tune ε, epoch count, and mini-batch size if needed
