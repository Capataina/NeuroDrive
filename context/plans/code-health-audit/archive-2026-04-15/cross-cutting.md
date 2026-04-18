# Cross-Cutting — Code Health Findings

**Systems covered:** Project-wide patterns
**Finding count:** 2 findings (0 high, 1 medium, 1 low)

---

## Configuration Drift

### PPO Hyperparameters Scattered Across Multiple Locations
- [x] Consolidate PPO hyperparameter defaults into a single `PpoConfig` struct instead of spreading them across `PpoBrain::default()`, `update.rs` constants, and `model.rs::ActorCritic::new()`

**Category:** Configuration Drift
**Severity:** Medium
**Effort:** Small
**Behavioural Impact:** None (verified — same default values, centralised location)

**Location:**
- `src/brain/ppo/mod.rs:46-61` — `PpoBrain::default()` — gamma, gae_lambda, max_steps, min_update_steps, ppo_epochs, clip_epsilon, samples_per_tick
- `src/brain/ppo/update.rs:10-13` — `VALUE_HUBER_DELTA`, `ACTOR_GRAD_CLIP_NORM`, `CRITIC_GRAD_CLIP_NORM`, `ENTROPY_COEF`
- `src/brain/ppo/model.rs:133-134` — actor LR (3e-4), critic LR (5e-4), critic weight decay (3e-4) hardcoded in `AdamOptimizer::new` calls
- `src/brain/ppo/update.rs:333` — log_std learning rate (3e-4) and clamp range (-1.0, 0.5) hardcoded in the log-std update loop
- `src/brain/ppo/model.rs:49-50` — actor hidden dim (64), critic hidden dim (128) hardcoded in `PpoBrain::default()`

**Current State:**
PPO hyperparameters are spread across 4 files:
1. **`PpoBrain::default()`** has gamma, gae_lambda, horizons, epochs, clip_epsilon, samples_per_tick, and hardcodes the network dimensions.
2. **`update.rs` file-level constants** have Huber delta, grad clip norms, and entropy coefficient.
3. **`model.rs` ActorCritic::new()`** has learning rates and weight decay hardcoded in `AdamOptimizer::new` calls.
4. **`update.rs` log-std update loop** has the log-std learning rate and clamp range as magic numbers.

An engineer tuning hyperparameters must search 4 files to find all the knobs. Some values that are logically part of the same configuration (e.g., actor LR and critic LR) live in the model constructor rather than in the brain config.

**Proposed Change:**
Define a `PpoConfig` struct that holds all tuneable parameters:
```rust
pub struct PpoConfig {
    // Rollout
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub min_update_steps: usize,
    pub ppo_epochs: usize,
    pub clip_epsilon: f32,
    pub samples_per_tick: usize,
    // Network
    pub actor_hidden_dim: usize,
    pub critic_hidden_dim: usize,
    // Optimiser
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub critic_weight_decay: f32,
    pub entropy_coef: f32,
    pub actor_grad_clip: f32,
    pub critic_grad_clip: f32,
    // Exploration
    pub log_std_floor: f32,
    pub log_std_ceil: f32,
    pub log_std_lr: f32,
}
```

Pass this config to `PpoBrain::new()`, `ActorCritic::new()`, and the update functions. Remove the scattered constants. `PpoBrain::default()` constructs a default `PpoConfig` and passes it through.

**Justification:**
Having hyperparameters in 4 files is a maintenance trap. When the project reaches the point of running hyperparameter sweeps or saving/loading configurations, having a single serialisable config struct will be essential. Centralising now prevents parameter drift (e.g., the log-std learning rate is 3e-4 but the actor LR is also 3e-4 — are these intentionally the same? Hard to tell when they live in different files).

This is not adding a configuration file or external config system — it is consolidating existing magic numbers into a single struct with defaults. The struct is internal to the brain module.

**Expected Benefit:**
All PPO hyperparameters visible in one place. Enables future config serialisation for experiment tracking. Makes hyperparameter sweeps straightforward. Reduces the risk of accidentally changing one parameter without noticing a related one in a different file.

**Impact Assessment:**
Zero functional change. Same default values, same runtime behaviour. Only the location of the defaults changes.

---

## Inconsistent Patterns

### `wrap_angle` in `sim/mod.rs` vs `wrap_to_pi` in `centerline.rs`
- [x] Replace the private `wrap_to_pi` in `centerline.rs` with the shared `wrap_angle` from `sim/mod.rs`

**Category:** Inconsistent Patterns
**Severity:** Low
**Effort:** Trivial
**Behavioural Impact:** None (verified — both functions produce identical output)

**Location:**
- `src/sim/mod.rs:14-22` — `wrap_angle()`
- `src/maps/centerline.rs:380-389` — `wrap_to_pi()`

**Current State:**
Two functions with identical behaviour exist:
- `sim::wrap_angle(angle) -> f32` — wraps to `[-PI, PI]` using a while loop.
- `centerline::wrap_to_pi(angle) -> f32` — wraps to `(-PI, PI]` using a while loop. The only difference is `wrap_angle` uses `>` and `<` bounds while `wrap_to_pi` uses `>` and `<=`.

For all practical inputs (angles produced by `atan2` which returns `(-PI, PI]`), both functions produce identical results. The boundary difference at exactly `-PI` is irrelevant because `atan2` never produces exactly `-PI` in normal operation.

**Proposed Change:**
Replace the call to `wrap_to_pi` in `push_corner_arc_samples` with `crate::sim::wrap_angle` and delete the private `wrap_to_pi` function.

**Justification:**
The previous audit (30 March) consolidated `wrap_angle` and `signed_angle_between` into `sim/mod.rs` specifically to eliminate triplicated geometry utilities. `wrap_to_pi` is a leftover that was missed because it has a different name.

**Expected Benefit:**
Removes 10 lines of duplicated code. Ensures all angle wrapping goes through the single shared utility.

**Impact Assessment:**
Zero functional change. The functions produce identical results for all inputs produced by `atan2` (the only source of angles passed to these functions). The theoretical boundary difference at exactly `-PI` is not reachable from the call site (which passes the difference of two `atan2` results).
