# The A2C Brain

## What This File Covers

The `brain/a2c/` module implements NeuroDrive's current autonomous controller: a handwritten Advantage Actor-Critic (A2C) system in Rust without any external ML framework. This file explains the full system from observation input to weight update, covering the model architecture, the rollout buffer, the GAE computation, and the training update in detail. It also explains the key ordering constraints and known limitations.

**Status:** Current implementation. This is the live learning system.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — MDP, returns, value functions
- `concepts/core/policy-gradient-methods.md` — policy gradient theorem, entropy bonus
- `concepts/core/advantage-estimation.md` — GAE recurrence, bootstrapping
- `concepts/core/actor-critic-architecture.md` — separate networks, Huber loss
- `concepts/core/continuous-control.md` — Gaussian policy, tanh squashing
- `concepts/foundations/neural-networks.md` — Linear layer, ReLU, backprop
- `concepts/foundations/optimization-and-gradients.md` — Adam optimiser

---

## The Model: ActorCritic

The policy is implemented as two separate networks that do not share weights.

### Architecture

```
ObservationVector (23-dim)
        │
        ├──────────────────────────┐
        ▼                          ▼
  Actor stack:               Critic stack:
  Linear(23 → 64)            Linear(23 → 64)
  ReLU                       ReLU
  Linear(64 → 64)            Linear(64 → 64)
  ReLU                       ReLU
  Linear(64 → 2)             Linear(64 → 1)
        │                          │
   [mean_steering,           [V(s): scalar
    mean_throttle]            value estimate]
        │
   Learnable log_std (2 params)
   (not observation-dependent)
```

### Key Implementation Details

**The actor outputs a mean** for each action dimension. The standard deviation is a learnable parameter that is independent of the observation — it is a scalar `log_std` for each of the two action dimensions. This means the policy's uncertainty is global (same for all states), not state-dependent. This is simpler than a full conditional Gaussian but is a known limitation.

**Initialisation:** All layers use Glorot uniform initialisation:
```
limit = sqrt(6.0 / (fan_in + fan_out))
weight ~ Uniform(-limit, limit)
```
Biases initialise to zero. `log_std` initialises to zero (so initial std = 1.0).

**ReLU activations** are used in all hidden layers. This is a known simplification relative to the literature, which sometimes favours tanh for continuous-control problems. ReLU is adequate for the baseline stage.

---

## The Gaussian Policy and Action Sampling

### Forward Pass (Inference)

At each tick, `a2c_act_system` runs the actor forward pass:
1. Compute `mean = actor.forward(obs)` — a 2-element vector
2. Compute `std = exp(log_std)` — element-wise
3. Sample from `N(mean, std)`: `latent = mean + std * N(0, 1)`
4. Apply tanh squashing: `squashed = tanh(latent)`
5. Map to action ranges:
   - `steering = squashed[0]` — already in `[-1, 1]`
   - `throttle = (squashed[1] + 1.0) / 2.0` — shifted to `[0, 1]`

### Why Tanh Squashing?

A Gaussian distribution has infinite support. Without squashing, a sample could be arbitrarily large, which would clamp at the action bounds and produce biologically incorrect gradients (the log-probability would be wrong for clamped actions because the Gaussian density was evaluated at the latent value, not the clamped applied value).

Tanh squashing maps the infinite-support Gaussian into the bounded interval `(-1, 1)` while keeping the function differentiable everywhere. The correct log-probability includes a Jacobian correction:

```
log π(a|s) = log N(latent | mean, std)
             - Σ_i log(1 - tanh²(latent_i))
```

The second term corrects for the change of variables. Without it, the policy gradient is biased — the network would learn incorrect credit assignment for actions near the boundaries.

### Storing Both Latent and Applied

The rollout buffer stores both:
- `latent` — the pre-tanh Gaussian sample (needed for log-probability computation in the update)
- `applied` — the post-tanh, post-rescaling executed action (stored for the record; physics uses this)

This matters because the log-probability is computed from the Gaussian density at the latent value, not at the applied value.

---

## The Rollout Buffer

```rust
pub struct RolloutBuffer {
    pub observations: Vec<ObservationVector>,
    pub actions: Vec<CarAction>,           // applied
    pub latent_actions: Vec<[f32; 2]>,     // pre-tanh samples
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub values: Vec<f32>,                  // V(s_t) from critic
    pub log_probs: Vec<f32>,               // log π(a_t | s_t)
    pub clamp_hits: Vec<[bool; 2]>,        // whether tanh was near ±1
}
```

### When the Buffer is Filled

The buffer is filled across multiple ticks. Each tick, `a2c_act_system` appends:
- current observation (before action)
- sampled action (latent and applied)
- critic value estimate V(s_t)
- log-probability log π(a_t|s_t)

Then `a2c_collect_reward_system` appends (in a later SimSet):
- reward r_t
- done flag

Note the split: the act system appends the state and action; the reward collector appends the reward and done. They operate in different SimSets because the reward is computed by `episode_loop_system` which runs between them.

### When an Update is Triggered

An update fires when either:
1. The rollout horizon is reached (`len >= horizon`, default often 512-1024 steps)
2. A terminal step occurs AND the buffer already has enough steps (`len >= min_batch_size`)

This means episodes that terminate early can trigger an update if enough data has accumulated. Episodes that run long will trigger at the horizon regardless.

### What Happens on Mode Toggle

When the user presses F4 to switch between keyboard and AI mode, the rollout buffer is **cleared** and the step counter is reset. This prevents mixed-controller trajectories: a buffer that contains actions from both keyboard and A2C cannot be used to compute valid policy gradients for either.

---

## GAE Computation

After an update is triggered, the buffer is processed to compute advantages. This is done by `RolloutBuffer::compute_gae()`.

### The Bootstrap Value

If the rollout ended at a non-terminal step (horizon was reached, not a crash/timeout/lap), a bootstrap value is needed:
```
V_bootstrap = critic.forward(last_observation)
```

If the rollout ended at a terminal step:
```
V_bootstrap = 0.0
```

This is correct because a terminal state has no future return.

### The GAE Recurrence

GAE is computed backwards from the end of the rollout:

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

A_T     = delta_T
A_{t-1} = delta_{t-1} + gamma * lambda * (1 - done_{t-1}) * A_t
```

The `(1 - done)` masking ensures that when a terminal step occurs, the advantage estimation does not bleed across episode boundaries. The next episode's value does not influence the current episode's advantage.

**Parameters:** `gamma = 0.99`, `lambda = 0.95` (from GAE paper; λ=0.95 is the standard baseline value).

### Advantage Normalisation

After computing all advantages in the rollout:

```
mean_A = mean(A_0, ..., A_{N-1})
std_A  = std(A_0, ..., A_{N-1})
A_norm = (A - mean_A) / (std_A + epsilon)
```

This ensures advantages are zero-centred and unit-variance within each update batch. Without normalisation, a batch with uniformly high returns would produce large policy gradient magnitudes unrelated to relative action quality.

---

## The Training Update: a2c_update()

The update function processes the entire rollout buffer in a single pass (no mini-batches in the current implementation).

### Value Loss (Critic)

The critic is trained to predict the Monte Carlo return (advantage + baseline):

```
target_t = A_t + V(s_t)
loss_value = mean( Huber(V_pred_t, target_t) )
```

**Why Huber loss instead of MSE?** Value targets can be noisy and outlier-prone, especially early in training when the critic knows nothing. Huber loss behaves like MSE for small errors (smooth gradient) and like L1 for large errors (bounded gradient). This prevents large value prediction errors from causing pathological weight updates.

### Policy Loss (Actor)

```
loss_policy = -mean( log_prob_t * A_norm_t )
```

This is the standard policy gradient loss with advantage weighting. A negative advantage (the action was worse than expected) produces a positive loss contribution, which pushes the gradient away from that action.

The log-probability is computed during the update (not during act) to get differentiable log-probs. The log-prob formula with tanh correction:

```
log_prob = log N(latent | mean, std) - sum_i log(1 - tanh²(latent_i))
```

### Entropy Bonus

```
entropy_t = 0.5 * (1 + log(2π)) + log_std[0] + log_std[1]
loss_entropy = -entropy_coeff * mean(entropy_t)
```

The entropy bonus rewards maintaining diverse action distributions. Without it, the policy can prematurely collapse to a deterministic action, ceasing to explore. The minus sign is because the loss is minimised: the entropy loss being negative means minimising it maximises entropy.

### Total Loss

```
loss_total = loss_value + loss_policy + loss_entropy
```

### Backpropagation and Gradient Update

`loss_total.backward()` propagates gradients through both actor and critic. Gradients are clipped with global norm clipping at 0.5 before the Adam step:

```
total_grad_norm = sqrt(sum of squared gradients across all params)
if total_grad_norm > 0.5:
    scale all gradients by (0.5 / total_grad_norm)
```

Then Adam steps separately for actor and critic (separate optimisers, separate learning rates: `actor_lr = 3e-4`, `critic_lr = 5e-4`).

**Note:** The `log_std` parameters use their own Adam momentum state, tracked separately from the network layer parameters.

### A2cTrainingStats

After each update, the following snapshot is written to `A2cTrainingStats`:

| Stat | Meaning |
|---|---|
| `value_loss` | Huber loss on value function |
| `policy_loss` | Policy gradient loss |
| `entropy` | Mean policy entropy |
| `explained_variance` | How much of return variance the critic explains |
| `action_mean`, `action_std` | Mean and std of applied actions this batch |
| `clamp_fraction` | Fraction of actions where tanh was near saturation |
| Per-layer weight norms | Sum of squared weights per layer |
| Per-layer gradient norms | Sum of squared gradients per layer |
| Dead ReLU fraction | Fraction of ReLU units that activated zero for the whole batch |

These stats are displayed in the HUD and persisted in the analytics export.

---

## Exit Handling

`a2c_flush_on_exit_system` runs in Bevy's `Last` schedule when the app exits. If the rollout buffer has accumulated steps but no update has been triggered (horizon not reached, no terminal), a final update is run on the partial rollout. This prevents the loss of learning signal from the final partial episode.

---

## Known Limitations

| Limitation | Consequence |
|---|---|
| No model persistence | Weights are lost when the app exits; training must restart from zero |
| No evaluation mode | Cannot run the policy deterministically (mean action only) for fair evaluation |
| No headless training | Requires a rendered window; cannot train faster than real-time |
| Ad hoc RNG (`rand::rng()`) | Sampling is not seeded; runs are not reproducible |
| No mini-batch updates | Entire rollout is processed as one batch; memory cost grows with horizon |
| State-independent std | Policy variance cannot adapt to observation context |
| Minimal Brain trait | The trait has no richer interface; plugging in a biological brain requires more scaffold |

---

## The A2C System in the Fixed Tick

For reference, the A2C-related systems and their positions in the fixed tick pipeline:

```
SimSet::Input:
  keyboard_action_input_system (if keyboard mode)
  a2c_act_system               ← writes desired action, appends obs/action/value to buffer
  action_smoothing_system

SimSet::Physics:
  car_physics_system

SimSet::Collision:
  collision_detection_system

SimSet::Measurement:
  update_track_progress_system
  episode_loop_system          ← computes reward and done
  update_sensor_readings_system
  build_observation_vector_system
  capture_episode_tick_trace_system
  snapshot_completed_episode_*_systems
  a2c_collect_reward_system    ← appends reward/done, maybe triggers update
  update_driving_hud_stats_system
  capture_driving_hud_episode_metrics_system
```

---

## Related Files

- `concepts/core/advantage-estimation.md` — GAE derivation and worked examples
- `concepts/core/actor-critic-architecture.md` — design decisions for the model structure
- `concepts/core/continuous-control.md` — Gaussian policy and tanh squashing theory
- `concepts/foundations/neural-networks.md` — Linear and ReLU implementations
- `concepts/foundations/optimization-and-gradients.md` — Adam optimiser derivation
- `project/decisions/a2c-as-baseline.md` — why A2C was chosen for the baseline role
- `project/decisions/tanh-squashed-actions.md` — why bounded action sampling matters
- `project/comparisons/a2c-vs-ppo.md` — how A2C compares to the obvious alternative
