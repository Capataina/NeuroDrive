# Exercise: Implement the Adam Optimiser

## Context

Adam is the optimiser used for both the actor and critic in NeuroDrive's A2C implementation. Unlike SGD, Adam maintains per-parameter moment estimates (first and second moments) that adaptively scale the learning rate for each parameter. Getting Adam right is essential for stable training; a subtle bug (e.g. missing bias correction) produces gradients that look correct but produce unstable learning.

NeuroDrive's A2C also runs Adam on the `log_std` parameters separately from the network layers — these are tracked with their own moment state. Understanding why requires implementing Adam yourself.

## Prerequisites

- `exercises/foundations/implement-relu-backprop.md`
- `concepts/foundations/optimization-and-gradients.md` — the Adam section

## The Task

Implement Adam with:

1. **Per-parameter moment initialisation**
2. **The Adam update step** given a parameter and its gradient
3. **Bias correction** (critical — do not skip)
4. **Global gradient norm clipping** applied before the Adam step

### Specification

**Parameters:**
```
η      = learning rate (e.g. 3e-4 for actor, 5e-4 for critic)
β1     = 0.9   (first moment decay)
β2     = 0.999 (second moment decay)
ε      = 1e-8  (denominator stability)
```

**State per parameter:**
```
m_t   = 0.0   (first moment estimate)
v_t   = 0.0   (second moment estimate)
t     = 0     (step counter)
```

**Update step (called once per parameter per training update):**
```
t     ← t + 1
m_t   ← β1 * m_{t-1} + (1 - β1) * g_t
v_t   ← β2 * v_{t-1} + (1 - β2) * g_t²

m̂_t  = m_t / (1 - β1^t)    (bias-corrected first moment)
v̂_t  = v_t / (1 - β2^t)    (bias-corrected second moment)

θ_t   ← θ_{t-1} - η * m̂_t / (sqrt(v̂_t) + ε)
```

**Global gradient norm clipping (applied before the Adam step):**
```
total_grad_norm = sqrt(Σ_i g_i²)    (sum over all parameters)

if total_grad_norm > clip_threshold (0.5):
    scale = clip_threshold / total_grad_norm
    g_i   ← g_i * scale   for all i
```

## Constraints

- Implement bias correction. Observe what happens early in training if you omit it.
- Implement norm clipping as a pre-step, not post-step.
- Do not look at `src/brain/common/mlp.rs` until you have a working implementation.

## Checkpoints

**Checkpoint 1:** Verify the bias correction matters in step 1.

At step `t = 1` with `g_1 = 1.0`, `β1 = 0.9`, `β2 = 0.999`:
```
m_1 = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
v_1 = 0.999 * 0.0 + 0.001 * 1.0 = 0.001

Without bias correction:
  effective_lr ≈ η * m_1 / sqrt(v_1) = 0.0003 * 0.1 / sqrt(0.001) ≈ 0.00000949

With bias correction:
  m̂_1 = 0.1 / (1 - 0.9) = 1.0
  v̂_1 = 0.001 / (1 - 0.999) = 1.0
  effective_lr ≈ η * 1.0 / sqrt(1.0) = 0.0003
```

The bias correction brings the effective learning rate close to `η` at step 1, rather than ~31× smaller. This is the critical warmup effect that bias correction provides.

**Checkpoint 2:** Verify gradient clipping. With five parameters having gradients `[3.0, 4.0, 0.0, 0.0, 0.0]`:
```
total_norm = sqrt(9 + 16) = 5.0
scale = 0.5 / 5.0 = 0.1
clipped = [0.3, 0.4, 0.0, 0.0, 0.0]
```

**Checkpoint 3:** Run 100 steps of Adam on the simple quadratic `f(θ) = θ²` starting from `θ = 1.0`, `η = 0.01`. The parameter should converge to near 0. Verify numerically.

## Hints

<details>
<summary>Hint 1 (step counter scope)</summary>

The step counter `t` is per-Adam-instance, not per-parameter. All parameters in the same network share the same step counter `t`, but each parameter has its own `m` and `v` state. Incrementing `t` once per update step (not once per parameter) is correct.

</details>

<details>
<summary>Hint 2 (the denominator epsilon)</summary>

The `ε = 1e-8` is added to `sqrt(v̂_t)`, not to `v̂_t`. The expression is:
```
θ ← θ - η * m̂_t / (sqrt(v̂_t) + ε)
```
Not:
```
θ ← θ - η * m̂_t / sqrt(v̂_t + ε)  ← wrong position
```
The difference is numerically small in practice but the formula in the paper uses the first form.

</details>

<details>
<summary>Hint 3 (gradient clipping must precede moment update)</summary>

Gradient clipping scales the raw gradients before they are fed into the moment estimates. If clipping happened after the moment update, the moments would be tracking the pre-clipped gradients and the effective update might not respect the clip threshold. The pipeline is: `compute gradients → clip → Adam moment update → parameter update`.

</details>

## A Note on the `log_std` Parameters

NeuroDrive's A2C maintains separate Adam moment state for the two `log_std` parameters (one per action dimension). These are not part of any `Linear` layer — they are standalone learnable scalars. When the actor's Adam step runs, it updates `log_std` using the same step counter and the same β1/β2, but separate m/v accumulators.

Why separate? Because `log_std` is a different kind of parameter than a weight matrix row. It controls exploration globally. Its gradient magnitude and typical update scale can differ significantly from the dense weight gradients. Sharing moment state would corrupt the normalisation.

## Reflection Questions

After completing the implementation:

1. What does Adam do when the gradient is consistently positive for many steps? What happens to the effective learning rate?

2. Gradient norm clipping at 0.5 is quite aggressive. What would happen if the clip threshold were set to 10.0? What about 0.01?

3. Two separate Adam instances run in NeuroDrive — one for the actor and one for the critic. Why not share one Adam instance? What would go wrong?

4. What happens to the bias correction `(1 - β1^t)` after 1000 training steps? Is bias correction still important then?

## Related Files

- `concepts/foundations/optimization-and-gradients.md` — the full Adam derivation
- `exercises/core/implement-gae.md` — the next major exercise
- `src/brain/common/mlp.rs` — reference implementation (after completing)
