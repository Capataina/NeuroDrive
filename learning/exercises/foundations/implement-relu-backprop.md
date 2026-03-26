# Exercise: Implement ReLU and Its Backward Pass

## Context

The A2C model uses ReLU activations in all four hidden layers (two in the actor, two in the critic). ReLU's backward pass is deceptively simple but has an important property: it hard-zeroes gradients for any unit that was inactive during the forward pass. This is the dead neuron problem, and it is tracked in `A2cTrainingStats` as `dead_relu_fraction`. Understanding why requires implementing the backward pass yourself.

## Prerequisites

- `exercises/foundations/implement-linear-layer.md`
- `concepts/foundations/neural-networks.md` — the activation functions section

## The Task

Implement the `Relu` struct with:

1. **Forward pass:** apply ReLU element-wise, store activation mask
2. **Backward pass:** gate the incoming gradient by the activation mask
3. **Dead neuron detection:** compute the fraction of units that were inactive over a batch

### Specification

**Forward:**
```
Input:  z ∈ R^n  (pre-activation, output of a Linear layer)
Output: x ∈ R^n  (post-activation)

x_i = max(0, z_i)

Store: mask_i = 1 if z_i > 0 else 0
```

**Backward:**
```
Input:  dx ∈ R^n  (gradient of loss w.r.t. post-activation x)
Output: dz ∈ R^n  (gradient of loss w.r.t. pre-activation z)

dz_i = dx_i * mask_i
```

**Dead neuron detection (across a batch of N forward passes):**
```
dead_fraction = count(units with mask_i = 0 for ALL N passes) / n
```

## Constraints

- Do not look at `src/brain/common/mlp.rs` until you have a working implementation.
- Implement dead neuron detection using a reasonable batch approximation.

## Checkpoints

**Checkpoint 1:** Verify your forward pass on a small example:
```
z = [3.0, -1.0, 0.0, -2.0, 5.0]
x = [3.0,  0.0, 0.0,  0.0, 5.0]  (expected)
```

**Checkpoint 2:** Verify your backward pass. Using the same z and an arbitrary `dx = [1.0, 1.0, 1.0, 1.0, 1.0]`:
```
dz = [1.0, 0.0, 0.0, 0.0, 1.0]  (expected — gradient zeroed at negative inputs)
```

Note: the behaviour at exactly `z = 0.0` is conventionally set to 0 (gradient undefined; we choose 0). Verify your implementation handles this consistently.

**Checkpoint 3:** Simulate dead neuron tracking. Construct a scenario where a unit is always inactive across a batch of 10 forward passes. Verify your detection reports the correct fraction.

## Hints

<details>
<summary>Hint 1 (backward pass key insight)</summary>

The backward pass is just pointwise multiplication of the incoming gradient by the mask you saved during the forward pass. If `mask_i = 0`, the gradient through unit `i` is completely cut off — that unit contributed nothing to the output, so it receives no gradient signal.

</details>

<details>
<summary>Hint 2 (dead neuron detection approach)</summary>

Over a batch of N samples, you need to know whether each unit was active in *any* sample. A simple approach: accumulate activation counts per unit (increment a counter when `z_i > 0`). After N samples, units with count 0 are dead. The dead fraction is `count(dead_units) / total_units`.

</details>

<details>
<summary>Hint 3 (the z = 0 boundary)</summary>

At exactly `z_i = 0.0`, the ReLU is not differentiable (left derivative = 0, right derivative = 1). The subgradient convention is to set the gradient to 0 at this point. Since 64-bit float operations rarely produce exactly 0.0 in practice, this boundary case does not significantly affect training — but your implementation should handle it consistently.

</details>

## The Dead Neuron Problem

After completing the implementation, consider this: if many units are dead, the network's effective capacity is reduced. Gradients cannot flow through dead units to update earlier layers. A2C tracks `dead_relu_fraction` for exactly this reason — if the fraction is high (e.g. > 20%), the network may be learning with significantly less capacity than its architecture nominally provides.

What causes dead neurons?
- Large learning rates that push weights too far negative in one update
- Initial log_std that causes the policy to produce consistently extreme (saturated) actions
- Very negative reward signals in early training that produce strongly negative gradient updates

## Reflection Questions

After completing the implementation:

1. ReLU was chosen over sigmoid or tanh for hidden layers. What is the practical advantage? What would change in the backward pass if sigmoid were used?

2. Consider a ReLU unit whose input `z_i` is almost always very small (between -0.001 and 0.001). Is it dead? Is it useful? Does it contribute to the network's output?

3. The `dead_relu_fraction` in `A2cTrainingStats` is computed per-update, not per-tick. Why might a unit be "dead" for one update batch but "alive" for another?

4. Leaky ReLU uses `max(0.01 * z, z)` instead of `max(0, z)`. How does this change the backward pass? Would dead neurons still be a problem?

## Related Files

- `concepts/foundations/neural-networks.md` — theory including dead neuron problem
- `exercises/foundations/implement-linear-layer.md` — the layer that feeds into ReLU
- `exercises/foundations/implement-adam-optimizer.md` — the next exercise
