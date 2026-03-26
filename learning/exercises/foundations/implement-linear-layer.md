# Exercise: Implement a Linear Layer

## Context

The `Linear` struct in `src/brain/common/mlp.rs` is the foundational building block of the entire A2C neural network. Every weight update, every policy gradient, every value prediction passes through `Linear` layers. If this implementation has a bug, the whole learning system is wrong in ways that may not produce obvious error messages.

This exercise asks you to implement a `Linear` layer from scratch, in Rust or in pseudocode, and verify that your implementation matches the mathematical specification exactly.

## Prerequisites

- `concepts/foundations/neural-networks.md` — especially the Linear layer and backpropagation sections
- Basic Rust or pseudocode ability

## The Task

Implement a `Linear` layer that supports:

1. **Forward pass:** `z = W * x + b`
2. **Backward pass:** given gradient of loss with respect to `z` (`dz`), compute:
   - `dW` — gradient with respect to weights
   - `db` — gradient with respect to biases
   - `dx` — gradient with respect to the input (needed to continue backprop to earlier layers)
3. **Weight initialisation:** Glorot uniform initialisation

The implementation must handle arbitrary `in_features` and `out_features`.

### Specification

**Forward:**
```
Input:  x ∈ R^{in_features}
Output: z ∈ R^{out_features}

z_j = Σ_i W_{ji} * x_i + b_j
```

**Backward:**
```
Input:  dz ∈ R^{out_features}  (gradient of loss w.r.t. z)
Output:
  dW_{ji} = dz_j * x_i         (outer product)
  db_j    = dz_j
  dx_i    = Σ_j W_{ji} * dz_j  (W^T * dz)
```

**Glorot uniform initialisation:**
```
limit = sqrt(6.0 / (in_features + out_features))
W_{ji} ~ Uniform(-limit, limit)
b_j = 0.0
```

## Constraints

- Do not look at `src/brain/common/mlp.rs` until you have a working implementation or have exhausted all hints.
- Your implementation can be in Rust, Python, or pseudocode — the mathematics is the same.
- You must implement all three components (forward, backward, init) to complete the exercise.

## Checkpoints

**Checkpoint 1:** Your forward pass produces the correct output for a 2×3 weight matrix, 1×2 bias, and a specific input. Verify by hand: pick W, b, x with small integer values and compute z manually.

**Checkpoint 2:** Your backward pass satisfies the numerical gradient check. Pick a small network (e.g. `Linear(2, 2)` followed by sum), perturb each weight by `ε = 1e-4`, and verify that:
```
(loss(W + ε*e_ji) - loss(W - ε*e_ji)) / (2ε)  ≈  dW_{ji}
```
Agreement to 4+ decimal places is expected for a correct implementation.

**Checkpoint 3:** Glorot initialisation for a `Linear(23, 64)` layer (the first actor layer) should produce weights in approximately `[-0.215, 0.215]`. Verify this.

## Hints

<details>
<summary>Hint 1 (general direction)</summary>

The forward pass is matrix multiplication with a bias. The key question is: which dimension of W is "output" and which is "input"? In NeuroDrive, W has shape `[out_features, in_features]`. So the j-th output neuron's weights are in row j of W.

</details>

<details>
<summary>Hint 2 (backward pass structure)</summary>

The backward pass uses two separate operations:
- `dW` is the outer product of `dz` and `x`: `dW = dz * x^T`
- `dx` is the transpose-matrix multiply: `dx = W^T * dz`
- `db` is just `dz` directly

The outer product `dz * x^T` computes: `dW_{ji} = dz_j * x_i`. This is because each weight W_{ji} was multiplied by x_i in the forward pass; by the chain rule, the gradient flows back proportionally to how the weight was used.

</details>

<details>
<summary>Hint 3 (Glorot limit computation)</summary>

For `Linear(23, 64)`:
```
fan_in = 23
fan_out = 64
limit = sqrt(6.0 / (23 + 64)) = sqrt(6.0 / 87) ≈ sqrt(0.0690) ≈ 0.2626
```

So weights are sampled from `Uniform(-0.2626, 0.2626)`.

</details>

## Reflection Questions

After completing the implementation:

1. Why is the gradient with respect to the input (`dx`) needed at all? Under what circumstances is it used?

2. What happens if you initialise all weights to zero? Try it — what does the network compute for any input?

3. Why does Glorot initialisation use `fan_in + fan_out` rather than just one of them? What would happen if you only used `fan_in`?

4. The forward pass accumulates a sum across `in_features`. What numerical issue could arise if `in_features` is very large?

## Related Files

- `concepts/foundations/neural-networks.md` — the theory
- `exercises/foundations/implement-relu-backprop.md` — the next exercise
- `src/brain/common/mlp.rs` — the reference implementation (after completing the exercise)
