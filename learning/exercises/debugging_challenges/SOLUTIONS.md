# Solutions: Debugging Challenges

---

## Challenge 1: Broken Backpropagation (`broken_backprop.py`)

### The Bug

In `linear_backward`, the weight gradient computation uses `grad_output` where it should use `cached_input`:

```python
# Buggy line:
grad_W = [[grad_output[i] * grad_output[j] for j in range(in_dim)] for i in range(out_dim)]

# Correct line:
grad_W = [[grad_output[i] * cached_input[j] for j in range(in_dim)] for i in range(out_dim)]
```

### Why This Causes the Observed Behaviour

The correct weight gradient for a linear layer `y = W @ x + b` is:

```
dL/dW[i][j] = dL/dy[i] * x[j] = grad_output[i] * cached_input[j]
```

The buggy version computes `grad_output[i] * grad_output[j]` instead. This is wrong in two ways:

1. **Wrong magnitude**: The gradient depends on `grad_output` squared rather than on the relationship between the upstream gradient and the input that produced the output. This means the weight update direction is not the true gradient descent direction.

2. **Shape mismatch is masked**: Because `grad_output` has length `out_dim` and `cached_input` has length `in_dim`, the buggy code indexes into `grad_output` with `j` up to `in_dim`. When `in_dim != out_dim`, this will produce an `IndexError`. When they happen to be equal (or when `in_dim < out_dim`), it silently computes the wrong values. In this exercise, the first layer maps 1→16, so `grad_output` has length 16 and `cached_input` has length 1 — the loop only runs `j=0`, which accesses `grad_output[0]`. This happens to be a valid index, so no error is raised, but the gradient is `grad_output[i] * grad_output[0]` instead of `grad_output[i] * input[0]`. The second layer maps 16→1, so `grad_output` has length 1 and `cached_input` has length 16 — here the loop runs `j` from 0 to 15, accessing `grad_output[0]` through `grad_output[15]`, which raises an `IndexError` for `j >= 1`.

3. **Observable symptoms**: Depending on the architecture dimensions, the network either crashes with an index error or silently trains with incorrect weight gradients. In the latter case, the loss plateaus because the weights are updated in a direction that is not the true gradient, so the network cannot fit the target function.

### The Fix

Replace the buggy line with:

```python
grad_W = [[grad_output[i] * cached_input[j] for j in range(in_dim)] for i in range(out_dim)]
```

After fixing, the loss should decrease steadily to < 0.01 over 1000 epochs, and predictions should closely match `sin(x)`.

---

## Challenge 2: Diverging Policy Gradient (`diverging_policy_gradient.py`)

### The Bug

In `compute_returns`, the discounted returns are computed correctly but are **not normalised** before being used as advantage weights in the policy gradient loss. The fix is to normalise them to zero mean and unit variance.

```python
# Buggy: returns used raw
returns = compute_returns(rewards, gamma)

# Fixed: normalise returns
returns = compute_returns(rewards, gamma)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### Why This Causes the Observed Behaviour

In CartPole-v1, the agent receives a reward of +1.0 for every timestep it survives. This means:

- A poor episode (pole falls after 20 steps) produces returns around 10–20.
- A good episode (survives 500 steps) produces returns around 200–500.

The policy gradient loss is `L = -Σ log π(a|s) · G_t`. When `G_t` ranges from 10 to 500, the gradient magnitude varies by 50×. This causes several problems:

1. **Gradient explosion**: When the agent has a good episode (high returns), the gradient is enormous. A single lucky episode can push the policy parameters far from their current values, overshooting the optimum and potentially making the policy degenerate.

2. **All actions reinforced equally**: When all returns are positive (as they always are in CartPole), every action has its probability increased. Good actions are reinforced slightly more than bad actions, but the signal-to-noise ratio is poor. Normalisation ensures roughly half the advantages are positive and half are negative, creating a clear signal about which actions were better than average.

3. **Effective learning rate instability**: The Adam optimiser adapts its step size based on gradient history. When gradient magnitudes swing wildly between episodes, Adam's moment estimates lag behind, and the effective learning rate becomes unstable. This manifests as reward spikes followed by crashes, or as NaN losses when gradients overflow.

### The Fix

Uncomment the normalisation line in the training loop:

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

This ensures:
- Returns have zero mean: roughly half the timesteps have positive advantage, half negative.
- Returns have unit variance: gradient magnitudes are consistent across episodes.
- The effective learning rate is stable regardless of episode length.

After fixing, the agent should solve CartPole (running reward > 195) within approximately 500–800 episodes.

### Connection to NeuroDrive

NeuroDrive performs advantage normalisation in `buffer.rs` after computing GAE — this is the same principle applied at a more sophisticated level. The concept file `advantage_estimation_gae.md` discusses normalisation in Phase 3 of the worked example and in Common Misconception #3.
