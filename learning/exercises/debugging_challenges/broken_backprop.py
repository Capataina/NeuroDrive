"""
DEBUGGING CHALLENGE: Broken Backpropagation

This is a 2-layer MLP that should learn to approximate sin(x), but it
does not converge. The loss plateaus or oscillates instead of decreasing.

There is ONE subtle bug in the backward pass. Find and fix it.

Expected behaviour (when fixed):
  - Loss should decrease steadily from ~0.5 to < 0.01 over 1000 epochs.
  - Predictions should closely match sin(x) at test points.

Actual behaviour (with bug):
  - Loss may decrease initially but plateaus around ~0.2-0.4 or oscillates.
  - Predictions are inaccurate, especially away from x = 0.

Hint 1: The bug is in linear_backward. Compare the weight gradient formula
         to the mathematical derivation: dL/dW[i][j] = grad_output[i] * input[j].
Hint 2: What variable is being used where input[j] should be?
"""

import math
import random

random.seed(42)


def glorot_init(fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [[random.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]


def zeros(n):
    return [0.0] * n


def linear_forward(W, b, x):
    out = []
    for i in range(len(W)):
        s = b[i]
        for j in range(len(x)):
            s += W[i][j] * x[j]
        out.append(s)
    return out, x


def relu_forward(x):
    out = [max(0.0, v) for v in x]
    return out, x


def linear_backward(W, grad_output, cached_input):
    in_dim = len(cached_input)
    out_dim = len(grad_output)

    grad_input = [0.0] * in_dim
    for j in range(in_dim):
        for i in range(out_dim):
            grad_input[j] += W[i][j] * grad_output[i]

    # BUG IS HERE: weight gradients should use cached_input, not grad_output
    grad_W = [[grad_output[i] * grad_output[j] for j in range(in_dim)] for i in range(out_dim)]

    grad_b = [grad_output[i] for i in range(out_dim)]
    return grad_input, grad_W, grad_b


def relu_backward(grad_output, cached_input):
    return [grad_output[i] if cached_input[i] > 0 else 0.0 for i in range(len(cached_input))]


def mse_loss(predicted, target):
    n = len(predicted)
    loss = sum((predicted[i] - target[i]) ** 2 for i in range(n)) / n
    grad = [2.0 * (predicted[i] - target[i]) / n for i in range(n)]
    return loss, grad


def sgd_update(param, grad, lr):
    if isinstance(param[0], list):
        for i in range(len(param)):
            for j in range(len(param[i])):
                param[i][j] -= lr * grad[i][j]
    else:
        for i in range(len(param)):
            param[i] -= lr * grad[i]


HIDDEN = 16
W1 = glorot_init(1, HIDDEN)
b1 = zeros(HIDDEN)
W2 = glorot_init(HIDDEN, 1)
b2 = zeros(1)

N_SAMPLES = 64
xs = [random.uniform(-math.pi, math.pi) for _ in range(N_SAMPLES)]
ys = [math.sin(x) for x in xs]

LR = 0.01
EPOCHS = 1000

for epoch in range(EPOCHS):
    total_loss = 0.0
    for i in range(N_SAMPLES):
        inp = [xs[i]]
        target = [ys[i]]

        h1, cache_l1 = linear_forward(W1, b1, inp)
        a1, cache_r1 = relu_forward(h1)
        out, cache_l2 = linear_forward(W2, b2, a1)

        loss, grad_loss = mse_loss(out, target)
        total_loss += loss

        grad_a1, gW2, gb2 = linear_backward(W2, grad_loss, cache_l2)
        grad_h1 = relu_backward(grad_a1, cache_r1)
        _, gW1, gb1 = linear_backward(W1, grad_h1, cache_l1)

        sgd_update(W2, gW2, LR)
        sgd_update(b2, gb2, LR)
        sgd_update(W1, gW1, LR)
        sgd_update(b1, gb1, LR)

    avg_loss = total_loss / N_SAMPLES
    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

print("\nFinal predictions:")
test_points = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]
for x in test_points:
    h1, _ = linear_forward(W1, b1, [x])
    a1, _ = relu_forward(h1)
    out, _ = linear_forward(W2, b2, a1)
    print(f"  x = {x:6.2f}  predicted: {out[0]:7.4f}  actual: {math.sin(x):7.4f}")
