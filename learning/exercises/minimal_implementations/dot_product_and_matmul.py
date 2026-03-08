"""
Minimal implementation: vector dot product and matrix-vector multiplication.

These are the two fundamental operations of a neural network forward pass.
A linear layer computes y = W @ x + b, which is a matrix-vector multiply
followed by a vector addition.

Implement both from scratch using only plain Python lists.

Expected output:
  dot_product([1, 2, 3], [4, 5, 6]) = 32
  mat_vec_mul([[1, 2], [3, 4], [5, 6]], [7, 8]) = [23, 53, 83]
  mat_vec_mul([[0.5, -0.3, 0.8], [-0.2, 0.7, 0.1]], [1.0, 2.0, 3.0]) = [2.3, 1.5]
"""


def dot_product(a, b):
    """Compute the dot product of two vectors of equal length."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def mat_vec_mul(W, x):
    """
    Multiply matrix W (list of row lists) by vector x.

    W has shape (out_dim, in_dim), x has shape (in_dim,).
    Returns a vector of shape (out_dim,).
    Each output element is the dot product of the corresponding row of W with x.
    """
    out_dim = len(W)
    in_dim = len(x)
    for row in W:
        assert len(row) == in_dim, f"Row length {len(row)} != vector length {in_dim}"

    result = []
    for i in range(out_dim):
        result.append(dot_product(W[i], x))
    return result


if __name__ == "__main__":
    print("=== Dot Product ===")
    r1 = dot_product([1, 2, 3], [4, 5, 6])
    print(f"dot_product([1, 2, 3], [4, 5, 6]) = {r1}")
    assert r1 == 32, f"Expected 32, got {r1}"

    print("\n=== Matrix-Vector Multiplication ===")
    r2 = mat_vec_mul([[1, 2], [3, 4], [5, 6]], [7, 8])
    print(f"mat_vec_mul([[1, 2], [3, 4], [5, 6]], [7, 8]) = {r2}")
    assert r2 == [23, 53, 83], f"Expected [23, 53, 83], got {r2}"

    r3 = mat_vec_mul(
        [[0.5, -0.3, 0.8], [-0.2, 0.7, 0.1]],
        [1.0, 2.0, 3.0],
    )
    print(f"mat_vec_mul([[0.5, -0.3, 0.8], [-0.2, 0.7, 0.1]], [1.0, 2.0, 3.0]) = {r3}")
    assert abs(r3[0] - 2.3) < 1e-9, f"Expected 2.3, got {r3[0]}"
    assert abs(r3[1] - 1.5) < 1e-9, f"Expected 1.5, got {r3[1]}"

    print("\nAll tests passed.")
