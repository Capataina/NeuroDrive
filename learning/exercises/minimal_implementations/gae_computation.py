"""
Minimal GAE (Generalised Advantage Estimation) implementation.

Given a fixed trajectory (rewards, values, dones), compute TD residuals,
GAE advantages, and returns using the recursive backward formula.

This exercise uses the exact worked example from the concept file:
  concepts/core/reinforcement_learning/advantage_estimation_gae.md

Pure Python, no external libraries.

Expected output:
  TD residuals: [0.092, 0.985, 0.091, -3.900, 0.493]
  GAE advantages: [-2.146, -2.379, -3.577, -3.900, 0.493]
  Returns: [-0.946, -1.579, -2.077, -3.000, 1.093]
  Normalised advantages: [0.101, -0.050, -0.822, -1.030, 1.803]
"""

import math


def compute_td_residuals(rewards, values, dones, next_value, gamma):
    """
    Compute one-step TD residuals: delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t).

    For the last timestep, V(s_{t+1}) is next_value (the bootstrap value).
    For earlier timesteps, V(s_{t+1}) is values[t+1].
    """
    T = len(rewards)
    deltas = [0.0] * T
    for t in range(T):
        if t + 1 < T:
            next_val = values[t + 1]
        else:
            next_val = next_value
        mask = 0.0 if dones[t] else 1.0
        deltas[t] = rewards[t] + gamma * mask * next_val - values[t]
    return deltas


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    Compute GAE advantages and returns via the recursive backward formula:
      GAE_t = delta_t + gamma * lambda * (1 - done_t) * GAE_{t+1}
      return_t = GAE_t + V(s_t)
    """
    T = len(rewards)
    deltas = compute_td_residuals(rewards, values, dones, next_value, gamma)

    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        gae = deltas[t] + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def normalise_advantages(advantages):
    """Normalise to zero mean, unit variance: A_norm = (A - mean) / (std + eps)."""
    n = len(advantages)
    mean = sum(advantages) / n
    variance = sum((a - mean) ** 2 for a in advantages) / n
    std = math.sqrt(variance + 1e-8)
    return [(a - mean) / std for a in advantages]


def round_list(lst, decimals=3):
    return [round(v, decimals) for v in lst]


if __name__ == "__main__":
    rewards = [0.5, 0.3, 0.7, -3.0, 0.4]
    values = [1.2, 0.8, 1.5, 0.9, 0.6]
    dones = [False, False, False, True, False]
    next_value = 0.7
    gamma = 0.99
    lam = 0.95

    deltas = compute_td_residuals(rewards, values, dones, next_value, gamma)
    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma, lam)
    normalised = normalise_advantages(advantages)

    print("TD residuals:          ", round_list(deltas))
    print("GAE advantages:        ", round_list(advantages))
    print("Returns:               ", round_list(returns))
    print("Normalised advantages: ", round_list(normalised))

    # --- Verification against hand-computed values from the concept file ---

    expected_deltas = [0.092, 0.985, 0.091, -3.900, 0.493]
    expected_advantages = [-2.146, -2.379, -3.577, -3.900, 0.493]
    expected_returns = [-0.946, -1.579, -2.077, -3.000, 1.093]
    expected_normalised = [0.101, -0.050, -0.822, -1.030, 1.803]

    def check(name, computed, expected, tol=0.01):
        for i, (c, e) in enumerate(zip(computed, expected)):
            assert abs(c - e) < tol, f"{name}[{i}]: expected {e}, got {c}"
        print(f"  ✓ {name} matches worked example")

    print("\nVerification:")
    check("TD residuals", deltas, expected_deltas)
    check("GAE advantages", advantages, expected_advantages)
    check("Returns", returns, expected_returns)
    check("Normalised advantages", normalised, expected_normalised)
    print("\nAll checks passed.")
