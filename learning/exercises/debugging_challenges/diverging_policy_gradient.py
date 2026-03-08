"""
DEBUGGING CHALLENGE: Diverging Policy Gradient

This is a REINFORCE implementation for CartPole-v1 that diverges or
oscillates wildly instead of converging. The reward may spike briefly
then crash to ~10, or the loss may become NaN.

There is ONE bug causing the instability. Find and fix it.

Expected behaviour (when fixed):
  - Running reward should increase steadily.
  - Should solve CartPole (reward > 195) within ~500-800 episodes.

Actual behaviour (with bug):
  - Reward is erratic: spikes and crashes.
  - May produce NaN losses after a few hundred episodes.
  - Policy becomes degenerate (always picks the same action regardless of state).

Hint 1: Look at what happens to the gradient magnitude when returns have
         high variance. What is the scale of raw returns in CartPole
         (values can range from ~10 to ~500)?
Hint 2: Policy gradient algorithms almost always normalise advantages/returns
         to zero mean and unit variance before computing the loss. Is that
         happening here?

Requirements: pip install torch gymnasium
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns. BUG: returns are NOT normalised."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def train():
    env = gym.make("CartPole-v1")
    policy = PolicyNet(obs_dim=4, hidden_dim=128, act_dim=2)
    optimiser = optim.Adam(policy.parameters(), lr=1e-2)
    gamma = 0.99
    running_reward = 0.0

    for episode in range(1000):
        obs, _ = env.reset(seed=episode)
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        returns = compute_returns(rewards, gamma)

        # BUG: using raw returns without normalisation causes gradient explosion
        # when returns have high variance (e.g. ranging from ~10 to ~500).
        # Fix: normalise returns to zero mean, unit variance.
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for lp, G in zip(log_probs, returns):
            policy_loss.append(-lp * G)
        policy_loss = torch.stack(policy_loss).sum()

        optimiser.zero_grad()
        policy_loss.backward()
        optimiser.step()

        ep_reward = sum(rewards)
        running_reward = 0.05 * ep_reward + 0.95 * running_reward

        if episode % 50 == 0:
            print(f"Episode {episode:4d} | Reward: {ep_reward:6.1f} | Running: {running_reward:6.1f}")

        if running_reward > 195:
            print(f"\nSolved at episode {episode}! Running reward: {running_reward:.1f}")
            break

    if running_reward <= 195:
        print(f"\nFailed to solve. Final running reward: {running_reward:.1f}")

    env.close()


if __name__ == "__main__":
    train()
