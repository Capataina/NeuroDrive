# Reward Design — Entertainment Constraint

## Core Principle

The simulation must be **entertaining to watch**. Cars should drive as aggressively and dangerously as possible while gradually learning to survive. This is the primary design constraint — it takes priority over convergence speed, sample efficiency, or clean reward engineering.

## What This Means in Practice

- **No crash penalties.** Episode termination is the cost of dying. The loss of future reward is the punishment — but only if the critic can predict it.
- **No survival bonuses.** A per-tick bonus for staying alive incentivises the policy to play safe, which produces boring behaviour.
- **No centreline reward large enough to dominate.** The centreline proximity reward (coef=0.3) exists as a gentle shaping signal, not a primary objective. If it dominates, cars learn to sit still on the centreline.
- **Velocity-projection reward is intentional.** It rewards speed along the track direction. This makes cars go fast, which is the entertaining part. The challenge is teaching them to go fast AND survive corners.

## Failure Modes We've Hit

| Change | What Happened | Lesson |
|--------|--------------|--------|
| Added crash penalty (-5) | Cars learned to stay still or brake constantly | Penalty outweighed reward for driving |
| Added centreline reward + crash penalty | Cars stopped moving entirely | "Do nothing on centreline" was optimal |
| Added braking (throttle [-1,1]) | Cars converged to "mostly brake" | Braking = safe local optimum |
| Velocity reward only, no crash penalty | Cars floor throttle and crash at first corner | Throttle exploration collapses before discovering cornering |

## How to Fix Learning Without Breaking Entertainment

When the policy isn't learning the right behaviour, fix it through:

1. **Critic capacity and accuracy** — if the critic can't distinguish "about to crash" from "driving safely", the advantage signal for crash-avoidance actions is too weak. Fix the critic, not the reward.
2. **Exploration mechanics** — if a dimension of the action space collapses (e.g., throttle std → 0.07), the policy can never discover better strategies. Prevent premature collapse through entropy bonuses, log-std floors, or wider initial std.
3. **Observation quality** — if the car doesn't have enough lookahead or the right features to anticipate corners, it can't learn to prepare for them. This was addressed with the 12-point lookahead expansion.

Never through reward penalties or bonuses that would make safe play optimal.

## Current Status

The velocity reward works — cars go fast (entertaining). The critic was widened (2x64 → 2x128), AdamW weight decay was added, and the log-std floor was raised from -2.0 to -1.0 to prevent throttle exploration collapse. Cars are confirmed to learn drifting corners. The remaining challenge is sustained corner survival at speed — the interplay between critic capacity, exploration, and the entertainment constraint (no crash penalties) remains the active research front within Milestone 1.
