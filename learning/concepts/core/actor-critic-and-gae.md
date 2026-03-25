# Actor-Critic And GAE

## Why This Matters Here

The current autonomous learner in NeuroDrive is a handwritten A2C baseline. If you do not understand actor-critic and GAE, the live learning path looks like arbitrary maths instead of a coherent baseline-validation system.

## Core Idea

Actor-critic splits the learning problem into two jobs:

- the actor chooses actions,
- the critic estimates how good the state is.

The actor improves when it learns that certain actions were better than expected. The critic helps define “better than expected” so the actor does not learn from raw reward noise alone.

GAE, or Generalised Advantage Estimation, is a practical way to compute that “better than expected” signal with a controllable bias-variance trade-off.

## Build-Up

### Step 1: Raw reward is too noisy

One crash or one good straight can swing reward sharply. If you update the actor from raw returns alone, learning becomes brittle.

### Step 2: Add a critic

The critic estimates state value. That lets the learner compare what happened against what was expected.

### Step 3: Estimate advantage

Advantage asks: “Was this action better or worse than the critic predicted?”

### Step 4: Smooth the estimate with GAE

GAE mixes multi-step TD errors using `gamma` and `lambda` so the signal is less noisy than one-step TD and less blunt than full Monte Carlo returns.

## Worked Examples

### Example 1: Straight-line progress

If the car makes smooth forward progress and the critic expected less, the advantage is positive and the actor should move slightly towards similar actions.

### Example 2: Crash after risky steering

If the car crashes and the critic expected a better outcome, the advantage is negative and the actor should move away from the action pattern that led there.

### Example 3: Why GAE matters

If a turn goes wrong over several ticks, one-step TD can be too local. GAE lets the training signal spread across the sequence more usefully.

## How This Appears In The Project

- `src/brain/a2c/mod.rs` owns the runtime act path, reward collection, and update triggering.
- `src/brain/a2c/buffer.rs` stores rollout data.
- `src/brain/a2c/update.rs` computes returns, advantages, and optimisation steps.

## Common Misunderstandings

❌ “Actor-critic means backprop is gone.”
✅ It still uses gradient-based optimisation. It is just a different RL structure from value-only learning.

❌ “GAE is the algorithm.”
✅ GAE is one part of the training target computation inside the wider actor-critic setup.

❌ “If losses look better, driving must be better.”
✅ Not in this project. Behavioural diagnostics still matter more than scalar optimisation health alone.

## Terms Used Here

- actor
- critic
- advantage
- return
- GAE
- entropy regularisation
