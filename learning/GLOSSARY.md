# Glossary

## A2C

Advantage Actor-Critic. In NeuroDrive this is the current handwritten baseline learner used to test whether the environment and observation contract are learnable.

## Action Boundary

The stable controller-facing contract between a brain and the car. NeuroDrive exposes steering and throttle through `CarAction` and routes them via `ActionState`.

## Advantage

How much better or worse an action was than the critic expected. NeuroDrive computes this through GAE rather than raw one-step TD error alone.

## Analytics

The post-run tracking and export subsystem. It records episode summaries, tick traces, and A2C update snapshots into JSON and Markdown reports.

## Determinism

Repeatable behaviour under the same setup. NeuroDrive has strong determinism in fixed-step physics and scheduling, but weaker determinism in the current A2C RNG path.

## Eligibility Trace

A short-lived memory of recent synaptic participation in biological-style learning. This is a target concept in the README, not part of the current A2C baseline.

## Fixed Timestep

A simulation model where updates run at a constant rate. NeuroDrive uses a fixed 60 Hz step for physics, measurement, reward, and learning alignment.

## GAE

Generalised Advantage Estimation. A method for computing lower-variance advantage targets using `gamma` and `lambda`.

## Lookahead Features

Observation features derived from future centreline samples. NeuroDrive currently includes heading-delta and curvature features at several lookahead distances.

## Neuromodulation

A broadcast reward-like teaching signal in biological learning theories. The README frames reward this way conceptually, even though the live implementation currently uses A2C updates instead.

## Observation Leakage

Giving the learner information that solves the task too directly rather than forcing meaningful control learning. NeuroDrive deliberately avoids direct progress input while still using centreline-relative geometry.

## Rollout Buffer

The temporary storage of states, actions, rewards, dones, and values collected before an A2C update.

## SimSet

The named fixed-update ordering contract in `src/sim/sets.rs`. NeuroDrive uses `Input -> Physics -> Collision -> Measurement`.

## Structural Plasticity

Growth and pruning of neural connections over time. This is part of the long-term project target and not yet implemented in the runtime brain.
