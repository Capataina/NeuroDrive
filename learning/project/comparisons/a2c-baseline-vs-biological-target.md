# A2C Baseline Vs Biological Target

## Why This Comparison Matters

NeuroDrive can be confusing if you only look at the current code or only read the README. The current runtime is an A2C-based validation harness, while the long-term project goal is biologically inspired local learning with eligibility traces, neuromodulation, and structural plasticity.

## What Stayed The Same

- The environment remains a deterministic 2D driving lab.
- The project still values watchable learning, observability, and from-scratch implementation in Rust.
- The controller boundary is still a fixed observation-to-action interface.

## What Changed

- The live learner is currently gradient-based A2C rather than local synaptic plasticity.
- The baseline already uses engineered geometry features and rich observability tooling.
- The project has become a validation-stage system rather than a pure environment prototype.

## Why The Project Preferred The Current Approach

The A2C baseline reduces ambiguity. If the task is not learnable under a competent baseline, it is hard to know whether later biological-learning failures come from the learning rule, the reward design, the observations, or the environment itself.

## What To Learn From The Older Or Future Approach

- The biological target explains the project's real research motivation.
- The A2C baseline explains the current engineering choices and validation discipline.
- You need both in mind to reason correctly about what code should be added next and what should stay modular.
