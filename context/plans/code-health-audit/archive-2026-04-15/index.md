# Code Health Audit

**Date:** 2026-04-15
**Scope:** Full repository
**Status:** complete

## Summary

Deep audit of the NeuroDrive codebase (11,149 lines of Rust across 50+ files). All 31 tests pass; zero compiler warnings. This is the second audit (the first on 30 March cleaned dead code, consolidated utilities, and renamed a2c to ppo). This audit goes deeper into algorithmic improvements, data layout wins, allocation elimination in hot paths, and subtle issues the first audit could not reach.

The highest-value findings are in the PPO hot path: the rollout buffer uses array-of-structs (`Vec<Vec<f32>>`) storage where flat struct-of-arrays would eliminate thousands of heap allocations per update; the `Linear::forward_batch` inner loop walks weights with a stride that defeats cache prefetching; and the `orthogonal_init` function produces `Vec<Vec<f32>>` only to immediately flatten it. The observation system creates a fresh `Normal` distribution object on every sample call (8 cars x 2 actions x every tick). The centreline projection runs a full linear scan of all segments for every car every tick when a cached hint would reduce it to 1-2 segment checks.

## Findings Overview

| File | System | Critical | High | Medium | Low | Total |
|------|--------|----------|------|--------|-----|-------|
| [brain-ppo.md](brain-ppo.md) | PPO hot path (model, buffer, update, common) | 0 | 6 | 3 | 1 | 10 |
| [environment.md](environment.md) | Game (physics, collision, episode, progress) + maps (centreline) | 0 | 2 | 2 | 1 | 5 |
| [agent.md](agent.md) | Agent (observation, action) | 0 | 1 | 1 | 0 | 2 |
| [analytics.md](analytics.md) | Analytics pipeline | 0 | 0 | 2 | 2 | 4 |
| [cross-cutting.md](cross-cutting.md) | Project-wide | 0 | 0 | 1 | 1 | 2 |
| **Total** | | **0** | **9** | **9** | **5** | **23** |

## Priority Actions

1. **[HIGH]** Flatten rollout buffer from AoS (`Vec<Vec<f32>>`) to SoA flat `Vec<f32>` storage -- [brain-ppo.md#flatten-rollout-buffer-to-soa](#flatten-rollout-buffer-to-soa)
2. **[HIGH]** Transpose `Linear::forward_batch` weight access to column-major for sequential cache reads -- [brain-ppo.md#transpose-weight-access-in-forward-batch](#transpose-weight-access-in-forward-batch)
3. **[HIGH]** Eliminate per-sample `Vec` allocations in `ppo_act_all_cars_system` -- [brain-ppo.md#eliminate-per-car-vec-allocations-in-act-system](#eliminate-per-car-vec-allocations-in-act-system)
4. **[HIGH]** Cache `Normal` distribution in `sample_normal` instead of constructing per call -- [brain-ppo.md#cache-normal-distribution-construction](#cache-normal-distribution-construction)
5. **[HIGH]** Eliminate gradient-seed clone allocations in `ppo_process_chunk` -- [brain-ppo.md#eliminate-gradient-seed-clone-allocations](#eliminate-gradient-seed-clone-allocations)
6. **[HIGH]** Eliminate obs-batch allocation in `ppo_process_chunk` by adding a scratch field -- [brain-ppo.md#eliminate-obs-batch-allocation-in-process-chunk](#eliminate-obs-batch-allocation-in-process-chunk)
7. **[HIGH]** Add cached-hint centreline projection to eliminate per-tick linear scan -- [environment.md#cached-hint-centreline-projection](#cached-hint-centreline-projection)
8. **[HIGH]** Batch all 8 critic forward passes in `ppo_act_all_cars_system` -- [brain-ppo.md#batch-critic-forward-passes-during-action-selection](#batch-critic-forward-passes-during-action-selection)
9. **[HIGH]** Raycast step size increase from 3.0 to adaptive for long-range rays -- [agent.md#adaptive-raycast-step-size](#adaptive-raycast-step-size)
10. **[MEDIUM]** Make `orthogonal_init` produce flat `Vec<f32>` directly -- [brain-ppo.md#flatten-orthogonal-init-output](#flatten-orthogonal-init-output)

## By Category

- Algorithm Optimisation: 3 findings
- Data Layout and Memory Access Patterns: 4 findings
- Performance Improvement: 8 findings
- Dead Code Removal: 1 finding
- Inconsistent Patterns: 2 findings
- Documentation Rot: 1 finding
- Complexity Hotspots: 1 finding
- Known Issues and Active Risks: 1 finding
- Triage Needed: 1 finding
- Configuration Drift: 1 finding
