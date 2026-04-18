# Pass 1 Checkpoint — 2026-04-18

## Scope

Full repository. Code base is 67 Rust files, ~11,362 LoC. Rust 2024 edition, Bevy 0.18, custom handwritten PPO. Prior audit on 2026-04-15 closed 22/23 findings; this audit is a fresh pass against the upgraded skill (commit `6417142`) and therefore must still satisfy research + diagnostic-test floors even though many surface issues are already resolved.

## Systems identified

| System | Root | Files / notable LoC | Notes |
|---|---|---|---|
| PPO hot path | `src/brain/ppo/` + `src/brain/common/` | `ppo/update.rs` 533, `ppo/mod.rs` 467, `ppo/model.rs` 295, `ppo/buffer.rs` 256; `common/mlp.rs` 206, `common/math.rs` 83, `common/optim.rs` 90 | Highest-leverage hot path. Last audit flattened buffer, added batched forward, pre-allocated scratch, removed per-sample allocations in process_chunk. Residual per-car actor forward still allocates. |
| Environment & reward | `src/game/` + `src/maps/` | `episode.rs` 415, `centerline.rs` 498, `grid.rs` 477, `observation.rs` 361, `physics.rs` 131, `progress.rs` 62 | Centreline projection cached by hint; arc samples hardcoded; obs vector build uses `let mut values = [0.0; OBSERVATION_DIM]` local then copies — already zero-allocating via fixed array. |
| Analytics pipeline | `src/analytics/` | `trackers/episode.rs` 351, `metrics/turns.rs` 314, `metrics/consistency.rs` 242, `exporters/markdown.rs` 547, `models.rs` 266 | Called on episode end + on exit; cost is amortised. Markdown exporter likely dominates on-exit cost but is off hot path. |
| Profiling | `src/profiling/` | `exporters/markdown.rs` 1003, `exporters/json.rs` 221 | Feature-gated (zero cost when disabled). Markdown exporter is very large; one-shot on exit. |
| Debug HUD/overlays | `src/debug/` | `hud.rs` 642, `leaderboard.rs` 186, `overlays.rs` 183 | Runs every Update frame but is skipped when `visible=false` by default. |
| Agent interface | `src/agent/` | `observation.rs` 361, `action.rs` 131 | Observation rebuild per tick, raycasts. Already optimised with adaptive step. |

## Pass 2 prioritisation (highest value first)

1. **PPO hot path** — actor action selection allocates via `Linear::forward` single-sample path (3 Vecs/car + 3 tanh outputs/car × 8 cars = 48 allocations/tick in action selection alone); `forward_actor` called per car per tick; `std` recomputed via `.collect()` per car. `a_log_std_grad` correctness across chunks within an epoch. `compute_gae_per_env` allocates HashMap every prepare. Possible vectorisation/SIMD or matmul ordering wins.
2. **Observation hot path** — `build_observation_vector_system` builds a local `[f32; OBSERVATION_DIM]` and copies out; write-through to the component field might be marginally cheaper. Ray/lookahead loops.
3. **Centreline projection** — `build_polyline_points` uses `HashSet<(usize,usize)>` in `traverse_cells` for visit detection on a bounded grid (one-shot at startup; not a hot path).
4. **Analytics** — `compute_trace_aggregates` and `trackers/episode.rs` fold functions; per-episode, not per-tick.
5. **Exporters (markdown)** — on-exit, not budget-relevant; scan for accumulation patterns only.
6. **Dead/obsolete** — reference files for earlier iterations may contain stale claims; plan files already archived.

Systems consciously deferred as non-substantive (reasoned omissions to be recorded in the map):
- `src/sim/` (12 lines of utility code),
- `src/maps/parts/mod.rs` (tile semantic enums),
- tiny plugin glue files under 100 LoC without branching logic (e.g. `profiling/mod.rs`, `brain/plugin.rs`, `game/plugin.rs`).

## Test baseline

Command: `cargo test`
Result: **31 passed; 0 failed; 0 ignored** (compiled from cold). No flaky tests observed on this run. No `tests/` directory or `benches/` directory exists — tests live as `#[cfg(test)]` modules inside their source files. The audit will write new diagnostic tests as either inline `#[cfg(test)]` modules inside the file-under-test's crate (project convention) or, where the test is cross-module, inside an adjacent `tests/` module.

## Pre-existing Known Issues surfaced from context

- `systems/brain-ppo.md` flags: no save/load, no PPO integration tests, alignment checks are `debug_assert!` only (inactive in release). These become baseline "Known Issues" findings to propagate — not audit-introduced.
- `notes/session-2026-04-15.md` flags `unsafe` in `update.rs` (lines 162 and 291–292) for verification this session. I have read the unsafe blocks: they construct shared slices from pointers to pre-allocated scratch arrays (`obs_batch`, `grad_seed_values`, `grad_seed_means`) that are only written outside the `forward_batch` / `backward_batch` call sites. The soundness argument holds **if and only if** the backward/forward internal implementation does not read-then-write the same scratch field via `self`. Evidence required: I will audit the call graph in the Pass-2 deep dive and record the result as either a "safe" attestation or a Known Issues finding.

## Entry conditions satisfied

- Pre-Pass-1 WebSearch performed and recorded in `obligation-evidence-map.md`.
- Architecture, notes, key systems docs, README, and Cargo.toml read.
- Cargo test baseline captured.
- Prior audit folder archived to `archive-2026-04-15/`.

Proceeding to Pass 2.
