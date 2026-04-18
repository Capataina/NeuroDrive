# NeuroDrive

## Project Description

**NeuroDrive** is a real-time, brain-inspired AI research project built around a custom 2D top-down racing environment.
The goal is _not_ to benchmark standard algorithms, chase leaderboard scores, or outsource learning to external ML frameworks.

Instead, NeuroDrive is a focused attempt to answer one question:

> **Can we build a learning system from scratch that mimics how the human brain learns, and watch it gradually acquire driving behaviour in real time?**

The project is written entirely in **Rust**, using **Bevy** for simulation and rendering.
All learning logic, plasticity rules, and structural adaptation mechanisms are implemented **from first principles** — no PyTorch, no TensorFlow, no external ML libraries.

---

## What Does the Human Brain Actually Do When It Learns?

### In Simple Terms

The human brain is a massive, sparsely connected graph of neurons.
Neurons communicate via synapses whose strengths change as a function of experience.

Learning happens when:

- **Co-activation strengthens connections** (useful correlations get reinforced).
- **Unhelpful connections weaken** (unused patterns fade).
- **Global reward signals modulate plasticity** (dopamine-like signals reinforce what led to better outcomes).
- **Over longer timescales, structure adapts** (connections can form, reorganise, or be pruned).

The brain does **not**:

- Run backpropagation.
- Compute global gradients.
- Train against a single static dataset.
- Reset itself after each failure.

Instead, it:

- Updates connections **locally** using only information available at the synapse.
- Uses global neuromodulation to **gate** which changes become lasting.
- Continually adapts while acting in the world.
- Slowly reshapes its structure through experience-driven plasticity.

**Hence, the brain is neither a typical reinforcement learning system nor an evolutionary algorithm; rather, it learns through ongoing, local adaptation of its own structure and connections, guided but not dictated by rewards, enabling continuous and flexible learning from experience.**

---

### In Scientific Terms (High Signal, Minimal Jargon)

Biological learning is believed to involve a few key mechanisms that compose together:

- **Hebbian plasticity**
  Synapses strengthen when presynaptic and postsynaptic activity are correlated ("fire together, wire together").

- **Spike-Timing Dependent Plasticity (STDP)**
  The _timing_ of spikes matters: pre-before-post tends to strengthen; post-before-pre tends to weaken.

- **Eligibility traces**
  Synapses maintain a short-lived "memory" of recent correlation, allowing reinforcement to arrive later.

- **Neuromodulation (dopamine-like signals)**
  A broadcast signal (reward prediction error) gates consolidation: _which changes should stick_.

- **Structural plasticity**
  Over longer timescales, synapses form/prune and circuits reorganise to allocate capacity where it matters.

Learning is therefore:

- **Local** (credit assignment is done using synapse-local signals, not global gradients)
- **Incremental** (continuous updates rather than rare re-training)
- **Dynamical** (neurons have internal state; behaviour depends on time)
- **Continual** (weights evolve during interaction, not just between episodes)

---

## Core Project Goal

NeuroDrive aims to replicate these principles in an engineered system:

- A **sparse neural graph** with neuron state and synapses
- **Local plasticity** rules (Hebbian / STDP-family)
- **Eligibility traces** for delayed credit assignment
- **Neuromodulation** (dopamine-like reward prediction errors)
- **Structural plasticity** (growth + pruning under constraints)
- **Continuous online learning** across episodes ("one brain, one lifetime")

We do **not** use:

- Genetic Algorithms / NEAT
- Evolution Strategies
- TensorFlow / PyTorch / JAX
- Backpropagation-based training loops

This is not evolution across generations.
This is **one persistent "brain"** learning within its lifetime.

---

## Environment Overview

The environment is intentionally minimal yet non-trivial:

- **Deterministic 60 Hz fixed-timestep** 2D top-down car physics
- **Steering** `[-1, 1]` + **throttle** `[0, 1]` control (coast to full thrust — no braking, drag is the sole deceleration mechanism)
- **Track boundaries** + corner-based collision detection
- **Cumulative forward progress** measured as arc-length along the centreline from spawn
- **Random spawn positions** — all cars spawn at random centreline positions, re-randomised on each episode reset
- **Episode boundaries**: crash or 30-second timeout only — there is no finish line, no lap concept

The car must learn to:

- Stay on track
- Maximise forward progress along the centreline
- Drive as fast as possible without crashing
- Survive corners at speed

The environment is designed to provide **dense, interpretable learning signals** without turning the task into scripted control.

### Design Decisions

Several environment design decisions were made through experimentation and are documented here because they are non-obvious:

| Decision | Why | What We Tried First |
|----------|-----|---------------------|
| **No braking** (throttle `[0, 1]`) | Braking creates a safe local optimum — the policy converges to "mostly brake" every time | `[-1, 1]` throttle with `brake_force = 400`; policy mean converged to -0.60 |
| **No finish line or laps** | With random spawns, a finish line creates perverse incentives (cars spawned near the line get easy completion bonuses) | Lap detection + lap completion bonus; removed entirely |
| **Random spawn positions** | Fixed spawn creates a privileged starting experience; random spawn forces generalisation across all track sections | Car 0 at canonical start, ghost cars random; now all cars fully random |
| **Crash penalty = 0** | Any crash penalty incentivises not moving; episode termination is already the cost of dying | Crash penalty of -5; cars learned to stay still or brake constantly |
| **No survival bonus** | A per-tick bonus for staying alive incentivises the policy to play safe, producing boring behaviour | Considered but rejected based on reward philosophy |
| **`rotation_speed = 8.0`** | The car needs to be physically capable of turning at speed; 4.0 was insufficient for tight corners | `rotation_speed = 4.0`; max turn rate was 3.8 degrees/tick, insufficient for U-turns |

---

## Reward Philosophy

Reward in NeuroDrive is treated as a **neuromodulatory teaching signal**, not a fitness score.

The primary design constraint is **entertainment**: the simulation must be entertaining to watch. Cars should drive as aggressively and dangerously as possible while gradually learning to survive. This takes priority over convergence speed, sample efficiency, or clean reward engineering.

### Current Reward Structure

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Velocity projection** | `dot(velocity, centreline_tangent) / speed_reference * velocity_reward_scale` | Rewards speed along the track direction — makes cars go fast |
| **Centreline proximity** | `centreline_reward_coef * (1 - (dist / max_dist)^2)` | Gentle shaping signal to keep cars near the racing line |
| **Crash penalty** | `0.0` | Episode termination is the cost; no explicit penalty |
| **Survival bonus** | None | Would incentivise safe, boring play |

### What Does Not Work (And Why)

When the policy is not learning the right behaviour, the fix is **never** reward penalties or bonuses that would make safe play optimal. Instead:

1. **Fix the critic** — if the critic cannot distinguish "about to crash" from "driving safely", the advantage signal for crash-avoidance actions is too weak.
2. **Fix exploration** — if an action dimension collapses (e.g., throttle std approaches zero), the policy can never discover better strategies. Prevent premature collapse through entropy bonuses, log-std floors, or wider initial distributions.
3. **Fix observations** — if the car does not have enough lookahead or the right features to anticipate corners, it cannot learn to prepare for them.

> In biology, reward signals guide plasticity but do not dictate behaviour directly.
> NeuroDrive uses reward to gate learning, not to define a brittle objective function.

---

## Current Implementation State

NeuroDrive is in a **transitional architecture state**. The long-term goal is brain-inspired local plasticity (Milestones 2–9). The current implementation is a handwritten **PPO baseline** used to validate that the environment, observation space, and reward structure are learnable before transitioning to biological learning rules.

### What Is Live Today

```
Environment (Milestone 0)          ████████████████████ Complete
PPO Baseline (Milestone 1)         ████████████████░░░░ ~90%
Brain-Inspired Learning (M2+)      ░░░░░░░░░░░░░░░░░░░░ Not started
```

The PPO baseline is not a toy. It is a substantial, optimised, from-scratch implementation:

| Component | Details |
|-----------|---------|
| **Algorithm** | PPO with clipped surrogate objective (epsilon = 0.2), 4 epochs per update |
| **Architecture** | Asymmetric actor-critic — actor 2x64, critic 2x128, tanh activations |
| **Initialisation** | Orthogonal (sqrt(2) hidden, 0.01x policy head, 1.0x value head) |
| **Optimiser** | Actor: Adam (LR 3e-4). Critic: AdamW with weight decay lambda = 3e-4 (LR 5e-4) |
| **Exploration** | Log-std floored at -1.0 (minimum sigma ~0.37), per-minibatch advantage normalisation, Fisher-Yates sample shuffling |
| **Training** | Multi-car vectorised: 8 cars, shared rollout buffer with env_id tagging, per-env GAE (no cross-env value leakage) |
| **Performance** | Amortised updates across ticks (64 samples/tick to avoid frame stutter), batched forward/backward passes, pre-allocated scratch buffers, flat `Vec<f32>` weight storage for cache-friendly traversal |
| **Observations** | 43 dimensions (see below) |
| **Actions** | Steering `[-1, 1]` via full tanh, throttle `[0, 1]` via `0.5*(tanh+1)` remapping |

### Observation Space (43 Dimensions)

```
Rays (11)
├── 11 normalised raycast distances

Kinematics (3)
├── v_forward      car-local forward velocity component
├── v_lateral      car-local lateral velocity component
└── speed_delta    frame-over-frame acceleration signal

Centreline (3)
├── offset         signed lateral distance from centreline
├── heading        heading error relative to centreline tangent
└── curvature      local centreline curvature

Lookahead (24)
├── 12 heading deltas    (upcoming heading changes at 30–650 units)
└── 12 curvatures        (upcoming curvature at 30–650 units)
    Spacing: dense near (~30 unit gaps) for steering, sparser far (~80 unit gaps) for anticipation
    650 units = ~2.17s warning at terminal velocity — enough to coast down through drag alone

Previous Actions (2)
├── previous_steering
└── previous_throttle
```

The observation space evolved significantly through experimentation:

| Version | Dims | What Changed | Why |
|---------|------|-------------|-----|
| Initial | ~15 | Basic rays + speed + heading | Starting point |
| + velocity decomposition | 23 | Replaced scalar speed with v_forward/v_lateral, added speed_delta | Car needs to know if it is sliding laterally vs moving forward |
| + previous actions | 25 | Added previous_steering, previous_throttle | One-step action memory helps policy learn momentum-aware control |
| + expanded lookahead | 43 | 4 samples (260 units) to 12 samples (650 units) | 4 points could not distinguish turn shapes (L vs C vs U vs S bends produce ambiguous patterns); 12 give an unambiguous sketch of road geometry |

### Multi-Car Vectorised Training

The runtime is not a single-car simulation. It is a **multi-car vectorised trainer**:

- **8 cars** run simultaneously (configurable via `TrainerConfig`)
- All cars spawn at **random centreline positions**, re-randomised on each episode reset
- Each car has its own colour from a 25-colour palette
- Per-car components: `EnvInstanceId`, `CarColour`, `ActionState`, `EpisodeState`, `EpisodeMovingAverages`, `PolicyOutput`
- One shared `TrainerRolloutBuffer` collects transitions from all cars with `env_id` tagging
- GAE is computed per-env (no cross-env value leakage)
- A `TrainerLiveRanking` resource tracks best/worst car with hysteresis
- A live leaderboard panel shows per-car performance with colour swatches

Running 8 cars produces ~2.5x more episodes per unit time than 3 cars, significantly accelerating learning.

---

## Brain Architecture

### Current: PPO Baseline

The current brain is a handwritten PPO implementation used for environment validation. It is intentionally gradient-based — the goal is to prove learnability before replacing it with biological learning rules.

```
Observation (43 dims) ──► Actor MLP (2x64, tanh) ──► Action means + log-stds ──► Gaussian sample ──► Action
                     └──► Critic MLP (2x128, tanh) ──► Value estimate ──► GAE advantage ──► PPO update
```

The actor and critic are **asymmetric** — the critic is wider because value estimation is harder than action selection in this domain. The critic needs to distinguish "about to crash at a corner" from "driving safely on a straight" using the same observations, which requires more representational capacity.

Key design choices:
- **Tanh activations** throughout (ReLU caused 34–57% dead neurons — permanently zero units that never recovered)
- **Orthogonal initialisation** preserves gradient norms at init, preventing early training instability
- **AdamW on the critic only** — weight decay prevents unbounded weight growth in the wider network
- **Log-std floor at -1.0** — prevents exploration collapse (discovered when throttle std dropped to 0.07, locking the policy at full throttle with no ability to discover deceleration)
- **Amortised PPO updates** — processing 64 samples per tick across multiple ticks avoids frame stutter during training

### Future: Brain-Inspired Local Plasticity (Milestone 2+)

The intended long-term architecture replaces the PPO MLP with:

- **Fixed input neurons** (sensor interface — the 43-dim observation)
- **Fixed output neurons** (motor interface — steering + throttle)
- A **dynamic sparse hidden graph**
- Local synapses with **eligibility traces**
- A global **neuromodulatory signal** (delta)

```
Observation ──► Brain ──► Action
```

External boundary remains stable. Internal topology may change over time.

> A brain can reorganise internally while still receiving sensory input and emitting motor commands.
> NeuroDrive mirrors this: I/O is fixed; internal structure is plastic.

### Learning Mechanism (Future)

**Local Plasticity + Eligibility:**

Each synapse maintains:

- Weight `w_ij`
- Eligibility trace `e_ij`

Eligibility accumulates "recent usefulness" locally:

```
e_ij <- lambda * e_ij + f(pre_i, post_j)
```

**Neuromodulation (Dopamine-like Teaching Signal):**

```
delta = r + gamma * V(s') - V(s)
delta_w_ij = eta * delta * e_ij
```

- `e_ij` says "this synapse participated recently"
- `delta` says "that participation led to better/worse outcomes than expected"
- Weight change is a gated consolidation mechanism

No gradients. No global loss. No backprop.

**Structural Plasticity (Topology Updates):**

- **Pruning**: remove synapses with persistently low magnitude and low eligibility contribution
- **Growth**: add synapses between recently co-active neurons when capacity is available
- **Constraints**: enforce bounded fan-in / fan-out to prevent graph blow-up

---

## Observability and Telemetry

NeuroDrive includes comprehensive observability because "looks like learning" is not evidence.

### Live Runtime

| Feature | Toggle | Description |
|---------|--------|-------------|
| **Geometry overlays** | F1 | Centreline, tangent vectors, forward vectors, velocity vectors |
| **Sensor overlays** | F2 | Raycast segments, hit points |
| **Diagnostics HUD** | F3 | Episode counter, progress metrics, moving averages, reward decomposition, PPO health (clip %, KL divergence), quarter summaries, run assessment |
| **Live leaderboard** | F3 | Per-car performance ranking with colour swatches, best/worst highlighting |
| **Agent mode toggle** | F4 | Switch between AI and keyboard control (clears rollout buffer on switch) |

All overlays default to off for clean viewing.

### Analytics Pipeline

A comprehensive post-run analytics system captures everything needed to diagnose learning:

- **16 tick-level trace fields**: position, velocity decomposition, drift angle, minimum ray distance, velocity projection, centreline reward, policy confidence (value prediction, action means/stds)
- **25 episode-level aggregates**: speed statistics, action distributions, crash forensics, value function diagnostics, exploration metrics
- **Crash classification system**: 5 crash types (Slide, HeadOn, Overshoot, Spin, Stall) diagnosed from terminal state kinematics
- **10-section Markdown report** with sparklines, heatmaps, sector breakdowns, and auto-generated takeaways
- **Two-tier JSON export**: compact (always) + full trace (opt-in)
- **Retention-limited cleanup**: auto-deletes oldest reports to prevent unbounded growth

### Profiling System

Feature-gated behind `--features profiling` (zero runtime cost when disabled):

- Per-system timing for all 17 FixedUpdate systems
- Per-SimSet breakdown (Input, Physics, Collision, Measurement)
- Auto-exit after configurable duration (default 30 seconds)
- Rich Markdown report with interpretation, stutter analysis, and recommendations
- JSON export with run context snapshot

### Planned (Later-Stage) Telemetry

- Dopamine delta visualisation (raw + smoothed)
- Weight statistics (mean |w|, histogram bins, clamp hits)
- Graph statistics (synapse count, sparsity, churn rate)
- Optional live graph view (nodes/edges)
- Optional live weight view (matrix/synapse list)

Learning must be measurable, not guessed.

---

## Development Constraints

NeuroDrive is developed on constrained hardware:

| Component | Detail |
|-----------|--------|
| **Machine** | MacBook Air M2 (2022) |
| **Memory** | 8 GB unified (shared CPU/GPU) |
| **Architecture** | ARM64 (Apple Silicon — NEON SIMD, not SSE/AVX) |
| **Display** | 60 Hz |

This means:
- No CUDA, no discrete GPU — all computation is CPU-bound
- The 16.67ms frame budget at 60 Hz is a hard constraint
- Memory-intensive work (rollout buffers, trace captures) competes with rendering
- Performance optimisation is not optional — it is a core engineering discipline

### Performance Journey

The PPO implementation went through significant optimisation to run 8 cars within the frame budget:

| Change | Impact |
|--------|--------|
| `Vec<Vec<f32>>` to flat `Vec<f32>` weight storage | Eliminated catastrophic cache misses (~43x theoretical improvement) |
| Pre-allocated scratch buffers | Zero heap allocations in the training loop |
| Batched forward/backward passes | Mat-mat instead of 128x mat-vec |
| Iterator-based inner loops | Enabled LLVM auto-vectorisation |
| Swap instead of clone for frozen rollout buffer | Eliminated full-buffer copy |
| Amortised PPO updates (64 samples/tick) | Spread training cost across ticks to avoid frame stutter |

Result: **426 stutters to 2**, mean frame time **17.3ms to 9.0ms** with 8 cars.

---

## Building and Running

NeuroDrive is a standard Cargo project. The only prerequisite is a recent Rust toolchain (edition 2024, tested on stable). On macOS the Apple Accelerate framework is used automatically — it ships with the OS, no separate install. On other platforms a portable pure-Rust backend is used automatically instead.

### Everyday commands

| Command | What it does |
|---------|--------------|
| `cargo run --release` | Start the simulation with all optimisations enabled. Release mode is **strongly recommended** — `cargo run` alone (debug mode) is ~10× slower and misses the actual performance story. |
| `cargo run` | Fast compile, slow runtime. Useful only when iterating on code changes and you don't care about frame rate. |
| `cargo test` | Run the full test suite (99 tests as of 2026-04-18). |
| `cargo test --release` | Tests in release mode — runs more slowly to compile but matches the optimiser flags of production code. |
| `cargo check` | Fast syntax/type check without producing a binary. |
| `cargo check --release` | Same but with release optimisations active (catches some LTO-specific issues). |

### GEMM backend selection

The PPO hot path (actor + critic forward and backward) spends most of its time in small single-precision matrix multiplications. NeuroDrive provides three interchangeable backends for that single operation, with one chosen automatically per platform:

| Command | Backend used | Notes |
|---------|--------------|-------|
| `cargo run --release` | **macOS:** Apple Accelerate (cblas_sgemm, AMX-accelerated). **Elsewhere:** `matrixmultiply` (pure Rust, NEON on ARM64). | The default. Picks the fastest available backend for your platform without any flags. |
| `cargo run --release --no-default-features --features force-accelerate` | Apple Accelerate, forced. **macOS only** — fails to build on other platforms. | Explicit opt-in; same as the macOS default. |
| `cargo run --release --no-default-features --features force-matrixmultiply` | `matrixmultiply` crate, forced on any platform. | Useful to A/B against Accelerate on macOS, or as the natural default on Linux/Windows. |
| `cargo run --release --no-default-features --features force-scalar` | Naive nested-loop Rust. Slowest by design. | Used as a **correctness reference** for the other two backends and as a fallback when neither is desired. |

Every performance report (see below) records which backend was active under its new `### Build` section, so benchmarks across different runs are directly comparable.

### Profiling

```bash
cargo run --release --features profiling
```

Enables the per-system frame-timing instrumentation. The app auto-exits after 30 seconds and writes two artefacts:

- `reports/performance/perf_<timestamp>.md` — Markdown report with per-system breakdown, stutter analysis, and auto-generated recommendations.
- `reports/json/performance/perf_<timestamp>.json` — raw timing data for custom post-processing.

The Markdown report's Run Context section includes the active GEMM backend, so you can tell at a glance whether a given profile was produced by Accelerate, matrixmultiply, or scalar.

### Benchmarking different backends

To compare backends on the same workload:

```bash
cargo run --release --features profiling
# → reports/performance/perf_A.md   (Accelerate on macOS by default)

cargo run --release --no-default-features --features "force-matrixmultiply,profiling"
# → reports/performance/perf_B.md   (matrixmultiply forced)

cargo run --release --no-default-features --features "force-scalar,profiling"
# → reports/performance/perf_C.md   (scalar reference)
```

Each report's `### Build` section records the backend; the frame-time and PPO Epoch timing tables can be compared directly.

### Test suite

```bash
cargo test                                                             # full suite, default backend
cargo test --no-default-features --features force-scalar              # scalar backend
cargo test --no-default-features --features force-matrixmultiply      # matrixmultiply backend
cargo test --no-default-features --features force-accelerate          # Accelerate backend (macOS only)
```

The suite includes cross-backend correctness tests in `tests/gemm_correctness.rs` that validate whichever backend is compiled in against an inline scalar reference for every matrix shape PPO actually uses.

### Feature flag reference

```toml
# Defined in Cargo.toml [features]
default = []
profiling             # Enable per-system timing instrumentation + auto-exit
force-scalar          # GEMM backend override — naive nested-loop reference
force-matrixmultiply  # GEMM backend override — portable pure-Rust BLIS kernel
force-accelerate      # GEMM backend override — Apple Accelerate (macOS only)
```

Constraints:

- At most one `force-*` backend flag may be enabled at a time (compile-time error if two or three are set together).
- `force-accelerate` on non-macOS platforms is a compile-time error (the Accelerate framework does not exist there).
- `profiling` is orthogonal to the backend flags — any combination is valid.

### Verified build matrix (2026-04-18)

All of the following pass `cargo test` with zero warnings and all 99 tests green:

- Default (Accelerate on this macOS M2 host)
- `--no-default-features --features force-scalar`
- `--no-default-features --features force-matrixmultiply`
- `--no-default-features --features force-accelerate`
- `--release`
- `--features profiling`

---

## Features and Roadmap

NeuroDrive follows a deliberate sequencing strategy:

1. Build a deterministic, observable environment.
2. Prove the task is learnable with a lightweight RL baseline.
3. Transition to brain-inspired local plasticity mechanisms.
4. Gradually increase biological fidelity and structural complexity.

This reduces debugging ambiguity and isolates representation issues from learning-rule issues.

---

## Milestone 0 — Environment Foundation (Complete)

This milestone established a fully deterministic, instrumented control environment before any learning algorithm was introduced.

- [x] Deterministic fixed-timestep 2D car physics (60 Hz)
- [x] Track representation (centreline polyline + boundaries)
- [x] Collision detection (corner-based off-road detection) + reset conditions
- [x] Progress metric via centreline projection (cumulative arc-length from spawn)
- [x] Raycast sensor system with on-screen debug overlays
- [x] Stable observation vector (normalised inputs, 43 dimensions)
- [x] Steering/throttle action interface with optional smoothing
- [x] Episode loop (crash / 30-second timeout)
- [x] Telemetry: reward, progress, crash count, moving averages
- [x] Debug visual overlays:
  - [x] Raycasts + hit points (F2)
  - [x] Closest centreline projection point (F1)
  - [x] Centreline tangent vector visualisation (F1)
  - [x] Car forward vector and velocity (F1)
  - [x] Heading error readout
  - [x] Progress percentage of the track
  - [x] F1/F2/F3 toggles (geometry, sensors, diagnostics)

**Status: Complete.** The environment is stable, deterministic, observable, and debuggable.

---

## Milestone 1 — RL Baseline: Learnability Validation (Active)

This milestone validates that the task is learnable using a from-scratch RL implementation. It began as a minimal A2C baseline and evolved into a substantially optimised PPO system through iterative experimentation.

This is not the final direction of the project. It answers:

> Is the observation space + reward structure sufficient for autonomous learning?

### Evolution

The baseline went through three major phases:

**Phase 1 — A2C Baseline**
The initial implementation: a minimal actor-critic with a single shared 2x64 MLP, on-policy rollout collection, GAE advantage estimation, and online updates. Cars learned to drive forward but could not reliably navigate corners.

**Phase 2 — PPO Upgrade**
A2C was replaced with PPO (clipped surrogate objective) for more stable policy updates. The observation space was expanded from ~15 to 43 dimensions. The finish line was removed in favour of cumulative progress. Random spawn positions replaced fixed start. The reward was simplified to velocity projection + centreline proximity with zero crash penalty.

**Phase 3 — Optimisation and Scaling**
The PPO implementation was heavily optimised for performance on constrained hardware. Weight storage was restructured for cache locality. Training was batched and amortised across ticks. The system was scaled from 1 car to 8 cars. The architecture was made asymmetric (wider critic). A comprehensive analytics pipeline and feature-gated profiling system were built to support data-driven iteration.

### Implementation Status

- [x] PPO with clipped surrogate objective and multi-epoch updates
- [x] Asymmetric actor-critic (actor 2x64, critic 2x128)
- [x] Tanh activations with orthogonal initialisation
- [x] AdamW optimiser with decoupled weight decay on critic
- [x] On-policy rollout buffer with env_id tagging and old log-probs
- [x] Per-env GAE (no cross-env value leakage)
- [x] Per-minibatch advantage normalisation with sample shuffling
- [x] Log-std floor preventing exploration collapse
- [x] Multi-car vectorised training (8 cars, random spawns)
- [x] Amortised PPO updates (64 samples/tick, no frame stutter)
- [x] Batched forward/backward with pre-allocated scratch buffers
- [x] Flat weight storage for cache-friendly traversal
- [x] 43-dimensional observation space (rays, kinematics, lookahead, previous actions)
- [x] Velocity-projection + centreline proximity reward
- [x] Entertainment-first reward philosophy (no crash penalties, no survival bonuses)
- [x] Comprehensive analytics pipeline (16 tick fields, 25 episode aggregates, crash classification, 10-section Markdown reports)
- [x] Feature-gated profiling system (per-system timing, auto-exit, Markdown + JSON reports)
- [x] Live diagnostics HUD with PPO metrics, quarter summaries, run assessment
- [x] Live leaderboard with per-car colour swatches and ranking
- [x] Real-time learning visualisation (watchable behaviour)
- [ ] Headless fast-training mode
- [ ] Policy snapshot + evaluation mode (save/load)

### Success Criteria

- [x] Measurable improvement in forward progress within minutes
- [x] Reduced crash frequency over time
- [x] No reward hacking
- [x] Learning visible in real time
- [ ] Stable extended driving behaviour (cars currently crash at first major corner)

### Active Learning Challenges

The PPO baseline has confirmed the task is learnable — cars demonstrably learn to steer, accelerate, and navigate gentle curves. The remaining challenge is corner survival at speed:

- Throttle exploration tends to collapse (std drops toward the floor), locking the policy at full throttle
- The critic must accurately distinguish "about to crash at a corner" from "driving safely on a straight" — this requires sufficient representational capacity
- The asymmetric architecture (wider critic) and AdamW weight decay were introduced to address this, but the interplay between critic capacity, exploration, and the entertainment constraint (no crash penalties) remains the active research front within Milestone 1

> Milestone 1 proves that the task is learnable.
> It isolates environment design from biological learning mechanics.

---

## Milestone 2 — Brain v1: Rate-Based Local Plasticity + Delta Gating

After learnability is validated, we replace gradient-based learning with biologically inspired mechanisms.

- [ ] Sparse neural graph (fixed I/O, sparse hidden connectivity)
- [ ] Neuron state dynamics (rate-based activations)
- [ ] Eligibility traces per synapse
- [ ] Reward-modulated weight updates (delta-gated plasticity)
- [ ] No backpropagation
- [ ] No global gradient computation
- [ ] Continuous online learning (single persistent brain)
- [ ] Episode metrics + moving averages
- [ ] Save/load brain state

**Success criteria:**
Observable behavioural improvement without gradients.

> This milestone transitions from optimisation to local plasticity.

---

## Milestone 3 — Scientific Control: Stability and Ablations

Prevent self-deception. Prove causality.

- [ ] Weight clamping + decay
- [ ] Learning rate schedules
- [ ] Deterministic episode replay
- [ ] First-half vs second-half statistics
- [ ] Ablations:
  - [ ] No dopamine gating
  - [ ] No eligibility traces
  - [ ] Frozen weights (control baseline)
- [ ] Training-speed controls (1x, 2x, 4x)

**Success criteria:**
Clear evidence that improvements arise from the intended mechanisms.

---

## Milestone 4 — Spiking Upgrade: SNN + STDP

Upgrade representation to spike-based dynamics.

- [ ] Spiking neuron model (membrane potential, threshold, reset)
- [ ] Spike encoding of inputs
- [ ] Spike decoding of outputs
- [ ] STDP-style eligibility traces
- [ ] Reward-modulated STDP
- [ ] Side-by-side comparison with rate-based version

**Success criteria:**
Comparable or improved learning with greater biological plausibility.

---

## Milestone 5 — Structural Plasticity: Growth + Pruning

Introduce constrained topology adaptation.

- [ ] Synapse pruning rules
- [ ] Synapse growth rules (co-activity driven)
- [ ] Bounded fan-in / fan-out
- [ ] Churn metrics (edges added/removed)
- [ ] Topology visualisation

**Success criteria:**
Structural adaptation improves efficiency or stability without graph explosion.

---

## Milestone 6 — Generalisation and Continual Learning

- [ ] Multiple curated tracks
- [ ] Interleaved training across tracks
- [ ] Held-out evaluation track
- [ ] Forgetting metrics
- [ ] Curriculum progression

**Success criteria:**
Skill transfers across tracks without catastrophic forgetting.

---

## Milestone 7 — Replay and Consolidation

- [ ] Trajectory buffer
- [ ] Offline replay ("sleep phase")
- [ ] Consolidation rules
- [ ] Sample efficiency analysis

**Success criteria:**
Replay improves learning speed or stability.

---

## Milestone 8 — Robustness and Perturbation Testing

- [ ] Sensor noise
- [ ] Physics randomisation
- [ ] Track perturbations
- [ ] Long-run stability testing
- [ ] Regression test suite

**Success criteria:**
Learning remains stable under controlled noise.

---

## Milestone 9 — Interpretability and Mechanistic Analysis

- [ ] Identify emergent motor primitives
- [ ] Synapse importance visualisation
- [ ] Activity clustering (turns vs straights)
- [ ] Export topology + activity traces

**Success criteria:**
The system becomes inspectable as a learning mechanism, not just a black box.

---

## What This Project Is Not

- Not a benchmark suite for mainstream RL.
- Not a competition between optimisation paradigms.
- Not a wrapper around PyTorch.
- Not an evolutionary algorithm playground.
- Not a racing game with AI glued on top.

It is a controlled experiment in building a brain-inspired learning system from first principles.

---

## Why Racing?

A racing environment provides:

- Continuous control (steering/throttle)
- Dense and interpretable progress signals
- Non-trivial stability constraints
- Clear measurable improvement (progress, speed, crash rate)
- Natural generalisation tests (new tracks)

It is complex enough to require learning,
but simple enough to keep the focus on the learning mechanism.

---

## Long-Term Vision

NeuroDrive is intended as a research-grade learning laboratory:

- Study synaptic vs structural plasticity in engineered systems
- Implement dopamine-modulated local learning without gradients
- Upgrade to spiking dynamics and STDP-family learning rules
- Evaluate generalisation and continual learning behaviour
- Build a system that _visibly learns_ and can be instrumented end-to-end

The ultimate goal is not the fastest racing agent.

It is to build a system that **visibly, measurably, and continuously learns**
using principles inspired by how biological brains adapt to the world.
