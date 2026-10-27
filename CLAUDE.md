# NeuroDrive

A Rust and Bevy laboratory in which cars on a 2D top-down track are driven by a brain-inspired learner (a sparse graph of rate-coded neurons learning through local three-factor plasticity, homeostasis and structural growth and pruning), beside a hand-written PPO learner that proves the environment is learnable. What the project is, for anyone, is in `README.md`; this file is what whoever works in the repository needs and the README does not carry. The repository is public, so nothing personal goes in here.

## How a tick runs, and where the depth lives

Everything runs on Bevy's fixed 60 Hz timestep, ordered by the system sets in `src/sim/`. Each tick the game layer (`src/game/`) advances physics, resolves collisions against the tile-grid track (`src/maps/`), measures progress along the centreline and computes the reward; the agent layer (`src/agent/`) builds the 43-dimensional observation every learner reads and applies the two-axis action every learner writes; the learner that owns each car, chosen by the zero-sized marker components `PpoCar`, `BrainCar` and `KeyboardCar`, turns the observation into an action (`src/brain/`); analytics (`src/analytics/`) records the episode and writes a layout-aware report under `reports/` when the run exits; the debug layer (`src/debug/`) draws the HUD, overlays and leaderboard. Episodes end on a crash or a 30-second timeout, never on a lap.

```
fixed tick (sim/ sets)
  game/     physics → collision → progress → reward → episode end?
  agent/    observation (43 dims) ─────────┐
  brain/    PpoCar → ppo/   BrainCar → inspired/   KeyboardCar → keys
  agent/    action (steering, throttle) ◀──┘
  analytics/  per-episode tracking → reports/ on exit
  debug/    HUD, overlays (F1, F2), diagnostics (F3); F4 cycles the layout
```

The detail is in `context/`, which is this repository's existing knowledge layer and is pointed at rather than copied: `context/architecture.md` (the layers, ownership and the tick), `context/systems/*.md` (one file per runtime system), `context/notes/brain-v1-decisions.md` (the per-dial implementation decisions of the brain), `context/notes/conventions.md` (the component-versus-resource rule, the `*Config` pattern, no Bevy events), `context/notes/biology-first-principle.md`, and the research behind the design in `context/references/`.

## Map

```
NeuroDrive/
├─ CLAUDE.md      this file
├─ README.md      what the project is, for anyone; the clips it embeds live in media/
├─ Cargo.toml     the crate and its feature flags: profiling, and one optional GEMM backend pin
├─ src/
│  ├─ sim/        tick ordering and shared geometry
│  ├─ maps/       the tile-grid track, its centreline and arc-length queries; monaco.rs is the one track
│  ├─ game/       cars, physics, collision, progress, reward and episodes
│  ├─ agent/      the observation and action contract both learners share
│  ├─ brain/      inspired/ (the brain), ppo/ (the baseline), common/ (shared maths and GEMM backends)
│  ├─ analytics/  run tracking, metrics and the Markdown and JSON report exporters
│  ├─ debug/      HUD, overlays and leaderboard
│  └─ profiling/  per-system timing behind the `profiling` feature
├─ tests/         pipeline tests for both learners and GEMM backend correctness
├─ context/       architecture, per-system notes, decisions, plans and research references
├─ learning/      a study guide to the concepts the project uses
└─ media/         the GIF and MP4 clips the README embeds
```

## Operating

```bash
cargo run --release                                   # the trainer; debug builds are ~10x slower
cargo run --release --features profiling              # runs 30 s, exits, writes reports/performance/
cargo test                                            # default backend
cargo test --no-default-features --features force-scalar           # the correctness reference
cargo test --no-default-features --features force-matrixmultiply   # the portable backend
cargo test --no-default-features --features force-accelerate       # macOS only
```

Success is every variant green with zero warnings. There is no `justfile` yet, so there is no `just test` and no CI workflow.

## Decisions: what was chosen, what lost, and what would revive it

The environment's own design decisions (no brake, no laps, random spawn, zero crash penalty) are in the README's Behaviour section and are not repeated here.

- **2026-09-07 · What is the project's identity: the biologically inspired brain, a PPO learner that changes its own architecture, or a quantum controller?** Chosen: the biologically inspired brain. Rejected: a quantum controller, because on this hardware it is a simulated variational circuit, and calling it quantum without a matched-parameter classical baseline beside it is a label, not a finding; a PPO learner with mid-training architecture change, because a brain that generalises across changing maps is the more impressive result. Revives if: the brain cannot be shown learning on a real run after a genuine attempt; then the PPO reframe is the fallback. The quantum branch is closed here.
- **2026-09-07 · Should procedural maps, obstacles and mid-run map mutation be built here?** Chosen: yes, as a later phase that doubles as the generalisation test. Rejected: leaving mutation out of NeuroDrive. Because: changing maps are the only real test of whether the brain generalises. Gated on: replacing the tile-grid map, which cannot express a track that changes shape.
- **2026-10-23 · Is NeuroDrive still under active development?** Chosen: no; the project is finished as far as its author is concerned. Revives if: the author picks it up again.

## Laws: what NeuroDrive will not violate, and why

- **No backpropagation and no global loss inside the brain-inspired learner.** Every weight change uses only information available at the synapse plus the broadcast modulator. The project's whole claim is local learning; one gradient breaks it.
- **No external ML library anywhere** (no PyTorch, TensorFlow, JAX, candle, burn). The PPO baseline is hand-written too, so both learners are built from the same first principles.
- **Biology first.** A pathology in the brain is answered from neuroscience (homeostasis, neuromodulation, structural plasticity, replay), never from the ML toolkit (dropout, batch norm, replay buffers). If biology has no clear answer, the next move is more research into the biology. Every new mechanism names its biological motivation and the pathology it addresses.
- **No evolution.** No genetic algorithms, NEAT, evolution strategies or inherited priors: one brain learns within its lifetime and is never reset between episodes.
- **PPO is permanent.** It is the diagnostic baseline that proves the environment learnable, and it stays when the brain improves; a learning brain is only a result beside it.
- **Never fix behaviour with reward penalties or bonuses that make safe play optimal.** Fix the value predictor, exploration, observations or the biology instead. A crash penalty of -5 taught cars to stand still.
- **The observation (43 dimensions) and action contracts are shared and stable.** Both learners consume them identically, so any behavioural difference is attributable to the learner. Changing one changes both and invalidates every prior comparison.
- **Depth is fixed; width and synaptic density grow.** tanh activations are incompatible with identity-preserving depth growth.

## Glossary: names that only matter inside the code

- **`AllBrain{n}`, `AllPpo{n}`, `SideBySide{ppo,brain}`, `Keyboard`**: the `TrainerLayout` variants. The default is `AllBrain{8}`; F4 cycles the first three; `Keyboard` is reachable only programmatically.
- **`BrainBrain`**: the resource holding the single shared brain graph; eligibility traces are per car.
- **`PpoCar`, `BrainCar`, `KeyboardCar`**: zero-sized marker components that partition cars between learners at compile time.
- **`PolicyOutput`**: per-car component whose fields mean the critic estimate under PPO and the modulator M under the brain.
- **RESEARCH-ANCHORED / TUNE**: tags on every `BrainInspiredConfig` dial, saying whether its value came from the literature or is free to tune.
- **Structural cadence**: the tick interval (128 by default) on which utility-based replacement, pruning and sprouting run.
- **Run slug**: `brain`, `side`, `ppo` or `keyboard` in an analytics report filename, from the layout that ran.
- **Monaco**: the hard-coded track module (`maps/monaco.rs`), despite the track being Sepang-inspired.

## Risks: what bites, and the trigger that makes it bite

- **The brain has never been observed learning on a real run.** The structural-plasticity module is implemented with green tests, but its acceptance bar (visible learning relative to PPO over a long side-by-side run) was never validated. Bites the moment anyone claims the brain works, or builds the brain view or procedural maps on top of it. Settled by one long side-by-side run with reward over time recorded for both learners.
- **The tile-grid map cannot express mutation.** Bites when the procedural or mutating-map phase starts: the map system must be rebuilt first.
- **Initial RNG state is not user-controllable.** `SpawnRng` and `PpoBrain.rng` seed from the thread-local RNG, so runs are internally deterministic but not reproducible across sessions. Bites when comparing two runs as if they were the same experiment; both seeding sites must change together.
- **The GitHub description claims STDP.** It says "STDP eligibility traces", but the brain is rate-coded with a three-factor rule; spiking neurons and true STDP are a long-term plan item. Bites any reader or agent who builds on the description.

## Traps

None recorded beyond the risks above.

## Design register

None yet. The only surface is the Bevy debug HUD, overlays and leaderboard, and no palette, type or motion decisions were ever recorded; write one in a `src/debug/CLAUDE.md` when the planned brain view is designed.

## Planned work

Open work is the repository's GitHub issues, `gh issue list -R Capataina/NeuroDrive`, and the private board, `gh project item-list 4 --owner @me --query 'repo:Capataina/NeuroDrive is:open'`.
