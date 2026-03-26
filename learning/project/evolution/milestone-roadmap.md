# Milestone Roadmap

## Overview

NeuroDrive follows a deliberate nine-milestone sequence from a deterministic environment to a biologically-inspired, generalising, interpretable learning system. Each milestone builds on the previous one. The sequencing strategy is not arbitrary — it reduces debugging ambiguity and isolates environment issues from learning-rule issues at each stage.

**Status:** Milestone 0 complete. Milestone 1 substantially complete (live A2C, with gaps in persistence and headless training). Milestones 2–9 are planned.

## The Sequencing Philosophy

```
1. Build a correct, observable environment.
2. Prove the task is learnable with a standard algorithm.
3. Replace gradient learning with local biological plasticity.
4. Gradually increase biological fidelity.
5. Test generalisation and stability.
6. Make the system interpretable.
```

The project never skips ahead. If Milestone 2 biology cannot learn the same task that A2C learned, the problem must be understood before adding STDP or structural plasticity. Each stage answers one question before the next question is asked.

---

## Milestone 0 — Environment Foundation

**Question answered:** Can we build a deterministic, observable racing environment that is reliable enough to run learning experiments?

**Status:** Complete.

### What Was Built

- Deterministic fixed-timestep (60 Hz) 2D car physics
- Closed-loop track with tile-based representation
- `TrackGrid` for collision detection and raycasting
- `TrackCenterline` for progress measurement and lookahead
- Collision detection with `CollisionEvent` emission
- Progress metric via centreline projection (dense, continuous)
- 23-dimensional normalised observation vector
- Steering + throttle action interface
- Episode lifecycle: crash / timeout / lap completion / reset
- Deterministic replay test: same actions → identical trajectory
- Full debug overlay suite: F1 (geometry), F2 (sensors), F3 (HUD)

### Success Criteria

A manually controlled or heuristic controller can complete laps reliably. All geometric quantities are visually verified. The environment is stable and debuggable before any learning is added.

**Note:** No learning occurs at Milestone 0. The entire goal is correctness and instrumentation.

---

## Milestone 1 — A2C Baseline (Learnability Validation)

**Question answered:** Is the observation space and reward structure sufficient for autonomous learning?

**Status:** Substantially complete. A2C is live and learning. Remaining gaps are in experiment discipline.

### What Was Built

- Handwritten A2C in Rust (no ML frameworks)
- Separate actor and critic MLP stacks (23 → 64 → 64 → 2 mean; 23 → 64 → 64 → 1 value)
- Learnable log-std parameters for the Gaussian policy
- Tanh-squashed bounded actions with Jacobian-corrected log-probability
- On-policy rollout buffer
- GAE (γ=0.99, λ=0.95) for advantage estimation
- Policy gradient + Huber value loss + entropy bonus
- Gradient clipping (global norm 0.5) + Adam optimiser
- Online updates at rollout horizon or episode terminal
- `a2c_flush_on_exit_system` for partial rollout updates on exit
- `A2cTrainingStats` snapshots (losses, entropy, explained variance, action spread, dead ReLU fraction)
- Analytics export: per-tick traces, episode summaries, training stats → JSON and Markdown reports
- F4 toggle between AI and keyboard mode

### Remaining Gaps

| Gap | Notes |
|---|---|
| No model persistence | Weights lost on exit; training must restart from zero |
| No headless mode | Requires window; can't run faster than real-time |
| No evaluation mode | Can't run deterministically for fair evaluation |
| Weak RNG ownership | Runs are not reproducible from a given seed |
| No run metadata in exports | RNG seed, config snapshot, git revision not recorded |

### Success Criteria

Measurable improvement in forward progress within minutes. Reduced crash frequency over episodes. Stable lap completion behaviour. No reward hacking. Learning visible in real time.

**Note:** Milestone 1 does not need to produce a perfect racing agent. It just needs to demonstrate that the environment contract enables autonomous learning. If A2C fails to learn, diagnose observation scaling, reward magnitudes, and timestep stability before proceeding to biological learning.

---

## Milestone 2 — Brain v1 (Rate-Based Local Plasticity)

**Question answered:** Can local plasticity rules without backpropagation produce observable behaviour improvement?

**Status:** Planned. The `src/brain/biological/` directory is an empty placeholder.

### What Will Be Built

- Sparse neural graph with fixed I/O nodes and sparse hidden connectivity
- Rate-based neuron activations (continuous, like current A2C activations, but with Hebbian rules)
- Per-synapse eligibility traces: `e_ij ← λ * e_ij + x_i * x_j`
- Reward prediction error δ computed from the value function: `δ = r + γV(s') - V(s)`
- Weight update: `Δw_ij = η * δ * e_ij`
- No backpropagation, no global gradient computation
- Continuous online learning (weights update during rollout, not just at update steps)
- Save/load brain state (persistent brain across sessions)
- Episode metrics and moving averages

### Key Design Constraints

- **One brain, one lifetime:** weights are never reset between episodes
- **Local updates only:** each synapse updates using only locally available information plus the global δ signal
- **A value function still needed:** δ requires V(s) and V(s'). In Milestone 2, this is likely a simple separately-learned module, making it a hybrid system.

### Success Criteria

Observable behavioural improvement without backpropagation. The learning curve should show the same qualitative shape as A2C (increasing returns, decreasing crash rate) but from local plasticity alone.

---

## Milestone 3 — Scientific Controls (Stability and Ablations)

**Question answered:** Are the improvements in Milestone 2 caused by the intended mechanisms, not by something else?

**Status:** Planned.

### What Will Be Built

- **Weight clamping and decay:** prevent weight explosion or collapse
- **Learning rate schedules:** reduce plasticity as the network stabilises
- **Deterministic episode replay:** given the same action stream, the network should produce identical state trajectories
- **First-half vs second-half statistics:** did the brain improve over the run's second half relative to the first?
- **Ablation suite:**
  - No δ gating (set δ = 1.0 always — pure Hebbian, no reward modulation)
  - No eligibility traces (set e_ij = instantaneous correlation only)
  - Frozen weights (the brain does nothing; pure environment baseline)
- **Training speed controls:** 1×, 2×, 4× simulation speed for longer experiments

### Why This Milestone Exists

Cognitive science and ML research are both plagued by results that look like causal proof but are not. "The brain improved after adding eligibility traces" is only meaningful if you can show that *removing* eligibility traces causes a measurable performance drop compared to the full system.

Milestone 3's ablations are the methodological insurance against confounding factors. Each removed component should produce a degradation. If removing δ gating does not hurt performance, either the gating is not working as intended or the task does not require it.

### Success Criteria

Clear, reproducible evidence that improvements arise from the intended mechanisms. Each ablated component should produce a measurable performance reduction.

---

## Milestone 4 — Spiking Upgrade (SNN + STDP)

**Question answered:** Do spike-based dynamics and timing-dependent plasticity produce comparable or better learning than rate-based Hebbian plasticity?

**Status:** Planned.

### What Will Be Built

- Spiking neuron model: leaky integrate-and-fire (LIF) dynamics
  - Membrane potential, threshold, reset, refractory period
- Spike encoding of sensor inputs (rate coding or population coding)
- Spike decoding of motor outputs
- STDP eligibility traces: timing-window-based, pre/post spike correlation
- Reward-modulated STDP: δ gates which STDP changes persist
- Side-by-side comparison with rate-based Milestone 2 version

### Key Challenges

- **Simulation timestep mismatch:** NeuroDrive runs at 60 Hz (16.7 ms per tick). LIF neurons typically require ~1 ms timesteps. Either multiple SNN micro-ticks per game tick, or accept lower temporal resolution.
- **Input encoding:** 23 continuous values → spike trains. Population coding is more biologically plausible but requires more neurons per input feature.
- **Output decoding:** Spike trains → steering and throttle. Must be smooth and bounded.

### Success Criteria

Comparable or improved learning relative to Milestone 2. A side-by-side performance comparison should quantify the tradeoffs. The STDP-based system should show evidence of the timing-based credit assignment that rate-based rules cannot capture.

---

## Milestone 5 — Structural Plasticity (Growth and Pruning)

**Question answered:** Does topology adaptation (forming and removing synapses) improve efficiency or stability compared to a fixed-topology sparse network?

**Status:** Planned.

### What Will Be Built

- **Synapse pruning:** remove synapses with persistently low weight magnitude AND low eligibility contribution
- **Synapse growth:** add synapses between recently co-active neurons when capacity is available
- **Bounded fan-in/fan-out:** prevents any neuron from accumulating too many connections
- **Churn metrics:** count of synapses added and removed per N ticks
- **Topology visualisation:** live display of graph connectivity changes over time

### Why This Is Not a Gimmick

A fixed-topology sparse network can represent only as many distinct functions as its topology allows. Structural plasticity allows the network to reallocate capacity — adding connections where the task demands them, pruning connections that carry no signal. This is the only mechanism that can adapt the representational structure, as opposed to just the strength of existing connections.

See `concepts/advanced/structural-plasticity.md` for a detailed treatment.

### Success Criteria

Structural adaptation improves efficiency or stability without graph explosion. Synapse count should stabilise over time. Churn rate should decrease as the network converges. Performance should match or exceed the fixed-topology equivalent.

---

## Milestone 6 — Generalisation and Continual Learning

**Question answered:** Does the biological brain generalise to new tracks without catastrophically forgetting old ones?

**Status:** Planned.

### What Will Be Built

- Multiple curated tracks (different layouts, turn radii, straight lengths)
- Interleaved training across tracks (e.g. alternating episodes across tracks)
- Held-out evaluation track (not seen during training)
- **Forgetting metrics:** `F = Performance_after_new_training(old_track) / Performance_before_new_training(old_track)`
- Curriculum progression (easy tracks first, harder tracks later)

### Why This Milestone Is Hard

The one-brain-one-lifetime principle means the same weights must handle all tracks. Catastrophic forgetting — where learning on Track B overwrites what was learned on Track A — is the primary risk. See `concepts/advanced/continual-learning.md` for the theory.

The biological mechanisms (eligibility traces, structural plasticity) provide some natural forgetting resistance. But for multi-track generalisation, explicit consolidation mechanisms (Milestone 7) may also be needed.

### Success Criteria

Skill transfers across tracks without catastrophic forgetting. The forgetting metric F should remain close to 1.0 as new tracks are introduced.

---

## Milestone 7 — Replay and Consolidation

**Question answered:** Does offline replay of stored trajectories (analogous to sleep-phase memory consolidation) improve learning speed or reduce forgetting?

**Status:** Planned.

### What Will Be Built

- Trajectory buffer: store recent episodes
- Offline replay phase: periodically replay buffered experiences (without live environment interaction)
- Consolidation rules: during replay, reinforce patterns that support good performance
- Sample efficiency analysis: does replay reduce the number of environment steps needed?

### Biological Motivation

The hippocampus rapidly encodes new experiences during waking, then replays them to the neocortex during sleep. This off-line replay transfers information from a fast-adapting, high-forgetting-risk system to a slow-changing, stable system. NeuroDrive's sleep phase mimics this principle.

### Success Criteria

Replay improves learning speed (fewer environment steps to a performance threshold) or reduces forgetting across tracks.

---

## Milestone 8 — Robustness and Perturbation Testing

**Question answered:** Does the biological brain maintain performance under sensor noise and physics variation?

**Status:** Planned.

### What Will Be Built

- Sensor noise (gaussian noise on ray distances, velocity)
- Physics randomisation (slight variation in max speed, steering gain between episodes)
- Track perturbations (slight changes to track width or corner radii)
- Long-run stability testing (hours of simulated time without performance collapse)
- Regression test suite (automated verification that performance meets a minimum threshold)

### Success Criteria

Learning remains stable under controlled noise. Performance degrades gracefully (not catastrophically) as noise increases. The regression suite passes.

---

## Milestone 9 — Interpretability and Mechanistic Analysis

**Question answered:** What did the brain actually learn? Can we identify the internal representations it uses?

**Status:** Planned.

### What Will Be Built

- Identification of emergent motor primitives (recurring activation patterns that correlate with specific behaviours)
- Synapse importance visualisation (which connections carry the most signal)
- Activity clustering: do groups of neurons specialise for turns vs straights?
- Export of topology and activity traces for offline analysis
- Optional live graph view (nodes and edges with weight coloring)

### Success Criteria

The system becomes inspectable as a learning mechanism, not just a black box. The internal representations should be human-interpretable enough to answer: "what did this brain learn to do, and how?"

---

## The Milestone Dependencies

```
Milestone 0 (Environment)
    ↓
Milestone 1 (A2C Baseline)
    ↓
Milestone 2 (Rate-Based Local Plasticity)
    ↓
Milestone 3 (Ablations)  ←─── validates Milestone 2 is real
    ↓
Milestone 4 (SNN + STDP)
    ↓
Milestone 5 (Structural Plasticity)
    ↓
Milestone 6 (Generalisation)  ←─── may co-develop with Milestone 7
Milestone 7 (Replay)
    ↓
Milestone 8 (Robustness)
    ↓
Milestone 9 (Interpretability)
```

None of the milestones are independent. Each one builds on the verified foundation of the previous. Skipping Milestone 3 and proceeding to Milestone 4 without ablation validation would be a scientific error.

---

## Related Files

- `project/evolution/from-baseline-to-brain.md` — the specific transition from A2C to biological learning
- `project/decisions/a2c-as-baseline.md` — why A2C is the right choice for Milestone 1
- `project/comparisons/rate-based-vs-spiking.md` — Milestones 2 and 4 compared
- `concepts/advanced/continual-learning.md` — the theory behind Milestone 6
- `concepts/advanced/structural-plasticity.md` — the theory behind Milestone 5
- `concepts/advanced/neuromodulation.md` — the δ signal that appears at Milestone 2
