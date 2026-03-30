# Observation and Action Space Design for Continuous-Control Racing RL

## Scope / Purpose

- Answer the repository-specific question: **is NeuroDrive's observation and action space well-designed for learning aggressive racing behaviour, and what changes would most improve learning?**
- Survey what the strongest racing RL implementations actually give their agents as inputs and outputs.
- Identify specific gaps between our observation/action design and research-backed best practice.
- This paper complements `context/references/reward-structure-design.md` (reward signal) and `context/references/ppo-optimisation.md` (PPO implementation details).

## Current Project Relevance

The agent is failing to learn basic cornering despite having lookahead curvature information, heading error, and braking capability. The observation and action spaces are the remaining unexplored axis — the reward structure has been revised (velocity projection), PPO implementation details have been addressed (tanh, orthogonal init, per-minibatch normalisation), and the physics now support braking. If the agent still cannot learn, the observation or action space may be the bottleneck.

## Current Implementation (Verified)

Verified by direct inspection of `src/agent/observation.rs` and `src/agent/action.rs`.

### Observation Vector (23 dimensions)

```text
Index   Feature                         Normalisation              Range
─────────────────────────────────────────────────────────────────────────
 0-10   Ray distances (11 rays)         /375.0, clamp [0,1]        [0, 1]
        Angles: ±150°, ±90°, ±60°,
        ±35°, ±15°, 0°

   11   Speed (scalar)                  /900.0, clamp [0,1]        [0, 1]
   12   Signed lateral offset           /75.0, clamp [-1,1]        [-1, 1]
   13   Heading error                   /π, clamp [-1,1]           [-1, 1]
   14   Angular velocity                /8.0, clamp [-1,1]         [-1, 1]

15-22   Lookahead (4 samples at         heading: /π                [-1, 1]
        50, 100, 175, 260 units)        curvature: /0.05           [-1, 1]
        2 features each:
        heading_delta, curvature
```

### Action Space (2 dimensions)

```text
Action      Range       Meaning
──────────────────────────────────────────
Steering    [-1, 1]     Left to right
Throttle    [-1, 1]     Full brake to full throttle
```

Both outputs are tanh-squashed from the policy network's Gaussian samples.

### Normalisation Method

Static fixed-range scaling with hard clamps. No running mean/variance normalisation.

---

## What The Research Says

### Observation Space Patterns Across Racing RL

The research reveals a clear hierarchy of observation quality for racing tasks:

```text
Effectiveness (source-backed consensus):

Waypoints + state     ████████████████████  Best generality
LiDAR + state         ████████████████░░░░  Strong for end-to-end
Raycasts + state      ██████████████░░░░░░  Good, our approach
Vision (raw pixels)   ████████████░░░░░░░░  Needs CNN + frame stacking
Vision (single frame) ██████░░░░░░░░░░░░░░  Violates Markov property
```

### What Top Implementations Include

| Feature | F1/10th | GT Sophy | TMRL | DeepRacer | NeuroDrive |
|---------|---------|----------|------|-----------|------------|
| Distance sensors | 20 LiDAR beams | No | 4-frame LiDAR | Camera | 11 raycasts |
| Speed/velocity | Yes | Yes (vector) | Yes | Implicit | Yes (scalar) |
| Acceleration | Yes | Yes (vector) | No | No | **No** |
| Lateral offset | Implicit | Yes | No | No | Yes |
| Heading error | Implicit | Yes | No | No | Yes |
| Angular velocity | No | Implicit | No | No | Yes |
| Track geometry ahead | Via planner | Centreline coords | No | Waypoints | Lookahead curvature |
| Previous actions | No | No | **Yes (last 2)** | No | **No** |
| Frame/state history | No | No | **4 frames** | No | **No** |
| Velocity components (vx, vy) | Sometimes | Yes | No | No | **No** |

### Key Findings That Matter For NeuroDrive

#### 1. Previous Actions Are More Important Than They Seem

**Source-backed finding:** TMRL (TrackMania RL) — one of the most successful MLP-based racing RL systems — includes the **previous 2 actions** (steering, throttle from t-1 and t-2) as observation features. This is notable because TMRL uses a similar architecture to ours (MLP, not RNN) and faces a similar task (continuous racing on a closed track).

**Why it matters:** A memoryless MLP sees one observation and produces one action. It has no way to know what it did last tick. Without this information:
- The car cannot distinguish "I'm already turning left" from "I haven't started turning yet" — both can have the same heading error
- The car cannot learn smooth braking sequences — it can't tell if it's currently braking or just started
- The car cannot learn throttle-brake transitions because it doesn't know its current control state

Including previous actions gives the MLP a form of **one-step memory** without needing an RNN. The policy can learn rules like "if I was already braking last tick and heading error is reducing, start releasing the brake."

**Cost:** +4 dimensions (2 actions × 2 timesteps) → observation dim goes from 23 to 27.

#### 2. Velocity Components (vx, vy) Beat Scalar Speed

**Source-backed finding:** Gran Turismo Sophy and several F1/10th papers include velocity as a 2D vector, not a scalar. Multiple papers note that scalar speed loses directional information that is critical for understanding slides, drifts, and post-collision trajectories.

**Why it matters for us:** Our car currently knows *how fast* it's going (scalar speed) but not *in what direction*. When sliding sideways into a wall, scalar speed reads "fast" but gives no indication of the slide direction. Velocity components in the car's local frame would tell the policy:
- `v_forward`: how fast it's going in the direction it's facing
- `v_lateral`: how fast it's sliding sideways

A car sliding sideways has high `v_lateral`, which is a clear "I'm in trouble" signal. With just scalar speed, this is invisible.

**Implementation:** Decompose `car.velocity` into car-local frame:
```
v_forward = dot(velocity, car_forward_vector)
v_lateral = dot(velocity, car_left_vector)
```

**Cost:** +1 dimension (replace scalar speed with v_forward and v_lateral, net +1) → observation dim goes from 23 to 24.

#### 3. The Action Space Question: 2D vs 3D

**Source-backed finding:** The research is split:
- **F1/10th, DeepRacer, CarRacing:** Use 2D (steering + throttle), no explicit brake
- **TORCS, realistic simulators:** Use 3D (steering + throttle + brake)
- **Key insight from multiple papers:** "The brake-throttle interplay is extremely challenging from an exploration perspective"

The consensus is that a **combined throttle-brake axis `[-1, 1]`** is the pragmatic choice for PPO with a small MLP. Separating them into a 3D space doubles the exploration burden for brake discovery — the policy must independently discover that the brake dimension exists and does something useful, which takes many episodes.

**Project inference:** Our current combined `[-1, 1]` throttle axis is the right choice. The user's observation that "cars barely move" after adding braking suggests the policy is exploring the negative throttle range and getting stuck, not that the action space is wrong. The fix is likely in the observation space (giving the car better information to decide *when* to brake) rather than removing braking.

#### 4. Observation Normalisation: Running Stats Outperform Fixed Scaling

**Source-backed finding:** The "37 Implementation Details of PPO" paper and Andrychowicz et al. (2020) both identify running observation normalisation as a high-impact implementation detail. The approach:
```
normalised = (obs - running_mean) / (running_std + 1e-8)
clip to [-10, 10]
```

Our current approach uses fixed scaling with different denominators per feature (375 for rays, 900 for speed, π for heading, etc.). This has two problems:
- The fixed scales are guesses that may not match actual feature distributions during training
- Different features may have very different effective ranges as the policy explores

**Severity for NeuroDrive:** Moderate. Our fixed scales are reasonable for the known feature ranges, but running normalisation is strictly better and could help with features whose distribution shifts during training (e.g., speed distribution changes as the car learns to drive faster).

#### 5. The Missing Acceleration Signal

**Source-backed finding:** F1/10th and Gran Turismo Sophy both include acceleration as an observation feature. Without it, the agent must infer its rate of speed change from consecutive speed observations — but with a memoryless MLP, it cannot do this (it only sees one tick at a time).

**Why it matters:** The car currently cannot distinguish "going 400 u/s and accelerating" from "going 400 u/s and braking." Both observations look identical. The appropriate action (keep throttle vs keep braking) is completely different. Including speed-delta or acceleration resolves this ambiguity.

**Implementation:** `speed_delta = current_speed - previous_speed`. Store `previous_speed` on `SensorReadings` alongside the existing `previous_heading`.

**Cost:** +1 dimension → minimal.

---

## Gap Analysis

```text
                    Research       NeuroDrive    Gap
                    practice       current       severity
──────────────────────────────────────────────────────────
Previous actions    Common         Missing       HIGH
  as observations   (TMRL, CARLA)               Policy has no memory
                                                of its own behaviour

Velocity            Common         Scalar        MEDIUM
  components        (2D vector)    speed only    Loses slide/drift
                                                direction info

Acceleration /      Common         Missing       MEDIUM-HIGH
  speed delta       (F1/10th,                   Can't distinguish
                     GT Sophy)                   accel from decel

Running obs         Standard       Fixed         MODERATE
  normalisation     practice       scaling       Reasonable but
                                                 suboptimal

Action space        2D combined    2D combined   NONE
  (steer+throttle)  is standard    [-1,1]        Correct choice

Ray count           20 typical     11            LOW
                    for LiDAR                    11 is adequate for
                                                 2D top-down

Lookahead           Varies         4 samples     NONE
  geometry                         w/ curvature  Strong feature set

Frame stacking /    Used with      N/A for       N/A
  temporal          vision, TMRL   our MLP       Prev actions solve
                                                 this differently
```

---

## Recommended Changes — Priority Order

### P0: Add Previous Actions to Observation (HIGH impact, LOW effort)

Add the previous tick's `[steering, throttle]` as 2 extra observation features. This is the single highest-leverage observation change.

```text
New observation layout (25 dimensions):
  [0-10]  Ray distances (11)
  [11]    Speed → v_forward (change in P1)
  [12]    Signed lateral offset
  [13]    Heading error
  [14]    Angular velocity
  [15-22] Lookahead (4 × 2 features)
  [23]    Previous steering         ← NEW
  [24]    Previous throttle         ← NEW
```

**Why P0:** Without previous actions, the policy cannot learn temporal control patterns like smooth braking sequences or throttle-steer coordination. This is the most likely explanation for the "barely moving" behaviour — the policy outputs random throttle values each tick because it has no continuity signal.

**Normalisation:** Previous actions are already in `[-1, 1]`. No normalisation needed.

### P1: Replace Scalar Speed with Velocity Components (MEDIUM impact, LOW effort)

Replace the single speed scalar with car-local velocity components:

```text
v_forward = dot(velocity, forward_vector)  // positive = going forward
v_lateral = dot(velocity, left_vector)     // positive = sliding left
```

**Why P1:** Gives the policy directional velocity information. A car sliding sideways into a wall now has a clear signal (`v_lateral` is large) that scalar speed alone cannot provide.

**Observation dim change:** 25 → 26 (one extra feature).

### P2: Add Speed Delta (MEDIUM impact, TRIVIAL effort)

```text
speed_delta = (current_speed - previous_speed) / speed_norm_max
```

Tells the policy whether it's accelerating or decelerating. Without this, a memoryless MLP cannot distinguish "400 u/s and speeding up" from "400 u/s and slowing down."

**Observation dim change:** 26 → 27.

### P3: Running Observation Normalisation (MODERATE impact, MEDIUM effort)

Replace fixed scaling with running mean/variance normalisation using Welford's online algorithm. Clip to `[-10, 10]` after normalisation.

**Why P3 not P0:** The current fixed scaling is reasonable and not wrong — just suboptimal. The P0–P2 changes address structural information gaps, which matter more than normalisation quality.

---

## What NOT to Change

- **Do not increase ray count.** 11 rays at the current angle spread provide adequate coverage for a 2D top-down track. F1/10th uses 20 for a physical car with a 270° LiDAR; our 11 with manually chosen angles are fine.
- **Do not add frame stacking.** With previous actions and speed delta, the MLP has the temporal context it needs. Frame stacking is for vision-based systems.
- **Do not add an RNN/LSTM.** The memoryless MLP is intentional for the PPO baseline. Previous actions give it one-step memory cheaply.
- **Do not separate brake into a third action dimension.** The combined `[-1, 1]` throttle axis is research-confirmed as the right choice for PPO with small MLPs.
- **Do not add opponent/multi-car observations.** Single-car learning must work first.

---

## Summary: The Likely Explanation for Current Failure

The car has **no memory of its own actions**. Every tick, the MLP sees the world fresh — it doesn't know if it was braking, accelerating, or turning. This makes it impossible to learn sequential control patterns like:

1. "I see a corner coming" → start braking
2. "I'm still going too fast" → keep braking
3. "Speed is now manageable" → release brake, start turning
4. "Through the corner" → accelerate

Without previous actions, step 2 is impossible — the car doesn't know it's already braking. It might output random throttle values because each tick is independent. This produces the "barely moving" or "jittery" behaviour observed.

Adding previous actions (P0) is the minimal change most likely to unlock sequential control. Velocity components (P1) and speed delta (P2) provide the momentum awareness needed for effective braking decisions.

---

## Relationship to Existing Context

- **Complements:** `context/references/reward-structure-design.md` — that paper addresses the reward signal. This paper addresses the information the policy receives.
- **Complements:** `context/references/ppo-optimisation.md` — that paper addresses PPO implementation details.
- **Reads from:** `context/architecture.md` for subsystem boundaries.
- **Verified against:** `src/agent/observation.rs`, `src/agent/action.rs`, `src/game/physics.rs`.

---

## Source List

### Primary Racing RL

- Evans et al., "Reward Signal Design for Autonomous Racing" (arXiv 2021): F1/10th observation space with 20 condensed LiDAR beams + velocity.
- Betz et al., "Unifying F1TENTH Autonomous Racing: Survey, Methods and Benchmarks" (arXiv 2024): Comprehensive survey of observation approaches.
- Wurman et al., "Outracing champion Gran Turismo drivers with deep reinforcement learning" (Nature 2022): Velocity vector + acceleration + track geometry observations.

### Temporal Observations

- TMRL (TrackMania RL): Previous 2 actions as observation features, 4-frame LiDAR history. GitHub: trackmania-rl/tmrl.
- OpenAI CarRacing-v2: 4-frame stacking required — single frame violates Markov property.

### Observation Normalisation

- Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" (ICLR Blog Track 2022): Running observation normalisation as high-impact PPO detail.
- Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning?" (ICLR 2021): Observation normalisation identified as significant for continuous control.

### Action Space Design

- Multiple F1/10th papers: 2D continuous (steering + throttle) is standard, 3D adds exploration cost for marginal benefit with small networks.
- "Observation Space Matters: Benchmark and Optimization Algorithm" (arXiv 2020): RL algorithms are sensitive to observation formulation.
