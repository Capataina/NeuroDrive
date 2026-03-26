# Reward Shaping

## Why This Matters Here

The reward signal is the primary teaching signal in NeuroDrive. Get it wrong and the agent learns the wrong behaviour — possibly achieving high reward while doing the opposite of what you intended. NeuroDrive's reward design has been deliberately revised to avoid common pitfalls, and understanding why it is structured the way it is requires understanding reward shaping principles.

**Status:** Current implementation. The reward structure described here reflects the live NeuroDrive code in `src/game/episode.rs`.

## Prerequisites

- `concepts/core/reinforcement-learning.md` — what rewards are optimised over

## The Current NeuroDrive Reward Structure

Every timestep, the total reward is:

```
r_t = progress_reward + time_penalty + terminal_reward
```

Where:

```
progress_reward      = (best_progress_gain) * 140.0
time_penalty         = -0.005 + heading_speed_penalty
heading_speed_penalty = -0.02 * heading_error_norm * speed_norm
terminal_reward      = -5.0  (on crash)
                     = +100.0 (on lap completion)
                     = 0.0   (on timeout)
```

Specific defaults from `EpisodeConfig`:

| Component | Value | Purpose |
|---|---|---|
| `progress_reward_scale` | 140.0 | Primary learning signal |
| `time_penalty_per_tick` | -0.005 | Penalise stalling |
| `heading_speed_penalty_scale` | 0.02 | Penalise high-speed misalignment |
| `crash_penalty` | -5.0 | One-off terminal penalty |
| `lap_bonus` | +100.0 | One-off completion reward |

---

## What Makes a Good Reward Signal

### 1. Density

Sparse rewards (e.g. +1 only on lap completion) are very difficult to learn from because early episodes have no signal at all. The agent may run for many episodes receiving only 0 reward, with no gradient information about what to do.

NeuroDrive uses **dense rewards** — the progress-based reward fires every tick the car makes forward progress. This gives the policy immediate feedback on whether its behaviour is helping.

### 2. Informativeness

The reward should reflect what you actually want, not a proxy that can be gamed. A classic failure mode: reward for *speed* produces a car that crashes at maximum speed, exploiting the reward without the intended behaviour.

NeuroDrive's **best-so-far progress** reward is:

```
progress_reward = max(progress.fraction - previous_best, 0.0) * 140.0
```

This only rewards *new* progress — progress the car has not made before in this episode. It prevents the agent from learning to oscillate back and forth at a progress value it has already achieved (which would give spurious progress gains if raw progress gain were used).

### 3. Avoiding Sprint-and-Crash Incentives

Early versions of the reward may have incentivised aggressive driving that maximises short-term progress at the cost of stability. The heading-speed penalty addresses this:

```
heading_speed_penalty = -0.02 * |heading_error / π| * (speed / 900.0)
```

- At low speed, heading error is barely penalised (the car can afford to be misaligned when it is slow).
- At high speed with large heading error, the penalty is significant.

This discourages the specific behaviour pattern of going fast in the wrong direction — which is exactly the pre-crash state.

---

## Potential-Based Reward Shaping

A formal result in RL (Ng et al., 1999) says that **potential-based shaping** preserves the optimal policy:

```
r'(s, a, s') = r(s, a, s') + γ * Φ(s') - Φ(s)
```

where `Φ(s)` is an arbitrary potential function. Adding such a shaping term does not change which policy is optimal — it only changes the *speed* of learning by providing more informative intermediate rewards.

NeuroDrive's progress reward approximates this: the progress fraction `Φ(s) = progress(s)` acts as a potential, and the reward `progress_gain * 140` is approximately `Φ(s') - Φ(s)` scaled up.

The key property: if the car eventually completes the lap, the total progress reward is approximately the same regardless of path (140 × total progress ≈ 140 × 1.0 for a lap). The shaping adds density without changing the terminal outcome signal.

---

## The Time Penalty

Without a time penalty, the agent is not penalised for stalling indefinitely. It could simply stay in one place and avoid the crash penalty forever, accumulating zero reward. The per-tick time penalty `-0.005` makes inactivity costly:

- After the 30-second timeout (30s × 60 ticks/s = 1800 ticks): cumulative time penalty ≈ -9.0
- This exceeds the crash penalty in magnitude over a full timeout episode

This means the optimal policy prefers crashing quickly over stalling forever — which is the correct incentive.

---

## Reward Decomposition in Analytics

NeuroDrive's analytics system tracks reward decomposition per episode:
- `progress_reward_sum` — total progress reward
- `time_penalty_sum` — total time penalty
- `crash_penalty_sum` — total crash penalties
- `lap_bonus_sum` — total lap bonuses

This decomposition makes it possible to diagnose agent behaviour from the reward breakdown:

| Pattern | Likely cause |
|---|---|
| `progress_reward` high, `crash_penalty` high | Agent is fast but reckless |
| `time_penalty` dominates | Agent is stalling, not making progress |
| `crash_penalty` decreasing, `progress` increasing | Positive learning signal |
| `lap_bonus` appearing | Agent is completing laps |

---

## Known Pitfalls in This Domain

### 1. Single-track Memorisation

With one fixed track, the agent can learn brittle circuit-specific habits. The reward could be maximised by a policy that only works on the training track — not genuine driving skill. This is addressed in Milestone 6 (multi-track generalisation).

### 2. Reward Hacking

Reward hacking is when the agent finds unexpected ways to maximise the reward metric without exhibiting the intended behaviour. Examples in driving:
- Spinning in place if heading error is not penalised at low speed
- Oscillating near a high-progress point
- Slowing to near-zero to avoid the heading penalty while barely advancing

The heading-speed penalty and best-so-far progress reward are defences against the most obvious forms of this.

### 3. Terminal Reward Magnitude

The crash penalty (-5.0) and lap bonus (+100.0) have quite different magnitudes. If a single crash produces a -5 reward but is followed by a fresh episode where the car can earn 100× more, early crashes are relatively cheap. This is intentional: the agent should not be paralysed by crash fear — it should learn to drive well, which means taking risks early.

---

## Reward Is Not the Policy Target

A critical framing difference in NeuroDrive's philosophy:

> "In biology, reward signals guide plasticity but do not dictate behaviour directly."

In the planned biological architecture, reward (via dopamine RPE) will gate which synaptic changes persist — not directly define what the agent does. The agent's behaviour emerges from the interplay of local plasticity rules and the modulating reward signal.

This is reflected in the design principle: the reward is intentionally kept interpretable and separated from the agent code. It is a teaching signal, not a fitness function the agent is trying to satisfy in a narrow computational sense.

---

## Common Misunderstandings

❌ "Denser reward is always better"
✅ Overly dense, easy reward can produce a policy that is good at getting the immediate reward but does not exhibit the larger-scale behaviour you want. The reward density must match the temporal scale of the intended behaviour.

❌ "The crash penalty should be very large to teach safety"
✅ An extremely large crash penalty can make the agent so risk-averse that it never explores far enough to learn driving. The crash penalty should be large enough to deter gratuitous crashing, but not so large that it dominates all other signals.

❌ "Reward shaping always improves learning"
✅ Shaping can introduce spurious incentives if not carefully designed. Only potential-based shaping guarantees policy invariance. Ad hoc shaping may produce unexpected behaviours.

---

## Related Files

- `concepts/core/reinforcement-learning.md` — returns as sums of rewards
- `project/systems/environment-system.md` — where the reward is computed
- `references/observation-vector-reference.md` — what the policy sees alongside the reward
