# Episode-Based RL Environments

## 1. What Is This Pattern?

An episode-based RL environment structures agent-environment interaction as a sequence of discrete episodes. Each episode begins from an initial state, progresses through a series of timesteps where the agent acts and receives rewards, and terminates when a condition is met. The environment then resets and a new episode begins.

This pattern provides the fundamental training loop for nearly all reinforcement learning: the agent collects experience within episodes, learns from that experience, and applies what it has learnt in subsequent episodes.

## 2. When To Use This Pattern

**Good for:**
- Any RL problem with natural start/end conditions (games, races, manipulation tasks)
- Training loops that need clean boundaries for rollout collection and parameter updates
- Environments where tracking per-episode metrics (return, length, success rate) is essential
- Problems where the agent should learn to avoid terminal failure states

**Not good for:**
- Continuing (non-episodic) tasks with no natural reset (e.g., process control, portfolio management)
- Real-world robotics where physical reset is expensive or dangerous
- Problems where episode boundaries would artificially truncate the learning signal

## 3. Core Concept

### Episode Lifecycle

Every episode follows the same structure:

```
SPAWN → [observe → act → step → reward → check terminal] → RESET → repeat
```

1. **Spawn:** Initialise the agent and environment to starting conditions.
2. **Interact:** The agent observes the state, selects an action, and the environment advances one timestep. A reward is emitted.
3. **Terminal check:** After each step, evaluate whether the episode should end.
4. **Reset:** Clear episode state, respawn the agent, increment the episode counter.

### Terminal Conditions

Episodes end for one of three reasons:
- **Crash / failure:** The agent enters an unrecoverable state (collision, out of bounds).
- **Timeout:** A maximum number of timesteps is reached without the agent completing or failing.
- **Goal completion:** The agent achieves the objective (lap completed, target reached).

Each terminal condition typically carries a different reward signal so the agent can distinguish them.

### Reward Design Principles

**Dense vs sparse:** Dense rewards provide signal every timestep (e.g., progress gain). Sparse rewards fire only at terminal events (e.g., +100 for winning). Dense rewards are much easier to learn from but risk reward hacking.

**Reward shaping:** Adding intermediate rewards that guide the agent towards the goal without changing the optimal policy. Must be designed carefully to avoid creating local optima.

**Reward decomposition:** Breaking reward into named components (progress, time, safety, completion) so each can be tuned and inspected independently.

**Anti-backtracking:** Only rewarding *new* best progress prevents the agent from earning reward by oscillating back and forth across a checkpoint.

### Moving Averages for Learning Trends

Raw episode returns are noisy. A moving average (typically over 50–100 episodes) smooths the signal, making it possible to detect whether the agent is actually improving. The formula:

```
ema_new = α * current_return + (1 - α) * ema_old
```

where `α` is the smoothing factor (e.g., `2 / (window + 1)`).

## 4. Key Design Decisions

| Decision | Option A | Option B |
|---|---|---|
| Reward density | Dense (every tick, fast learning) | Sparse (goal only, harder to learn) |
| Timeout length | Short (fast episodes, less exploration) | Long (more exploration, slower training) |
| Reset strategy | Fixed start (reproducible) | Randomised start (generalisation) |
| Terminal reward | Large bonus/penalty (strong signal) | No terminal reward (rely on shaping) |

**Key trade-off:** Reward density vs reward hacking. Dense rewards accelerate learning but create opportunities for the agent to exploit unintended reward pathways. For example, rewarding any forward progress might cause the agent to drive in tiny circles if "progress" is poorly defined. Anti-backtracking mechanisms and careful progress definitions mitigate this.

## 5. Simplified Example Implementation

```python
class Episode:
    def __init__(self, max_ticks, crash_penalty, time_penalty, lap_bonus):
        self.max_ticks = max_ticks
        self.crash_penalty = crash_penalty
        self.time_penalty = time_penalty
        self.lap_bonus = lap_bonus
        self.reset()

    def reset(self):
        self.tick = 0
        self.total_reward = 0.0
        self.best_progress = 0.0
        self.done = False

    def step(self, progress, crashed, lap_complete):
        self.tick += 1
        reward = self.time_penalty  # small negative each tick

        # Only reward NEW best progress (anti-backtracking)
        if progress > self.best_progress:
            reward += progress - self.best_progress
            self.best_progress = progress

        if crashed:
            reward += self.crash_penalty
            self.done = True
        elif lap_complete:
            reward += self.lap_bonus
            self.done = True
        elif self.tick >= self.max_ticks:
            self.done = True

        self.total_reward += reward
        return reward, self.done

class MovingAverage:
    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value
```

## 6. How NeuroDrive Implements This

**Reward structure:** NeuroDrive decomposes the reward into four components:
- **Progress gain:** `current_progress - best_progress`, awarded only when the car reaches a new best. This prevents reward from backtracking.
- **Time penalty:** `-0.001` per tick. Encourages efficient driving rather than dawdling.
- **Crash penalty:** `-5.0` on collision. Strong enough to discourage wall-hitting but not so large as to dominate the learning signal.
- **Lap bonus:** `+50.0` on completing a full lap. Terminal reward for goal completion.

**The `episode_loop_system`:** This Bevy system runs every fixed tick and orchestrates the full episode lifecycle:
1. Increments the tick counter.
2. Computes reward from progress, time, and terminal events.
3. Checks terminal conditions (crash, timeout, lap completion).
4. On termination: finalises episode metrics, logs the result, resets the car to the start position, clears episode state.

**Lap-wrap detection:** Progress is measured as arc-length along the track centreline, which wraps from ~maximum back to ~0 at the start/finish line. NeuroDrive uses an **arm/trigger hysteresis pattern**:
- An "arm" flag activates when the car passes a threshold (e.g., 90% of total track length).
- The lap is only counted as complete when the car subsequently crosses the start line *after* the arm flag is set.
- This prevents false lap triggers from the car reversing across the start line.

**Moving averages:** Episode returns are tracked via an exponential moving average. The analytics system reports both the raw return and the smoothed trend, making it easy to see whether the agent's performance is improving across training runs.

**Episode metadata:** Each completed episode records: episode number, tick count, total reward, terminal reason (crash/timeout/lap), best progress achieved, and the smoothed return. This metadata feeds into the analytics and reporting pipeline.

## 7. Variations

- **Curriculum episodes:** Gradually increase difficulty (shorter timeout, longer track, faster required speed) as the agent improves.
- **Multi-agent episodes:** Multiple agents share an environment; episodes end based on global or per-agent conditions.
- **Hindsight Experience Replay (HER):** After a failed episode, retroactively relabel the goal to match what the agent actually achieved, generating additional training signal.
- **Continuing tasks with pseudo-episodes:** Artificially segment a continuing task into episodes at fixed intervals for the purpose of computing returns and metrics.

## 8. Common Pitfalls

- **Reward scale mismatch:** If the crash penalty is 1000× larger than the progress reward, the agent learns "don't crash" but never learns "drive forward." Balance component magnitudes.
- **Timeout too short:** If most episodes hit the timeout, the agent never experiences goal completion and cannot learn to pursue it. Start with generous timeouts and tighten them as performance improves.
- **Progress reward for backtracking:** Without an anti-backtracking mechanism, the agent can earn unbounded reward by oscillating over a progress boundary. Always track *best* progress.
- **Reset distribution mismatch:** If the agent always starts from the same state, it may overfit to that starting condition. Consider randomised spawn positions once basic behaviour is stable.
- **Ignoring episode length:** Tracking only total reward ignores efficiency. An agent that earns +10 in 100 ticks is better than one that earns +10 in 10,000 ticks. Track both return and episode length.

## 9. Projects That Use This Pattern

- **OpenAI Gymnasium (formerly Gym):** The standard RL environment interface. Every environment implements `reset()` and `step()` — the canonical episode-based API.
- **DeepMind Control Suite (dm_control):** Physics-based RL tasks with episode boundaries, reward decomposition, and timeout handling.
- **MuJoCo Locomotion Tasks:** Humanoid, Ant, HalfCheetah — each uses episode-based training with crash/timeout terminals and dense progress rewards.

## 10. Glossary

| Term | Definition |
|---|---|
| **Episode** | One complete run from initial state to terminal condition |
| **Terminal condition** | A criterion that ends the current episode (crash, timeout, goal) |
| **Dense reward** | Reward signal provided at every timestep |
| **Sparse reward** | Reward signal provided only at specific events (e.g., goal completion) |
| **Reward shaping** | Adding intermediate rewards to guide learning without changing the optimal policy |
| **Anti-backtracking** | Only rewarding new best progress to prevent oscillation exploits |
| **Moving average** | A smoothed statistic that tracks trends by weighting recent values more heavily |
| **Arm/trigger hysteresis** | A two-phase detection pattern that requires passing a threshold before a trigger is valid |

## 11. Recommended Materials

- **"Reinforcement Learning: An Introduction"** by Sutton & Barto, Chapter 3 — The formal definition of episodic vs continuing tasks, returns, and the agent-environment interface.
- **"Gymnasium Documentation"** (gymnasium.farama.org) — The practical standard for episode-based RL environments. Study the `Env` API (reset, step, render) and wrapper patterns.
- **"Reward Shaping in Reinforcement Learning"** by Andrew Ng (1999) — The foundational paper on potential-based reward shaping that preserves optimal policies.
