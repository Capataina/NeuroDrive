# Project Phase — Baseline → Brain-Inspired

## Current Understanding

As of 2026-04-19, NeuroDrive has **completed its Milestone-1 PPO baseline validation**. The environment, observation contract, reward structure, and analytics pipeline are all confirmed healthy by the round-2 training run (`reports/analytics/run_1776556719.md`):

- all 8 cars complete the full track loop within ~2,000 episodes,
- fleet converges tightly (max-progress spread 1.1%),
- mean speed rises monotonically across training,
- crash rate collapsed from 100% → ~56% in the best chunk,
- critic now genuinely anticipates (96% of crashes had throttle released > 0.25 s before impact).

This was the goal of the baseline phase: prove the environment is learnable and the observation/reward contract is sound. **That proof is complete.**

The project's intent per `README.md` has always been **brain-inspired local plasticity** rather than backprop-driven PPO. The PPO path existed specifically to validate the environment before committing to the harder learning-rules work. The transition to that phase is now pending.

## Rationale

The baseline was necessary because introducing brain-inspired plasticity rules on an *unvalidated* environment would have confounded every failure: "is the environment broken, the observation wrong, the reward mis-shaped, or is the learning rule genuinely incapable of solving this?" Now when a biological-plasticity implementation fails to learn, we can rule out the environment as the cause — PPO has demonstrated that the same observation → action mapping **is** learnable under standard RL machinery.

## What This Means for `context/`

- The PPO subsystem is **stable reference machinery**, not active development. `systems/brain-ppo.md` captures its current reality; further PPO polish items (LR annealing, log-std Adam extraction) are deferred indefinitely.
- The four round-1 research references (`ppo-critic-architecture.md`, `value-target-normalisation.md`, `observation-horizon-racing-rl.md`, `ppo-tuning-knobs-racing.md`) now document *why the baseline works*, not active intervention candidates.
- The environment, agent interface, and analytics subsystems are load-bearing for the next phase — whatever replaces PPO will still consume the same observation contract and produce the same `ActionState.desired` writes.
- New `systems/*.md` files will likely appear for biological-plasticity substrates, but the shape isn't committed yet. **Do not create speculative system files.**

## Guiding Principles for the Transition

- **Preserve the stable boundary.** The agent interface (`CarAction` ↔ `ActionState`, `ObservationVector` 43-dim) is the stable contract. Whatever learning rule replaces PPO must consume the same inputs and produce the same outputs — otherwise the environment validation has to be redone.
- **PPO stays until the replacement works end-to-end.** The `AgentMode` toggle (F4 keyboard vs Ai) becomes a three-way toggle (keyboard, PPO, brain-inspired), or PPO is retired only when the brain-inspired agent has been shown to learn on the same track.
- **Entertainment-first reward constraint carries forward.** `notes/reward-and-entertainment.md` applies regardless of learning rule — no crash penalty, no survival bonus, velocity projection + centreline proximity.
- **Analytics also carry forward.** The 15-section Markdown report and `PpoUpdateRecord` schema assume PPO internals, but the *episode-level* fields (`EpisodeRecord`, `TickTraceRecord`) are learning-rule-agnostic and should be reused.

## What Was Tried

- **PPO baseline from scratch in handwritten Rust** (no PyTorch, no external ML libs). Succeeded.
- Throughout PPO baseline development, multiple dead ends were explored and captured durably:
  - ReLU critic (reverted to tanh — dead-neuron fraction starved the actor).
  - Brake axis (reverted — policy collapsed into idle basin).
  - Crash penalty in reward (never shipped — would produce boring driving).
  - Progress-bonus reward (superseded by velocity projection).

## What Comes Next

Pending discussion (2026-04-19): what specifically counts as the first brain-inspired increment. Candidates the user has mentioned in `README.md`:

- Local Hebbian / STDP-style synaptic updates.
- Dopamine-like neuromodulation gating plasticity.
- Structural adaptation (connection formation and pruning).
- Continual online learning without episode resets on the learning side.

This note will be updated or superseded once the first brain-inspired commit lands.
