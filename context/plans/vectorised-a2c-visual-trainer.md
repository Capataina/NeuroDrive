# Implement Now — Vectorised A2C Visual Trainer

## Header

- [ ] Status: Proposed, not yet executed.
- [ ] Scope: Replace NeuroDrive’s singleton A2C runtime with a 25-car synchronous vectorised trainer that keeps all cars visible in one scene, highlights the current best performer, and upgrades analytics from single-episode summaries to per-instance and cohort-level training summaries.
- [ ] Exit rule: stop when the runtime can spawn and train `25` independent cars at once under one shared A2C policy, the best current car is visually emphasised, analytics and HUD report both per-car and aggregate cohort behaviour correctly, and the old singleton-only assumptions are removed or isolated behind compatibility shims.

## Implementation Structure

- [ ] Modules / files affected (expected):
  - `src/main.rs`
  - `src/agent/action.rs`
  - `src/agent/observation.rs`
  - `src/agent/plugin.rs`
  - `src/brain/plugin.rs`
  - `src/brain/types.rs`
  - `src/brain/a2c/mod.rs`
  - `src/brain/a2c/buffer.rs`
  - `src/brain/a2c/update.rs`
  - `src/game/car.rs`
  - `src/game/collision.rs`
  - `src/game/episode.rs`
  - `src/game/plugin.rs`
  - `src/game/progress.rs`
  - `src/debug/hud.rs`
  - `src/debug/overlays.rs`
  - `src/debug/plugin.rs`
  - `src/analytics/models.rs`
  - `src/analytics/plugin.rs`
  - `src/analytics/trackers/action.rs`
  - `src/analytics/trackers/trace.rs`
  - `src/analytics/trackers/episode.rs`
  - `src/analytics/metrics/*`
  - `src/analytics/exporters/json.rs`
  - `src/analytics/exporters/markdown.rs`
  - `context/systems/brain-a2c.md`
  - `context/systems/analytics.md`
  - `context/systems/debug-runtime.md`
  - `context/systems/environment.md`
- [ ] Responsibility boundaries:
  - `maps` stays singleton unless later evidence demands per-instance tracks.
  - `game` owns per-car environment truth and per-car episode truth.
  - `agent` owns per-car action and observation boundaries, but no longer as singleton resources/components-only assumptions.
  - `brain/a2c` owns the shared policy, the trainer rollout buffer, rollout scheduling, update cadence, and trainer-level selection of “best current car”.
  - `analytics` owns both per-car episode facts and trainer-level cohort aggregation.
  - `debug` owns visual differentiation and trainer dashboard surfaces only; it must not define “best” truth independently.
- [ ] Recommended default runtime shape:
  - one shared track entity
  - twenty-five independent car entities
  - one shared policy/value network
  - one trainer resource aggregating transitions across all cars
  - one visible scene containing all cars
  - cars rendered with reduced alpha by default
  - one best-performing current car rendered at full opacity and stronger colour
- [ ] Alternative architecture to keep in reserve:
  - one visible environment plus many headless environments or separate worlds
  - worse for the current goal because the user explicitly wants to see all 25 cars at once
  - better only if visual clutter or ECS scaling makes one-world training too awkward later
- [ ] Function / type inventory:
  - `TrainerConfig` in a new or expanded brain module
    - Inputs/outputs: stores trainer-wide constants such as `num_envs`, visual ranking window, update horizon, evaluation cadence, cohort-percentile cutoffs.
    - Kind: resource.
    - Called by: startup, brain, analytics, debug.
  - `EnvInstanceId` in `src/game/` or a small shared runtime-types file
    - Inputs/outputs: tags every car-scoped runtime component with one stable environment instance identity.
    - Kind: component/value type.
    - Called by: car spawn, analytics grouping, HUD focus logic.
  - `TrainerCar` or expanded `Car` metadata in `src/game/car.rs`
    - Inputs/outputs: attaches per-car trainer metadata such as instance id, render role, and reset bookkeeping.
    - Kind: component.
    - Called by: car spawn, debug styling, trainer ranking.
  - `spawn_training_cars_system` replacing singleton spawn path
    - Inputs/outputs: spawns 25 cars with deterministic offsets near spawn or in a small start fanout that avoids exact overlap.
    - Kind: orchestrator.
    - Called by: `GamePlugin` startup.
  - `PerCarEpisodeState` replacing singleton `EpisodeState`
    - Inputs/outputs: stores reward, end reason, progress, crash counts, and summary fields per car.
    - Kind: component or indexed resource record.
    - Called by: episode loop, analytics, debug, trainer ranking.
  - `PerCarActionState` replacing singleton `ActionState`
    - Inputs/outputs: desired and applied action per car.
    - Kind: component or indexed resource record.
    - Called by: brain act path, smoothing, physics, analytics.
  - `a2c_act_all_cars_system`
    - Inputs/outputs: reads all car observations, runs the shared policy for each active car, writes each car’s desired action, appends pre-step rollout entries tagged by env id.
    - Kind: orchestrator.
    - Called by: `A2cPlugin` during `SimSet::Input`.
  - `a2c_collect_rewards_all_cars_system`
    - Inputs/outputs: reads every car’s per-tick reward/done state, appends aligned reward/done data by env id, triggers one shared update when trainer horizon is reached.
    - Kind: orchestrator.
    - Called by: `A2cPlugin` during `SimSet::Measurement`.
  - `TrainerRolloutBuffer`
    - Inputs/outputs: stores flattened batch data plus env ids, per-env episode fragment alignment, and batch-boundary bookkeeping.
    - Kind: data structure.
    - Called by: A2C systems and update path.
  - `TrainerLiveRanking`
    - Inputs/outputs: computes current best car, worst car, and ranking snapshots from recent performance windows.
    - Kind: resource.
    - Called by: debug styling, analytics summaries, HUD.
  - `update_car_visual_roles_system`
    - Inputs/outputs: adjusts sprite colour/alpha based on current rank role.
    - Kind: helper/orchestrator.
    - Called by: debug/update schedule.
  - `TrainerAnalyticsRecord` and cohort summary builders
    - Inputs/outputs: derive top 25%, middle 50%, bottom 25%, best, worst, global mean, spread, and error summaries from per-car episode/update data.
    - Kind: schema + metrics layer.
    - Called by: analytics trackers and exporters.
- [ ] Wiring summary:
  - spawn 25 cars
  - every fixed tick each car receives its own action and observation
  - all cars step independently on the same track geometry
  - each car computes reward and done independently
  - the shared trainer buffer aggregates transitions from all cars
  - one A2C update consumes the aggregate batch
  - trainer ranking marks the current best car
  - analytics exports both per-car and cohort-level summaries
  - HUD surfaces trainer-wide status rather than singleton-car status only

## Algorithm / System Sections

### 1. Instance-scoped environment state

The first job is to break the singleton assumptions cleanly. Right now the runtime assumes one car, one progress state, one action state, and one episode state. Vectorised A2C cannot be layered on top of those assumptions safely; trying to do so would produce hidden coupling and misaligned analytics.

The recommended default is to keep one visible track entity and to make every car-scoped runtime concept explicitly instance-scoped. That means per-car components or per-car indexed state for action, progress, observation, and episode truth.

- [ ] Discovery (bounded):
  - [ ] Read every `single()` / `single_mut()` query in `src/game/`, `src/agent/`, `src/debug/`, and `src/brain/a2c/`.
  - [ ] Inventory every singleton resource that currently stores car-specific truth.
  - [ ] Identify which singleton resources are truly trainer-wide versus only singleton because the repo currently has one car.
- [ ] Implementation playbook:
  - [ ] Introduce a stable `EnvInstanceId` type.
  - [ ] Decide the storage strategy for per-car state:
    - recommended default: car-scoped components for action/progress/observation/episode state to stay idiomatic with ECS queries.
    - alternative: indexed trainer resources keyed by entity or instance id, which is worse here because the existing systems already reason in entity queries.
  - [ ] Replace singleton queries with `for` loops over all cars.
  - [ ] Keep one shared `Track` query unless per-instance tracks become necessary.
  - [ ] Replace or isolate any helper that assumes “the one car”.
- [ ] Stop-and-verify checkpoints:
  - [ ] The runtime can spawn 25 cars without panic from `single()` assumptions.
  - [ ] Each car has independent progress, reward, and reset behaviour.
  - [ ] One car crashing does not reset any other car.
- [ ] Invariants / sanity checks:
  - [ ] Every training car has exactly one `EnvInstanceId`.
  - [ ] Every training car has exactly one action, observation, progress, and episode state surface.
  - [ ] There are no remaining singleton-car queries in the training path.
- [ ] Minimal explicit test requirements:
  - [ ] Add at least one ECS-level test or runtime assertion proving that two cars can terminate independently in the same tick range without shared-state corruption.

### 2. Multi-car spawn, reset, and visibility model

The visual goal changes the implementation significantly. This plan is not for a mostly headless trainer with one representative viewport. The user wants to see all 25 cars at once, with the best current performer fully coloured and the rest semi-transparent.

The recommended default is one shared track with 25 non-colliding cars because car-to-car collisions are not part of the current environment truth. That keeps the environment semantics stable while making the trainer visually inspectable.

- [ ] Discovery (bounded):
  - [ ] Read `src/game/car.rs`, `src/game/plugin.rs`, and all code that assumes one camera target or one player car.
  - [ ] Decide how to stagger initial spawn positions and rotations without making starts unfair or instantly overlapping.
- [ ] Implementation playbook:
  - [ ] Replace singleton car spawn with a loop over `num_envs`.
  - [ ] Use a deterministic spawn fanout around the canonical spawn point:
    - small lateral offsets and perhaps tiny heading jitter
    - keep offsets small enough that the task remains the same problem
  - [ ] Add a render-role concept:
    - default car
    - best-current car
    - optionally focused car for HUD/overlay details
  - [ ] Update sprites so all cars render with reduced alpha by default.
  - [ ] Render the best-current car at full opacity and stronger colour.
  - [ ] Ensure reset restores the car to its own configured spawn offset, not just the canonical track spawn.
- [ ] Stop-and-verify checkpoints:
  - [ ] All 25 cars are visible at startup.
  - [ ] Default opacity is visibly lower for non-best cars.
  - [ ] Best-car highlight updates when rankings change.
  - [ ] Visual role changes do not affect physics or training logic.
- [ ] Invariants / sanity checks:
  - [ ] Rendering role is derived from trainer ranking, never hand-authored independently in debug code.
  - [ ] Car-to-car overlap must not create gameplay truth because collisions remain track-only.
  - [ ] Spawn staggering must be deterministic given a seed/config.
- [ ] Minimal explicit test requirements:
  - [ ] Add at least one test or assertion proving per-car reset returns to that car’s assigned spawn transform.

### 3. Shared-policy synchronous rollout collection

This is the actual vectorised A2C core. The shared policy must act for all cars each tick, and all per-car transitions must be gathered into one coherent trainer batch before one update is taken. That is the part that makes the trainer “A2C” rather than 25 unrelated single-agent runs.

The recommended default is one shared policy and one shared critic, with a trainer rollout buffer that flattens all transitions but preserves `env_id` tagging so debugging and per-instance analytics remain possible.

- [ ] Discovery (bounded):
  - [ ] Read the current `a2c_act_system`, `a2c_collect_reward_system`, `RolloutBuffer`, and `a2c_update`.
  - [ ] Identify every place where the rollout currently assumes exactly one pending reward for one stored state.
- [ ] Implementation playbook:
  - [ ] Redesign `RolloutBuffer` into a trainer batch structure that stores:
    - states
    - actions
    - latent actions
    - values
    - rewards
    - dones
    - safety-clamp hits
    - `env_id`
    - optional per-transition episode id
  - [ ] Decide update triggering semantics:
    - recommended default: update when total collected transitions across all cars reaches a configured horizon
    - alternative: per-env horizon, which is worse because it complicates synchronous batching with little gain
  - [ ] Keep bootstrapping per transition based on whether that env instance terminated.
  - [ ] Ensure multiple cars terminating in the same tick are handled without buffer misalignment.
  - [ ] Continue to clear rollout state on mode switches, but now at trainer scope.
- [ ] Stop-and-verify checkpoints:
  - [ ] Batch sizes scale roughly with `num_envs`.
  - [ ] The number of rewards always matches the number of stored state/action transitions.
  - [ ] Partial terminal episodes from some cars do not corrupt non-terminal fragments from others.
- [ ] Invariants / sanity checks:
  - [ ] For every batch index `i`, all rollout fields share the same `env_id`.
  - [ ] Done masking is applied per transition, not globally per update.
  - [ ] A trainer update never mixes missing reward entries with live state entries.
- [ ] Minimal explicit test requirements:
  - [ ] Unit test for trainer-buffer alignment with two or more env ids interleaved.
  - [ ] Unit test for GAE / return logic on mixed terminal and non-terminal env fragments.

### 4. Per-car episode logic and ranking model

The old singleton `EpisodeState` currently mixes environment truth, reward decomposition, and last-episode summaries for one car. In a vectorised trainer, that must become per-car truth first, with trainer-level ranking built on top of it.

The ranking should not be based on one noisy scalar only. The recommended default is to rank cars by a short rolling performance score built primarily from best progress and return, with explicit tie-breakers and stable hysteresis so the highlight does not flicker every tick.

- [ ] Discovery (bounded):
  - [ ] Read the current episode loop and identify all fields that are current-tick facts versus last-episode summaries.
  - [ ] Decide which metrics are safe to use for live best-car selection.
- [ ] Implementation playbook:
  - [ ] Split per-car episode truth from trainer aggregate summaries.
  - [ ] Create a `TrainerLiveRanking` resource with:
    - current best env id
    - current worst env id
    - recent performance window per env id
    - tie-breaker and hysteresis rules
  - [ ] Define recommended ranking score:
    - primary: recent best progress
    - secondary: recent total return
    - tertiary: fewer crashes / longer survival
  - [ ] Recompute ranking at a bounded cadence, not every visual frame if that causes flicker.
- [ ] Stop-and-verify checkpoints:
  - [ ] The highlighted car remains stable long enough to be visually meaningful.
  - [ ] Cars can terminate and reset independently without affecting trainer ranking bookkeeping.
  - [ ] Worst-car and percentile groups update correctly as episodes accumulate.
- [ ] Invariants / sanity checks:
  - [ ] Ranking source of truth lives in trainer/analytics logic, not in debug code.
  - [ ] A car that just reset is not automatically considered best or worst without actual data.
  - [ ] Live ranking windows and exported analytics windows are documented separately if they differ.
- [ ] Minimal explicit test requirements:
  - [ ] Add at least one deterministic ranking test covering ties, resets, and flicker-prevention behaviour.

### 5. Analytics redesign for cohort summaries

The user’s requested analytics are no longer only “how did the run do over time?”. They now also need “how did the population of concurrently running cars distribute performance?”. That means the analytics model must support per-car traces plus trainer-level cohort summaries in the same run.

The recommended default is to keep raw per-car records, then derive grouped cohort summaries in the metrics/export layer. Do not throw away raw per-car data just because the top/bottom quartile summaries are the main user-facing outputs.

- [ ] Discovery (bounded):
  - [ ] Read current `EpisodeRecord`, `EpisodeTrace`, `A2cUpdateRecord`, and the Markdown exporter.
  - [ ] Identify where singleton assumptions appear in trackers and derived metrics.
- [ ] Implementation playbook:
  - [ ] Extend analytics schemas with `env_id` and trainer-wide update context.
  - [ ] Record per-car episode summaries and per-car traces separately from trainer aggregate metrics.
  - [ ] Add cohort summary metrics for:
    - best car
    - worst car
    - top 25%
    - middle 50%
    - bottom 25%
    - total average
    - standard deviation / spread
    - error averages and deviance measures
  - [ ] Define “error averages” explicitly so they are not vague report prose:
    - mean centreline distance
    - mean absolute heading error
    - curvature-steering mismatch
    - understeer rate
    - crash-speed error context
  - [ ] Keep per-car raw traces available for later investigation when cohort summaries hide failure modes.
  - [ ] Update Markdown report structure so trainer-wide sections and per-car sections are clearly separated.
- [ ] Stop-and-verify checkpoints:
  - [ ] Exported JSON contains stable `env_id` tags for every per-car record.
  - [ ] Markdown report clearly shows requested cohort groupings.
  - [ ] Aggregate means and percentile splits reconcile with raw per-car data.
- [ ] Invariants / sanity checks:
  - [ ] Best/worst reported in analytics uses the same documented metric family as live trainer ranking, or the distinction is made explicit.
  - [ ] Quartile buckets are well-defined even when the number of completed cars/episodes is not divisible cleanly.
  - [ ] No per-car episode is recorded twice.
- [ ] Minimal explicit test requirements:
  - [ ] Unit tests for cohort bucketing and aggregate statistics.
  - [ ] At least one exporter test or golden-output check for best/worst and percentile sections.

### 6. HUD and overlay redesign for trainer-wide observability

The current HUD is a single-car driving diagnostics panel. In a 25-car trainer, that is no longer enough. The HUD must answer two different questions:

1. how is the trainer doing overall?
2. what is the currently highlighted best car doing right now?

The recommended default is a split HUD:

- trainer summary panel
- focused-car detail line for the current best car

- [ ] Discovery (bounded):
  - [ ] Read all current HUD assumptions about “the” car and “the” episode.
  - [ ] Decide which per-car details remain worth showing live.
- [ ] Implementation playbook:
  - [ ] Replace singleton car queries in HUD systems with trainer summary plus best-car focus queries.
  - [ ] Keep the current quarter-summary spirit, but adapt it to trainer-wide cohort progress.
  - [ ] Add trainer panel sections for:
    - total active cars
    - current best / worst env ids
    - mean and spread of progress/return
    - crash distribution
    - update cadence
    - latest A2C health
  - [ ] Add focused best-car detail line for:
    - current progress
    - offset
    - heading error
    - reward
    - current life duration
  - [ ] Update geometry/sensor overlays to target the best car by default, or allow explicit cycling later.
- [ ] Stop-and-verify checkpoints:
  - [ ] HUD remains readable with 25 cars on screen.
  - [ ] Best-car focus data updates correctly when leadership changes.
  - [ ] Overlay target follows the best car consistently.
- [ ] Invariants / sanity checks:
  - [ ] HUD must not compute its own competing trainer statistics if analytics/trainer resources already own them.
  - [ ] Focus-car overlay target must be derived from trainer ranking.
- [ ] Minimal explicit test requirements:
  - [ ] At least one small test for trainer assessment logic if the current heuristic is rewritten.

### 7. Compatibility cleanup and staged migration

This change touches almost every runtime layer. A direct big-bang rewrite is risky. The recommended execution strategy is staged migration with temporary compatibility shims only where they reduce risk, followed by explicit cleanup once the trainer path works.

- [ ] Discovery (bounded):
  - [ ] Mark which old singleton resources/systems can be removed immediately and which need temporary adapters.
- [ ] Implementation playbook:
  - [ ] Stage 1: introduce instance ids and multi-car spawn while keeping training disabled if needed.
  - [ ] Stage 2: migrate action/observation/episode state to per-car truth.
  - [ ] Stage 3: migrate A2C rollout and update logic.
  - [ ] Stage 4: migrate analytics and HUD.
  - [ ] Stage 5: remove singleton-only compatibility paths that are no longer useful.
- [ ] Stop-and-verify checkpoints:
  - [ ] Each migration stage compiles before the next begins.
  - [ ] No stale singleton path silently drives production behaviour at the end.
- [ ] Invariants / sanity checks:
  - [ ] Temporary shims must be clearly labelled and removed before completion.
- [ ] Minimal explicit test requirements:
  - [ ] Run `cargo check` and `cargo test` at each major stage.

## Integration Points

- [ ] Where it plugs into the existing pipeline:
  - `GamePlugin` startup must spawn 25 cars instead of one.
  - `AgentPlugin` systems must process all cars.
  - `BrainPlugin` keeps one shared `AgentMode`, but A2C systems now act over all cars.
  - `AnalyticsPlugin` must aggregate trainer-wide records from per-car data.
  - `DebugPlugin` must present trainer-wide summaries plus best-car focus.
- [ ] Order of execution and lifecycle placement:
  - startup spawns track once and cars many times
  - every fixed tick:
    - policy acts for all cars
    - smoothing applies per car
    - physics runs per car
    - collisions run per car
    - progress and episode logic run per car
    - observations rebuild per car
    - analytics capture per car
    - trainer reward collector appends per-car rewards and updates policy when batch horizon is met
  - update:
    - trainer ranking refreshes
    - visual highlight refreshes
    - HUD refreshes
  - last:
    - partial trainer rollout flushes
    - analytics export writes trainer and per-car summaries
- [ ] Pre-conditions:
  - all singleton-car assumptions are identified
  - ranking metric and analytic cohort semantics are defined before exporter work
  - visual opacity/highlight rules are documented before HUD/overlay rewrites
- [ ] Post-conditions:
  - all training cars share one policy
  - all training cars own independent environment truth
  - one trainer batch aggregates transitions across cars
  - exported analytics contain both raw per-car facts and cohort summaries
  - one best-performing current car is visually obvious at runtime

## Debugging / Verification

- [ ] Required logs, assertions, or inspection steps:
  - log trainer startup with `num_envs`
  - log per-update batch size and env-id distribution
  - assert rollout alignment across all fields
  - assert per-car reset only changes that car
  - log best-car changes with rank score and env id
  - inspect exported analytics for best/worst/quartile consistency
- [ ] Manual inspection steps:
  - visually confirm 25 cars are rendered
  - confirm non-best cars are semi-transparent
  - confirm the best car changes highlight when another car outperforms it
  - confirm multiple cars can crash/reset in the same general period without affecting each other
  - confirm trainer HUD shows mean plus spread, not just one-car stats
- [ ] Focused runtime signals to check:
  - update batch size should be much larger than singleton mode
  - explained variance and losses should remain finite
  - quartile progress gaps should be plausible, not all identical
  - best/worst car IDs should not flap every frame without underlying metric changes
- [ ] Common failure patterns:
  - leftover `single()` queries panic once more than one car exists
  - one car’s terminal reward resets or overwrites another car’s episode state
  - rollout buffer aligns by insertion order but loses `env_id` coherence
  - HUD or overlays accidentally follow an arbitrary first car instead of the best car
  - analytics bucket cars by entity order rather than actual performance
  - spawn offsets create unfair starts that dominate ranking noise

## Completion Criteria

- [ ] Functional correctness: 25 cars can run, learn, terminate, and reset independently under one shared A2C policy.
- [ ] Visual correctness: all cars are visible, non-best cars are reduced-opacity, and the best current car is clearly highlighted.
- [ ] Trainer correctness: rollout collection is synchronous and aggregate across cars, with one coherent update path.
- [ ] Analytics correctness: exports include per-car records plus best/worst/top-25%/mid-50%/bottom-25%/overall summaries with spread and error metrics.
- [ ] Integration correctness: singleton-only runtime assumptions are removed or isolated away from the trainer path.
- [ ] Tests passing: at minimum `cargo check`, `cargo test`, and targeted tests for rollout alignment, ranking, and cohort aggregation.
- [ ] Context updates completed: relevant system docs are updated to reflect the new vectorised trainer reality.
- [ ] File removal or archival condition: once the feature is implemented and documented in system files, archive or delete this plan so it does not become stale long-term memory.
