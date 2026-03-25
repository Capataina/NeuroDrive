# ECS Plugin Scheduling

## Why This Matters Here

NeuroDrive uses Bevy ECS. Understanding the repo means understanding that “what happens” depends on both data ownership and schedule placement.

## Core Idea

In an ECS app, behaviour is not just inside one call stack. Systems read and write shared world state across schedules. Plugins and named system sets are how you keep that behaviour coherent.

## Build-Up

### Step 1: Plugins define subsystem boundaries

Each major NeuroDrive subsystem has a plugin:

- `AgentPlugin`
- `BrainPlugin`
- `GamePlugin`
- `AnalyticsPlugin`
- `DebugPlugin`

### Step 2: Named sets define broad ordering

Instead of relying on accidental registration order, NeuroDrive uses named fixed-update sets.

### Step 3: Local `.after()` and `.before()` express tighter dependencies

Some systems need extra precision beyond the broad set ordering. A2C reward collection and analytics trace capture are examples.

## Worked Examples

### Example 1: Action smoothing

The learner writes `desired` action before smoothing. Physics then reads the `applied` action after smoothing has run.

### Example 2: Analytics capture

Trace capture runs after observation rebuild and episode truth, but before A2C reward collection, so the exported trace sits at a specific point in the tick lifecycle.

### Example 3: Mode toggling

`Update` handles F4 mode toggling separately from the fixed simulation tick, because it is UI/input control rather than physics truth.

## How This Appears In The Project

- `src/main.rs` wires plugins.
- `src/sim/sets.rs` defines the cross-plugin fixed-update contract.
- `src/game/plugin.rs`, `src/agent/plugin.rs`, `src/brain/plugin.rs`, `src/analytics/plugin.rs`, and `src/debug/plugin.rs` each contribute systems into that contract.

## Common Misunderstandings

❌ “Plugins are just folders.”
✅ In Bevy, plugins are runtime wiring units that decide resources, systems, and schedules.

❌ “If two systems are in the same schedule they are safely ordered.”
✅ Only if you make that ordering explicit where it matters.

## Terms Used Here

- ECS
- plugin
- schedule
- system set
- ordering contract
