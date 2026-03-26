# Learning Map

## What This Archive Is

`learning/` is the educational archive for the NeuroDrive project. Its job is to teach — not document in the narrow implementation-memory sense, but to teach thoroughly enough that a motivated engineer can understand what the project is, why it is built the way it is, how it works right now, and where it is going.

This archive is deliberately large. NeuroDrive sits at the intersection of several demanding intellectual domains: reinforcement learning, computational neuroscience, real-time simulation, and systems programming in Rust. None of those domains can be summarised in a bullet list without losing the understanding that actually matters. The archive reflects that.

## How This Differs from `context/`

`context/` is implementation memory. It answers questions like: what does this module own, what is its current state, what are the known risks? It is written for an engineer who already understands the project and needs to recall or update a specific system.

`learning/` is teaching material. It answers questions like: what is GAE and why does it matter here, how does eligibility-trace learning work at the synapse level, why does this architecture look the way it does, and what would the project look like if Milestone 2 were complete? It does not assume existing familiarity. It builds understanding from first principles.

Both sources are valuable. `context/` is authoritative for current implementation facts. `learning/` is authoritative for explanatory depth.

## Coverage Areas

The archive covers five major learning surfaces:

1. **Foundations** — neural network mechanics, optimisation, probability, the Bevy ECS model. These are the prerequisites needed to understand the current A2C implementation at a mathematical and structural level.

2. **Core RL concepts** — reinforcement learning from first principles, policy gradients, advantage estimation, actor-critic architectures, continuous control. These are the domain concepts that the current A2C baseline is built on.

3. **Biological learning concepts** — Hebbian plasticity, STDP, eligibility traces, neuromodulation, structural plasticity, continual learning. These are the concepts that define the project's long-term research direction and everything Milestones 2–9 are building toward.

4. **Project systems** — deep dives into each runtime subsystem, the architecture as a whole, key decisions, comparisons, and the evolutionary trajectory from the current baseline to the intended brain-inspired architecture.

5. **Practice** — exercises that build mastery through reconstruction, debugging, extension, and design reasoning grounded in the real project.

## Status Labels

Files and sections in this archive use status labels where there is a risk of confusion between present reality and future direction:

- **Current** — this exists in the live runtime.
- **Foundational** — domain knowledge required to understand the project, not specific to NeuroDrive's implementation.
- **Planned** — described in the README roadmap but not yet implemented in code.
- **Historical** — superseded in the current implementation but still educationally valuable.

If a file contains no status label, its content applies to the current implementation unless stated otherwise.

## How to Navigate

There are multiple entry points depending on your goal:

| Goal | Start here |
|---|---|
| Understand what the project is and where it is going | `STUDY_GUIDE.md` |
| See the full archive structure at a glance | `DIRECTORY_TREE.md` |
| Follow a focused study route | `paths/PATH_INDEX.md` |
| Practise with exercises | `exercises/EXERCISE_ORDER.md` |
| Look up a term | `GLOSSARY.md` |
| Find reading material | `materials/` |
| Quick notation or status reference | `references/` |

## Where Progress Is Tracked

Learner progress checkboxes live in:

- `STUDY_GUIDE.md` — route selection
- `paths/*.md` — step-by-step sequence within each path
- `exercises/EXERCISE_ORDER.md` — recommended practice sequence

Concept files, system deep-dives, and glossary entries intentionally do not contain progress checkboxes — they are reference material, not linear curricula.

## A Note on Scale

This archive is long because the project is complex. Do not feel you must read it sequentially from beginning to end. Use `STUDY_GUIDE.md` to find a route that matches your goal, and follow it. Most paths are designed to take you from a standing start to genuine understanding of a specific aspect of the project, and they point outward to related areas when you are ready to go further.
