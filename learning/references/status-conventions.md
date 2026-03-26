# Status Conventions

All content files in this archive use a **Status** line near the top to communicate whether the material describes current reality, planned future work, or foundational theory. Use this guide to interpret status labels correctly.

---

## Status Labels

### `Current implementation`

The material describes something that exists in the codebase right now. The code matches the description. If you read the relevant source files, you will find the systems, structures, or behaviours described.

Example files with this status:
- `project/systems/environment-system.md`
- `project/architecture/fixed-tick-pipeline.md`
- `project/systems/a2c-brain.md`

**When reading these files:** Trust the description as reflecting current reality. If you find a discrepancy between the file and the code, the code is authoritative — the file may have drifted.

---

### `Foundational domain knowledge`

The material covers theory that the project depends on conceptually but that is not directly "implemented" in any file. Mathematical foundations, neuroscience concepts, and RL theory fall into this category.

Example files with this status:
- `concepts/foundations/neural-networks.md`
- `concepts/core/reinforcement-learning.md`
- `concepts/advanced/hebbian-plasticity.md`

**When reading these files:** The material is durable — it will remain relevant regardless of how the project evolves. These files do not describe code; they describe the ideas that inform the code.

---

### `Planned for Milestone N`

The material describes something that is intended to exist in the future but is not yet implemented. These files are useful for understanding where the project is going and the theory you will need when that milestone arrives.

Example files with this status:
- `concepts/advanced/structural-plasticity.md` (Milestone 5)
- `concepts/advanced/spike-timing-dependent-plasticity.md` (Milestone 4)

**When reading these files:** The design and theory are described as if the system exists, to make the material concrete and learnable. But do not expect to find the described code in `src/brain/biological/` — that directory is currently an empty placeholder.

---

### `Future direction`

The material describes the project's long-term architectural evolution — the transition from the current implementation to the eventual biological brain. These files are directional rather than specifying a concrete milestone.

Example files with this status:
- `project/evolution/from-baseline-to-brain.md`
- `project/evolution/milestone-roadmap.md`

**When reading these files:** Treat the content as the intended direction rather than a committed specification. Some details will change as the milestones are actually implemented.

---

### `Implemented decision`

The material explains a specific design choice that was made and is reflected in the current codebase. Unlike "current implementation" (which describes what a system does), "implemented decision" files explain *why* it was done that way.

Example files with this status:
- `project/decisions/a2c-as-baseline.md`
- `project/decisions/tanh-squashed-actions.md`

---

## Checking Currency

All learning archive files were generated at a specific point in the project's history. The code evolves faster than the documentation. When in doubt:

1. Read the file's status label.
2. If the label is `Current implementation`, open the referenced source file and verify.
3. If the label is `Foundational domain knowledge`, the material is theory and does not require code verification.
4. If the label is `Planned for Milestone N`, check the milestone checklist in `README.md` to see if that milestone has since been completed.

The `context/` folder is the canonical implementation-facing memory and will be more up-to-date than `learning/` for fast-moving implementation details. Use `context/` to resolve discrepancies.
