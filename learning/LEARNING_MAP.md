# Learning Map

`learning/` is the project's teaching layer. It explains NeuroDrive from first principles, then connects those ideas back to the current Rust/Bevy implementation, the maintained `context/` memory, and the long-term project direction in `README.md`.

Use this folder differently from `context/`:

- `context/` is the implementation-facing memory layer.
- `learning/` is the learner-facing explanation and practice layer.

This rebuild starts from current project reality:

- a deterministic 2D racing environment is implemented,
- a handwritten A2C baseline is live,
- analytics and debug tooling are already part of the runtime,
- the long-term direction is still biological local plasticity rather than “make A2C the permanent centre”.

## Start Here

- If you want to understand the shape of the folder: see `DIRECTORY_TREE.md`
- If you want route guidance: see `STUDY_GUIDE.md`
- If you want a focused study sequence: see `paths/PATH_INDEX.md`
- If you want project-specific practice: see `exercises/EXERCISE_ORDER.md`
- If you want terminology first: see `GLOSSARY.md`

## How To Use This Folder

- Start with a path if you want guided study.
- Jump straight into `project/` if you already know the domain and want NeuroDrive-specific understanding.
- Use `concepts/` to fill foundation gaps when a project file assumes knowledge you do not yet have.
- Use `materials/` for topic-grouped external study prompts and reading suggestions.

## Progress Tracking

Checkboxes live mainly in:

- `STUDY_GUIDE.md`
- `paths/*.md`
- `exercises/EXERCISE_ORDER.md`

Concept files, system files, and glossary entries are explanation surfaces, not progress trackers.
