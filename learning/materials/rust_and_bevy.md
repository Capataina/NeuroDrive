# Materials: Rust and Bevy

Resources for learning Rust programming and the Bevy game engine, focused on the patterns NeuroDrive uses: ownership-based memory safety, trait-based polymorphism, ECS architecture, and deterministic simulation loops.

---

## Rust Fundamentals

- [ ] **The Rust Programming Language ("The Rust Book") — Chapters 4, 5, 6, 10**
  Link: https://doc.rust-lang.org/book/
  Section: Chapter 4 "Understanding Ownership" (ownership, borrowing, lifetimes), Chapter 5 "Using Structs" (data modelling), Chapter 6 "Enums and Pattern Matching" (algebraic types, Option, match), Chapter 10 "Generic Types, Traits, and Lifetimes."
  Why: These four chapters cover the Rust concepts that appear on every page of NeuroDrive. Ownership and borrowing determine how data flows between Bevy systems. Traits are used for the `Brain` abstraction and layer interfaces. Enums model state machines (episode states, action types). Read these carefully — skimming will not suffice.
  Difficulty: Beginner–Intermediate | Time: 6–8 hours

- [ ] **Rustlings exercises**
  Link: https://github.com/rust-lang/rustlings
  Section: Complete at minimum: `move_semantics` (1–6), `structs` (1–3), `enums` (1–3), `traits` (1–5), `generics` (1–2), `options` (1–3), `error_handling` (1–6).
  Why: Hands-on practice with compiler errors. Rustlings teaches you to read the borrow checker's messages, which is essential when modifying NeuroDrive's systems that share data through Bevy resources and queries.
  Difficulty: Beginner–Intermediate | Time: 3–4 hours

- [ ] **Rust By Example — "Error Handling" and "Traits" sections**
  Link: https://doc.rust-lang.org/rust-by-example/
  Section: Chapter 15 "Traits" and Chapter 18 "Error handling" (Result, unwrap, ? operator).
  Why: Concise, example-driven coverage of patterns that appear throughout NeuroDrive. The trait section shows how to define and implement shared behaviour — compare directly to NeuroDrive's `Brain` trait.
  Difficulty: Beginner | Time: 1–2 hours

- [ ] **Jon Gjengset — "Crust of Rust: Lifetime Annotations"**
  Link: https://www.youtube.com/watch?v=rAl-9HwD858
  Section: Full video (1:42:00).
  Why: The clearest explanation of Rust lifetimes on the internet. While NeuroDrive avoids complex lifetime annotations (Bevy's ECS handles most ownership patterns), understanding lifetimes helps you read any Rust code confidently.
  Difficulty: Intermediate–Advanced | Time: 1.5 hours

---

## Bevy ECS

- [ ] **Official Bevy Book — Getting Started**
  Link: https://bevyengine.org/learn/book/introduction/
  Section: Full introductory guide. Focus on "ECS" concepts: entities, components, systems, and how Bevy's `App` builder composes them.
  Why: The official starting point. NeuroDrive is built as a set of Bevy plugins — understanding how `App::add_plugins`, `add_systems`, and system sets work is prerequisite to reading any NeuroDrive source file.
  Difficulty: Beginner–Intermediate | Time: 2–3 hours

- [ ] **Bevy Cheatbook — Core sections**
  Link: https://bevy-cheatbook.github.io/
  Section: "Bevy Programming Framework" chapter, specifically: "Intro to ECS," "Resources" (Res/ResMut), "Systems" (system parameters, ordering), "Plugins" (modular composition), "Schedules" (FixedUpdate, system sets, ordering constraints).
  Why: The most practical Bevy reference. NeuroDrive uses `Resource` extensively (for episode state, agent brains, analytics accumulators), and all game logic runs in `FixedUpdate` with explicit system ordering via `SystemSet`. The cheatbook explains these patterns with runnable examples.
  Difficulty: Intermediate | Time: 3–4 hours

- [ ] **Bevy Cheatbook — "System Order of Execution"**
  Link: https://bevy-cheatbook.github.io/programming/system-order.html
  Section: Full page, including system sets, `before`/`after` constraints, and run conditions.
  Why: NeuroDrive's determinism depends entirely on system execution order. The `SimSet` system set enforces a specific pipeline: physics → collision → progress → observation → reward → action. Understanding how Bevy guarantees this ordering is critical.
  Difficulty: Intermediate | Time: 1 hour

- [ ] **Official Bevy Examples — Repository**
  Link: https://github.com/bevyengine/bevy/tree/main/examples
  Section: Start with `ecs/` examples: `component.rs`, `system_param.rs`, `fixed_timestep.rs`, `state.rs`. Then look at `games/` for complete game patterns.
  Why: Runnable code demonstrating each Bevy concept in isolation. When NeuroDrive's usage of a Bevy feature is unclear, find the corresponding official example and study it.
  Difficulty: Intermediate | Time: 2–3 hours (selective reading)

---

## Game Physics and Simulation Patterns

- [ ] **Robert Nystrom — *Game Programming Patterns*, "Game Loop" chapter**
  Link: https://gameprogrammingpatterns.com/game-loop.html
  Section: Full chapter. Focus on the "Play catch up" pattern (fixed timestep with variable rendering) and the "Fixed time step, variable rendering" implementation.
  Why: NeuroDrive uses Bevy's `FixedUpdate` schedule, which implements exactly this pattern. Understanding the distinction between simulation time and wall-clock time explains why NeuroDrive can run headless at thousands of episodes per second.
  Difficulty: Beginner–Intermediate | Time: 30 minutes

- [ ] **Robert Nystrom — *Game Programming Patterns*, "Update Method" chapter**
  Link: https://gameprogrammingpatterns.com/update-method.html
  Section: Full chapter. Focus on how entities update themselves and the implications for determinism.
  Why: The update method pattern is the imperative-style precursor to ECS. Understanding what it solves (and its limitations) makes Bevy's ECS approach feel motivated rather than arbitrary. NeuroDrive's migration from a monolithic update to ordered ECS systems follows this exact evolution.
  Difficulty: Beginner–Intermediate | Time: 30 minutes

- [ ] **Robert Nystrom — *Game Programming Patterns*, "Component" chapter**
  Link: https://gameprogrammingpatterns.com/component.html
  Section: Full chapter.
  Why: This chapter explains the component pattern — the "C" in ECS — using game development examples. It motivates composition over inheritance, which is exactly how NeuroDrive structures its car entity (Transform + Physics + Sensors + Progress as separate components).
  Difficulty: Beginner | Time: 30 minutes

- [ ] **Glenn Fiedler — "Fix Your Timestep!"**
  Link: https://gafferongames.com/post/fix_your_timestep/
  Section: Full post. Focus on the final "The Final Touch" implementation.
  Why: The definitive article on fixed-timestep simulation. NeuroDrive's deterministic replay system relies on this principle: identical inputs at identical fixed timesteps must produce identical outputs. Fiedler explains why variable timesteps make this impossible.
  Difficulty: Intermediate | Time: 30 minutes

- [ ] **Catherine West — "RustConf 2018: Using Rust for Game Development" (talk)**
  Link: https://www.youtube.com/watch?v=aKLntZcp27M
  Section: Full talk (40 min). Focus on the ECS section starting at ~15:00.
  Why: The most influential talk on ECS in Rust. West explains why inheritance hierarchies fail for game entities and how ECS solves the expression problem. Directly motivates Bevy's design choices and NeuroDrive's entity structure.
  Difficulty: Intermediate | Time: 40 minutes
