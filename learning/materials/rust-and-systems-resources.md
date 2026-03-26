# Rust and Systems Resources

A curated set of resources for studying Rust and the systems concepts used in NeuroDrive. Organised by topic.

---

## Rust Language

### The Rust Book (official)

The primary Rust learning resource. Free online at doc.rust-lang.org/book.

**Chapters directly relevant to NeuroDrive:**
- Chapter 4 (Ownership) — understanding why Rust is memory-safe without GC
- Chapter 6 (Enums and Pattern Matching) — used throughout NeuroDrive (AgentMode, SimSet, etc.)
- Chapter 13 (Iterators and Closures) — idiomatic Rust iteration
- Chapter 15 (Smart Pointers) — relevant for understanding Bevy's resource and component ownership
- Chapter 17 (Object-Oriented Patterns) — trait objects, the `Brain` trait in NeuroDrive

### Rust by Example (official)

Companion to The Rust Book. Useful for looking up specific syntax patterns quickly.

### Programming Rust — Blandy, Orendorff & Tindall (3rd ed., 2021)

A more comprehensive Rust reference than the Rust Book. Particularly useful for:
- Traits and generics in depth
- Lifetimes and borrow checker edge cases
- Closures and iterators
- Writing idiomatic, production-quality Rust

---

## Bevy ECS

### Bevy Book (official)

The official Bevy learning resource at bevyengine.org/learn/book.

**Sections relevant to NeuroDrive:**
- ECS Intro — entities, components, resources, systems
- App (plugins, schedules) — how NeuroDrive's plugin architecture is registered
- Events — how `CollisionEvent` works
- Time — `Time<Fixed>` and fixed timestep (core to NeuroDrive's 60 Hz loop)

### Bevy Cheatbook (unofficial, bevy-cheatbook.github.io)

A practical reference for Bevy patterns and idioms. More immediately useful than the official book for:
- System ordering constraints (`before`, `after`, `.chain()`)
- Bevy schedules (`FixedUpdate`, `Update`, `Last`)
- Working with resources and components in systems
- Event reading and writing

Covers Bevy 0.14–0.18. NeuroDrive uses **Bevy 0.18** — verify version compatibility for any patterns you look up.

### Bevy 0.18 Release Notes and Migration Guide

Essential when upgrading Bevy versions or when reading older Bevy examples. The schedule API, event API, and plugin API all changed significantly between Bevy 0.12 and 0.18.

---

## Numerical Computing in Rust

### ndarray crate documentation

NeuroDrive does not use ndarray, but if you want to understand n-dimensional array operations in Rust (for comparison with the handwritten matrix operations), the ndarray crate is the standard choice. Understanding it helps clarify why NeuroDrive's handwritten approach stores weights as `Vec<f32>` with index arithmetic rather than using a matrix type.

### Fixed-Point and Float Arithmetic

For understanding the numerical stability properties of NeuroDrive's handwritten operations:
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (Goldberg 1991) — canonical reference for floating-point behaviour

Relevant to NeuroDrive: the `log(1 - tanh²(x))` Jacobian correction near saturation is a floating-point precision issue, not just a mathematical one. Near `|x| = 3`, `tanh(x) ≈ 0.995` and `1 - tanh²(x) ≈ 0.01`. The logarithm is approximately `-4.6`. This is representable, but in production implementations, the numerically stable form `log(1 - tanh²(x)) = log(4) - 2 * log(exp(|x|) + exp(-|x|)) + 2*|x|` avoids catastrophic cancellation. NeuroDrive uses the straightforward form; the stable form is mentioned here for awareness.

---

## Algorithms in Rust

### The Algorithms (GitHub: the-algorithms/rust)

A reference implementations repository. Useful for checking Rust idioms for common algorithms — sorting, searching, mathematical operations.

Not directly relevant to NeuroDrive, but useful background.

---

## Performance and Profiling

NeuroDrive runs at 60 Hz with a full forward pass, trace update (Milestone 2+), and analytics capture on every tick. Performance may matter in the later milestones.

### cargo flamegraph

A profiling tool for Rust programs. Produces flamegraph visualisations of CPU time. Useful for identifying hot paths in the fixed-tick loop.

Installation: `cargo install flamegraph`

### criterion crate

The standard Rust benchmarking crate. Used for microbenchmarks of specific functions. Useful for benchmarking the neural network forward pass, GAE computation, or eligibility trace update.

---

## Topics by NeuroDrive Relevance

| Topic | Resource |
|---|---|
| Understanding Bevy plugins and systems | Bevy Book (App section), Bevy Cheatbook |
| Fixed timestep and deterministic simulation | Bevy Book (Time section) |
| ECS ownership model (why `Res<>`, `ResMut<>`) | Rust Book Chapter 4, Bevy Book (ECS) |
| Trait objects and the Brain trait | Rust Book Chapter 17 |
| Pattern matching on enums (AgentMode, SimSet) | Rust Book Chapter 6 |
| Vec<f32> as matrix storage | Rust standard library docs, Programming Rust |
| Floating-point correctness | Goldberg 1991 |
| Profiling the fixed tick loop | cargo flamegraph |

---

## Minimum Viable Rust for NeuroDrive

If you are new to Rust but want to read the NeuroDrive source code without extensive Rust study first, focus on:

1. **Rust Book Chapters 4, 6, 10** — ownership, enums/match, generics and traits
2. **Bevy Cheatbook: ECS and System sections** — how Bevy's query/resource system works
3. **Bevy Cheatbook: Schedules section** — understanding FixedUpdate and SimSet

These three areas cover the majority of what makes NeuroDrive Bevy code look different from general Rust.
