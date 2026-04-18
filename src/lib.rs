//! NeuroDrive — library target.
//!
//! This library mirrors the binary's module tree (`src/main.rs`) as `pub`
//! modules so integration tests in `tests/*.rs` and external benchmarks can
//! import the crate's types. The binary remains the primary entry point;
//! this file exists solely to unblock test-writing against internal systems
//! (notably PPO hot paths and environment logic).
//!
//! See `context/plans/code-health-audit/cross-cutting.md` finding #1 for the
//! motivation.

pub mod agent;
pub mod analytics;
pub mod brain;
pub mod debug;
pub mod game;
pub mod maps;
#[cfg(feature = "profiling")]
pub mod profiling;
pub mod sim;
