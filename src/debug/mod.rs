//! Debug and observability tools.
//!
//! This module is intentionally isolated from core simulation logic so that
//! debug overlays and instrumentation cannot accidentally become dependencies
//! of the environment or agent interfaces.

pub mod hud;
pub mod overlays;
pub mod plugin;

pub use plugin::DebugPlugin;
