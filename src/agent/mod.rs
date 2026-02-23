//! Agent-facing interfaces.
//!
//! This module defines stable, normalised input/output surfaces for whatever
//! agent is driving the car (manual keyboard, scripted controller, replay, or a
//! learned policy).
//!
//! Milestone 0 focus:
//! - A stable action interface (`steering`, `throttle`) that is updated on the
//!   fixed simulation tick.

pub mod action;
pub mod observation;
pub mod plugin;

pub use plugin::AgentPlugin;
