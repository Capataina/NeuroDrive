//! Simulation scheduling primitives.
//!
//! This module defines shared system sets for the fixed-timestep simulation
//! pipeline, keeping ordering explicit without creating cross-module
//! dependencies (e.g. agent code depending on game code).

pub mod sets;

