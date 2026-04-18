mod agent;
mod analytics;
mod brain;
mod debug;
mod game;
mod maps;
#[cfg(feature = "profiling")]
mod profiling;
mod sim;

use agent::AgentPlugin;
use analytics::plugin::AnalyticsPlugin;
use bevy::prelude::*;
use bevy::time::Fixed;
use brain::plugin::BrainPlugin;
use debug::DebugPlugin;
use game::GamePlugin;
use maps::MonacoPlugin;

fn main() {
    // Pin Apple Accelerate to single-threaded before any cblas_sgemm call.
    // Accelerate's default is to spin up worker threads for larger matrices,
    // but our GEMMs are small enough (critic fc2 is 64×128×128) that thread
    // spawn overhead dominates the useful work AND the worker threads
    // compete with Bevy's render pipeline for CPU cores — a net loss.
    //
    // Must be set before the first Accelerate call. `App::new()` triggers
    // bevy plugin registration which may indirectly warm up system caches;
    // set it first to be safe. No-op on non-macOS builds.
    #[cfg(target_os = "macos")]
    // SAFETY: Rust 2024 marks env::set_var as unsafe because it is racy
    // against concurrent reads. We call it here at process start, before any
    // thread has been spawned, so there is no concurrent reader.
    unsafe {
        std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    }

    let mut app = App::new();

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "NeuroDrive".to_string(),
            resolution: (1600, 900).into(),
            ..default()
        }),
        ..default()
    }))
    // Fixed timestep: required for determinism, replay, and stable metrics.
    .insert_resource(Time::<Fixed>::from_hz(60.0))
    // Track must be spawned before game systems query it
    .add_plugins(MonacoPlugin)
    .add_plugins(AgentPlugin)
    .add_plugins(BrainPlugin)
    .add_plugins(AnalyticsPlugin)
    .add_plugins(GamePlugin)
    .add_plugins(DebugPlugin);

    #[cfg(feature = "profiling")]
    app.add_plugins(profiling::ProfilingPlugin);

    app.run();
}
