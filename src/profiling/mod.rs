pub mod capture;
pub mod config;
pub mod exporters;
pub mod timers;

use bevy::prelude::*;

use crate::sim::sets::SimSet;

use self::capture::{
    auto_exit_system, collision_end_system, frame_end_system, frame_start_system,
    input_end_system, physics_end_system, start_timer, stop_timer,
};
use self::config::ProfilingConfig;
use self::exporters::on_exit_export_system;
use self::timers::{FrameTimings, SystemTimers};

// ── System imports for per-system timing instrumentation ─────────────────

use crate::agent::action::{action_smoothing_system, keyboard_action_input_system};
use crate::agent::observation::{build_observation_vector_system, update_sensor_readings_system};
use crate::analytics::trackers::action::{
    capture_episode_action_stats_system, snapshot_completed_episode_action_stats_system,
};
use crate::analytics::trackers::trace::{
    capture_episode_tick_trace_system, snapshot_completed_episode_trace_system,
};
use crate::brain::ppo::{
    ppo_act_all_cars_system, ppo_collect_rewards_all_cars_system, ppo_epoch_system,
};
use crate::debug::hud::{
    capture_driving_hud_episode_metrics_system, update_driving_hud_stats_system,
};
use crate::game::collision::collision_detection_system;
use crate::game::episode::episode_loop_system;
use crate::game::physics::car_physics_system;
use crate::game::progress::update_track_progress_system;

/// Performance profiling plugin. Only compiled when `--features profiling` is active.
///
/// Instruments the fixed-tick pipeline with lightweight timing guards,
/// accumulates per-frame metrics into a ring buffer, auto-exits after a
/// configurable duration, and exports a performance report to `reports/`.
///
/// Per-system timing works by registering a `start_timer` system `.before()`
/// each target and a `stop_timer` system `.after()` it, both in the same
/// `SimSet`. The `SystemTimers` resource accumulates microsecond durations
/// which are drained into each `FrameRecord` at frame end.
pub struct ProfilingPlugin;

/// Convenience macro that registers a start/stop timer pair around a target
/// system in a given `SimSet`. Each expansion produces two `add_systems` calls.
macro_rules! instrument {
    ($app:expr, $name:literal, $target:expr, $set:expr) => {
        $app.add_systems(
            FixedUpdate,
            start_timer($name).before($target).in_set($set),
        );
        $app.add_systems(
            FixedUpdate,
            stop_timer($name).after($target).in_set($set),
        );
    };
}

impl Plugin for ProfilingPlugin {
    fn build(&self, app: &mut App) {
        let config = ProfilingConfig::default();
        let timings = FrameTimings::new(config.ring_buffer_frames);

        info!(
            "Profiling enabled — will auto-exit after {:.0}s ({} ticks at 60 Hz).",
            config.duration_seconds,
            config.duration_ticks(),
        );

        app.insert_resource(config)
            .insert_resource(timings)
            .init_resource::<SystemTimers>()
            .add_systems(
                FixedUpdate,
                (
                    // Frame start: capture Instant::now() before anything else.
                    frame_start_system.before(SimSet::Input),
                    // Set boundary markers — capture timestamps at each set transition.
                    input_end_system
                        .after(SimSet::Input)
                        .before(SimSet::Physics),
                    physics_end_system
                        .after(SimSet::Physics)
                        .before(SimSet::Collision),
                    collision_end_system
                        .after(SimSet::Collision)
                        .before(SimSet::Measurement),
                    // Frame end: compute deltas and push the completed frame record.
                    frame_end_system.after(SimSet::Measurement),
                    // Auto-exit countdown.
                    auto_exit_system.after(frame_end_system),
                ),
            )
            .add_systems(Last, on_exit_export_system);

        // ── Per-system timing instrumentation ────────────────────────────
        //
        // Each macro invocation registers a start_timer before and stop_timer
        // after the target system, both constrained to the same SimSet so
        // Bevy's scheduler places them correctly.

        // SimSet::Input
        instrument!(app, "Keyboard Input", keyboard_action_input_system, SimSet::Input);
        instrument!(app, "PPO Action Selection", ppo_act_all_cars_system, SimSet::Input);
        instrument!(app, "Action Smoothing", action_smoothing_system, SimSet::Input);

        // SimSet::Physics
        instrument!(app, "Car Physics", car_physics_system, SimSet::Physics);
        instrument!(app, "Action Stats Capture", capture_episode_action_stats_system, SimSet::Physics);

        // SimSet::Collision
        instrument!(app, "Collision Detection", collision_detection_system, SimSet::Collision);

        // SimSet::Measurement
        instrument!(app, "Track Progress", update_track_progress_system, SimSet::Measurement);
        instrument!(app, "Episode Loop (Reward + Reset)", episode_loop_system, SimSet::Measurement);
        instrument!(app, "Sensor Raycasting", update_sensor_readings_system, SimSet::Measurement);
        instrument!(app, "Observation Vector", build_observation_vector_system, SimSet::Measurement);
        instrument!(app, "Trace Capture", capture_episode_tick_trace_system, SimSet::Measurement);
        instrument!(app, "Trace Snapshot", snapshot_completed_episode_trace_system, SimSet::Measurement);
        instrument!(app, "Action Stats Snapshot", snapshot_completed_episode_action_stats_system, SimSet::Measurement);
        instrument!(app, "PPO Reward Collection", ppo_collect_rewards_all_cars_system, SimSet::Measurement);
        instrument!(app, "PPO Epoch (Training)", ppo_epoch_system, SimSet::Measurement);
        instrument!(app, "HUD Stats", update_driving_hud_stats_system, SimSet::Measurement);
        instrument!(app, "HUD Episode Capture", capture_driving_hud_episode_metrics_system, SimSet::Measurement);
    }
}
