use std::time::Instant;

use bevy::app::AppExit;
use bevy::prelude::*;

use crate::profiling::config::ProfilingConfig;
use crate::profiling::timers::{FrameRecord, FrameTimings, SetBoundaryKind, SystemTimers};

/// Creates a system that starts a timer for the named system.
pub fn start_timer(name: &'static str) -> impl Fn(ResMut<SystemTimers>) {
    move |mut timers: ResMut<SystemTimers>| {
        timers.start(name);
    }
}

/// Creates a system that stops a timer for the named system.
pub fn stop_timer(name: &'static str) -> impl Fn(ResMut<SystemTimers>) {
    move |mut timers: ResMut<SystemTimers>| {
        timers.stop(name);
    }
}

/// Captures `Instant::now()` at the very start of each fixed tick.
pub fn frame_start_system(
    mut timings: ResMut<FrameTimings>,
    mut sys_timers: ResMut<SystemTimers>,
) {
    timings.reset_scratch();
    sys_timers.reset();
    timings.frame_start = Some(Instant::now());
}

/// Records a boundary timestamp for the end of SimSet::Input.
pub fn input_end_system(mut timings: ResMut<FrameTimings>) {
    timings.boundary_instants[SetBoundaryKind::InputEnd as usize] = Some(Instant::now());
}

/// Records a boundary timestamp for the end of SimSet::Physics.
pub fn physics_end_system(mut timings: ResMut<FrameTimings>) {
    timings.boundary_instants[SetBoundaryKind::PhysicsEnd as usize] = Some(Instant::now());
}

/// Records a boundary timestamp for the end of SimSet::Collision.
pub fn collision_end_system(mut timings: ResMut<FrameTimings>) {
    timings.boundary_instants[SetBoundaryKind::CollisionEnd as usize] = Some(Instant::now());
}

/// Computes per-set durations, drains per-system timings, and pushes the
/// completed FrameRecord.
pub fn frame_end_system(
    config: Res<ProfilingConfig>,
    mut timings: ResMut<FrameTimings>,
    mut sys_timers: ResMut<SystemTimers>,
    rollout_buffer: Option<Res<crate::brain::ppo::buffer::TrainerRolloutBuffer>>,
    tracker: Option<Res<crate::analytics::models::EpisodeTracker>>,
) {
    let Some(frame_start) = timings.frame_start else {
        return;
    };

    let frame_end = Instant::now();
    let total_us = frame_end.duration_since(frame_start).as_micros() as u64;

    let mut input_us = 0u64;
    let mut physics_us = 0u64;
    let mut collision_us = 0u64;
    let measurement_us;

    if config.track_set_timings {
        let input_end = timings.boundary_instants[SetBoundaryKind::InputEnd as usize];
        let physics_end = timings.boundary_instants[SetBoundaryKind::PhysicsEnd as usize];
        let collision_end = timings.boundary_instants[SetBoundaryKind::CollisionEnd as usize];

        if let Some(ie) = input_end {
            input_us = ie.duration_since(frame_start).as_micros() as u64;
        }
        if let (Some(ie), Some(pe)) = (input_end, physics_end) {
            physics_us = pe.duration_since(ie).as_micros() as u64;
        }
        if let (Some(pe), Some(ce)) = (physics_end, collision_end) {
            collision_us = ce.duration_since(pe).as_micros() as u64;
        }
        let measurement_start = collision_end.or(physics_end).or(input_end).unwrap_or(frame_start);
        measurement_us = frame_end.duration_since(measurement_start).as_micros() as u64;
    } else {
        measurement_us = 0;
    }

    let rollout_buffer_len = rollout_buffer
        .as_ref()
        .map(|b| b.len() as u32)
        .unwrap_or(0);
    let trace_count = tracker
        .as_ref()
        .map(|t| t.episode_traces.len() as u32)
        .unwrap_or(0);

    // Drain per-system timings into a sorted Vec for deterministic output order.
    let mut system_timings: Vec<(String, u64)> = sys_timers
        .durations_us
        .drain()
        .map(|(name, us)| (name.to_string(), us))
        .collect();
    system_timings.sort_by(|a, b| a.0.cmp(&b.0));

    let tick = timings.tick_counter;
    timings.tick_counter += 1;

    timings.push(FrameRecord {
        tick,
        total_us,
        input_us,
        physics_us,
        collision_us,
        measurement_us,
        rollout_buffer_len,
        trace_count,
        system_timings,
    });
}

/// Counts fixed ticks and sends `AppExit` when the profiling duration is reached.
pub fn auto_exit_system(
    config: Res<ProfilingConfig>,
    timings: Res<FrameTimings>,
    mut exit_writer: bevy::ecs::message::MessageWriter<AppExit>,
) {
    if timings.tick_counter >= config.duration_ticks() {
        info!(
            "Profiling complete — {} ticks captured. Exiting.",
            timings.tick_counter
        );
        exit_writer.write(AppExit::Success);
    }
}
