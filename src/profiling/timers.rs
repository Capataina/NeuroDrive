use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::Instant;

use bevy::prelude::*;
use serde::Serialize;

/// Identifies a set boundary timestamp. The `u8` discriminant is used as a
/// const generic parameter so Bevy treats each boundary as a distinct system.
#[repr(u8)]
pub enum SetBoundaryKind {
    InputEnd = 0,
    PhysicsEnd = 1,
    CollisionEnd = 2,
}

/// Per-system timing accumulator. Resets each frame.
///
/// Each system we want to profile gets a start/stop pair registered in the
/// `ProfilingPlugin`. The start system records `Instant::now()` into `starts`,
/// and the stop system computes the elapsed duration and stores it in
/// `durations_us`. At frame end, the completed durations are drained into the
/// `FrameRecord.system_timings` vector and the accumulator is reset.
#[derive(Resource, Default)]
pub struct SystemTimers {
    /// Start instants keyed by system name.
    starts: HashMap<&'static str, Instant>,
    /// Completed durations in microseconds keyed by system name.
    pub durations_us: HashMap<&'static str, u64>,
}

impl SystemTimers {
    pub fn start(&mut self, name: &'static str) {
        self.starts.insert(name, Instant::now());
    }

    pub fn stop(&mut self, name: &'static str) {
        if let Some(start) = self.starts.remove(name) {
            let elapsed = start.elapsed().as_micros() as u64;
            *self.durations_us.entry(name).or_insert(0) += elapsed;
        }
    }

    pub fn reset(&mut self) {
        self.starts.clear();
        self.durations_us.clear();
    }
}

/// Per-frame timing record. All durations are in microseconds (u64) to
/// avoid floating-point noise. `system_timings` uses a `Vec` of tuples
/// because `HashMap` is not `Serialize`-friendly and we do not need `Copy`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FrameRecord {
    /// Tick index since profiling started.
    pub tick: u64,

    /// Total fixed-tick time in microseconds.
    pub total_us: u64,

    /// Time spent in SimSet::Input (microseconds).
    pub input_us: u64,
    /// Time spent in SimSet::Physics (microseconds).
    pub physics_us: u64,
    /// Time spent in SimSet::Collision (microseconds).
    pub collision_us: u64,
    /// Time spent in SimSet::Measurement (microseconds).
    pub measurement_us: u64,

    /// Rollout buffer length at end of frame.
    pub rollout_buffer_len: u32,
    /// Number of episode traces held in memory.
    pub trace_count: u32,

    /// Per-system timings for this frame, keyed by human-readable name.
    pub system_timings: Vec<(String, u64)>,
}

/// Accumulates per-frame timing data in a fixed-capacity ring buffer.
#[derive(Resource)]
pub struct FrameTimings {
    /// Ring buffer of frame records.
    pub frames: VecDeque<FrameRecord>,
    /// Maximum ring buffer capacity.
    pub capacity: usize,
    /// Running tick counter.
    pub tick_counter: u64,

    // ── Scratch state for the current frame (not serialised) ──────────

    /// Instant captured at frame start.
    pub frame_start: Option<Instant>,
    /// Boundary timestamps: [InputEnd, PhysicsEnd, CollisionEnd].
    pub boundary_instants: [Option<Instant>; 3],
}

impl FrameTimings {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
            tick_counter: 0,
            frame_start: None,
            boundary_instants: [None; 3],
        }
    }

    /// Pushes a completed frame record, evicting the oldest if at capacity.
    pub fn push(&mut self, record: FrameRecord) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(record);
    }

    /// Resets per-frame scratch state for the next tick.
    pub fn reset_scratch(&mut self) {
        self.frame_start = None;
        self.boundary_instants = [None; 3];
    }
}
