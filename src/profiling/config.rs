use bevy::prelude::*;

/// Controls what the profiling system captures and how long it runs.
///
/// All fields can be adjusted before the plugin runs. The defaults
/// provide a good baseline for a quick profiling pass.
#[derive(Resource, Clone, Debug)]
pub struct ProfilingConfig {
    /// How many seconds to run before auto-exiting.
    pub duration_seconds: f32,

    /// Number of frames to keep in the ring buffer.
    /// At 60 Hz, 600 frames = 10 seconds of history.
    pub ring_buffer_frames: usize,

    // ── Category toggles ──────────────────────────────────────────────

    /// Track per-SimSet timing breakdown each fixed tick.
    pub track_set_timings: bool,

    /// Track rollout buffer and trace accumulator sizes.
    pub track_buffer_sizes: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            duration_seconds: 30.0,
            ring_buffer_frames: 1800, // 30s at 60 Hz — full run history
            track_set_timings: true,
            track_buffer_sizes: true,
        }
    }
}

impl ProfilingConfig {
    /// Number of fixed ticks for the configured duration at 60 Hz.
    pub fn duration_ticks(&self) -> u64 {
        (self.duration_seconds * 60.0).ceil() as u64
    }
}
