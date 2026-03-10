use crate::analytics::models::{EpisodeTrace, NUM_PROGRESS_SECTORS};

#[derive(Clone, Copy, Debug, Default)]
pub struct SectorDiagnosticsRow {
    pub sector_index: usize,
    pub samples: usize,
    pub tick_share: f32,
    pub speed_mean: f32,
    pub heading_abs_mean_rad: f32,
    pub throttle_mean: f32,
    pub centerline_distance_mean: f32,
    pub terminal_count: usize,
    pub crash_terminal_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct SectorAccumulator {
    samples: usize,
    speed_sum: f32,
    heading_abs_sum: f32,
    throttle_sum: f32,
    centerline_distance_sum: f32,
    terminal_count: usize,
    crash_terminal_count: usize,
}

pub fn compute_sector_diagnostics(traces: &[EpisodeTrace]) -> Vec<SectorDiagnosticsRow> {
    if traces.is_empty() {
        return Vec::new();
    }

    let mut accumulators = vec![SectorAccumulator::default(); NUM_PROGRESS_SECTORS];
    let mut total_ticks = 0usize;

    for trace in traces {
        for tick in &trace.ticks {
            let sector = (tick.sector_index as usize).min(NUM_PROGRESS_SECTORS.saturating_sub(1));
            let accumulator = &mut accumulators[sector];
            accumulator.samples += 1;
            accumulator.speed_sum += tick.speed;
            accumulator.heading_abs_sum += tick.heading_error.abs();
            accumulator.throttle_sum += tick.throttle;
            accumulator.centerline_distance_sum += tick.centerline_distance;
            if tick.done {
                accumulator.terminal_count += 1;
                if tick.done_reason.as_deref() == Some("Crash") {
                    accumulator.crash_terminal_count += 1;
                }
            }
            total_ticks += 1;
        }
    }

    accumulators
        .into_iter()
        .enumerate()
        .filter(|(_, accumulator)| accumulator.samples > 0)
        .map(|(sector_index, accumulator)| {
            let count = accumulator.samples as f32;
            SectorDiagnosticsRow {
                sector_index,
                samples: accumulator.samples,
                tick_share: accumulator.samples as f32 / total_ticks.max(1) as f32,
                speed_mean: accumulator.speed_sum / count,
                heading_abs_mean_rad: accumulator.heading_abs_sum / count,
                throttle_mean: accumulator.throttle_sum / count,
                centerline_distance_mean: accumulator.centerline_distance_sum / count,
                terminal_count: accumulator.terminal_count,
                crash_terminal_count: accumulator.crash_terminal_count,
            }
        })
        .collect()
}
