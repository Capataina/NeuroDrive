use crate::analytics::models::{EpisodeTrace, NUM_PROGRESS_SECTORS};

#[derive(Clone, Copy, Debug, Default)]
pub struct SectorDiagnosticsRow {
    pub sector_index: usize,
    pub speed_mean: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct SectorAccumulator {
    samples: usize,
    speed_sum: f32,
}

pub fn compute_sector_diagnostics(traces: &[EpisodeTrace]) -> Vec<SectorDiagnosticsRow> {
    if traces.is_empty() {
        return Vec::new();
    }

    let mut accumulators = vec![SectorAccumulator::default(); NUM_PROGRESS_SECTORS];

    for trace in traces {
        for tick in &trace.ticks {
            let sector = (tick.sector_index as usize).min(NUM_PROGRESS_SECTORS.saturating_sub(1));
            let accumulator = &mut accumulators[sector];
            accumulator.samples += 1;
            accumulator.speed_sum += tick.speed;
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
                speed_mean: accumulator.speed_sum / count,
            }
        })
        .collect()
}
