use crate::analytics::metrics::turns::CURVATURE_DEMAND_THRESHOLD;
use crate::analytics::models::{EpisodeTrace, NUM_PROGRESS_SECTORS, TickTraceRecord};

#[derive(Clone, Copy, Debug, Default)]
pub struct CriticStats {
    pub count: usize,
    target_sum: f32,
    target_sumsq: f32,
    error_sum: f32,
    error_sumsq: f32,
    abs_error_sum: f32,
}

impl CriticStats {
    pub fn push(&mut self, target: f32, prediction: f32) {
        let error = target - prediction;
        self.count += 1;
        self.target_sum += target;
        self.target_sumsq += target * target;
        self.error_sum += error;
        self.error_sumsq += error * error;
        self.abs_error_sum += error.abs();
    }

    pub fn mae(self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.abs_error_sum / self.count as f32
        }
    }

    pub fn bias(self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.error_sum / self.count as f32
        }
    }

    pub fn explained_variance(self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f32;
        let target_mean = self.target_sum / n;
        let error_mean = self.error_sum / n;
        let target_var = (self.target_sumsq / n) - target_mean * target_mean;
        let error_var = (self.error_sumsq / n) - error_mean * error_mean;
        if target_var <= 1e-8 {
            0.0
        } else {
            1.0 - (error_var / target_var)
        }
    }
}

#[derive(Clone, Debug)]
pub struct CriticDiagnostics {
    pub straight: CriticStats,
    pub high_curvature: CriticStats,
    pub terminal: CriticStats,
    pub by_sector: Vec<CriticStats>,
}

impl Default for CriticDiagnostics {
    fn default() -> Self {
        Self {
            straight: CriticStats::default(),
            high_curvature: CriticStats::default(),
            terminal: CriticStats::default(),
            by_sector: vec![CriticStats::default(); NUM_PROGRESS_SECTORS],
        }
    }
}

pub fn compute_critic_diagnostics(traces: &[EpisodeTrace]) -> CriticDiagnostics {
    let mut diagnostics = CriticDiagnostics::default();

    for trace in traces {
        let mut return_to_go = 0.0;
        for tick in trace.ticks.iter().rev() {
            return_to_go += tick.reward;
            let Some(prediction) = tick.value_prediction else {
                continue;
            };

            let sector = (tick.sector_index as usize).min(NUM_PROGRESS_SECTORS.saturating_sub(1));
            diagnostics.by_sector[sector].push(return_to_go, prediction);

            if max_abs_curvature(tick) >= CURVATURE_DEMAND_THRESHOLD {
                diagnostics.high_curvature.push(return_to_go, prediction);
            } else {
                diagnostics.straight.push(return_to_go, prediction);
            }
            if tick.done {
                diagnostics.terminal.push(return_to_go, prediction);
            }
        }
    }

    diagnostics
}

fn max_abs_curvature(tick: &TickTraceRecord) -> f32 {
    tick.lookahead_curvatures
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max)
}
