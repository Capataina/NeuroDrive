use crate::analytics::metrics::chunking::ChunkMetrics;
use crate::analytics::metrics::critic::CriticDiagnostics;
use crate::analytics::metrics::inputs::{InputLearningChunk, InputSignalTrend};
use crate::analytics::metrics::sectors::SectorDiagnosticsRow;
use crate::analytics::metrics::turns::{FailureModeCount, TurnExecutionSummary};
use crate::analytics::models::{A2cUpdateRecord, EpisodeRecord};

#[derive(Clone, Debug, Default)]
pub struct ReportInsights {
    pub overview: Vec<String>,
    pub performance: Vec<String>,
    pub inputs: Vec<String>,
    pub turns: Vec<String>,
    pub sectors: Vec<String>,
    pub critic: Vec<String>,
    pub optimisation: Vec<String>,
}

pub fn build_report_insights(
    episodes: &[EpisodeRecord],
    chunks: &[ChunkMetrics],
    input_chunks: &[InputLearningChunk],
    input_trends: &[InputSignalTrend],
    turn_summary: &TurnExecutionSummary,
    failure_modes: &[FailureModeCount],
    sector_rows: &[SectorDiagnosticsRow],
    critic: &CriticDiagnostics,
    a2c_updates: &[A2cUpdateRecord],
) -> ReportInsights {
    let mut insights = ReportInsights::default();

    if let (Some(first), Some(last)) = (chunks.first(), chunks.last()) {
        insights.overview.push(format!(
            "Average progress moved from {:.2}% to {:.2}% across the run.",
            first.avg_progress * 100.0,
            last.avg_progress * 100.0
        ));
        insights.overview.push(format!(
            "Average return moved from {:.2} to {:.2}.",
            first.avg_reward, last.avg_reward
        ));

        let plateau = chunks.len() >= 3
            && chunks[chunks.len() - 3..]
                .iter()
                .map(|chunk| chunk.avg_progress)
                .fold(f32::MIN, f32::max)
                - chunks[chunks.len() - 3..]
                    .iter()
                    .map(|chunk| chunk.avg_progress)
                    .fold(f32::MAX, f32::min)
                <= 0.003
            && chunks[chunks.len() - 3..]
                .iter()
                .map(|chunk| chunk.max_progress)
                .fold(f32::MIN, f32::max)
                - chunks[chunks.len() - 3..]
                    .iter()
                    .map(|chunk| chunk.max_progress)
                    .fold(f32::MAX, f32::min)
                <= 0.003;
        if plateau {
            insights.overview.push(
                "Progress appears plateaued: the last three chunks are tightly clustered in both average and max progress.".to_string(),
            );
        }
    }

    if let Some(best_episode) = episodes
        .iter()
        .max_by(|a, b| a.progress.total_cmp(&b.progress))
    {
        insights.performance.push(format!(
            "Best progress episode is {} at {:.2}%, ending in {}.",
            best_episode.episode_id,
            best_episode.progress * 100.0,
            best_episode.end_reason
        ));
    }
    if episodes.iter().all(|episode| !episode.lap_completed) {
        insights.performance.push(
            "No laps completed yet; current learning remains local rather than task-complete."
                .to_string(),
        );
    }

    if let (Some(first), Some(last)) = (input_chunks.first(), input_chunks.last()) {
        insights.inputs.push(format!(
            "Mean centreline gap moved from {:.2} to {:.2}.",
            first.mean_centerline_distance, last.mean_centerline_distance
        ));
        insights.inputs.push(format!(
            "Mean absolute lateral offset moved from {:.2} to {:.2}.",
            first.mean_abs_lateral_offset, last.mean_abs_lateral_offset
        ));
        insights.inputs.push(format!(
            "Mean absolute heading error moved from {:.2}° to {:.2}°.",
            first.mean_abs_heading_error_deg, last.mean_abs_heading_error_deg
        ));
    }
    for trend in input_trends {
        let improved = match trend.desired_direction {
            "lower" => trend.delta < 0.0,
            "higher" => trend.delta > 0.0,
            _ => false,
        };
        if improved {
            insights.inputs.push(format!(
                "{} improved in the desired direction ({:.2} -> {:.2}).",
                trend.name, trend.early_value, trend.late_value
            ));
        }
    }

    if let Some(entry_speed) = turn_summary.turn_entry_speed_mean {
        insights.turns.push(format!(
            "Mean turn-entry speed is {:.1}, with high-curvature throttle mean {:.3}.",
            entry_speed, turn_summary.high_curvature_throttle_mean
        ));
    }
    insights.turns.push(format!(
        "Steering adequacy is {:.3}; curvature steering error mean is {:.3}.",
        turn_summary.steering_adequacy_mean, turn_summary.curvature_steering_error_mean
    ));
    insights.turns.push(format!(
        "Understeer rate across strong-demand ticks is {:.1}%.",
        turn_summary.understeer_rate_mean * 100.0
    ));
    if let Some(mode) = failure_modes.first() {
        insights.turns.push(format!(
            "Dominant classified crash mode is '{}' at {:.1}% of classified failures.",
            mode.label,
            mode.share * 100.0
        ));
    }

    if let Some(top_sector) = sector_rows
        .iter()
        .max_by_key(|row| row.crash_terminal_count)
        .filter(|row| row.crash_terminal_count > 0)
    {
        insights.sectors.push(format!(
            "Crash hotspot is sector {} with {} crash terminals.",
            top_sector.sector_index + 1,
            top_sector.crash_terminal_count
        ));
        insights.sectors.push(format!(
            "Sector {} mean line gap is {:.2} and mean |heading| is {:.2}°.",
            top_sector.sector_index + 1,
            top_sector.centerline_distance_mean,
            top_sector.heading_abs_mean_rad.to_degrees()
        ));
    }

    insights.critic.push(format!(
        "Critic MAE is {:.4} on straight ticks versus {:.4} on high-curvature ticks.",
        critic.straight.mae(),
        critic.high_curvature.mae()
    ));
    insights.critic.push(format!(
        "Explained variance is {:.4} on straight ticks versus {:.4} on high-curvature ticks.",
        critic.straight.explained_variance(),
        critic.high_curvature.explained_variance()
    ));

    if let Some(latest) = a2c_updates.last() {
        insights.optimisation.push(format!(
            "Latest critic explained variance is {:.4} with value loss {:.4}.",
            latest.explained_variance, latest.value_loss
        ));
        insights.optimisation.push(format!(
            "Policy entropy remains at {:.4}; exploration has not collapsed yet.",
            latest.policy_entropy
        ));
    }

    insights
}
