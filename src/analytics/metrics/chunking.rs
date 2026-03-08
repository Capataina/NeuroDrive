use crate::analytics::trackers::episode::EpisodeRecord;

#[derive(Debug, Default)]
pub struct ChunkMetrics {
    pub chunk_index: usize,
    pub start_episode: u32,
    pub end_episode: u32,
    pub avg_progress: f32,
    pub progress_std: f32,
    pub progress_median: f32,
    pub progress_p90: f32,
    pub max_progress: f32,
    pub avg_reward: f32,
    pub reward_std: f32,
    pub avg_pre_terminal_return: f32,
    pub avg_progress_reward: f32,
    pub avg_time_penalty: f32,
    pub avg_terminal_reward: f32,
    pub avg_crash_penalty: f32,
    pub avg_lap_bonus: f32,
    pub avg_ticks: f32,
    pub crash_rate: f32,
    pub lap_completion_rate: f32,
    pub steering_mean: f32,
    pub steering_std: f32,
    pub throttle_mean: f32,
    pub throttle_std: f32,
}

pub fn calculate_chunks(records: &[EpisodeRecord], chunk_size: usize) -> Vec<ChunkMetrics> {
    if records.is_empty() {
        return vec![];
    }

    let mut metrics = Vec::new();
    for (i, chunk) in records.chunks(chunk_size).enumerate() {
        let count = chunk.len() as f32;
        let progress_values: Vec<f32> = chunk.iter().map(|r| r.progress).collect();
        let reward_values: Vec<f32> = chunk.iter().map(|r| r.reward).collect();
        let pre_terminal_values: Vec<f32> = chunk.iter().map(|r| r.pre_terminal_return).collect();
        let progress_reward_values: Vec<f32> =
            chunk.iter().map(|r| r.progress_reward_sum).collect();
        let time_penalty_values: Vec<f32> =
            chunk.iter().map(|r| r.time_penalty_sum).collect();
        let terminal_reward_values: Vec<f32> =
            chunk.iter().map(|r| r.terminal_reward_sum).collect();
        let crash_penalty_values: Vec<f32> =
            chunk.iter().map(|r| r.crash_penalty_sum).collect();
        let lap_bonus_values: Vec<f32> = chunk.iter().map(|r| r.lap_bonus_sum).collect();
        let steering_values: Vec<f32> = chunk.iter().map(|r| r.steering_mean).collect();
        let throttle_values: Vec<f32> = chunk.iter().map(|r| r.throttle_mean).collect();
        let avg_progress = progress_values.iter().sum::<f32>() / count;
        let max_progress = chunk.iter().map(|r| r.progress).fold(0.0, f32::max);
        let avg_reward = reward_values.iter().sum::<f32>() / count;
        let avg_ticks = chunk.iter().map(|r| r.ticks as f32).sum::<f32>() / count;
        let crashes = chunk.iter().filter(|r| r.end_reason.contains("Crash")).count() as f32;
        let crash_rate = crashes / count;
        let lap_completion_rate =
            chunk.iter().filter(|r| r.lap_completed).count() as f32 / count;

        metrics.push(ChunkMetrics {
            chunk_index: i,
            start_episode: chunk.first().unwrap().episode_id,
            end_episode: chunk.last().unwrap().episode_id,
            avg_progress,
            progress_std: std_dev(&progress_values),
            progress_median: percentile(progress_values.clone(), 0.50),
            progress_p90: percentile(progress_values, 0.90),
            max_progress,
            avg_reward,
            reward_std: std_dev(&reward_values),
            avg_pre_terminal_return: pre_terminal_values.iter().sum::<f32>() / count,
            avg_progress_reward: progress_reward_values.iter().sum::<f32>() / count,
            avg_time_penalty: time_penalty_values.iter().sum::<f32>() / count,
            avg_terminal_reward: terminal_reward_values.iter().sum::<f32>() / count,
            avg_crash_penalty: crash_penalty_values.iter().sum::<f32>() / count,
            avg_lap_bonus: lap_bonus_values.iter().sum::<f32>() / count,
            avg_ticks,
            crash_rate,
            lap_completion_rate,
            steering_mean: steering_values.iter().sum::<f32>() / count,
            steering_std: std_dev(&steering_values),
            throttle_mean: throttle_values.iter().sum::<f32>() / count,
            throttle_std: std_dev(&throttle_values),
        });
    }

    metrics
}

fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

fn percentile(mut values: Vec<f32>, q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    values[index]
}
