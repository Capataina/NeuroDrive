use crate::analytics::metrics::stats::{percentile, std_dev};
use crate::analytics::models::EpisodeRecord;

pub const DEFAULT_CHUNK_COUNT: usize = 10;

#[derive(Clone, Copy, Debug)]
pub struct ChunkWindow {
    pub chunk_index: usize,
    pub start_index: usize,
    pub end_index: usize,
    pub start_episode: u32,
    pub end_episode: u32,
}

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

/// Splits the episode history into at most `chunk_count` non-empty windows.
pub fn build_chunk_windows(records: &[EpisodeRecord], chunk_count: usize) -> Vec<ChunkWindow> {
    if records.is_empty() {
        return Vec::new();
    }

    let bucket_count = chunk_count.max(1).min(records.len());
    let mut windows = Vec::with_capacity(bucket_count);
    for chunk_index in 0..bucket_count {
        let start_index = records.len() * chunk_index / bucket_count;
        let end_index = records.len() * (chunk_index + 1) / bucket_count;
        if start_index >= end_index {
            continue;
        }

        windows.push(ChunkWindow {
            chunk_index,
            start_index,
            end_index,
            start_episode: records[start_index].episode_id,
            end_episode: records[end_index - 1].episode_id,
        });
    }

    windows
}

pub fn calculate_chunks(records: &[EpisodeRecord], chunk_count: usize) -> Vec<ChunkMetrics> {
    build_chunk_windows(records, chunk_count)
        .into_iter()
        .map(|window| {
            let chunk = &records[window.start_index..window.end_index];
            let count = chunk.len() as f32;
            let progress_values: Vec<f32> = chunk.iter().map(|r| r.progress).collect();
            let reward_values: Vec<f32> = chunk.iter().map(|r| r.reward).collect();
            let pre_terminal_values: Vec<f32> =
                chunk.iter().map(|r| r.pre_terminal_return).collect();
            let progress_reward_values: Vec<f32> =
                chunk.iter().map(|r| r.progress_reward_sum).collect();
            let time_penalty_values: Vec<f32> = chunk.iter().map(|r| r.time_penalty_sum).collect();
            let terminal_reward_values: Vec<f32> =
                chunk.iter().map(|r| r.terminal_reward_sum).collect();
            let crash_penalty_values: Vec<f32> =
                chunk.iter().map(|r| r.crash_penalty_sum).collect();
            let lap_bonus_values: Vec<f32> = chunk.iter().map(|r| r.lap_bonus_sum).collect();
            let steering_values: Vec<f32> = chunk.iter().map(|r| r.steering_mean).collect();
            let throttle_values: Vec<f32> = chunk.iter().map(|r| r.throttle_mean).collect();
            let crashes = chunk
                .iter()
                .filter(|r| r.end_reason.contains("Crash"))
                .count() as f32;
            let lap_completion_rate =
                chunk.iter().filter(|r| r.lap_completed).count() as f32 / count;

            ChunkMetrics {
                chunk_index: window.chunk_index,
                start_episode: window.start_episode,
                end_episode: window.end_episode,
                avg_progress: progress_values.iter().sum::<f32>() / count,
                progress_std: std_dev(&progress_values),
                progress_median: percentile(progress_values.clone(), 0.50),
                progress_p90: percentile(progress_values.clone(), 0.90),
                max_progress: progress_values.iter().copied().fold(0.0, f32::max),
                avg_reward: reward_values.iter().sum::<f32>() / count,
                reward_std: std_dev(&reward_values),
                avg_pre_terminal_return: pre_terminal_values.iter().sum::<f32>() / count,
                avg_progress_reward: progress_reward_values.iter().sum::<f32>() / count,
                avg_time_penalty: time_penalty_values.iter().sum::<f32>() / count,
                avg_terminal_reward: terminal_reward_values.iter().sum::<f32>() / count,
                avg_crash_penalty: crash_penalty_values.iter().sum::<f32>() / count,
                avg_lap_bonus: lap_bonus_values.iter().sum::<f32>() / count,
                avg_ticks: chunk.iter().map(|r| r.ticks as f32).sum::<f32>() / count,
                crash_rate: crashes / count,
                lap_completion_rate,
                steering_mean: steering_values.iter().sum::<f32>() / count,
                steering_std: std_dev(&steering_values),
                throttle_mean: throttle_values.iter().sum::<f32>() / count,
                throttle_std: std_dev(&throttle_values),
            }
        })
        .collect()
}
