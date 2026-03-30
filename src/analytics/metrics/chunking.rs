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

    // Progress
    pub avg_distance_driven: f32,

    // Reward
    pub avg_reward: f32,
    pub avg_progress_reward: f32,
    pub avg_time_penalty: f32,
    pub avg_crash_penalty: f32,
    pub avg_reward_per_second: f32,

    // Timing
    pub avg_life_seconds: f32,
    pub crash_rate: f32,

    // Action behaviour
    pub avg_braking_fraction: f32,
    pub avg_acceleration_fraction: f32,
    pub avg_coasting_fraction: f32,
    pub avg_action_change: f32,

    // Speed and momentum
    pub avg_speed: f32,
    pub avg_peak_speed: f32,
    pub avg_velocity_projection: f32,
    pub avg_drift_angle_deg: f32,

    // Crash forensics
    pub avg_crash_speed: f32,
    pub slide_crash_fraction: f32,
    pub overshoot_crash_fraction: f32,
    pub headon_crash_fraction: f32,

    // Exploration
    pub avg_wall_proximity_fraction: f32,

    // Policy confidence
    pub avg_policy_steering_std: f32,
    pub avg_policy_throttle_std: f32,
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

fn opt_mean(values: &[Option<f32>]) -> f32 {
    let (sum, count) = values.iter().fold((0.0f32, 0u32), |(s, c), v| {
        if let Some(val) = v { (s + val, c + 1) } else { (s, c) }
    });
    if count > 0 { sum / count as f32 } else { 0.0 }
}

pub fn calculate_chunks(records: &[EpisodeRecord], chunk_count: usize) -> Vec<ChunkMetrics> {
    build_chunk_windows(records, chunk_count)
        .into_iter()
        .map(|window| {
            let chunk = &records[window.start_index..window.end_index];
            let count = chunk.len() as f32;
            let reward_values: Vec<f32> = chunk.iter().map(|r| r.reward).collect();
            let progress_reward_values: Vec<f32> =
                chunk.iter().map(|r| r.progress_reward_sum).collect();
            let time_penalty_values: Vec<f32> = chunk.iter().map(|r| r.time_penalty_sum).collect();
            let crash_penalty_values: Vec<f32> =
                chunk.iter().map(|r| r.crash_penalty_sum).collect();
            let crashes = chunk
                .iter()
                .filter(|r| r.end_reason.contains("Crash"))
                .count() as f32;

            // Crash type classification counts.
            let crash_episodes: Vec<&EpisodeRecord> = chunk.iter()
                .filter(|r| r.end_reason.contains("Crash"))
                .collect();
            let crash_count = crash_episodes.len().max(1) as f32;
            let slide_crashes = crash_episodes.iter()
                .filter(|r| r.crash_type.as_deref() == Some("Slide"))
                .count() as f32;
            let overshoot_crashes = crash_episodes.iter()
                .filter(|r| r.crash_type.as_deref() == Some("Overshoot"))
                .count() as f32;
            let headon_crashes = crash_episodes.iter()
                .filter(|r| r.crash_type.as_deref() == Some("HeadOn"))
                .count() as f32;

            let avg_crash_speed = {
                let speeds: Vec<f32> = crash_episodes.iter()
                    .filter_map(|r| r.crash_speed)
                    .collect();
                if speeds.is_empty() { 0.0 } else { speeds.iter().sum::<f32>() / speeds.len() as f32 }
            };

            ChunkMetrics {
                chunk_index: window.chunk_index,
                start_episode: window.start_episode,
                end_episode: window.end_episode,

                avg_distance_driven: chunk.iter().map(|r| r.distance_driven).sum::<f32>() / count,

                avg_reward: reward_values.iter().sum::<f32>() / count,
                avg_progress_reward: progress_reward_values.iter().sum::<f32>() / count,
                avg_time_penalty: time_penalty_values.iter().sum::<f32>() / count,
                avg_crash_penalty: crash_penalty_values.iter().sum::<f32>() / count,
                avg_reward_per_second: chunk.iter().map(|r| r.reward_per_second).sum::<f32>() / count,

                avg_life_seconds: chunk.iter().map(|r| r.ticks as f32 / 60.0).sum::<f32>() / count,
                crash_rate: crashes / count,

                avg_braking_fraction: chunk.iter().map(|r| r.braking_fraction).sum::<f32>() / count,
                avg_acceleration_fraction: chunk.iter().map(|r| r.acceleration_fraction).sum::<f32>() / count,
                avg_coasting_fraction: chunk.iter().map(|r| r.coasting_fraction).sum::<f32>() / count,
                avg_action_change: chunk.iter().map(|r| r.mean_action_change).sum::<f32>() / count,

                avg_speed: chunk.iter().map(|r| r.mean_speed).sum::<f32>() / count,
                avg_peak_speed: chunk.iter().map(|r| r.peak_speed).sum::<f32>() / count,
                avg_velocity_projection: chunk.iter().map(|r| r.mean_velocity_projection).sum::<f32>() / count,
                avg_drift_angle_deg: chunk.iter().map(|r| r.mean_drift_angle_deg).sum::<f32>() / count,

                avg_crash_speed,
                slide_crash_fraction: slide_crashes / crash_count,
                overshoot_crash_fraction: overshoot_crashes / crash_count,
                headon_crash_fraction: headon_crashes / crash_count,

                avg_wall_proximity_fraction: chunk.iter().map(|r| r.wall_proximity_fraction).sum::<f32>() / count,

                avg_policy_steering_std: opt_mean(&chunk.iter().map(|r| r.mean_policy_steering_std).collect::<Vec<_>>()),
                avg_policy_throttle_std: opt_mean(&chunk.iter().map(|r| r.mean_policy_throttle_std).collect::<Vec<_>>()),
            }
        })
        .collect()
}
