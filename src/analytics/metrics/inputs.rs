use crate::analytics::metrics::chunking::{DEFAULT_CHUNK_COUNT, build_chunk_windows};
use crate::analytics::models::EpisodeRecord;

#[derive(Clone, Debug, Default)]
pub struct InputLearningChunk {
    pub chunk_index: usize,
    pub start_episode: u32,
    pub end_episode: u32,
    pub mean_centerline_distance: f32,
    pub mean_abs_lateral_offset: f32,
    pub mean_abs_heading_error_deg: f32,
    pub mean_all_ray_distance: f32,
    pub mean_front_ray_distance: f32,
    pub mean_side_ray_distance: f32,
}

#[derive(Clone, Debug)]
pub struct InputSignalTrend {
    pub name: &'static str,
    pub desired_direction: &'static str,
    pub early_value: f32,
    pub late_value: f32,
    pub delta: f32,
}

pub fn calculate_input_learning_chunks(records: &[EpisodeRecord]) -> Vec<InputLearningChunk> {
    build_chunk_windows(records, DEFAULT_CHUNK_COUNT)
        .into_iter()
        .map(|window| {
            let chunk = &records[window.start_index..window.end_index];
            let count = chunk.len() as f32;
            InputLearningChunk {
                chunk_index: window.chunk_index,
                start_episode: window.start_episode,
                end_episode: window.end_episode,
                mean_centerline_distance: chunk
                    .iter()
                    .map(|record| record.mean_centerline_distance)
                    .sum::<f32>()
                    / count,
                mean_abs_lateral_offset: chunk
                    .iter()
                    .map(|record| record.mean_abs_lateral_offset)
                    .sum::<f32>()
                    / count,
                mean_abs_heading_error_deg: chunk
                    .iter()
                    .map(|record| record.mean_abs_heading_error_deg)
                    .sum::<f32>()
                    / count,
                mean_all_ray_distance: chunk
                    .iter()
                    .map(|record| record.mean_all_ray_distance)
                    .sum::<f32>()
                    / count,
                mean_front_ray_distance: chunk
                    .iter()
                    .map(|record| record.mean_front_ray_distance)
                    .sum::<f32>()
                    / count,
                mean_side_ray_distance: chunk
                    .iter()
                    .map(|record| record.mean_side_ray_distance)
                    .sum::<f32>()
                    / count,
            }
        })
        .collect()
}

pub fn summarize_input_signal_trends(chunks: &[InputLearningChunk]) -> Vec<InputSignalTrend> {
    let Some(first) = chunks.first() else {
        return Vec::new();
    };
    let Some(last) = chunks.last() else {
        return Vec::new();
    };

    vec![
        InputSignalTrend {
            name: "Centreline distance",
            desired_direction: "lower",
            early_value: first.mean_centerline_distance,
            late_value: last.mean_centerline_distance,
            delta: last.mean_centerline_distance - first.mean_centerline_distance,
        },
        InputSignalTrend {
            name: "Abs lateral offset",
            desired_direction: "lower",
            early_value: first.mean_abs_lateral_offset,
            late_value: last.mean_abs_lateral_offset,
            delta: last.mean_abs_lateral_offset - first.mean_abs_lateral_offset,
        },
        InputSignalTrend {
            name: "Abs heading error",
            desired_direction: "lower",
            early_value: first.mean_abs_heading_error_deg,
            late_value: last.mean_abs_heading_error_deg,
            delta: last.mean_abs_heading_error_deg - first.mean_abs_heading_error_deg,
        },
        InputSignalTrend {
            name: "All-ray clearance",
            desired_direction: "higher",
            early_value: first.mean_all_ray_distance,
            late_value: last.mean_all_ray_distance,
            delta: last.mean_all_ray_distance - first.mean_all_ray_distance,
        },
        InputSignalTrend {
            name: "Front-ray clearance",
            desired_direction: "higher",
            early_value: first.mean_front_ray_distance,
            late_value: last.mean_front_ray_distance,
            delta: last.mean_front_ray_distance - first.mean_front_ray_distance,
        },
        InputSignalTrend {
            name: "Side-ray clearance",
            desired_direction: "higher",
            early_value: first.mean_side_ray_distance,
            late_value: last.mean_side_ray_distance,
            delta: last.mean_side_ray_distance - first.mean_side_ray_distance,
        },
    ]
}
