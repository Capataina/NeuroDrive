/// Returns the arithmetic mean or `0.0` for an empty slice.
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Returns the population standard deviation or `0.0` for an empty slice.
pub fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

/// Returns a percentile sampled by rounded index or `0.0` when empty.
pub fn percentile(mut values: Vec<f32>, q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    values[index]
}
