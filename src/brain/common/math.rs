use rand::Rng;
use rand::RngExt;
use rand_distr::{Distribution, Normal};

/// Initializes a weight matrix with Glorot (Xavier) uniform distribution.
pub fn glorot_uniform(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    let limit = (6.0 / (rows as f32 + cols as f32)).sqrt();
    let mut weights = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            weights[r][c] = rng.random_range(-limit..limit);
        }
    }
    weights
}

/// Initializes a vector with zeros.
pub fn zeros(size: usize) -> Vec<f32> {
    vec![0.0; size]
}

/// Computes the log probability of a value given mean and std deviation (normal distribution).
pub fn normal_log_prob(value: f32, mean: f32, std: f32) -> f32 {
    let variance = std * std;
    let diff = value - mean;
    -0.5 * (diff * diff / variance + (2.0 * std::f32::consts::PI).ln() + 2.0 * std.ln())
}

/// Samples from a normal distribution.
pub fn sample_normal(mean: f32, std: f32, rng: &mut impl Rng) -> f32 {
    let normal = Normal::new(mean, std).unwrap();
    normal.sample(rng)
}

/// Computes entropy of a normal distribution.
pub fn normal_entropy(std: f32) -> f32 {
    0.5 + 0.5 * (2.0 * std::f32::consts::PI).ln() + std.ln()
}
