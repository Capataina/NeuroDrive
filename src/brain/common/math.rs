use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

/// Initializes a vector with zeros.
pub fn zeros(size: usize) -> Vec<f32> {
    vec![0.0; size]
}

/// Generates an orthogonal matrix of shape `[rows × cols]` in flat row-major
/// order, scaled by `scale`. Uses a random matrix followed by iterative
/// Gram-Schmidt orthogonalisation.
pub fn orthogonal_init(rows: usize, cols: usize, scale: f32, rng: &mut impl Rng) -> Vec<f32> {
    let (m, n) = if rows >= cols {
        (rows, cols)
    } else {
        (cols, rows)
    };

    // Generate random Gaussian matrix in column-major flat storage [m × n]
    // mat[k * n + i] corresponds to old mat[k][i]
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let mut mat: Vec<f32> = (0..m * n).map(|_| normal.sample(rng)).collect();

    // Gram-Schmidt orthogonalisation (operates on columns of the m×n matrix)
    for i in 0..n.min(m) {
        // Two passes for numerical stability
        for _pass in 0..2 {
            // Subtract projections onto previous columns
            for j in 0..i {
                let dot: f32 = (0..m).map(|k| mat[k * n + i] * mat[k * n + j]).sum();
                for k in 0..m {
                    mat[k * n + i] -= dot * mat[k * n + j];
                }
            }
            // Normalise
            let norm: f32 = (0..m).map(|k| mat[k * n + i] * mat[k * n + i]).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for k in 0..m {
                    mat[k * n + i] /= norm;
                }
            }
        }
    }

    // Scale
    for val in mat.iter_mut() {
        *val *= scale;
    }

    // Produce flat row-major [rows × cols] output
    if rows >= cols {
        // mat is [m×n] = [rows×cols], just truncate to rows*cols elements
        mat.truncate(rows * cols);
        mat
    } else {
        // mat is [cols × rows], transpose to [rows × cols]
        let mut result = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] = mat[j * n + i];
            }
        }
        result
    }
}

/// Computes the log probability of a value given mean and std deviation (normal distribution).
pub fn normal_log_prob(value: f32, mean: f32, std: f32) -> f32 {
    let variance = std * std;
    let diff = value - mean;
    -0.5 * (diff * diff / variance + (2.0 * std::f32::consts::PI).ln() + 2.0 * std.ln())
}

/// Samples from a normal distribution using the standard normal + affine transform.
pub fn sample_normal(mean: f32, std: f32, rng: &mut impl Rng) -> f32 {
    let z: f32 = StandardNormal.sample(rng);
    mean + std * z
}

/// Computes entropy of a normal distribution.
pub fn normal_entropy(std: f32) -> f32 {
    0.5 + 0.5 * (2.0 * std::f32::consts::PI).ln() + std.ln()
}
