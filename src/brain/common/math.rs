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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── orthogonal_init ──────────────────────────────────────────────────

    #[test]
    fn orthogonal_init_square_matrix_has_unit_row_norms() {
        let mut rng = StdRng::seed_from_u64(11);
        let w = orthogonal_init(4, 4, 1.0, &mut rng);
        // Verify each row of the resulting matrix has norm ≈ scale = 1.
        for i in 0..4 {
            let norm_sq: f32 = (0..4).map(|j| w[i * 4 + j].powi(2)).sum();
            assert!(approx(norm_sq.sqrt(), 1.0, 1e-3),
                "row {} norm: {}", i, norm_sq.sqrt());
        }
    }

    #[test]
    fn orthogonal_init_scale_is_applied() {
        let mut rng = StdRng::seed_from_u64(13);
        let w = orthogonal_init(4, 4, 3.0, &mut rng);
        for i in 0..4 {
            let norm_sq: f32 = (0..4).map(|j| w[i * 4 + j].powi(2)).sum();
            assert!(approx(norm_sq.sqrt(), 3.0, 1e-2));
        }
    }

    #[test]
    fn orthogonal_init_rectangular_output_matches_shape() {
        let mut rng = StdRng::seed_from_u64(17);
        // Fat rectangular: 2 rows × 8 cols (rows < cols case)
        let w = orthogonal_init(2, 8, 1.0, &mut rng);
        assert_eq!(w.len(), 16);
        // Tall rectangular: 8 rows × 2 cols (rows > cols case)
        let w = orthogonal_init(8, 2, 1.0, &mut rng);
        assert_eq!(w.len(), 16);
    }

    #[test]
    fn orthogonal_init_different_seeds_produce_different_matrices() {
        let mut rng_a = StdRng::seed_from_u64(1);
        let mut rng_b = StdRng::seed_from_u64(2);
        let a = orthogonal_init(4, 4, 1.0, &mut rng_a);
        let b = orthogonal_init(4, 4, 1.0, &mut rng_b);
        let any_diff = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
        assert!(any_diff);
    }

    // ── normal_log_prob ──────────────────────────────────────────────────

    #[test]
    fn normal_log_prob_matches_analytical_peak_at_mean() {
        // log prob at value=mean is: -0.5 * (log(2π) + 2*log(std))
        let mean = 0.0;
        let std = 1.0;
        let lp = normal_log_prob(mean, mean, std);
        let expected = -0.5 * (2.0 * std::f32::consts::PI).ln();
        assert!(approx(lp, expected, 1e-4));
    }

    #[test]
    fn normal_log_prob_symmetric_around_mean() {
        // log_prob(mean+d, mean, std) == log_prob(mean-d, mean, std)
        let mean = 1.5;
        let std = 0.7;
        let d = 0.3;
        assert!(approx(
            normal_log_prob(mean + d, mean, std),
            normal_log_prob(mean - d, mean, std),
            1e-6,
        ));
    }

    #[test]
    fn normal_log_prob_finite_for_reasonable_inputs() {
        let lp = normal_log_prob(0.5, 0.0, 1.0);
        assert!(lp.is_finite());
        let lp = normal_log_prob(-10.0, 0.0, 1.0);
        assert!(lp.is_finite());
    }

    // ── sample_normal ────────────────────────────────────────────────────

    #[test]
    fn sample_normal_recovers_mean_and_std_over_many_samples() {
        let mut rng = StdRng::seed_from_u64(37);
        let mean = 2.5;
        let std = 0.8;
        let n = 5000;

        let samples: Vec<f32> = (0..n).map(|_| sample_normal(mean, std, &mut rng)).collect();
        let empirical_mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let empirical_var: f32 = samples
            .iter()
            .map(|x| (x - empirical_mean).powi(2))
            .sum::<f32>()
            / n as f32;
        let empirical_std = empirical_var.sqrt();

        // With 5000 samples, SE on mean is std/√n ≈ 0.011; accept 0.05 tolerance.
        assert!(approx(empirical_mean, mean, 0.05),
            "empirical mean {} vs expected {}", empirical_mean, mean);
        assert!(approx(empirical_std, std, 0.05),
            "empirical std {} vs expected {}", empirical_std, std);
    }

    #[test]
    fn sample_normal_deterministic_for_same_seed() {
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let a = sample_normal(0.0, 1.0, &mut rng_a);
            let b = sample_normal(0.0, 1.0, &mut rng_b);
            assert!(approx(a, b, 1e-8));
        }
    }

    // ── normal_entropy ───────────────────────────────────────────────────

    #[test]
    fn normal_entropy_positive_for_reasonable_std() {
        assert!(normal_entropy(0.5) > 0.0);
        assert!(normal_entropy(1.0) > 0.0);
        assert!(normal_entropy(5.0) > 0.0);
    }

    #[test]
    fn normal_entropy_increases_with_std() {
        assert!(normal_entropy(2.0) > normal_entropy(1.0));
        assert!(normal_entropy(1.0) > normal_entropy(0.5));
    }
}
