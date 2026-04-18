//! GEMM backend dispatch — three single-precision matrix-multiply implementations
//! selected at compile time via Cargo features, with a platform-aware default.
//!
//! The PPO hot path (actor and critic forward+backward) spends the vast
//! majority of its time in three GEMM-shaped operations inside `Linear`:
//! forward, input-gradient, and weight-gradient. This module centralises all
//! three backends behind a stable signature so `Linear::forward_batch` and
//! `Linear::backward_batch` never have to know which one is active.
//!
//! # Selection order (highest priority first)
//!
//! | Priority | Condition                                                             | Backend          |
//! |----------|-----------------------------------------------------------------------|------------------|
//! | 1        | `--features force-scalar`                                             | scalar           |
//! | 2        | `--features force-matrixmultiply`                                     | matrixmultiply   |
//! | 3        | `--features force-accelerate`                                         | accelerate       |
//! | 4        | Default + `target_os = "macos"`                                       | accelerate       |
//! | 5        | Default + other platforms                                             | matrixmultiply   |
//!
//! # Conventions
//!
//! All operations use **row-major** storage (same as `Linear`'s flat
//! `Vec<f32>` weight layout). Shapes follow the convention `sgemm(m, k, n)`:
//! an `m×k` times `k×n` multiplication producing an `m×n` result.
//!
//! Any `unsafe` block here is confined to FFI entry points. The scalar and
//! matrixmultiply paths are fully safe Rust (matrixmultiply internally uses
//! unsafe but the crate's API is safe at call sites).

// ── Backend selection resolves to exactly one of these cfg-gated modules ──

#[cfg(feature = "force-scalar")]
#[path = "gemm_scalar.rs"]
mod active;

#[cfg(all(
    not(feature = "force-scalar"),
    any(
        feature = "force-matrixmultiply",
        all(
            not(feature = "force-accelerate"),
            not(target_os = "macos"),
        ),
    )
))]
#[path = "gemm_matrixmultiply.rs"]
mod active;

#[cfg(all(
    not(feature = "force-scalar"),
    not(feature = "force-matrixmultiply"),
    any(
        feature = "force-accelerate",
        target_os = "macos",
    ),
))]
#[path = "gemm_accelerate.rs"]
mod active;

// Refuse to build if two or three force-* features are simultaneously enabled.
#[cfg(any(
    all(feature = "force-scalar", feature = "force-matrixmultiply"),
    all(feature = "force-scalar", feature = "force-accelerate"),
    all(feature = "force-matrixmultiply", feature = "force-accelerate"),
))]
compile_error!(
    "At most one of `force-scalar`, `force-matrixmultiply`, `force-accelerate` \
     may be enabled. Use `--no-default-features --features <one>` to pick."
);

// Refuse to build on non-macOS platforms if the caller forced accelerate.
#[cfg(all(feature = "force-accelerate", not(target_os = "macos")))]
compile_error!(
    "The `force-accelerate` feature is only available on macOS. Use \
     `force-matrixmultiply` or `force-scalar` on other platforms."
);

// ── Public API — identical shape regardless of active backend ──

/// Row-major matrix multiplication with accumulate:
///
/// ```text
/// c[m, n] := alpha * a[m, k] * b[k, n] + beta * c[m, n]
/// ```
///
/// Used internally by `Linear::forward_batch` and `Linear::backward_batch`.
///
/// # Panics
/// The scalar backend debug-asserts slice lengths. Release builds trust the
/// caller. BLAS/matrixmultiply backends assume the caller provides slices
/// with `>= m*k`, `>= k*n`, and `>= m*n` elements respectively.
pub fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    active::sgemm(m, k, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// Same as `sgemm` but with `b` treated as transposed. Used by
/// `Linear::forward_batch` (input × weights^T) and the input-gradient path
/// of `Linear::backward_batch` (grad_out × weights^T).
pub fn sgemm_nt(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    active::sgemm_nt(m, k, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// Same as `sgemm` but with `a` treated as transposed. Used by the
/// weight-gradient path of `Linear::backward_batch` (grad_out^T × input).
pub fn sgemm_tn(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    active::sgemm_tn(m, k, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// Human-readable identifier for the active backend. Included in profiling
/// reports so every perf artefact records which backend produced its numbers.
pub fn backend_name() -> &'static str {
    active::NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn sgemm_known_values_2x2() {
        //    a = [1 2 ; 3 4]    b = [5 6 ; 7 8]     expected = [19 22 ; 43 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        sgemm(2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert!(approx_slice(&c, &expected, 1e-4), "got {:?}, want {:?}", c, expected);
    }

    #[test]
    fn sgemm_alpha_beta_accumulate() {
        // c := 2 * a*b + 3 * c
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let mut c = vec![5.0, 5.0, 5.0, 5.0];
        // a*b = [2 2 ; 2 2]   2*a*b = [4 4 ; 4 4]   3*c = [15 15 ; 15 15]
        // result = [19 19 ; 19 19]
        sgemm(2, 2, 2, 2.0, &a, 2, &b, 2, 3.0, &mut c, 2);
        for v in &c {
            assert!((*v - 19.0).abs() < 1e-4, "got {}", v);
        }
    }

    #[test]
    fn sgemm_nt_is_transpose_of_b() {
        // a = [1 2 3; 4 5 6], b = [1 0 0; 0 1 0], b^T = [1 0; 0 1; 0 0]
        // a × b^T = [1 2; 4 5]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 × 3 row-major
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 × 3 row-major (will be transposed)
        let mut c = vec![0.0; 4];
        sgemm_nt(2, 3, 2, 1.0, &a, 3, &b, 3, 0.0, &mut c, 2);
        let expected = vec![1.0, 2.0, 4.0, 5.0];
        assert!(approx_slice(&c, &expected, 1e-4), "got {:?}", c);
    }

    #[test]
    fn sgemm_tn_is_transpose_of_a() {
        // a = [1 2; 3 4; 5 6], a^T = [1 3 5; 2 4 6]
        // b = [1 0; 0 1; 0 0]    but note b is k × n where k=3, n=2, so (3 × 2)
        // Actually simpler test: a^T × I = a^T
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 × 2 row-major
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3 × 2 row-major (I_2 padded with zero row)
        // sgemm_tn computes c := a^T × b  where a is treated as (k × m) = (3 × 2), so a^T is (m × k) = (2 × 3)
        // But we want a^T × b where a^T is (2 × 3) and b is (3 × 2)... b is stored as (k × n) row-major = (3 × 2)
        let mut c = vec![0.0; 4]; // 2 × 2
        sgemm_tn(2, 3, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        // c[i,j] = sum_p a[p, i] * b[p, j]
        // c[0,0] = a[0,0]*b[0,0] + a[1,0]*b[1,0] + a[2,0]*b[2,0] = 1*1 + 3*0 + 5*0 = 1
        // c[0,1] = a[0,0]*b[0,1] + a[1,0]*b[1,1] + a[2,0]*b[2,1] = 1*0 + 3*1 + 5*0 = 3
        // c[1,0] = a[0,1]*b[0,0] + a[1,1]*b[1,0] + a[2,1]*b[2,0] = 2*1 + 4*0 + 6*0 = 2
        // c[1,1] = a[0,1]*b[0,1] + a[1,1]*b[1,1] + a[2,1]*b[2,1] = 2*0 + 4*1 + 6*0 = 4
        let expected = vec![1.0, 3.0, 2.0, 4.0];
        assert!(approx_slice(&c, &expected, 1e-4), "got {:?}", c);
    }

    #[test]
    fn sgemm_overwrite_vs_accumulate() {
        // beta=0 should overwrite pre-existing garbage in c; beta=1 should add.
        let a = vec![2.0, 0.0, 0.0, 2.0]; // 2I
        let b = vec![3.0, 0.0, 0.0, 3.0]; // 3I
        // A*B = 6I
        let mut c = vec![100.0; 4];
        sgemm(2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert!((c[0] - 6.0).abs() < 1e-4);
        assert!((c[3] - 6.0).abs() < 1e-4);
        assert!(c[1].abs() < 1e-4);

        let mut c = vec![1.0; 4];
        sgemm(2, 2, 2, 1.0, &a, 2, &b, 2, 1.0, &mut c, 2);
        // 6I + 1-everywhere = [7 1 ; 1 7]
        assert!((c[0] - 7.0).abs() < 1e-4);
        assert!((c[1] - 1.0).abs() < 1e-4);
        assert!((c[2] - 1.0).abs() < 1e-4);
        assert!((c[3] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn sgemm_rectangular_non_square_shapes() {
        // 4x3 × 3x2 → 4x2
        let a: Vec<f32> = (1..=12).map(|x| x as f32).collect(); // 4 × 3
        let b: Vec<f32> = (1..=6).map(|x| x as f32).collect(); // 3 × 2
        let mut c = vec![0.0; 8];
        sgemm(4, 3, 2, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2);
        // Hand-computed:
        // row 0 of a = [1, 2, 3] dot col 0 of b = [1, 3, 5] = 1+6+15 = 22
        // row 0 of a = [1, 2, 3] dot col 1 of b = [2, 4, 6] = 2+8+18 = 28
        // row 1 of a = [4, 5, 6] dot col 0 of b = [1, 3, 5] = 4+15+30 = 49
        // row 1 of a = [4, 5, 6] dot col 1 of b = [2, 4, 6] = 8+20+36 = 64
        // row 2 of a = [7, 8, 9] dot col 0 = 7+24+45 = 76
        // row 2 of a = [7, 8, 9] dot col 1 = 14+32+54 = 100
        // row 3 of a = [10, 11, 12] dot col 0 = 10+33+60 = 103
        // row 3 of a = [10, 11, 12] dot col 1 = 20+44+72 = 136
        let expected = vec![22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0];
        assert!(approx_slice(&c, &expected, 1e-3), "got {:?}", c);
    }

    #[test]
    fn backend_name_is_nonempty() {
        let name = backend_name();
        assert!(!name.is_empty());
        // Must be one of the three known backend identifiers (prefix check)
        let known_prefixes = ["scalar", "matrixmultiply", "accelerate"];
        assert!(
            known_prefixes.iter().any(|p| name.contains(p)),
            "unexpected backend name: {}", name
        );
    }
}
