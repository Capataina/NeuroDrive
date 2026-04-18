//! Scalar GEMM backend — naive row-major nested-loop multiplication.
//!
//! This is the reference implementation. Every other backend must produce
//! numerically-equivalent results (up to floating-point rounding order).
//! It is the slowest path by construction but useful as a correctness oracle
//! and as a fallback where platform-specific acceleration is unavailable.
//!
//! No dependencies. No unsafe. Compiles everywhere.

pub const NAME: &str = "scalar (naive Rust nested loops, fallback / reference)";

/// `C := alpha * A * B + beta * C` — row-major, no transposes.
pub fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            let dst = &mut c[i * ldc + j];
            *dst = alpha * sum + beta * *dst;
        }
    }
}

/// `C := alpha * A * B^T + beta * C` — row-major, B transposed.
/// `B` is logically `(n × k)` row-major but is treated as `(k × n)` here.
pub fn sgemm_nt(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * lda + p] * b[j * ldb + p];
            }
            let dst = &mut c[i * ldc + j];
            *dst = alpha * sum + beta * *dst;
        }
    }
}

/// `C := alpha * A^T * B + beta * C` — row-major, A transposed.
/// `A` is logically `(k × m)` row-major but is treated as `(m × k)` here.
pub fn sgemm_tn(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[p * lda + i] * b[p * ldb + j];
            }
            let dst = &mut c[i * ldc + j];
            *dst = alpha * sum + beta * *dst;
        }
    }
}
