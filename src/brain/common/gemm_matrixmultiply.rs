//! `matrixmultiply` crate GEMM backend.
//!
//! Pure-Rust BLIS-style microkernel with AArch64 NEON intrinsics on Apple
//! Silicon and SSE/AVX on x86. No C dependency. Default backend on non-macOS
//! platforms. Available on macOS behind `--features force-matrixmultiply`
//! for A/B correctness testing against Accelerate.
//!
//! The `matrixmultiply::sgemm` function takes explicit row and column
//! strides, so row-major storage is expressed as `(rsa = lda, csa = 1)`
//! and transposed views as `(rsa = 1, csa = lda)`.

pub const NAME: &str = "matrixmultiply (pure Rust, BLIS-style NEON microkernel)";

/// `C := alpha * A * B + beta * C` — row-major, no transposes.
pub fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    // SAFETY: matrixmultiply::sgemm requires valid non-null pointers to
    // regions of at least `m*k`, `k*n`, `m*n` contiguous f32 elements
    // respectively, under the given strides. Our callers (Linear) always
    // supply backing Vec<f32> slices sized from in_dim/out_dim/batch_size,
    // so these invariants hold. Strides are positive and non-zero.
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            alpha,
            a.as_ptr(), lda as isize, 1,
            b.as_ptr(), ldb as isize, 1,
            beta,
            c.as_mut_ptr(), ldc as isize, 1,
        );
    }
}

/// `C := alpha * A * B^T + beta * C` — row-major, B transposed.
///
/// `B` is logically `(n × k)` row-major but is consumed as its transpose:
/// we swap B's row and column strides at the matrixmultiply boundary.
pub fn sgemm_nt(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    // SAFETY: see `sgemm`. The transposed view of B is expressed by swapping
    // its row and column strides (1, ldb) instead of (ldb, 1).
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            alpha,
            a.as_ptr(), lda as isize, 1,
            b.as_ptr(), 1, ldb as isize,
            beta,
            c.as_mut_ptr(), ldc as isize, 1,
        );
    }
}

/// `C := alpha * A^T * B + beta * C` — row-major, A transposed.
pub fn sgemm_tn(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    // SAFETY: see `sgemm`. A's transposed view swaps its strides to (1, lda).
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            alpha,
            a.as_ptr(), 1, lda as isize,
            b.as_ptr(), ldb as isize, 1,
            beta,
            c.as_mut_ptr(), ldc as isize, 1,
        );
    }
}
