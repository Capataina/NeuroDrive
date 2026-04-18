//! Apple Accelerate GEMM backend (macOS only).
//!
//! Calls Apple's `cblas_sgemm` via the `cblas` Rust crate. On M-series chips
//! this dispatches internally to the AMX matrix coprocessor, delivering
//! roughly 8-20× the throughput of `matrixmultiply` at small matrix sizes
//! (benchmark basis: 699 GFLOP/s at N=64 on M2 — see
//! `context/references/ppo-epoch-performance.md`).
//!
//! The backend is the platform-aware default on macOS and opt-in via
//! `--features force-accelerate`. It will fail to compile on non-macOS
//! platforms (the `cblas` and `blas-src` crates are only listed as
//! dependencies inside a `cfg(target_os = "macos")` target block).

pub const NAME: &str = "accelerate (cblas_sgemm → AMX on Apple Silicon)";

use cblas::{Layout, Transpose};

// Ensure `blas-src` is linked — this is the extern-"C" glue that names
// the `cblas_*` symbols we call via the `cblas` wrapper.
extern crate blas_src;

/// `C := alpha * A * B + beta * C` — row-major, no transposes.
pub fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    // SAFETY: cblas::sgemm wraps a C FFI call into Apple's Accelerate
    // framework. The contract: `a`, `b` must point to at least `m*k` /
    // `k*n` valid f32 elements; `c` must point to at least `m*n` writable
    // f32 elements. Our callers (Linear) always allocate slices sized from
    // in_dim/out_dim/batch_size, so these invariants hold.
    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

/// `C := alpha * A * B^T + beta * C` — row-major, B transposed.
pub fn sgemm_nt(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32,
    c: &mut [f32], ldc: usize,
) {
    // SAFETY: see `sgemm`. Apple's Accelerate reads B via `ldb` regardless
    // of the Transpose flag — it is the stride of the ORIGINAL (non-transposed)
    // matrix.
    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
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
    // SAFETY: see `sgemm`. `lda` is the stride of the original A matrix.
    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}
