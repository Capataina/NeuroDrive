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
