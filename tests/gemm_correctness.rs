//! Integration test: whichever GEMM backend is active must produce results
//! numerically-equivalent to a scalar reference for every shape PPO uses in
//! practice. Tolerances are generous enough to accommodate f32 ULP drift
//! between BLIS-style tiled summation and naive nested-loop summation.
//!
//! This exercises the backend's public API (sgemm / sgemm_nt / sgemm_tn).
//! Run it per backend:
//!
//!   cargo test --test gemm_correctness                               # default backend
//!   cargo test --test gemm_correctness --no-default-features \
//!       --features force-scalar
//!   cargo test --test gemm_correctness --no-default-features \
//!       --features force-matrixmultiply
//!   cargo test --test gemm_correctness --no-default-features \
//!       --features force-accelerate                                  # macOS only

use neurodrive::brain::common::gemm_backend;

// ── Scalar reference implementation (inline to keep the test self-contained) ─

fn scalar_sgemm(
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

fn scalar_sgemm_nt(
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

fn scalar_sgemm_tn(
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

// ── Test helpers ─────────────────────────────────────────────────────────

fn deterministic_fill(len: usize, seed: u64) -> Vec<f32> {
    // Linear congruential generator — avoids the rand dependency on the
    // integration-test side and keeps the test hermetic.
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 32) as u32;
            // Map to [-1, 1] roughly
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel = diff / e.abs().max(1.0);
        assert!(
            diff < tol || rel < tol,
            "{} at [{}]: active={}, reference={}, abs_diff={}, rel_diff={}",
            label, i, a, e, diff, rel,
        );
    }
}

// ── The shapes PPO actually uses (from ActorCritic architecture) ─────────

const OBS_DIM: usize = 43;
const ACTOR_HIDDEN: usize = 64;
const CRITIC_HIDDEN: usize = 128;

// Batch sizes from the hot paths: 8 (action selection) and 32 (training chunk).
const ACTION_BATCH: usize = 8;
const TRAINING_BATCH: usize = 32;

// Generous but principled tolerance — the BLIS-style tiled kernel in
// matrixmultiply accumulates differently from the naive i-j-p loop, and
// Accelerate's AMX path is different again. 5e-3 is roughly two ULPs per
// mul-add × the longest reduction length (128) for well-behaved inputs.
const TOL: f32 = 5e-3;

#[test]
fn sgemm_active_matches_scalar_on_actor_forward_shape() {
    // Actor hidden layer: [batch × actor_hidden] × [actor_hidden × actor_hidden]
    let a = deterministic_fill(ACTION_BATCH * ACTOR_HIDDEN, 1);
    let b = deterministic_fill(ACTOR_HIDDEN * ACTOR_HIDDEN, 2);
    let mut c_active = vec![0.0; ACTION_BATCH * ACTOR_HIDDEN];
    let mut c_scalar = vec![0.0; ACTION_BATCH * ACTOR_HIDDEN];

    gemm_backend::sgemm(
        ACTION_BATCH, ACTOR_HIDDEN, ACTOR_HIDDEN,
        1.0, &a, ACTOR_HIDDEN, &b, ACTOR_HIDDEN,
        0.0, &mut c_active, ACTOR_HIDDEN,
    );
    scalar_sgemm(
        ACTION_BATCH, ACTOR_HIDDEN, ACTOR_HIDDEN,
        1.0, &a, ACTOR_HIDDEN, &b, ACTOR_HIDDEN,
        0.0, &mut c_scalar, ACTOR_HIDDEN,
    );

    assert_close(&c_active, &c_scalar, TOL, "actor hidden SGEMM");
}

#[test]
fn sgemm_nt_active_matches_scalar_on_forward_batch_shape() {
    // forward_batch shape: [batch × in_dim] × [out_dim × in_dim]^T
    // Uses forward through a Linear(in=43, out=64) layer on batch=8.
    let a = deterministic_fill(ACTION_BATCH * OBS_DIM, 3);
    let b = deterministic_fill(ACTOR_HIDDEN * OBS_DIM, 4);
    let mut c_active = vec![0.0; ACTION_BATCH * ACTOR_HIDDEN];
    let mut c_scalar = vec![0.0; ACTION_BATCH * ACTOR_HIDDEN];

    gemm_backend::sgemm_nt(
        ACTION_BATCH, OBS_DIM, ACTOR_HIDDEN,
        1.0, &a, OBS_DIM, &b, OBS_DIM,
        0.0, &mut c_active, ACTOR_HIDDEN,
    );
    scalar_sgemm_nt(
        ACTION_BATCH, OBS_DIM, ACTOR_HIDDEN,
        1.0, &a, OBS_DIM, &b, OBS_DIM,
        0.0, &mut c_scalar, ACTOR_HIDDEN,
    );

    assert_close(&c_active, &c_scalar, TOL, "forward_batch NT SGEMM");
}

#[test]
fn sgemm_tn_active_matches_scalar_on_backward_weight_shape() {
    // backward_batch weight-gradient shape: grad_output^T × input_cache
    // [out_dim × batch] × [batch × in_dim] (produced by sgemm_tn)
    let a = deterministic_fill(TRAINING_BATCH * ACTOR_HIDDEN, 5); // grad_output
    let b = deterministic_fill(TRAINING_BATCH * OBS_DIM, 6); // input_cache
    let mut c_active = vec![0.0; ACTOR_HIDDEN * OBS_DIM];
    let mut c_scalar = vec![0.0; ACTOR_HIDDEN * OBS_DIM];

    gemm_backend::sgemm_tn(
        ACTOR_HIDDEN, TRAINING_BATCH, OBS_DIM,
        1.0, &a, ACTOR_HIDDEN, &b, OBS_DIM,
        0.0, &mut c_active, OBS_DIM,
    );
    scalar_sgemm_tn(
        ACTOR_HIDDEN, TRAINING_BATCH, OBS_DIM,
        1.0, &a, ACTOR_HIDDEN, &b, OBS_DIM,
        0.0, &mut c_scalar, OBS_DIM,
    );

    assert_close(&c_active, &c_scalar, TOL, "backward weight-grad TN SGEMM");
}

#[test]
fn sgemm_active_matches_scalar_on_critic_training_shape() {
    // Critic training uses 128-wide hidden with training batch 32.
    let a = deterministic_fill(TRAINING_BATCH * CRITIC_HIDDEN, 7);
    let b = deterministic_fill(CRITIC_HIDDEN * CRITIC_HIDDEN, 8);
    let mut c_active = vec![0.0; TRAINING_BATCH * CRITIC_HIDDEN];
    let mut c_scalar = vec![0.0; TRAINING_BATCH * CRITIC_HIDDEN];

    gemm_backend::sgemm(
        TRAINING_BATCH, CRITIC_HIDDEN, CRITIC_HIDDEN,
        1.0, &a, CRITIC_HIDDEN, &b, CRITIC_HIDDEN,
        0.0, &mut c_active, CRITIC_HIDDEN,
    );
    scalar_sgemm(
        TRAINING_BATCH, CRITIC_HIDDEN, CRITIC_HIDDEN,
        1.0, &a, CRITIC_HIDDEN, &b, CRITIC_HIDDEN,
        0.0, &mut c_scalar, CRITIC_HIDDEN,
    );

    assert_close(&c_active, &c_scalar, TOL, "critic training SGEMM");
}

#[test]
fn sgemm_alpha_beta_accumulate_matches_scalar() {
    // Non-trivial alpha / beta values should be handled identically.
    let a = deterministic_fill(4 * 3, 9);
    let b = deterministic_fill(3 * 5, 10);
    let c_init: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();

    let mut c_active = c_init.clone();
    let mut c_scalar = c_init.clone();

    let alpha = 2.5f32;
    let beta = -0.75f32;

    gemm_backend::sgemm(4, 3, 5, alpha, &a, 3, &b, 5, beta, &mut c_active, 5);
    scalar_sgemm(4, 3, 5, alpha, &a, 3, &b, 5, beta, &mut c_scalar, 5);

    assert_close(&c_active, &c_scalar, TOL, "alpha/beta accumulate");
}

#[test]
fn backend_name_identifies_one_of_the_known_backends() {
    let name = gemm_backend::backend_name();
    let known = ["scalar", "matrixmultiply", "accelerate"];
    assert!(
        known.iter().any(|k| name.contains(k)),
        "unknown backend name: {}",
        name
    );
}
