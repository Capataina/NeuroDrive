//! Integration test: PPO forward/backward/optimiser pipeline produces
//! well-formed gradients and updates weights. Uses the `[lib]` target to
//! import internal types — this test could not exist before 2026-04-18
//! when the library target was added (see cross-cutting audit finding).
//!
//! These tests are deliberately coarse: they verify that running a forward
//! and a backward pass produces finite, non-zero gradients on every layer
//! that should have them, and that the optimiser steps update the weights.
//! Unit-level correctness of the PPO math itself is covered in
//! `src/brain/ppo/update.rs` tests.

use neurodrive::brain::common::mlp::{Linear, Tanh};
use neurodrive::brain::common::optim::AdamOptimizer;
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;

// ── Helpers ──────────────────────────────────────────────────────────────

fn fill_random(buf: &mut [f32], rng: &mut StdRng) {
    for v in buf.iter_mut() {
        *v = rng.random_range(-1.0..1.0);
    }
}

fn all_finite(slice: &[f32]) -> bool {
    slice.iter().all(|v| v.is_finite())
}

// ── Forward + backward produce finite, non-zero gradients ────────────────

#[test]
fn linear_forward_then_backward_produces_nonzero_gradients() {
    let mut rng = StdRng::seed_from_u64(101);
    let mut layer = Linear::new_orthogonal(8, 4, 1.0, &mut rng);

    let batch_size = 16;
    let mut input = vec![0.0f32; batch_size * 8];
    fill_random(&mut input, &mut rng);

    let mut output = vec![0.0f32; batch_size * 4];
    layer.forward_batch(&input, &mut output, batch_size);

    let mut grad_output = vec![0.0f32; batch_size * 4];
    fill_random(&mut grad_output, &mut rng);

    let mut grad_input = vec![0.0f32; batch_size * 8];
    layer.backward_batch(&grad_output, &mut grad_input, batch_size);

    // All outputs and gradients must be finite — any NaN/Inf means the
    // backend broke something numerically.
    assert!(all_finite(&output), "output has NaN/Inf");
    assert!(all_finite(&grad_input), "grad_input has NaN/Inf");
    assert!(all_finite(&layer.grad_weights), "grad_weights has NaN/Inf");
    assert!(all_finite(&layer.grad_biases), "grad_biases has NaN/Inf");

    // Some gradient should be non-zero — a fully-zero result would mean
    // backward_batch silently no-op'd.
    assert!(layer.grad_weights.iter().any(|g| g.abs() > 1e-6),
        "grad_weights all near zero");
    assert!(layer.grad_biases.iter().any(|g| g.abs() > 1e-6),
        "grad_biases all near zero");
    assert!(grad_input.iter().any(|g| g.abs() > 1e-6),
        "grad_input all near zero");
}

#[test]
fn linear_plus_tanh_chain_forward_backward_is_finite() {
    let mut rng = StdRng::seed_from_u64(102);
    let mut fc = Linear::new_orthogonal(4, 4, 1.0, &mut rng);
    let mut tanh = Tanh::new();

    let batch_size = 8;
    let mut input = vec![0.0f32; batch_size * 4];
    fill_random(&mut input, &mut rng);

    // Forward: Linear → Tanh
    let mut h = vec![0.0f32; batch_size * 4];
    fc.forward_batch(&input, &mut h, batch_size);
    let mut out = vec![0.0f32; batch_size * 4];
    tanh.forward_batch(&h, &mut out, batch_size, 4);

    // Backward: Tanh → Linear
    let mut grad_out = vec![0.0f32; batch_size * 4];
    fill_random(&mut grad_out, &mut rng);
    let mut grad_h = vec![0.0f32; batch_size * 4];
    tanh.backward_batch(&grad_out, &mut grad_h, batch_size, 4);
    let mut grad_input = vec![0.0f32; batch_size * 4];
    fc.backward_batch(&grad_h, &mut grad_input, batch_size);

    assert!(all_finite(&out));
    assert!(all_finite(&grad_input));
    assert!(all_finite(&fc.grad_weights));
}

// ── Optimiser step moves weights ─────────────────────────────────────────

#[test]
fn adam_step_after_backward_moves_weights() {
    let mut rng = StdRng::seed_from_u64(103);
    let mut layer = Linear::new_orthogonal(4, 3, 1.0, &mut rng);
    let weights_before = layer.weights.clone();

    // Simulate a non-zero gradient accumulation
    let batch_size = 8;
    let mut input = vec![0.0f32; batch_size * 4];
    fill_random(&mut input, &mut rng);
    let mut output = vec![0.0f32; batch_size * 3];
    layer.forward_batch(&input, &mut output, batch_size);

    let mut grad_output = vec![0.1f32; batch_size * 3];
    let mut grad_input = vec![0.0f32; batch_size * 4];
    layer.backward_batch(&grad_output, &mut grad_input, batch_size);

    let mut opt = AdamOptimizer::new(&[&layer], 0.01, 0.0);
    opt.step(&mut [&mut layer]);

    let any_change = layer.weights.iter().zip(weights_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(any_change, "Adam step did not change any weight");

    // Also: post-step state must remain finite.
    assert!(all_finite(&layer.weights));
    assert!(all_finite(&layer.biases));

    // Clear for the next step
    let _ = grad_output.fill(0.0);
    let _ = grad_input.fill(0.0);
}

// ── Cache / state integrity across multiple calls ────────────────────────

#[test]
fn linear_forward_batch_handles_varying_batch_sizes() {
    // Running forward_batch with different batch sizes in sequence must not
    // corrupt state — a regression guard against cache-sizing bugs.
    let mut rng = StdRng::seed_from_u64(104);
    let mut layer = Linear::new_orthogonal(5, 3, 1.0, &mut rng);

    for &batch_size in &[4, 16, 8, 32, 2] {
        let input = vec![0.1f32; batch_size * 5];
        let mut output = vec![0.0f32; batch_size * 3];
        layer.forward_batch(&input, &mut output, batch_size);

        // Output must be well-formed for the new batch size.
        assert!(all_finite(&output), "output not finite for batch={}", batch_size);
        // And backward_batch must also work — it depends on the cached input.
        let grad_out = vec![0.01f32; batch_size * 3];
        let mut grad_in = vec![0.0f32; batch_size * 5];
        layer.backward_batch(&grad_out, &mut grad_in, batch_size);
        assert!(all_finite(&grad_in), "grad_in not finite for batch={}", batch_size);
    }
}

#[test]
fn tanh_forward_batch_preserves_sign_at_nonzero_inputs() {
    let mut tanh = Tanh::new();
    let batch_size = 4;
    let input: Vec<f32> = vec![-2.0, -1.0, 0.5, 1.0,
                                -0.5, 0.5, 1.5, -1.5,
                                 0.1, 0.2, 0.3, 0.4,
                                -0.1, -0.2, -0.3, -0.4];
    let mut output = vec![0.0f32; 16];
    tanh.forward_batch(&input, &mut output, batch_size, 4);

    // tanh preserves sign for non-zero inputs (signs match for non-zero x).
    for (i, (o, x)) in output.iter().zip(input.iter()).enumerate() {
        if x.abs() > 1e-6 {
            assert!(
                (o.signum() - x.signum()).abs() < 1e-6,
                "sign mismatch at {}: in={}, out={}", i, x, o,
            );
        }
    }
}
