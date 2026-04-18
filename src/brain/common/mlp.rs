use crate::brain::common::gemm_backend;
use crate::brain::common::math::zeros;
use rand::Rng;

/// Fully-connected linear layer with flat contiguous weight storage.
///
/// Weights are stored as a single `Vec<f32>` in row-major order:
/// `weights[i * in_dim + j]` is the weight from input `j` to output `i`.
/// This gives sequential memory access during matrix-vector and matrix-matrix
/// multiplies, enabling cache-friendly traversal and LLVM auto-vectorisation.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weights: Vec<f32>,      // [out_dim × in_dim], row-major
    pub biases: Vec<f32>,       // [out_dim]
    pub grad_weights: Vec<f32>, // [out_dim × in_dim]
    pub grad_biases: Vec<f32>,  // [out_dim]
    pub in_dim: usize,
    pub out_dim: usize,
    /// Cached input from the most recent single forward pass (for backward).
    input_cache: Vec<f32>,
    /// Cached batch input from the most recent batch forward pass (for backward_batch).
    batch_input_cache: Vec<f32>,
    batch_size_cached: usize,
}

impl Linear {
    pub fn new_orthogonal(in_dim: usize, out_dim: usize, scale: f32, rng: &mut impl Rng) -> Self {
        let weights = crate::brain::common::math::orthogonal_init(out_dim, in_dim, scale, rng);

        Self {
            weights,
            biases: zeros(out_dim),
            grad_weights: vec![0.0; out_dim * in_dim],
            grad_biases: zeros(out_dim),
            in_dim,
            out_dim,
            input_cache: vec![0.0; in_dim],
            batch_input_cache: Vec::new(),
            batch_size_cached: 0,
        }
    }

    // ── Single-sample forward / backward ─────────────────────────────

    /// Single-sample forward, writing into a caller-supplied `output` slice.
    /// `output.len()` must be `>= self.out_dim`. Allocation-free on the hot
    /// path — see `ActorCritic::forward_actor` / `forward_critic` / `forward`
    /// for the scratch-buffer strategy.
    pub fn forward_into(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert!(input.len() >= self.in_dim);
        debug_assert!(output.len() >= self.out_dim);
        self.input_cache.copy_from_slice(&input[..self.in_dim]);
        for i in 0..self.out_dim {
            let row = &self.weights[i * self.in_dim..(i + 1) * self.in_dim];
            output[i] = self.biases[i]
                + row.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f32>();
        }
    }

    // ── Batch forward / backward ─────────────────────────────────────

    /// Batch forward: `output = input × weights^T + bias (broadcast)`.
    ///
    /// `input`  is row-major `[batch_size × in_dim]`.
    /// `output` is row-major `[batch_size × out_dim]`, must be pre-allocated.
    ///
    /// Caches `input` for `backward_batch`. The mat-mat itself goes through
    /// the selected `gemm_backend` (Accelerate on macOS by default,
    /// matrixmultiply elsewhere, scalar fallback available behind
    /// `--features force-scalar`).
    pub fn forward_batch(&mut self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Cache input for backward_batch
        self.batch_input_cache.resize(batch_size * self.in_dim, 0.0);
        self.batch_input_cache.copy_from_slice(&input[..batch_size * self.in_dim]);
        self.batch_size_cached = batch_size;

        // Broadcast biases into each output row. This is a memcpy, not a GEMM —
        // GEMM handles the weight-contribution accumulation below with beta=1.
        for s in 0..batch_size {
            let out_row = &mut output[s * self.out_dim..(s + 1) * self.out_dim];
            out_row.copy_from_slice(&self.biases);
        }

        // output += input × weights^T
        //   input shape   [batch_size × in_dim]   (m × k)
        //   weights shape [out_dim × in_dim]      (n × k, transposed inside)
        //   output shape  [batch_size × out_dim]  (m × n)
        gemm_backend::sgemm_nt(
            batch_size, self.in_dim, self.out_dim,
            1.0,
            input, self.in_dim,
            &self.weights, self.in_dim,
            1.0,
            output, self.out_dim,
        );
    }

    /// Batch backward. Accumulates into `grad_weights` / `grad_biases` and
    /// writes `grad_input`.
    ///
    /// `grad_output` is `[batch_size × out_dim]`.
    /// `grad_input`  is `[batch_size × in_dim]`, pre-allocated (overwritten).
    ///
    /// Two of the three updates go through `gemm_backend`; the bias gradient
    /// is a simple per-element reduction, not a GEMM.
    pub fn backward_batch(&mut self, grad_output: &[f32], grad_input: &mut [f32], batch_size: usize) {
        // 1. Bias gradient: grad_biases += Σ_s grad_output[s, :]
        for s in 0..batch_size {
            let go_row = &grad_output[s * self.out_dim..(s + 1) * self.out_dim];
            for i in 0..self.out_dim {
                self.grad_biases[i] += go_row[i];
            }
        }

        // 2. Weight gradient: grad_weights += grad_output^T × input_cache
        //    grad_output transposed shape: [out_dim × batch_size]  (m × k)
        //    input_cache shape:            [batch_size × in_dim]   (k × n)
        //    grad_weights shape:           [out_dim × in_dim]      (m × n, accumulated with beta=1)
        gemm_backend::sgemm_tn(
            self.out_dim, batch_size, self.in_dim,
            1.0,
            grad_output, self.out_dim,
            &self.batch_input_cache, self.in_dim,
            1.0,
            &mut self.grad_weights, self.in_dim,
        );

        // 3. Input gradient: grad_input := grad_output × weights  (beta=0 → overwrite)
        //    grad_output shape: [batch_size × out_dim]  (m × k)
        //    weights shape:     [out_dim × in_dim]     (k × n)
        //    grad_input shape:  [batch_size × in_dim]  (m × n)
        gemm_backend::sgemm(
            batch_size, self.out_dim, self.in_dim,
            1.0,
            grad_output, self.out_dim,
            &self.weights, self.in_dim,
            0.0,
            grad_input, self.in_dim,
        );
    }

    // ── Utilities ────────────────────────────────────────────────────

    pub fn zero_grad(&mut self) {
        self.grad_weights.iter_mut().for_each(|v| *v = 0.0);
        self.grad_biases.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn weight_l2_norm(&self) -> f32 {
        let w: f32 = self.weights.iter().map(|v| v * v).sum();
        let b: f32 = self.biases.iter().map(|v| v * v).sum();
        (w + b).sqrt()
    }

    pub fn grad_l2_norm(&self) -> f32 {
        let w: f32 = self.grad_weights.iter().map(|v| v * v).sum();
        let b: f32 = self.grad_biases.iter().map(|v| v * v).sum();
        (w + b).sqrt()
    }
}

/// Tanh activation with cached output for backward pass.
#[derive(Clone, Debug)]
pub struct Tanh {
    /// Output cache for batch backward: row-major [batch × dim].
    batch_output_cache: Vec<f32>,
    batch_size_cached: usize,
    batch_dim_cached: usize,
}

impl Tanh {
    pub fn new() -> Self {
        Self {
            batch_output_cache: Vec::new(),
            batch_size_cached: 0,
            batch_dim_cached: 0,
        }
    }

    // ── Single ───────────────────────────────────────────────────────

    /// Single-sample forward, writing into a caller-supplied `output` slice.
    /// Allocation-free — the hot path passes a reusable scratch buffer.
    ///
    /// There is no single-sample backward pass in the PPO training code
    /// (all gradients flow through `backward_batch`), so no output cache
    /// is maintained on this path.
    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) {
        debug_assert!(output.len() >= input.len());
        for i in 0..input.len() {
            output[i] = input[i].tanh();
        }
    }

    /// Returns a reference to the batch output cache (for saturation diagnostics).
    pub fn batch_cache(&self) -> &[f32] {
        &self.batch_output_cache
    }

    // ── Batch ────────────────────────────────────────────────────────

    /// Batch forward: element-wise tanh, caches output for backward.
    /// `data` is `[batch_size × dim]`, `output` same shape (pre-allocated).
    pub fn forward_batch(&mut self, input: &[f32], output: &mut [f32], batch_size: usize, dim: usize) {
        let n = batch_size * dim;
        for i in 0..n {
            output[i] = input[i].tanh();
        }
        self.batch_output_cache.resize(n, 0.0);
        self.batch_output_cache[..n].copy_from_slice(&output[..n]);
        self.batch_size_cached = batch_size;
        self.batch_dim_cached = dim;
    }

    /// Batch backward: element-wise `g * (1 - o²)`.
    pub fn backward_batch(&self, grad_output: &[f32], grad_input: &mut [f32], batch_size: usize, dim: usize) {
        let n = batch_size * dim;
        let cache = &self.batch_output_cache;
        for i in 0..n {
            grad_input[i] = grad_output[i] * (1.0 - cache[i] * cache[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── Linear ───────────────────────────────────────────────────────────

    #[test]
    fn linear_forward_batch_writes_bias_when_input_is_zero() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut linear = Linear::new_orthogonal(4, 3, 1.0, &mut rng);
        linear.biases = vec![1.0, 2.0, 3.0];

        let input = vec![0.0f32; 8]; // 2 × 4 zeros
        let mut output = vec![0.0f32; 6]; // 2 × 3

        linear.forward_batch(&input, &mut output, 2);

        // Zero input → output = bias broadcast
        for s in 0..2 {
            for i in 0..3 {
                assert!(approx(output[s * 3 + i], linear.biases[i], 1e-6),
                    "zero input should give bias at [s={}, i={}]: got {}, want {}",
                    s, i, output[s * 3 + i], linear.biases[i]);
            }
        }
    }

    #[test]
    fn linear_forward_batch_known_handcrafted() {
        // Hand-computed 2×2 → 1 linear: y = [1,2]·[x0,x1] + 10
        let mut rng = StdRng::seed_from_u64(0);
        let mut linear = Linear::new_orthogonal(2, 1, 1.0, &mut rng);
        linear.weights = vec![1.0, 2.0]; // [out_dim=1 × in_dim=2]
        linear.biases = vec![10.0];

        let input = vec![3.0, 4.0, 5.0, 6.0]; // 2 samples
        let mut output = vec![0.0; 2];
        linear.forward_batch(&input, &mut output, 2);

        // Sample 0: 1*3 + 2*4 + 10 = 21
        // Sample 1: 1*5 + 2*6 + 10 = 27
        assert!(approx(output[0], 21.0, 1e-5), "sample 0: {}", output[0]);
        assert!(approx(output[1], 27.0, 1e-5), "sample 1: {}", output[1]);
    }

    #[test]
    fn linear_forward_batch_caches_input() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut linear = Linear::new_orthogonal(3, 2, 1.0, &mut rng);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut output = vec![0.0; 4];

        linear.forward_batch(&input, &mut output, 2);

        assert_eq!(linear.batch_size_cached, 2);
        assert_eq!(linear.batch_input_cache, input);
    }

    #[test]
    fn linear_backward_batch_finite_difference_gradient_check() {
        // Central-difference gradient check against the analytical backward.
        let mut rng = StdRng::seed_from_u64(7);
        let mut linear = Linear::new_orthogonal(3, 2, 1.0, &mut rng);
        let input = vec![0.3, -0.4, 0.2, 0.1, 0.5, -0.7]; // 2 × 3

        // Forward to populate cache
        let mut output = vec![0.0; 4]; // 2 × 2
        linear.forward_batch(&input, &mut output, 2);

        // Analytical gradients: grad_output = all ones (sum loss)
        let grad_output = vec![1.0f32; 4];
        let mut grad_input_analytical = vec![0.0f32; 6];
        linear.backward_batch(&grad_output, &mut grad_input_analytical, 2);

        // Numerical gradient: ∂(sum(output))/∂input[j] for each j
        let eps = 1e-3;
        for s in 0..2 {
            for j in 0..3 {
                let mut input_plus = input.clone();
                let mut input_minus = input.clone();
                input_plus[s * 3 + j] += eps;
                input_minus[s * 3 + j] -= eps;

                let mut out_plus = vec![0.0; 4];
                let mut out_minus = vec![0.0; 4];
                linear.forward_batch(&input_plus, &mut out_plus, 2);
                linear.forward_batch(&input_minus, &mut out_minus, 2);

                let loss_plus: f32 = out_plus.iter().sum();
                let loss_minus: f32 = out_minus.iter().sum();
                let numerical = (loss_plus - loss_minus) / (2.0 * eps);

                let analytical = grad_input_analytical[s * 3 + j];
                assert!(approx(numerical, analytical, 5e-3),
                    "grad mismatch at [s={}, j={}]: analytical={}, numerical={}",
                    s, j, analytical, numerical);
            }
        }
    }

    #[test]
    fn linear_zero_grad_clears_all_gradients() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut linear = Linear::new_orthogonal(3, 2, 1.0, &mut rng);

        // Populate gradients
        linear.grad_weights.iter_mut().for_each(|v| *v = 1.0);
        linear.grad_biases.iter_mut().for_each(|v| *v = 1.0);

        linear.zero_grad();

        assert!(linear.grad_weights.iter().all(|&v| v == 0.0));
        assert!(linear.grad_biases.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn linear_weight_l2_norm_is_sqrt_of_sum_of_squares() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut linear = Linear::new_orthogonal(2, 2, 1.0, &mut rng);
        linear.weights = vec![3.0, 4.0, 0.0, 0.0];
        linear.biases = vec![0.0, 0.0];

        // sqrt(9 + 16 + 0 + 0) = 5
        assert!(approx(linear.weight_l2_norm(), 5.0, 1e-6));
    }

    // ── Tanh ─────────────────────────────────────────────────────────────

    #[test]
    fn tanh_forward_into_writes_tanh_of_input() {
        let tanh = Tanh::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        tanh.forward_into(&input, &mut output);

        for (i, x) in input.iter().enumerate() {
            assert!(approx(output[i], x.tanh(), 1e-6));
        }
    }

    #[test]
    fn tanh_forward_into_output_saturates_in_closed_minus_one_to_one() {
        let tanh = Tanh::new();
        let input = vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
        let mut output = vec![0.0; input.len()];
        tanh.forward_into(&input, &mut output);
        // tanh is mathematically in (-1, 1) but in f32 saturates to ±1 for
        // inputs past ~9 — so the closed-range assertion is what holds.
        for o in output.iter() {
            assert!(*o >= -1.0 && *o <= 1.0, "out of range: {}", o);
        }
        // Sanity: mid-range inputs produce strictly-interior outputs.
        assert!(output[3] == 0.0, "tanh(0) should be exactly 0");
        assert!(output[2] > -1.0 && output[2] < 0.0, "tanh(-1) interior");
        assert!(output[4] > 0.0 && output[4] < 1.0, "tanh(1) interior");
    }

    #[test]
    fn tanh_forward_batch_caches_outputs() {
        let mut tanh = Tanh::new();
        let input = vec![0.5, -0.5, 0.1, -0.1]; // 2 samples × 2 dims
        let mut output = vec![0.0; 4];
        tanh.forward_batch(&input, &mut output, 2, 2);

        for (i, x) in input.iter().enumerate() {
            assert!(approx(tanh.batch_cache()[i], x.tanh(), 1e-6));
        }
    }

    #[test]
    fn tanh_backward_batch_finite_difference_check() {
        // Check ∂tanh/∂x = 1 - tanh(x)² against numerical estimate.
        let mut tanh = Tanh::new();
        let input = vec![0.3, -0.4, 0.2, 0.1];
        let mut output = vec![0.0; 4];
        tanh.forward_batch(&input, &mut output, 2, 2);

        let grad_output = vec![1.0, 1.0, 1.0, 1.0];
        let mut grad_input_analytical = vec![0.0; 4];
        tanh.backward_batch(&grad_output, &mut grad_input_analytical, 2, 2);

        for (i, x) in input.iter().enumerate() {
            let expected = 1.0 - x.tanh().powi(2); // analytical derivative
            assert!(approx(grad_input_analytical[i], expected, 1e-5),
                "grad at i={}: analytical={}, expected={}",
                i, grad_input_analytical[i], expected);
        }
    }
}
