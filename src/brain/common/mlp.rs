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
