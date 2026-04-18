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

    /// Batch forward: output = input × weights^T + bias (broadcast).
    ///
    /// `input`  is row-major `[batch_size × in_dim]`.
    /// `output` is row-major `[batch_size × out_dim]`, must be pre-allocated.
    ///
    /// Caches `input` for `backward_batch`.
    pub fn forward_batch(&mut self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Cache input for backward
        self.batch_input_cache.resize(batch_size * self.in_dim, 0.0);
        self.batch_input_cache.copy_from_slice(&input[..batch_size * self.in_dim]);
        self.batch_size_cached = batch_size;

        // output[s, i] = bias[i] + Σ_j input[s, j] * weights[i, j]
        // Using s-i-j loop order so `weights[i * in_dim .. (i+1) * in_dim]` is
        // read sequentially per output neuron, giving cache-friendly access on
        // the row-major weight layout.
        // First: fill with biases (broadcast)
        for s in 0..batch_size {
            let out_row = &mut output[s * self.out_dim..(s + 1) * self.out_dim];
            out_row.copy_from_slice(&self.biases);
        }
        // Then: accumulate weight contributions
        for s in 0..batch_size {
            let in_row = &input[s * self.in_dim..(s + 1) * self.in_dim];
            let out_row = &mut output[s * self.out_dim..(s + 1) * self.out_dim];
            for i in 0..self.out_dim {
                let w_row = &self.weights[i * self.in_dim..(i + 1) * self.in_dim];
                let mut sum = 0.0f32;
                for j in 0..self.in_dim {
                    sum += w_row[j] * in_row[j];
                }
                out_row[i] += sum;
            }
        }
    }

    /// Batch backward. Accumulates into grad_weights / grad_biases.
    ///
    /// `grad_output` is `[batch_size × out_dim]`.
    /// `grad_input`  is `[batch_size × in_dim]`, must be pre-allocated (will be overwritten).
    pub fn backward_batch(&mut self, grad_output: &[f32], grad_input: &mut [f32], batch_size: usize) {
        // Zero grad_input
        grad_input[..batch_size * self.in_dim].iter_mut().for_each(|x| *x = 0.0);

        // grad_biases += sum over batch of grad_output
        // grad_weights += grad_output^T × input_cache  (accumulated across batch)
        // grad_input = grad_output × weights
        for s in 0..batch_size {
            let go_row = &grad_output[s * self.out_dim..(s + 1) * self.out_dim];
            let in_row = &self.batch_input_cache[s * self.in_dim..(s + 1) * self.in_dim];
            let gi_row = &mut grad_input[s * self.in_dim..(s + 1) * self.in_dim];

            for i in 0..self.out_dim {
                let g = go_row[i];
                self.grad_biases[i] += g;
                let row_start = i * self.in_dim;
                let w_row = &self.weights[row_start..row_start + self.in_dim];
                let gw_row = &mut self.grad_weights[row_start..row_start + self.in_dim];

                for j in 0..self.in_dim {
                    gw_row[j] += g * in_row[j];
                    gi_row[j] += w_row[j] * g;
                }
            }
        }
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
