use crate::brain::common::mlp::Linear;

pub struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    /// Decoupled weight decay coefficient (AdamW-style). 0.0 = standard Adam.
    weight_decay: f32,
    t: f32,

    /// First moment (mean) for weights — one flat Vec per layer, mirroring Linear::weights.
    m_weights: Vec<Vec<f32>>,
    /// Second moment (variance) for weights.
    v_weights: Vec<Vec<f32>>,
    /// First moment for biases — one Vec per layer.
    m_biases: Vec<Vec<f32>>,
    /// Second moment for biases.
    v_biases: Vec<Vec<f32>>,
}

impl AdamOptimizer {
    pub fn new(layers: &[&Linear], lr: f32, weight_decay: f32) -> Self {
        let mut m_weights = Vec::new();
        let mut v_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_biases = Vec::new();

        for l in layers {
            let weight_count = l.in_dim * l.out_dim;
            m_weights.push(vec![0.0; weight_count]);
            v_weights.push(vec![0.0; weight_count]);
            m_biases.push(vec![0.0; l.out_dim]);
            v_biases.push(vec![0.0; l.out_dim]);
        }

        Self {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-5,
            weight_decay,
            t: 0.0,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
        }
    }

    pub fn step(&mut self, layers: &mut [&mut Linear]) {
        self.t += 1.0;

        // Precompute bias correction factors once per step (not per weight).
        let bc1 = 1.0 / (1.0 - self.beta1.powi(self.t as i32));
        let bc2 = 1.0 / (1.0 - self.beta2.powi(self.t as i32));

        for (l_idx, layer) in layers.iter_mut().enumerate() {
            let weight_count = layer.in_dim * layer.out_dim;

            // Weight update — flat contiguous iteration
            let mw = &mut self.m_weights[l_idx];
            let vw = &mut self.v_weights[l_idx];
            for k in 0..weight_count {
                let g = layer.grad_weights[k];
                mw[k] = self.beta1 * mw[k] + (1.0 - self.beta1) * g;
                vw[k] = self.beta2 * vw[k] + (1.0 - self.beta2) * g * g;
                let m_hat = mw[k] * bc1;
                let v_hat = vw[k] * bc2;
                layer.weights[k] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                // Decoupled weight decay (AdamW): applied after the Adam step
                if self.weight_decay > 0.0 {
                    layer.weights[k] -= self.learning_rate * self.weight_decay * layer.weights[k];
                }
            }

            // Bias update (no weight decay on biases — standard practice)
            let mb = &mut self.m_biases[l_idx];
            let vb = &mut self.v_biases[l_idx];
            for k in 0..layer.out_dim {
                let g = layer.grad_biases[k];
                mb[k] = self.beta1 * mb[k] + (1.0 - self.beta1) * g;
                vb[k] = self.beta2 * vb[k] + (1.0 - self.beta2) * g * g;
                let m_hat = mb[k] * bc1;
                let v_hat = vb[k] * bc2;
                layer.biases[k] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn adam_step_reduces_loss_on_quadratic_optimisation() {
        // Set up a 1-wide Linear acting as a scalar "weight"; optimise w toward 0
        // by feeding the gradient dL/dw = 2w (from loss = w²).
        let mut rng = StdRng::seed_from_u64(0);
        let mut layer = Linear::new_orthogonal(1, 1, 1.0, &mut rng);
        layer.weights = vec![3.0]; // start far from optimum
        layer.biases = vec![0.0];
        let initial = layer.weights[0].abs();

        let mut opt = AdamOptimizer::new(&[&layer], 0.1, 0.0);

        // 200 steps: at each step, gradient = 2*w (derivative of w²).
        // Adam should drive w toward 0.
        for _ in 0..200 {
            layer.grad_weights = vec![2.0 * layer.weights[0]];
            layer.grad_biases = vec![0.0];
            opt.step(&mut [&mut layer]);
        }

        let final_abs = layer.weights[0].abs();
        assert!(final_abs < initial, "Adam failed to descend: start={}, end={}", initial, final_abs);
        assert!(final_abs < 0.2, "Adam did not converge close to optimum: end={}", final_abs);
    }

    #[test]
    fn adam_step_zero_gradient_leaves_weights_essentially_unchanged() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut layer = Linear::new_orthogonal(2, 2, 1.0, &mut rng);
        let initial_weights = layer.weights.clone();
        layer.grad_weights = vec![0.0; 4];
        layer.grad_biases = vec![0.0; 2];

        let mut opt = AdamOptimizer::new(&[&layer], 0.1, 0.0);
        opt.step(&mut [&mut layer]);

        // m and v remain 0 → update is 0; weights should be bit-identical.
        for (a, b) in layer.weights.iter().zip(initial_weights.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
    }

    #[test]
    fn adamw_weight_decay_shrinks_weights_toward_zero() {
        // With grad=0 and weight_decay>0, AdamW should shrink weights.
        let mut rng = StdRng::seed_from_u64(2);
        let mut layer = Linear::new_orthogonal(2, 2, 1.0, &mut rng);
        layer.weights = vec![1.0, 1.0, 1.0, 1.0];
        layer.biases = vec![0.0, 0.0];
        layer.grad_weights = vec![0.0; 4];
        layer.grad_biases = vec![0.0; 2];

        let mut opt = AdamOptimizer::new(&[&layer], 0.1, 0.01);
        opt.step(&mut [&mut layer]);

        // Each weight shrunk by 0.1 * 0.01 = 0.001 per step
        for w in &layer.weights {
            assert!(*w < 1.0, "weight did not decay: {}", w);
            assert!(*w > 0.998, "weight decayed too much: {}", w);
        }
    }

    #[test]
    fn adam_weight_decay_zero_is_pure_adam() {
        // With weight_decay=0 and grad=0, weights should be bit-unchanged.
        let mut rng = StdRng::seed_from_u64(3);
        let mut layer = Linear::new_orthogonal(3, 3, 1.0, &mut rng);
        let initial = layer.weights.clone();
        layer.grad_weights = vec![0.0; 9];
        layer.grad_biases = vec![0.0; 3];

        let mut opt = AdamOptimizer::new(&[&layer], 0.5, 0.0);
        opt.step(&mut [&mut layer]);
        opt.step(&mut [&mut layer]);

        for (a, b) in layer.weights.iter().zip(initial.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
