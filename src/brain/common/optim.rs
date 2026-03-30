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
