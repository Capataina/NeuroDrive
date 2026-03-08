use crate::brain::common::mlp::Linear;

pub struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: f32,
    
    // State arrays mirroring weights and biases
    m_weights: Vec<Vec<Vec<f32>>>, // [layer][out][in]
    v_weights: Vec<Vec<Vec<f32>>>,
    m_biases: Vec<Vec<f32>>, // [layer][out]
    v_biases: Vec<Vec<f32>>,
}

impl AdamOptimizer {
    pub fn new(layers: &[&Linear], lr: f32) -> Self {
        let mut m_weights = Vec::new();
        let mut v_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_biases = Vec::new();

        for l in layers {
            let out_dim = l.weights.len();
            let in_dim = if out_dim > 0 { l.weights[0].len() } else { 0 };

            m_weights.push(vec![vec![0.0; in_dim]; out_dim]);
            v_weights.push(vec![vec![0.0; in_dim]; out_dim]);
            m_biases.push(vec![0.0; out_dim]);
            v_biases.push(vec![0.0; out_dim]);
        }

        Self {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0.0,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
        }
    }

    pub fn step(&mut self, layers: &mut [&mut Linear]) {
        self.t += 1.0;

        for (l_idx, layer) in layers.iter_mut().enumerate() {
            let out_dim = layer.weights.len();
            let in_dim = if out_dim > 0 { layer.weights[0].len() } else { 0 };

            for i in 0..out_dim {
                // Bias update
                let g_b = layer.grad_biases[i];
                self.m_biases[l_idx][i] = self.beta1 * self.m_biases[l_idx][i] + (1.0 - self.beta1) * g_b;
                self.v_biases[l_idx][i] = self.beta2 * self.v_biases[l_idx][i] + (1.0 - self.beta2) * g_b * g_b;

                let m_hat_b = self.m_biases[l_idx][i] / (1.0 - self.beta1.powf(self.t));
                let v_hat_b = self.v_biases[l_idx][i] / (1.0 - self.beta2.powf(self.t));

                layer.biases[i] -= self.learning_rate * m_hat_b / (v_hat_b.sqrt() + self.epsilon);

                // Weights update
                for j in 0..in_dim {
                    let g_w = layer.grad_weights[i][j];
                    self.m_weights[l_idx][i][j] = self.beta1 * self.m_weights[l_idx][i][j] + (1.0 - self.beta1) * g_w;
                    self.v_weights[l_idx][i][j] = self.beta2 * self.v_weights[l_idx][i][j] + (1.0 - self.beta2) * g_w * g_w;

                    let m_hat_w = self.m_weights[l_idx][i][j] / (1.0 - self.beta1.powf(self.t));
                    let v_hat_w = self.v_weights[l_idx][i][j] / (1.0 - self.beta2.powf(self.t));

                    layer.weights[i][j] -= self.learning_rate * m_hat_w / (v_hat_w.sqrt() + self.epsilon);
                }
            }
        }
    }
}
