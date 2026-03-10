use crate::brain::common::math::{glorot_uniform, zeros};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Linear {
    pub weights: Vec<Vec<f32>>, // [out_dim][in_dim]
    pub biases: Vec<f32>,       // [out_dim]
    pub grad_weights: Vec<Vec<f32>>,
    pub grad_biases: Vec<f32>,
    pub input_cache: Option<Vec<f32>>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            weights: glorot_uniform(out_dim, in_dim, rng),
            biases: zeros(out_dim),
            grad_weights: vec![vec![0.0; in_dim]; out_dim],
            grad_biases: zeros(out_dim),
            input_cache: None,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_cache = Some(input.to_vec());
        let mut output = vec![0.0; self.biases.len()];
        for i in 0..self.biases.len() {
            let mut sum = self.biases[i];
            for j in 0..input.len() {
                sum += self.weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        output
    }

    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let input = self.input_cache.as_ref().expect("Must call forward first");
        let out_dim = self.biases.len();
        let in_dim = input.len();

        let mut grad_input = vec![0.0; in_dim];

        for i in 0..out_dim {
            self.grad_biases[i] += grad_output[i];
            for j in 0..in_dim {
                self.grad_weights[i][j] += grad_output[i] * input[j];
                grad_input[j] += self.weights[i][j] * grad_output[i];
            }
        }
        grad_input
    }

    pub fn zero_grad(&mut self) {
        for row in &mut self.grad_weights {
            for val in row {
                *val = 0.0;
            }
        }
        for val in &mut self.grad_biases {
            *val = 0.0;
        }
    }

    pub fn weight_l2_norm(&self) -> f32 {
        let weight_sum = self
            .weights
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| value * value)
            .sum::<f32>();
        let bias_sum = self.biases.iter().map(|value| value * value).sum::<f32>();
        (weight_sum + bias_sum).sqrt()
    }

    pub fn grad_l2_norm(&self) -> f32 {
        let weight_sum = self
            .grad_weights
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| value * value)
            .sum::<f32>();
        let bias_sum = self
            .grad_biases
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        (weight_sum + bias_sum).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct Relu {
    pub input_cache: Option<Vec<f32>>,
}

impl Relu {
    pub fn new() -> Self {
        Self { input_cache: None }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_cache = Some(input.to_vec());
        input
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect()
    }

    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let input = self.input_cache.as_ref().unwrap();
        input
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect()
    }
}
