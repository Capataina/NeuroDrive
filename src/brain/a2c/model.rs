use crate::brain::common::mlp::{Linear, Tanh};
use crate::brain::common::optim::AdamOptimizer;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct ActionDist {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

pub struct ActorCritic {
    // Actor
    pub a_fc1: Linear,
    pub a_tanh1: Tanh,
    pub a_fc2: Linear,
    pub a_tanh2: Tanh,
    pub a_mean: Linear,
    pub a_log_std: Vec<f32>, // Learnable parameters
    pub a_log_std_grad: Vec<f32>,

    // Critic
    pub c_fc1: Linear,
    pub c_tanh1: Tanh,
    pub c_fc2: Linear,
    pub c_tanh2: Tanh,
    pub c_value: Linear,

    // Optimizer
    pub a_opt: AdamOptimizer,
    pub c_opt: AdamOptimizer,
    pub log_std_opt_m: Vec<f32>,
    pub log_std_opt_v: Vec<f32>,
    pub opt_t: f32,
}

impl ActorCritic {
    pub fn new(in_dim: usize, hidden_dim: usize, act_dim: usize, rng: &mut impl Rng) -> Self {
        let sqrt2 = 2.0f32.sqrt();
        let a_fc1 = Linear::new_orthogonal(in_dim, hidden_dim, sqrt2, rng);
        let a_fc2 = Linear::new_orthogonal(hidden_dim, hidden_dim, sqrt2, rng);
        let a_mean = Linear::new_orthogonal(hidden_dim, act_dim, 0.01, rng);

        let c_fc1 = Linear::new_orthogonal(in_dim, hidden_dim, sqrt2, rng);
        let c_fc2 = Linear::new_orthogonal(hidden_dim, hidden_dim, sqrt2, rng);
        let c_value = Linear::new_orthogonal(hidden_dim, 1, 1.0, rng);

        let a_opt = AdamOptimizer::new(&[&a_fc1, &a_fc2, &a_mean], 3e-4);
        let c_opt = AdamOptimizer::new(&[&c_fc1, &c_fc2, &c_value], 5e-4);

        Self {
            a_fc1,
            a_tanh1: Tanh::new(),
            a_fc2,
            a_tanh2: Tanh::new(),
            a_mean,
            a_log_std: vec![0.0; act_dim],
            a_log_std_grad: vec![0.0; act_dim],

            c_fc1,
            c_tanh1: Tanh::new(),
            c_fc2,
            c_tanh2: Tanh::new(),
            c_value,

            a_opt,
            c_opt,
            log_std_opt_m: vec![0.0; act_dim],
            log_std_opt_v: vec![0.0; act_dim],
            opt_t: 0.0,
        }
    }

    pub fn forward(&mut self, obs: &[f32]) -> (ActionDist, f32) {
        // Actor
        let a1 = self.a_fc1.forward(obs);
        let a1_r = self.a_tanh1.forward(&a1);
        let a2 = self.a_fc2.forward(&a1_r);
        let a2_r = self.a_tanh2.forward(&a2);
        let mean = self.a_mean.forward(&a2_r);
        let std = self.a_log_std.iter().map(|&ls| ls.exp()).collect();

        // Critic
        let c1 = self.c_fc1.forward(obs);
        let c1_r = self.c_tanh1.forward(&c1);
        let c2 = self.c_fc2.forward(&c1_r);
        let c2_r = self.c_tanh2.forward(&c2);
        let value = self.c_value.forward(&c2_r)[0];

        (ActionDist { mean, std }, value)
    }

    pub fn zero_grad(&mut self) {
        self.a_fc1.zero_grad();
        self.a_fc2.zero_grad();
        self.a_mean.zero_grad();
        self.c_fc1.zero_grad();
        self.c_fc2.zero_grad();
        self.c_value.zero_grad();
        for g in &mut self.a_log_std_grad {
            *g = 0.0;
        }
    }
}
