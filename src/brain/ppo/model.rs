use crate::brain::common::mlp::{Linear, Tanh};
use crate::brain::common::optim::AdamOptimizer;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct ActionDist {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// Pre-allocated scratch buffers for batched forward and backward passes.
/// Allocated once at construction; reused every training chunk.
pub struct BatchScratch {
    pub actor_hidden_dim: usize,
    pub critic_hidden_dim: usize,
    pub act_dim: usize,
    pub obs_dim: usize,

    // Actor forward intermediates  [batch × actor_hidden]
    pub a_h1: Vec<f32>,
    pub a_h1_act: Vec<f32>,
    pub a_h2: Vec<f32>,
    pub a_h2_act: Vec<f32>,
    pub a_out: Vec<f32>,

    // Critic forward intermediates [batch × critic_hidden]
    pub c_h1: Vec<f32>,
    pub c_h1_act: Vec<f32>,
    pub c_h2: Vec<f32>,
    pub c_h2_act: Vec<f32>,
    pub c_out: Vec<f32>,

    // Backward intermediates (actor) — uses actor_hidden
    pub ga_out: Vec<f32>,
    pub ga_h2_act: Vec<f32>,
    pub ga_h2: Vec<f32>,
    pub ga_h1_act: Vec<f32>,
    pub ga_h1: Vec<f32>,
    pub ga_input: Vec<f32>,

    // Backward intermediates (critic) — uses critic_hidden
    pub gc_out: Vec<f32>,
    pub gc_h2_act: Vec<f32>,
    pub gc_h2: Vec<f32>,
    pub gc_h1_act: Vec<f32>,
    pub gc_h1: Vec<f32>,
    pub gc_input: Vec<f32>,

    // Pre-allocated buffers for PPO loss computation to avoid per-chunk allocations
    pub obs_batch: Vec<f32>,
    pub grad_seed_values: Vec<f32>,
    pub grad_seed_means: Vec<f32>,
}

impl BatchScratch {
    fn new(max_batch: usize, obs_dim: usize, actor_hidden: usize, critic_hidden: usize, act_dim: usize) -> Self {
        let bah = max_batch * actor_hidden;
        let bch = max_batch * critic_hidden;
        let ba = max_batch * act_dim;
        let bo = max_batch * obs_dim;
        let bv = max_batch;
        Self {
            actor_hidden_dim: actor_hidden,
            critic_hidden_dim: critic_hidden,
            act_dim,
            obs_dim,
            a_h1: vec![0.0; bah],
            a_h1_act: vec![0.0; bah],
            a_h2: vec![0.0; bah],
            a_h2_act: vec![0.0; bah],
            a_out: vec![0.0; ba],
            c_h1: vec![0.0; bch],
            c_h1_act: vec![0.0; bch],
            c_h2: vec![0.0; bch],
            c_h2_act: vec![0.0; bch],
            c_out: vec![0.0; bv],
            ga_out: vec![0.0; ba],
            ga_h2_act: vec![0.0; bah],
            ga_h2: vec![0.0; bah],
            ga_h1_act: vec![0.0; bah],
            ga_h1: vec![0.0; bah],
            ga_input: vec![0.0; bo],
            gc_out: vec![0.0; bv],
            gc_h2_act: vec![0.0; bch],
            gc_h2: vec![0.0; bch],
            gc_h1_act: vec![0.0; bch],
            gc_h1: vec![0.0; bch],
            gc_input: vec![0.0; bo],

            obs_batch: vec![0.0; bo],
            grad_seed_values: vec![0.0; bv],
            grad_seed_means: vec![0.0; ba],
        }
    }
}

pub struct ActorCritic {
    // Actor
    pub a_fc1: Linear,
    pub a_tanh1: Tanh,
    pub a_fc2: Linear,
    pub a_tanh2: Tanh,
    pub a_mean: Linear,
    pub a_log_std: Vec<f32>,
    pub a_log_std_grad: Vec<f32>,

    // Critic
    pub c_fc1: Linear,
    pub c_tanh1: Tanh,
    pub c_fc2: Linear,
    pub c_tanh2: Tanh,
    pub c_value: Linear,

    // Optimizers
    pub a_opt: AdamOptimizer,
    pub c_opt: AdamOptimizer,
    pub log_std_opt_m: Vec<f32>,
    pub log_std_opt_v: Vec<f32>,
    pub opt_t: f32,

    // Batch scratch buffers
    pub scratch: BatchScratch,
}

impl ActorCritic {
    pub fn new(
        in_dim: usize,
        actor_hidden_dim: usize,
        critic_hidden_dim: usize,
        act_dim: usize,
        actor_lr: f32,
        critic_lr: f32,
        critic_weight_decay: f32,
        rng: &mut impl Rng,
    ) -> Self {
        let sqrt2 = 2.0f32.sqrt();
        let a_fc1 = Linear::new_orthogonal(in_dim, actor_hidden_dim, sqrt2, rng);
        let a_fc2 = Linear::new_orthogonal(actor_hidden_dim, actor_hidden_dim, sqrt2, rng);
        let a_mean = Linear::new_orthogonal(actor_hidden_dim, act_dim, 0.01, rng);

        let c_fc1 = Linear::new_orthogonal(in_dim, critic_hidden_dim, sqrt2, rng);
        let c_fc2 = Linear::new_orthogonal(critic_hidden_dim, critic_hidden_dim, sqrt2, rng);
        let c_value = Linear::new_orthogonal(critic_hidden_dim, 1, 1.0, rng);

        let a_opt = AdamOptimizer::new(&[&a_fc1, &a_fc2, &a_mean], actor_lr, 0.0);
        let c_opt = AdamOptimizer::new(&[&c_fc1, &c_fc2, &c_value], critic_lr, critic_weight_decay);

        let max_batch = 512;
        let scratch = BatchScratch::new(max_batch, in_dim, actor_hidden_dim, critic_hidden_dim, act_dim);

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

            scratch,
        }
    }

    /// Actor-only single-sample forward pass (used during action selection).
    /// Skips the critic — saves ~50% of the forward cost per car.
    pub fn forward_actor(&mut self, obs: &[f32]) -> ActionDist {
        let a1 = self.a_fc1.forward(obs);
        let a1_r = self.a_tanh1.forward(&a1);
        let a2 = self.a_fc2.forward(&a1_r);
        let a2_r = self.a_tanh2.forward(&a2);
        let mean = self.a_mean.forward(&a2_r);
        let std = self.a_log_std.iter().map(|&ls| ls.exp()).collect();
        ActionDist { mean, std }
    }

    /// Critic-only single-sample forward pass (used for bootstrap values).
    pub fn forward_critic(&mut self, obs: &[f32]) -> f32 {
        let c1 = self.c_fc1.forward(obs);
        let c1_r = self.c_tanh1.forward(&c1);
        let c2 = self.c_fc2.forward(&c1_r);
        let c2_r = self.c_tanh2.forward(&c2);
        self.c_value.forward(&c2_r)[0]
    }

    /// Full single-sample forward pass (actor + critic).
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

    /// Batched critic-only forward pass for action selection.
    ///
    /// `obs_batch` is row-major `[batch_size × obs_dim]`.
    /// Returns value predictions in `scratch.c_out[0..batch_size]`.
    ///
    /// Does **not** cache intermediates for backward — this is inference only.
    pub fn forward_critic_batch(&mut self, obs_batch: &[f32], batch_size: usize) {
        let ch = self.scratch.critic_hidden_dim;
        self.c_fc1.forward_batch(obs_batch, &mut self.scratch.c_h1, batch_size);
        self.c_tanh1.forward_batch(&self.scratch.c_h1, &mut self.scratch.c_h1_act, batch_size, ch);
        self.c_fc2.forward_batch(&self.scratch.c_h1_act, &mut self.scratch.c_h2, batch_size);
        self.c_tanh2.forward_batch(&self.scratch.c_h2, &mut self.scratch.c_h2_act, batch_size, ch);
        self.c_value.forward_batch(&self.scratch.c_h2_act, &mut self.scratch.c_out, batch_size);
    }

    /// Batched forward pass through actor + critic.
    ///
    /// `obs_batch` is row-major `[batch_size × obs_dim]`.
    ///
    /// After this call, results are in:
    /// - `scratch.a_out[s * act_dim + j]` = actor mean for sample `s`, action `j`
    /// - `scratch.c_out[s]` = critic value for sample `s`
    ///
    /// Caches all intermediates for `backward_batch`.
    pub fn forward_batch(&mut self, obs_batch: &[f32], batch_size: usize) {
        let ah = self.scratch.actor_hidden_dim;
        let ch = self.scratch.critic_hidden_dim;
        // Actor path
        self.a_fc1.forward_batch(obs_batch, &mut self.scratch.a_h1, batch_size);
        self.a_tanh1.forward_batch(&self.scratch.a_h1, &mut self.scratch.a_h1_act, batch_size, ah);
        self.a_fc2.forward_batch(&self.scratch.a_h1_act, &mut self.scratch.a_h2, batch_size);
        self.a_tanh2.forward_batch(&self.scratch.a_h2, &mut self.scratch.a_h2_act, batch_size, ah);
        self.a_mean.forward_batch(&self.scratch.a_h2_act, &mut self.scratch.a_out, batch_size);

        // Critic path
        self.c_fc1.forward_batch(obs_batch, &mut self.scratch.c_h1, batch_size);
        self.c_tanh1.forward_batch(&self.scratch.c_h1, &mut self.scratch.c_h1_act, batch_size, ch);
        self.c_fc2.forward_batch(&self.scratch.c_h1_act, &mut self.scratch.c_h2, batch_size);
        self.c_tanh2.forward_batch(&self.scratch.c_h2, &mut self.scratch.c_h2_act, batch_size, ch);
        self.c_value.forward_batch(&self.scratch.c_h2_act, &mut self.scratch.c_out, batch_size);
    }

    /// Batched backward pass through critic.
    ///
    /// `grad_values` is `[batch_size × 1]` — the gradient seed for each sample's value output.
    pub fn backward_batch_critic(&mut self, grad_values: &[f32], batch_size: usize) {
        let h = self.scratch.critic_hidden_dim;
        self.c_value.backward_batch(grad_values, &mut self.scratch.gc_h2_act, batch_size);
        self.c_tanh2.backward_batch(&self.scratch.gc_h2_act, &mut self.scratch.gc_h2, batch_size, h);
        self.c_fc2.backward_batch(&self.scratch.gc_h2, &mut self.scratch.gc_h1_act, batch_size);
        self.c_tanh1.backward_batch(&self.scratch.gc_h1_act, &mut self.scratch.gc_h1, batch_size, h);
        self.c_fc1.backward_batch(&self.scratch.gc_h1, &mut self.scratch.gc_input, batch_size);
        // gc_input is grad w.r.t. observations — not needed, just discarded.
    }

    /// Batched backward pass through actor.
    ///
    /// `grad_means` is `[batch_size × act_dim]` — the gradient seed for each sample's mean output.
    pub fn backward_batch_actor(&mut self, grad_means: &[f32], batch_size: usize) {
        let h = self.scratch.actor_hidden_dim;
        self.a_mean.backward_batch(grad_means, &mut self.scratch.ga_h2_act, batch_size);
        self.a_tanh2.backward_batch(&self.scratch.ga_h2_act, &mut self.scratch.ga_h2, batch_size, h);
        self.a_fc2.backward_batch(&self.scratch.ga_h2, &mut self.scratch.ga_h1_act, batch_size);
        self.a_tanh1.backward_batch(&self.scratch.ga_h1_act, &mut self.scratch.ga_h1, batch_size, h);
        self.a_fc1.backward_batch(&self.scratch.ga_h1, &mut self.scratch.ga_input, batch_size);
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
