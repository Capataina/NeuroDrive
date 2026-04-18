use crate::brain::common::mlp::{Linear, Tanh};
use crate::brain::common::optim::AdamOptimizer;
use rand::Rng;

/// Policy distribution parameters emitted by a single-sample actor forward.
///
/// Stored as fixed-size arrays because the environment's action dimension
/// is structurally 2 (steering + throttle); a `Vec<f32>` here would incur a
/// heap allocation every tick per car purely to hold two floats.
#[derive(Clone, Copy, Debug)]
pub struct ActionDist {
    pub mean: [f32; 2],
    pub std: [f32; 2],
}

/// Per-sample scratch buffers for the single-sample forward paths
/// (`forward_actor`, `forward_critic`, `forward`). These paths run on the
/// action-selection hot tick (every car, every tick, 60 Hz), so fresh
/// `Vec` allocations per call were costly. Buffers are sized for the
/// actor/critic hidden dims at construction and reused every call.
///
/// Separate from `BatchScratch` because the single-sample path uses
/// `forward_into` (slice-based) while the batched path uses `forward_batch`
/// (different buffer sizes and cache semantics).
pub struct SampleScratch {
    // Actor intermediates [actor_hidden]
    pub a_h1: Vec<f32>,
    pub a_h1_act: Vec<f32>,
    pub a_h2: Vec<f32>,
    pub a_h2_act: Vec<f32>,
    // Actor output [act_dim]
    pub a_out: Vec<f32>,
    // Critic intermediates [critic_hidden]
    pub c_h1: Vec<f32>,
    pub c_h1_act: Vec<f32>,
    pub c_h2: Vec<f32>,
    pub c_h2_act: Vec<f32>,
}

impl SampleScratch {
    fn new(actor_hidden: usize, critic_hidden: usize, act_dim: usize) -> Self {
        Self {
            a_h1: vec![0.0; actor_hidden],
            a_h1_act: vec![0.0; actor_hidden],
            a_h2: vec![0.0; actor_hidden],
            a_h2_act: vec![0.0; actor_hidden],
            a_out: vec![0.0; act_dim],
            c_h1: vec![0.0; critic_hidden],
            c_h1_act: vec![0.0; critic_hidden],
            c_h2: vec![0.0; critic_hidden],
            c_h2_act: vec![0.0; critic_hidden],
        }
    }
}

/// Input-side scratch: batch-stacked observations and the gradient seeds
/// that feed `backward_batch_*`. Kept as a separate struct from
/// `BatchScratch` so Rust's disjoint-field borrow inference lets the update
/// loop write into these buffers while the forward/backward methods
/// simultaneously borrow `&mut self.scratch`.
///
/// This is the structural fix for the previous `unsafe { slice::from_raw_parts }`
/// pattern — keeping inputs and intermediates as sibling fields on
/// `ActorCritic` is the canonical Rust idiom (cf. nomicon §"Splitting Borrows").
pub struct BatchIo {
    pub obs_batch: Vec<f32>,
    pub grad_seed_values: Vec<f32>,
    pub grad_seed_means: Vec<f32>,
}

impl BatchIo {
    fn new(max_batch: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            obs_batch: vec![0.0; max_batch * obs_dim],
            grad_seed_values: vec![0.0; max_batch],
            grad_seed_means: vec![0.0; max_batch * act_dim],
        }
    }
}

/// Pre-allocated scratch buffers for batched forward and backward passes.
/// Allocated once at construction; reused every training chunk. Contains
/// only the forward and backward *intermediate* buffers — inputs and
/// gradient seeds live in the sibling `BatchIo` struct so the two can be
/// borrowed independently.
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

    // Single-sample scratch (hot-tick action selection / bootstrap).
    pub sample_scratch: SampleScratch,

    // Batch scratch buffers — split into inputs vs. intermediates so
    // update.rs can borrow inputs and intermediates disjointly without
    // needing raw-pointer aliasing.
    pub batch_io: BatchIo,
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
        let sample_scratch = SampleScratch::new(actor_hidden_dim, critic_hidden_dim, act_dim);
        let batch_io = BatchIo::new(max_batch, in_dim, act_dim);
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

            sample_scratch,
            batch_io,
            scratch,
        }
    }

    /// Actor-only single-sample forward pass (used during action selection).
    /// Skips the critic — saves ~50% of the forward cost per car.
    ///
    /// Allocation-free: all intermediates live in `self.sample_scratch`.
    /// At 8 cars × 60 Hz that previously amounted to 2,880 heap allocations
    /// per second on the hot path; now zero.
    pub fn forward_actor(&mut self, obs: &[f32]) -> ActionDist {
        self.a_fc1.forward_into(obs, &mut self.sample_scratch.a_h1);
        self.a_tanh1.forward_into(&self.sample_scratch.a_h1, &mut self.sample_scratch.a_h1_act);
        self.a_fc2.forward_into(&self.sample_scratch.a_h1_act, &mut self.sample_scratch.a_h2);
        self.a_tanh2.forward_into(&self.sample_scratch.a_h2, &mut self.sample_scratch.a_h2_act);
        self.a_mean.forward_into(&self.sample_scratch.a_h2_act, &mut self.sample_scratch.a_out);
        ActionDist {
            mean: [self.sample_scratch.a_out[0], self.sample_scratch.a_out[1]],
            std: [self.a_log_std[0].exp(), self.a_log_std[1].exp()],
        }
    }

    /// Critic-only single-sample forward pass (used for bootstrap values).
    /// Allocation-free on the hot path.
    pub fn forward_critic(&mut self, obs: &[f32]) -> f32 {
        self.c_fc1.forward_into(obs, &mut self.sample_scratch.c_h1);
        self.c_tanh1.forward_into(&self.sample_scratch.c_h1, &mut self.sample_scratch.c_h1_act);
        self.c_fc2.forward_into(&self.sample_scratch.c_h1_act, &mut self.sample_scratch.c_h2);
        self.c_tanh2.forward_into(&self.sample_scratch.c_h2, &mut self.sample_scratch.c_h2_act);
        let mut value_out = [0.0f32; 1];
        self.c_value.forward_into(&self.sample_scratch.c_h2_act, &mut value_out);
        value_out[0]
    }

    /// Full single-sample forward pass (actor + critic).
    /// Used only off the hot path (on-exit flush bootstrap); retained for
    /// API parity with `forward_actor` / `forward_critic`. Allocation-free.
    pub fn forward(&mut self, obs: &[f32]) -> (ActionDist, f32) {
        // Actor path
        self.a_fc1.forward_into(obs, &mut self.sample_scratch.a_h1);
        self.a_tanh1.forward_into(&self.sample_scratch.a_h1, &mut self.sample_scratch.a_h1_act);
        self.a_fc2.forward_into(&self.sample_scratch.a_h1_act, &mut self.sample_scratch.a_h2);
        self.a_tanh2.forward_into(&self.sample_scratch.a_h2, &mut self.sample_scratch.a_h2_act);
        self.a_mean.forward_into(&self.sample_scratch.a_h2_act, &mut self.sample_scratch.a_out);
        let dist = ActionDist {
            mean: [self.sample_scratch.a_out[0], self.sample_scratch.a_out[1]],
            std: [self.a_log_std[0].exp(), self.a_log_std[1].exp()],
        };

        // Critic path
        self.c_fc1.forward_into(obs, &mut self.sample_scratch.c_h1);
        self.c_tanh1.forward_into(&self.sample_scratch.c_h1, &mut self.sample_scratch.c_h1_act);
        self.c_fc2.forward_into(&self.sample_scratch.c_h1_act, &mut self.sample_scratch.c_h2);
        self.c_tanh2.forward_into(&self.sample_scratch.c_h2, &mut self.sample_scratch.c_h2_act);
        let mut value_out = [0.0f32; 1];
        self.c_value.forward_into(&self.sample_scratch.c_h2_act, &mut value_out);

        (dist, value_out[0])
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
    /// Reads the batch-stacked observations from `self.batch_io.obs_batch`
    /// (caller fills it before the call, up to `batch_size * obs_dim`).
    ///
    /// After this call, results are in:
    /// - `scratch.a_out[s * act_dim + j]` = actor mean for sample `s`, action `j`
    /// - `scratch.c_out[s]` = critic value for sample `s`
    ///
    /// Caches all intermediates for `backward_batch`.
    pub fn forward_batch(&mut self, batch_size: usize) {
        let ah = self.scratch.actor_hidden_dim;
        let ch = self.scratch.critic_hidden_dim;
        let obs_dim = self.scratch.obs_dim;
        let obs_batch = &self.batch_io.obs_batch[..batch_size * obs_dim];
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
    /// Reads gradient seeds from `self.batch_io.grad_seed_values[..batch_size]`;
    /// caller writes them before the call.
    pub fn backward_batch_critic(&mut self, batch_size: usize) {
        let h = self.scratch.critic_hidden_dim;
        let grad_values = &self.batch_io.grad_seed_values[..batch_size];
        self.c_value.backward_batch(grad_values, &mut self.scratch.gc_h2_act, batch_size);
        self.c_tanh2.backward_batch(&self.scratch.gc_h2_act, &mut self.scratch.gc_h2, batch_size, h);
        self.c_fc2.backward_batch(&self.scratch.gc_h2, &mut self.scratch.gc_h1_act, batch_size);
        self.c_tanh1.backward_batch(&self.scratch.gc_h1_act, &mut self.scratch.gc_h1, batch_size, h);
        self.c_fc1.backward_batch(&self.scratch.gc_h1, &mut self.scratch.gc_input, batch_size);
        // gc_input is grad w.r.t. observations — not needed, just discarded.
    }

    /// Batched backward pass through actor.
    ///
    /// Reads gradient seeds from `self.batch_io.grad_seed_means[..batch_size * act_dim]`;
    /// caller writes them before the call.
    pub fn backward_batch_actor(&mut self, batch_size: usize) {
        let h = self.scratch.actor_hidden_dim;
        let act_dim = self.scratch.act_dim;
        let grad_means = &self.batch_io.grad_seed_means[..batch_size * act_dim];
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
