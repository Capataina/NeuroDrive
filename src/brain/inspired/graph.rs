//! Sparse directed graph storage for the brain-inspired learner.
//!
//! The graph is stored as two slot-stable `Vec`s — one for neurons, one for
//! synapses — with `alive` flags to support low-cost apoptosis / pruning
//! without invalidating IDs. Freed slots accumulate in free-lists and are
//! reused by neurogenesis / sprouting.
//!
//! ## Design decisions
//!
//! - **Slot stability**: `NeuronId` / `SynapseId` are stable for the lifetime
//!   of a node or edge. Structural plasticity marks them dead and pushes the
//!   slot onto the free-list; new neurons / synapses reuse slots before
//!   growing the `Vec`s.
//! - **No self-loops**: the seed graph and sprouting forbid `source == target`.
//! - **Eligibility per synapse per car**: each `Synapse` owns a `Vec<f32>`
//!   indexed by `env_id` so the learner tracks one eligibility trace per
//!   embodiment of the single shared graph.
//!
//! See `context/notes/brain-v1-design.md` for the biological rationale.

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Normal};

use super::config::BrainInspiredConfig;

/// Stable index into `BrainGraph.neurons`.
pub type NeuronId = u32;
/// Stable index into `BrainGraph.synapses`.
pub type SynapseId = u32;

/// What role a neuron plays in the I/O contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuronRole {
    /// Input neuron bound to observation dimension `idx`.
    Input(u8),
    /// Plastic hidden neuron.
    Hidden,
    /// Output neuron bound to action dimension `idx` (0 = steering, 1 = throttle).
    Output(u8),
}

impl NeuronRole {
    pub fn is_input(self) -> bool {
        matches!(self, Self::Input(_))
    }
    pub fn is_output(self) -> bool {
        matches!(self, Self::Output(_))
    }
    pub fn is_hidden(self) -> bool {
        matches!(self, Self::Hidden)
    }
}

/// A neuron in the sparse graph.
#[derive(Clone, Debug)]
pub struct Neuron {
    pub role: NeuronRole,
    /// Intrinsic excitability bias added to pre-activation.
    pub bias: f32,
    /// EMA of `|tanh(z)|` — used by the intrinsic homeostat (S3).
    pub mean_rate: f32,
    /// Continual-backprop utility (S4). Running EMA of `|h_i| · Σ|w_out|`.
    pub utility: f32,
    /// Tick count since the neuron was born / last replaced.
    pub age_ticks: u64,
    /// `false` indicates the slot is dead and awaits recycling.
    pub alive: bool,
    /// Indices of incoming synapses. Non-authoritative but kept for fast
    /// forward-pass traversal; maintained alongside `outgoing` via
    /// `push_synapse` / `kill_synapse`.
    pub incoming: Vec<SynapseId>,
    /// Indices of outgoing synapses. Used by the utility computation.
    pub outgoing: Vec<SynapseId>,
}

impl Neuron {
    pub fn new(role: NeuronRole) -> Self {
        Self {
            role,
            bias: 0.0,
            mean_rate: 0.0,
            utility: 0.0,
            age_ticks: 0,
            alive: true,
            incoming: Vec::new(),
            outgoing: Vec::new(),
        }
    }
}

/// A directed synapse connecting `source → target`.
#[derive(Clone, Debug)]
pub struct Synapse {
    pub source: NeuronId,
    pub target: NeuronId,
    pub weight: f32,
    /// Per-car eligibility trace. `eligibility[car]` tracks the Hebbian
    /// product for the car indexed by its `EnvInstanceId`.
    pub eligibility: Vec<f32>,
    pub alive: bool,
}

impl Synapse {
    pub fn new(source: NeuronId, target: NeuronId, weight: f32, num_cars: usize) -> Self {
        Self {
            source,
            target,
            weight,
            eligibility: vec![0.0; num_cars],
            alive: true,
        }
    }
}

/// The shared brain graph. Owned by the `BrainBrain` resource.
#[derive(Clone, Debug)]
pub struct BrainGraph {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub free_neuron_slots: Vec<NeuronId>,
    pub free_synapse_slots: Vec<SynapseId>,
    pub input_neurons: Vec<NeuronId>,
    pub output_neurons: Vec<NeuronId>,
    /// Car count the graph is sized for (synapse eligibility vector length).
    pub num_cars: usize,
}

impl BrainGraph {
    /// Builds a seed graph: `obs_dim` inputs, `initial_hidden_neurons` hidden,
    /// `action_dim` outputs, with edges sampled at `initial_edge_density` from
    /// allowed connections (input→hidden, hidden→hidden excluding self, hidden→output,
    /// input→output) and weights drawn from `Normal(0, initial_weight_sigma)`.
    pub fn seed(config: &BrainInspiredConfig, num_cars: usize, rng: &mut impl Rng) -> Self {
        let mut graph = Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            free_neuron_slots: Vec::new(),
            free_synapse_slots: Vec::new(),
            input_neurons: Vec::with_capacity(config.obs_dim),
            output_neurons: Vec::with_capacity(config.action_dim),
            num_cars,
        };

        // Inputs
        for i in 0..config.obs_dim {
            let id = graph.allocate_neuron(NeuronRole::Input(i as u8));
            graph.input_neurons.push(id);
        }
        // Hidden
        let mut hidden_ids: Vec<NeuronId> = Vec::with_capacity(config.initial_hidden_neurons);
        for _ in 0..config.initial_hidden_neurons {
            let id = graph.allocate_neuron(NeuronRole::Hidden);
            hidden_ids.push(id);
        }
        // Outputs
        for i in 0..config.action_dim {
            let id = graph.allocate_neuron(NeuronRole::Output(i as u8));
            graph.output_neurons.push(id);
        }

        // Random edge sampling.
        // Candidate set: every allowed (source, target) pair. Forbid self-loops
        // and synapses that target an input neuron or originate from an output
        // neuron (keeps the I/O contract pinned). Hidden→hidden excluding self
        // is allowed — cyclic connections are intentional.
        let normal = Normal::new(0.0f32, config.initial_weight_sigma).expect("sigma > 0");

        let mut candidate_pairs: Vec<(NeuronId, NeuronId)> = Vec::new();
        for s_idx in 0..graph.neurons.len() {
            let s_role = graph.neurons[s_idx].role;
            if s_role.is_output() {
                continue; // outputs do not project backwards in the seed
            }
            for t_idx in 0..graph.neurons.len() {
                if s_idx == t_idx {
                    continue;
                }
                let t_role = graph.neurons[t_idx].role;
                if t_role.is_input() {
                    continue; // inputs have no incoming synapses
                }
                candidate_pairs.push((s_idx as NeuronId, t_idx as NeuronId));
            }
        }

        for (source, target) in candidate_pairs {
            if rng.random::<f32>() < config.initial_edge_density {
                let weight = normal.sample(rng);
                graph.add_synapse(source, target, weight);
            }
        }

        graph
    }

    /// Reserves or allocates a neuron slot and returns its stable id.
    pub fn allocate_neuron(&mut self, role: NeuronRole) -> NeuronId {
        if let Some(id) = self.free_neuron_slots.pop() {
            let neuron = &mut self.neurons[id as usize];
            *neuron = Neuron::new(role);
            id
        } else {
            let id = self.neurons.len() as NeuronId;
            self.neurons.push(Neuron::new(role));
            id
        }
    }

    /// Reserves or allocates a synapse slot and links the source/target
    /// adjacency lists. No dedup — callers must ensure no existing live
    /// synapse already occupies (source, target) if uniqueness is required.
    pub fn add_synapse(&mut self, source: NeuronId, target: NeuronId, weight: f32) -> SynapseId {
        debug_assert!(source != target, "self-loops are forbidden");
        let id = if let Some(id) = self.free_synapse_slots.pop() {
            let syn = &mut self.synapses[id as usize];
            *syn = Synapse::new(source, target, weight, self.num_cars);
            id
        } else {
            let id = self.synapses.len() as SynapseId;
            self.synapses.push(Synapse::new(source, target, weight, self.num_cars));
            id
        };
        self.neurons[source as usize].outgoing.push(id);
        self.neurons[target as usize].incoming.push(id);
        id
    }

    /// Count of live neurons across all slots.
    pub fn live_neuron_count(&self) -> usize {
        self.neurons.iter().filter(|n| n.alive).count()
    }

    /// Count of live synapses.
    pub fn live_synapse_count(&self) -> usize {
        self.synapses.iter().filter(|s| s.alive).count()
    }

    /// Count of live hidden neurons (excludes inputs/outputs).
    pub fn live_hidden_count(&self) -> usize {
        self.neurons
            .iter()
            .filter(|n| n.alive && n.role.is_hidden())
            .count()
    }
}
