//! One-tick forward pass over the brain graph.
//!
//! **Key design decision — one-step propagation with previous-tick reads.**
//! Each tick, every non-input neuron's new activation is computed from the
//! *previous* tick's activations of its source neurons. This makes the
//! forward pass order-independent (any iteration order produces the same
//! result) and makes cyclic connections trivially well-defined. It is also
//! biologically plausible: real neurons have non-zero integration time
//! constants, so propagation through a circuit takes multiple ticks by
//! default.
//!
//! See `context/notes/brain-v1-design.md` §Forward Pass.

use super::graph::BrainGraph;

/// Per-car activation buffers.
///
/// `prev` holds activations from the previous tick and is read by the forward
/// pass. `curr` is written during this tick's forward pass.
///
/// At the end of each forward pass `prev` and `curr` are conceptually swapped;
/// this is implemented as `prev <- curr; curr is zeroed or overwritten for the
/// next tick`.
///
/// Length of both vectors tracks `BrainGraph::neurons.len()` (including dead
/// slots — simpler than remapping). Grown lazily in the act system when the
/// graph grows via structural plasticity (S4).
#[derive(bevy::prelude::Component, Debug, Default, Clone)]
pub struct NeuronActivations {
    pub prev: Vec<f32>,
    pub curr: Vec<f32>,
}

impl NeuronActivations {
    /// Ensures both buffers are sized to `neuron_count`, preserving any
    /// existing values and zero-extending when the graph has grown.
    pub fn ensure_sized(&mut self, neuron_count: usize) {
        if self.prev.len() != neuron_count {
            self.prev.resize(neuron_count, 0.0);
        }
        if self.curr.len() != neuron_count {
            self.curr.resize(neuron_count, 0.0);
        }
    }
}

/// Runs the forward pass for one car for one tick.
///
/// Protocol:
/// 1. Rotate buffers: `prev <- curr`.
/// 2. Write the observation into the `curr` slots of input neurons (inputs do
///    not pass through `tanh` — they carry the raw normalised observation).
/// 3. For every live non-input neuron, compute `z = bias + Σ prev[src] · w`
///    over incoming synapses and set `curr = tanh(z)`.
/// 4. Read output neurons from `curr` and return `(steering, throttle)`.
///
/// Returns `(steering, throttle)` where steering is in `[-1, 1]` and throttle
/// is remapped from the neuron's `[-1, 1]` tanh range to `[0, 1]` via
/// `0.5·(x+1)`, matching the PPO baseline's throttle remap.
pub fn forward_tick(
    graph: &BrainGraph,
    activations: &mut NeuronActivations,
    observation: &[f32],
) -> (f32, f32) {
    activations.ensure_sized(graph.neurons.len());

    // Step 1: rotate buffers. After this, `prev` holds this-tick's soon-to-be
    // inputs; `curr` gets overwritten below.
    activations.prev.copy_from_slice(&activations.curr);

    // Step 2: set input neurons directly from observation. Inputs are written
    // to `curr` so downstream reads in the same tick (and plasticity) see the
    // current observation.
    for (&neuron_id, &obs_value) in graph.input_neurons.iter().zip(observation.iter()) {
        activations.curr[neuron_id as usize] = obs_value;
    }

    // Step 3: compute activations for every live non-input neuron.
    // Order does not matter because we read from `prev`, never from the
    // just-written `curr` of another non-input neuron on the same tick.
    for (idx, neuron) in graph.neurons.iter().enumerate() {
        if !neuron.alive || neuron.role.is_input() {
            continue;
        }
        let mut z = neuron.bias;
        for &syn_id in &neuron.incoming {
            let syn = &graph.synapses[syn_id as usize];
            if !syn.alive {
                continue;
            }
            z += activations.prev[syn.source as usize] * syn.weight;
        }
        activations.curr[idx] = z.tanh();
    }

    // Step 4: read outputs.
    // graph.output_neurons is assumed to hold [steering_neuron, throttle_neuron]
    // when action_dim == 2. Any missing entry falls back to 0.
    let steering = graph
        .output_neurons
        .first()
        .map(|&id| activations.curr[id as usize])
        .unwrap_or(0.0);
    let throttle_raw = graph
        .output_neurons
        .get(1)
        .map(|&id| activations.curr[id as usize])
        .unwrap_or(0.0);
    // Remap tanh [-1, 1] → throttle [0, 1] to match PPO's action-space contract.
    let throttle = 0.5 * (throttle_raw + 1.0);

    (steering, throttle)
}
