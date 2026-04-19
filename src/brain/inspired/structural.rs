//! Structural plasticity (Milestone 6, Stage S4).
//!
//! Continual-backprop-style utility tracking + replacement:
//!
//!   u_i ← η_u · u_i + (1 − η_u) · |h_i| · Σ_outgoing |w|
//!
//! Low-utility mature neurons get "replaced" — outgoing weights zeroed,
//! incoming weights resampled, utility/age/mean_rate/bias reset. Slot-stable.
//!
//! Plateau-triggered neurogenesis grows the graph's width when running-mean
//! reward stops improving.
//!
//! Synapse sprouting + pruning maintain target density.
//!
//! Runs every `structural_cadence` ticks from the learn system. Utility is
//! updated every tick (cheap) from `update_utility_tick`.

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Normal};

use super::graph::{BrainGraph, NeuronId, NeuronRole};
use super::config::BrainInspiredConfig;

/// Per-tick utility EMA update.
///
/// `activations_by_car` holds slices of each car's `curr` activations (one
/// per car). The utility uses the mean of `|h_i|` across cars for robustness.
pub fn update_utility_tick(graph: &mut BrainGraph, activations_by_car: &[&[f32]], eta_u: f32) {
    // Pre-compute Σ|w_outgoing| per neuron so the inner loop is cheap.
    // Because outgoing synapses contribute to a neuron's downstream role, this
    // captures how much a neuron's activation actually influences the network.
    let mut outgoing_sum = vec![0.0f32; graph.neurons.len()];
    for syn in graph.synapses.iter() {
        if !syn.alive {
            continue;
        }
        outgoing_sum[syn.source as usize] += syn.weight.abs();
    }

    for (idx, neuron) in graph.neurons.iter_mut().enumerate() {
        if !neuron.alive {
            continue;
        }

        // Mean |activation| across cars.
        let mut sum_abs = 0.0f32;
        let mut cnt = 0u32;
        for car_acts in activations_by_car.iter() {
            if let Some(a) = car_acts.get(idx).copied() {
                sum_abs += a.abs();
                cnt += 1;
            }
        }
        let mean_abs = if cnt > 0 { sum_abs / cnt as f32 } else { 0.0 };
        let contribution = mean_abs * outgoing_sum[idx];

        neuron.utility = eta_u * neuron.utility + (1.0 - eta_u) * contribution;
    }
}

/// Replaces the `ρ · hidden_count` lowest-utility mature hidden neurons.
///
/// Returns the number of neurons replaced.
pub fn replace_low_utility(
    graph: &mut BrainGraph,
    config: &BrainInspiredConfig,
    rng: &mut impl Rng,
) -> u32 {
    let hidden_count = graph.live_hidden_count();
    if hidden_count == 0 {
        return 0;
    }
    let target_replacements = (config.replace_fraction * hidden_count as f32).ceil() as usize;
    if target_replacements == 0 {
        return 0;
    }

    // Collect mature hidden candidates.
    let mut candidates: Vec<(NeuronId, f32)> = graph
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            n.alive && matches!(n.role, NeuronRole::Hidden) && n.age_ticks >= config.maturity_ticks
        })
        .map(|(i, n)| (i as NeuronId, n.utility))
        .collect();

    if candidates.is_empty() {
        return 0;
    }

    // Sort ascending by utility — lowest first.
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(target_replacements);

    let normal = Normal::new(0.0f32, config.initial_weight_sigma).expect("sigma > 0");
    let mut replaced = 0u32;

    for (nid, _) in candidates {
        // Zero outgoing weights (behaviour-preserving at the moment of replacement).
        let outgoing: Vec<u32> = graph.neurons[nid as usize].outgoing.clone();
        for syn_id in outgoing {
            let syn = &mut graph.synapses[syn_id as usize];
            if syn.alive {
                syn.weight = 0.0;
                // Zero eligibility across all cars so the stale correlation
                // history does not leak into future plasticity on the new
                // neuron's downstream edges.
                for e in syn.eligibility.iter_mut() {
                    *e = 0.0;
                }
            }
        }

        // Resample incoming weights.
        let incoming: Vec<u32> = graph.neurons[nid as usize].incoming.clone();
        for syn_id in incoming {
            let syn = &mut graph.synapses[syn_id as usize];
            if syn.alive {
                syn.weight = normal.sample(rng);
                for e in syn.eligibility.iter_mut() {
                    *e = 0.0;
                }
            }
        }

        // Reset per-neuron state.
        let neuron = &mut graph.neurons[nid as usize];
        neuron.utility = 0.0;
        neuron.age_ticks = 0;
        neuron.mean_rate = 0.0;
        neuron.bias = 0.0;

        replaced += 1;
    }

    replaced
}

/// Detects a reward plateau over the last `plateau_episode_window` episodes.
///
/// Returns `true` when the mean over the most-recent half of the window is
/// within `plateau_threshold` (relative) of the mean over the older half.
pub fn detect_plateau(window: &std::collections::VecDeque<f32>, episodes: usize, threshold: f32) -> bool {
    if window.len() < episodes {
        return false;
    }
    let half = episodes / 2;
    if half == 0 {
        return false;
    }
    let start = window.len() - episodes;
    let old_mean: f32 = window.iter().skip(start).take(half).copied().sum::<f32>() / half as f32;
    let new_mean: f32 = window.iter().skip(start + half).take(episodes - half).copied().sum::<f32>()
        / (episodes - half) as f32;
    let denom = old_mean.abs().max(1e-6);
    ((new_mean - old_mean).abs() / denom) < threshold
}

/// Grows the graph by one hidden neuron, wiring it into ~10 existing
/// neurons in each direction with small random weights. Returns the new
/// neuron's id.
pub fn grow_hidden_neuron(
    graph: &mut BrainGraph,
    config: &BrainInspiredConfig,
    rng: &mut impl Rng,
) -> NeuronId {
    let new_id = graph.allocate_neuron(NeuronRole::Hidden);
    let normal = Normal::new(0.0f32, config.initial_weight_sigma).expect("sigma > 0");

    let degree = 10usize;

    // Incoming: choose from inputs + other hidden.
    let mut candidate_sources: Vec<NeuronId> = graph
        .neurons
        .iter()
        .enumerate()
        .filter(|(i, n)| *i as NeuronId != new_id && n.alive && !n.role.is_output())
        .map(|(i, _)| i as NeuronId)
        .collect();
    if !candidate_sources.is_empty() {
        let take = degree.min(candidate_sources.len());
        // Partial shuffle then take first `take`.
        for i in 0..take {
            let j = i + rng.random_range(0..(candidate_sources.len() - i));
            candidate_sources.swap(i, j);
        }
        for &src in candidate_sources.iter().take(take) {
            let w = normal.sample(rng);
            graph.add_synapse(src, new_id, w);
        }
    }

    // Outgoing: choose from hidden + outputs.
    let mut candidate_targets: Vec<NeuronId> = graph
        .neurons
        .iter()
        .enumerate()
        .filter(|(i, n)| *i as NeuronId != new_id && n.alive && !n.role.is_input())
        .map(|(i, _)| i as NeuronId)
        .collect();
    if !candidate_targets.is_empty() {
        let take = degree.min(candidate_targets.len());
        for i in 0..take {
            let j = i + rng.random_range(0..(candidate_targets.len() - i));
            candidate_targets.swap(i, j);
        }
        for &tgt in candidate_targets.iter().take(take) {
            let w = normal.sample(rng);
            graph.add_synapse(new_id, tgt, w);
        }
    }

    new_id
}

/// Prunes synapses whose magnitude is below `prune_weight_threshold`.
/// Returns the number pruned.
pub fn prune_synapses(graph: &mut BrainGraph, config: &BrainInspiredConfig) -> u32 {
    let thresh = config.prune_weight_threshold;
    let mut pruned = 0u32;
    let synapse_count = graph.synapses.len();
    for syn_id in 0..synapse_count {
        let syn = &mut graph.synapses[syn_id];
        if !syn.alive {
            continue;
        }
        if syn.weight.abs() < thresh {
            syn.alive = false;
            syn.weight = 0.0;
            for e in syn.eligibility.iter_mut() {
                *e = 0.0;
            }
            // Remove from source/target adjacency lists.
            let src = syn.source;
            let tgt = syn.target;
            graph.neurons[src as usize].outgoing.retain(|&s| s != syn_id as u32);
            graph.neurons[tgt as usize].incoming.retain(|&s| s != syn_id as u32);
            graph.free_synapse_slots.push(syn_id as u32);
            pruned += 1;
        }
    }
    pruned
}

/// Samples `candidates` random (source, target) pairs that are not currently
/// connected, creating a synapse between each with a small random weight.
/// Skips self-loops, input-target, and output-source pairs.
///
/// Returns the number of synapses created.
pub fn sprout_synapses(
    graph: &mut BrainGraph,
    config: &BrainInspiredConfig,
    rng: &mut impl Rng,
    candidates: usize,
) -> u32 {
    if rng.random::<f32>() >= config.sprout_probability {
        return 0;
    }

    let normal = Normal::new(0.0f32, config.initial_weight_sigma).expect("sigma > 0");
    let mut created = 0u32;

    let live_neurons: Vec<NeuronId> = graph
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.alive)
        .map(|(i, _)| i as NeuronId)
        .collect();
    if live_neurons.len() < 2 {
        return 0;
    }

    for _ in 0..candidates {
        let src_idx = rng.random_range(0..live_neurons.len());
        let tgt_idx = rng.random_range(0..live_neurons.len());
        if src_idx == tgt_idx {
            continue;
        }
        let src = live_neurons[src_idx];
        let tgt = live_neurons[tgt_idx];
        let src_role = graph.neurons[src as usize].role;
        let tgt_role = graph.neurons[tgt as usize].role;
        if src_role.is_output() || tgt_role.is_input() {
            continue;
        }
        // Skip if an existing live synapse already covers this pair.
        let already_connected = graph.neurons[src as usize]
            .outgoing
            .iter()
            .any(|&sid| {
                let s = &graph.synapses[sid as usize];
                s.alive && s.target == tgt
            });
        if already_connected {
            continue;
        }

        let w = normal.sample(rng);
        graph.add_synapse(src, tgt, w);
        created += 1;
    }

    created
}
