//! Three-factor plasticity with eligibility traces (Milestone 6, Stage S2).
//!
//! Learning rule applied per-tick to every live synapse:
//!   e_ij[c] ← λ · e_ij[c] + pre_i[c] · post_j[c]     (per-car eligibility)
//!   Δw_ij  = η · M_c · e_ij[c]                       (per-car weight update)
//!
//! With M_c = per-car raw reward (Option C from the research synthesis — no
//! critic in v1). Per-car Δw contributions are summed across cars and applied
//! to the shared graph weights (or averaged, controlled by
//! `sum_per_car_updates`).
//!
//! On episode terminal, the car's eligibility traces are zeroed to prevent
//! stale correlations bleeding across episode resets.

use super::graph::BrainGraph;

/// One-tick snapshot of a single car's observable state, needed for plasticity.
///
/// Activations are passed as slices (not `&NeuronActivations`) so the caller
/// can collect per-car data from a Bevy query without fighting the borrow
/// checker when another query needs `&mut PolicyOutput` on the same entities.
#[derive(Clone, Copy)]
pub struct CarLearnSample<'a> {
    pub env_id: u32,
    pub modulator: f32,
    pub episode_done: bool,
    pub prev: &'a [f32],
    pub curr: &'a [f32],
}

/// Applies one tick of three-factor plasticity for every car in `samples`.
///
/// Returns the number of synapse weights actually mutated (non-zero Δw).
///
/// Ordering:
/// 1. For every live synapse, update eligibility per car and accumulate the
///    per-car Δw contribution into a per-synapse scratch vector.
/// 2. Apply accumulated Δw to shared weights (after all cars are visited, so
///    within a tick every car sees the same weights).
/// 3. For cars that terminated this tick, zero their eligibility across all
///    synapses (eligibility discontinuity on episode reset).
pub fn apply_plasticity_tick(
    graph: &mut BrainGraph,
    samples: &[CarLearnSample],
    lambda: f32,
    eta: f32,
    sum_per_car: bool,
) -> u64 {
    if samples.is_empty() {
        return 0;
    }

    // Scratch: per-synapse accumulated Δw across cars.
    let mut accumulated_delta: Vec<f32> = vec![0.0; graph.synapses.len()];

    // Step 1: eligibility update + Δw accumulation, in a single pass over
    // synapses. Cars are visited in the inner loop so we can exploit cache
    // locality for the synapse struct.
    for (syn_idx, syn) in graph.synapses.iter_mut().enumerate() {
        if !syn.alive {
            continue;
        }
        for sample in samples {
            let car = sample.env_id as usize;
            if car >= syn.eligibility.len() {
                continue;
            }
            let pre = sample.prev.get(syn.source as usize).copied().unwrap_or(0.0);
            let post = sample.curr.get(syn.target as usize).copied().unwrap_or(0.0);

            // Eligibility update — the "slow" factor of the three-factor rule.
            let e = &mut syn.eligibility[car];
            *e = lambda * *e + pre * post;

            // Accumulate Δw contribution from this car.
            accumulated_delta[syn_idx] += eta * sample.modulator * *e;
        }
    }

    // Step 2: apply accumulated Δw to shared weights.
    let divisor = if sum_per_car {
        1.0
    } else {
        samples.len() as f32
    };
    let mut applied = 0u64;
    for (syn_idx, syn) in graph.synapses.iter_mut().enumerate() {
        if !syn.alive {
            continue;
        }
        let delta = accumulated_delta[syn_idx] / divisor;
        if delta != 0.0 {
            syn.weight += delta;
            applied += 1;
        }
    }

    // Step 3: terminal eligibility reset for any car that crashed or timed out.
    for sample in samples {
        if !sample.episode_done {
            continue;
        }
        let car = sample.env_id as usize;
        for syn in graph.synapses.iter_mut() {
            if syn.alive && car < syn.eligibility.len() {
                syn.eligibility[car] = 0.0;
            }
        }
    }

    applied
}

/// Scans the graph and fills out a snapshot of plasticity health metrics.
/// Called from the learn system to keep `BrainRunningStats` fresh; drained
/// into `BrainTrainingStats` in S5.
pub fn sample_plasticity_health(graph: &BrainGraph) -> PlasticitySample {
    let mut abs_w_sum = 0.0f64;
    let mut w_sq_sum = 0.0f64;
    let mut live_syn = 0u64;
    for syn in graph.synapses.iter() {
        if syn.alive {
            abs_w_sum += syn.weight.abs() as f64;
            w_sq_sum += (syn.weight as f64) * (syn.weight as f64);
            live_syn += 1;
        }
    }
    let mean_abs_weight = if live_syn > 0 {
        (abs_w_sum / live_syn as f64) as f32
    } else {
        0.0
    };
    let weight_sigma = if live_syn > 0 {
        (w_sq_sum / live_syn as f64).sqrt() as f32
    } else {
        0.0
    };

    let mut abs_e_sum = 0.0f64;
    let mut e_count = 0u64;
    for syn in graph.synapses.iter() {
        if !syn.alive {
            continue;
        }
        for &e in syn.eligibility.iter() {
            abs_e_sum += e.abs() as f64;
            e_count += 1;
        }
    }
    let mean_abs_eligibility = if e_count > 0 {
        (abs_e_sum / e_count as f64) as f32
    } else {
        0.0
    };

    let mut dead = 0u64;
    let mut alive = 0u64;
    for n in graph.neurons.iter() {
        if !n.alive {
            continue;
        }
        alive += 1;
        if n.mean_rate < 0.01 && n.age_ticks > 200 {
            dead += 1;
        }
    }
    let dead_neuron_fraction = if alive > 0 {
        dead as f32 / alive as f32
    } else {
        0.0
    };

    PlasticitySample {
        mean_abs_weight,
        weight_sigma,
        mean_abs_eligibility,
        dead_neuron_fraction,
        live_synapse_count: live_syn,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlasticitySample {
    pub mean_abs_weight: f32,
    pub weight_sigma: f32,
    pub mean_abs_eligibility: f32,
    pub dead_neuron_fraction: f32,
    pub live_synapse_count: u64,
}
