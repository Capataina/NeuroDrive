//! Three-factor plasticity with eligibility traces (Milestone 6, Stage S2).
//!
//! Implements the learning rule:
//!   e_ij[c] ← λ · e_ij[c] + pre_i · post_j           (per-car eligibility)
//!   Δw_ij  = η · M_c · e_ij[c]                       (weight update)
//!
//! With M_c = per-car raw per-tick reward (Option C from the research
//! synthesis — no critic in v1). The per-car weight-update contributions are
//! either summed or averaged across cars (controlled by `sum_per_car_updates`)
//! and applied to the shared graph weights.
//!
//! On episode terminal, the car's eligibility traces are zeroed to prevent
//! stale correlations from bleeding across the reset boundary.
//!
//! S2 exposes a single entry point called from `brain_learn_all_cars_system`.
//! S3 and S4 add homeostasis and structural plasticity on top.

use super::graph::BrainGraph;
use super::NeuronActivations;

/// Observed car state needed for one tick of the learn step.
#[derive(Clone, Debug)]
pub struct CarLearnSample<'a> {
    pub env_id: u32,
    pub modulator: f32,
    pub episode_done: bool,
    pub activations: &'a NeuronActivations,
}

/// Applies one tick of three-factor plasticity for every car in `samples`.
///
/// Returns the number of weight updates actually applied (one per live
/// synapse). Stats on |Δw|, |e|, |w| are written by the caller from the
/// running graph after this returns.
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
    // Allocating per-call is acceptable for now because `apply_plasticity_tick`
    // runs once per tick, not once per car; the buffer is O(synapse_count)
    // which is ~140 at seed and grows modestly with structural plasticity.
    let mut accumulated_delta: Vec<f32> = vec![0.0; graph.synapses.len()];

    for sample in samples {
        let car = sample.env_id as usize;

        for syn in graph.synapses.iter_mut() {
            if !syn.alive {
                continue;
            }
            if car >= syn.eligibility.len() {
                // Car index outside the capacity the graph was seeded for.
                // This should never happen in practice but is a defensive guard;
                // skipping is behaviourally equivalent to zero contribution.
                continue;
            }
            let pre = sample.activations.prev.get(syn.source as usize).copied().unwrap_or(0.0);
            let post = sample.activations.curr.get(syn.target as usize).copied().unwrap_or(0.0);

            // Eligibility update.
            let e = &mut syn.eligibility[car];
            *e = lambda * *e + pre * post;

            // Accumulate Δw contribution from this car for the shared weights.
            // We defer the weight mutation until after all cars are visited
            // so that within a tick every car sees the same weights.
            accumulated_delta[syn.source as usize * 0 + 0] += 0.0; // placeholder
        }

        // Separately build the per-car Δw contribution keyed by synapse index.
        // (The inner loop above did not touch `accumulated_delta[syn_idx]`
        // directly because we needed `syn_idx` — loop again with enumerate.)
        for (syn_idx, syn) in graph.synapses.iter().enumerate() {
            if !syn.alive {
                continue;
            }
            if car >= syn.eligibility.len() {
                continue;
            }
            accumulated_delta[syn_idx] += eta * sample.modulator * syn.eligibility[car];
        }

        // Episode terminal → zero this car's eligibility on every synapse.
        if sample.episode_done {
            for syn in graph.synapses.iter_mut() {
                if syn.alive && car < syn.eligibility.len() {
                    syn.eligibility[car] = 0.0;
                }
            }
        }
    }

    // Apply accumulated Δw to shared weights.
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
    applied
}

/// Scans the graph and fills out stats on mean |w| / mean |e| / dead-neuron
/// fraction. Called from the learn system after plasticity has been applied.
pub fn sample_plasticity_health(
    graph: &BrainGraph,
) -> PlasticitySample {
    let mut abs_w_sum = 0.0f64;
    let mut live_syn = 0u64;
    for syn in graph.synapses.iter() {
        if syn.alive {
            abs_w_sum += syn.weight.abs() as f64;
            live_syn += 1;
        }
    }
    let mean_abs_weight = if live_syn > 0 {
        (abs_w_sum / live_syn as f64) as f32
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
        mean_abs_eligibility,
        dead_neuron_fraction,
        live_synapse_count: live_syn,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlasticitySample {
    pub mean_abs_weight: f32,
    pub mean_abs_eligibility: f32,
    pub dead_neuron_fraction: f32,
    pub live_synapse_count: u64,
}
