//! Homeostatic plasticity — synaptic scaling + intrinsic excitability
//! (Milestone 6, Stage S3).
//!
//! Two biological mechanisms run alongside the three-factor plasticity rule:
//!
//! - **Synaptic scaling** (Turrigiano): per-neuron total incoming weight
//!   magnitude is nudged toward `synaptic_scaling_target`. Prevents weight
//!   explosion and weight death.
//!
//! - **Intrinsic excitability** (Marder): per-neuron EMA of `|tanh(z)|` tracks
//!   recent firing rate; if the neuron drifts outside the target band, its
//!   bias is nudged in the correcting direction.
//!
//! Both are slow relative to per-tick plasticity — they run every
//! `structural_cadence` ticks, not every tick.
//!
//! See `context/notes/brain-v1-design.md` §Homeostasis.

use super::graph::{BrainGraph, NeuronRole};
use super::config::BrainInspiredConfig;

/// Applies synaptic scaling to every non-input neuron.
///
/// For each neuron j, compute `s = Σ_i |w_ij|` over incoming synapses. If
/// `s > 0`, scale all incoming weights by `(1 + rate * (target - s) / target)`
/// clamped to `[0.5, 2.0]` to prevent catastrophic corrections if the graph
/// is far from target.
pub fn apply_synaptic_scaling(graph: &mut BrainGraph, config: &BrainInspiredConfig) {
    let target = config.synaptic_scaling_target;
    let rate = config.synaptic_scaling_rate;

    // Collect (neuron_id, s) first because we need a fresh borrow of the
    // synapses for the mutation pass.
    let mut scaling_factors: Vec<(usize, f32)> = Vec::new();
    for (nid, neuron) in graph.neurons.iter().enumerate() {
        if !neuron.alive || neuron.role.is_input() {
            continue;
        }
        let mut s = 0.0f32;
        for &syn_id in &neuron.incoming {
            let syn = &graph.synapses[syn_id as usize];
            if syn.alive {
                s += syn.weight.abs();
            }
        }
        if s <= 1e-8 {
            continue;
        }
        let delta = rate * (target - s) / target;
        let factor = (1.0 + delta).clamp(0.5, 2.0);
        if (factor - 1.0).abs() > 1e-6 {
            scaling_factors.push((nid, factor));
        }
    }

    for (nid, factor) in scaling_factors {
        // Collect the incoming synapse ids first to dodge the aliasing borrow.
        let incoming: Vec<u32> = graph.neurons[nid].incoming.clone();
        for syn_id in incoming {
            let syn = &mut graph.synapses[syn_id as usize];
            if syn.alive {
                syn.weight *= factor;
            }
        }
    }
}

/// Updates each live neuron's intrinsic excitability homeostat.
///
/// Called each tick with the current per-car activations so the mean-rate
/// EMA reflects activity. Bias is nudged in the correcting direction on
/// every call — the rate is small enough (`intrinsic_bias_rate` ~ 1e-4) that
/// per-tick invocation is fine.
pub fn update_intrinsic_homeostat(
    graph: &mut BrainGraph,
    activations_by_car: &[&[f32]],
    config: &BrainInspiredConfig,
) {
    let ema_alpha = 0.01f32;
    let (lo, hi) = config.intrinsic_rate_band;
    let rate = config.intrinsic_bias_rate;

    for (idx, neuron) in graph.neurons.iter_mut().enumerate() {
        if !neuron.alive {
            continue;
        }
        // Inputs have no intrinsic bias adjustment — they carry the raw
        // observation and adjusting their bias would corrupt the I/O contract.
        if matches!(neuron.role, NeuronRole::Input(_)) {
            continue;
        }

        // Mean |activation| across cars this tick.
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for car_acts in activations_by_car.iter() {
            if let Some(a) = car_acts.get(idx).copied() {
                sum += a.abs();
                cnt += 1;
            }
        }
        if cnt > 0 {
            let mean_tick = sum / cnt as f32;
            neuron.mean_rate = (1.0 - ema_alpha) * neuron.mean_rate + ema_alpha * mean_tick;
        }

        // Nudge bias if outside target band.
        if neuron.mean_rate < lo {
            neuron.bias += rate;
        } else if neuron.mean_rate > hi {
            neuron.bias -= rate;
        }

        // Age tick.
        neuron.age_ticks = neuron.age_ticks.saturating_add(1);
    }
}
