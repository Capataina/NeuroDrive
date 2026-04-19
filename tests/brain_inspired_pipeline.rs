//! Integration tests for the brain-inspired learner (Milestone 6).
//!
//! Stage-gated: each stage in the M6 plan adds a section here.
//!
//! - S1: graph construction + forward pass sanity.
//! - S2: eligibility trace math + weight update stability.
//! - S3: homeostasis invariants.
//! - S4: structural plasticity invariants.
//! - S5: analytics serialisation.
//! - S6: side-by-side partition + cross-contamination guards.

use neurodrive::brain::inspired::config::BrainInspiredConfig;
use neurodrive::brain::inspired::graph::{BrainGraph, NeuronRole};
use neurodrive::brain::inspired::homeostasis::{
    apply_synaptic_scaling, update_intrinsic_homeostat,
};
use neurodrive::brain::inspired::plasticity::{CarLearnSample, apply_plasticity_tick};
use neurodrive::brain::inspired::{NeuronActivations, forward_tick};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ── S1: Plumbing + forward pass ──────────────────────────────────────────

#[test]
fn seed_graph_has_correct_io_counts() {
    let config = BrainInspiredConfig::default();
    let mut rng = StdRng::seed_from_u64(7);
    let graph = BrainGraph::seed(&config, 8, &mut rng);

    // Input/output role counts match the observation/action contract.
    assert_eq!(graph.input_neurons.len(), config.obs_dim);
    assert_eq!(graph.output_neurons.len(), config.action_dim);

    let input_count = graph
        .neurons
        .iter()
        .filter(|n| matches!(n.role, NeuronRole::Input(_)))
        .count();
    let output_count = graph
        .neurons
        .iter()
        .filter(|n| matches!(n.role, NeuronRole::Output(_)))
        .count();
    let hidden_count = graph
        .neurons
        .iter()
        .filter(|n| matches!(n.role, NeuronRole::Hidden))
        .count();

    assert_eq!(input_count, config.obs_dim);
    assert_eq!(output_count, config.action_dim);
    assert_eq!(hidden_count, config.initial_hidden_neurons);

    // Every synapse's endpoints must refer to live neuron slots.
    for syn in &graph.synapses {
        if syn.alive {
            assert!(graph.neurons[syn.source as usize].alive);
            assert!(graph.neurons[syn.target as usize].alive);
            // No self-loops.
            assert_ne!(syn.source, syn.target);
            // No input-target / output-source.
            assert!(!matches!(
                graph.neurons[syn.target as usize].role,
                NeuronRole::Input(_)
            ));
            assert!(!matches!(
                graph.neurons[syn.source as usize].role,
                NeuronRole::Output(_)
            ));
        }
    }

    // Eligibility is sized for the requested car count.
    for syn in &graph.synapses {
        if syn.alive {
            assert_eq!(syn.eligibility.len(), 8);
        }
    }
}

#[test]
fn forward_pass_is_deterministic_with_fixed_seed() {
    let config = BrainInspiredConfig::default();

    // Two independent graphs with the same seed should produce identical
    // outputs given identical observations and activation state.
    let mut rng_a = StdRng::seed_from_u64(42);
    let graph_a = BrainGraph::seed(&config, 1, &mut rng_a);
    let mut rng_b = StdRng::seed_from_u64(42);
    let graph_b = BrainGraph::seed(&config, 1, &mut rng_b);

    let mut act_a = NeuronActivations::default();
    let mut act_b = NeuronActivations::default();

    let obs: Vec<f32> = (0..config.obs_dim)
        .map(|i| (i as f32).sin() * 0.3)
        .collect();

    for _ in 0..5 {
        let out_a = forward_tick(&graph_a, &mut act_a, &obs);
        let out_b = forward_tick(&graph_b, &mut act_b, &obs);
        assert!((out_a.0 - out_b.0).abs() < 1e-6, "steering diverged");
        assert!((out_a.1 - out_b.1).abs() < 1e-6, "throttle diverged");
    }
}

#[test]
fn forward_pass_output_is_in_action_range() {
    let config = BrainInspiredConfig::default();
    let mut rng = StdRng::seed_from_u64(1);
    let graph = BrainGraph::seed(&config, 1, &mut rng);

    let mut activations = NeuronActivations::default();
    let obs: Vec<f32> = vec![0.5; config.obs_dim];

    // Run several ticks so the network has a chance to propagate.
    let mut saw_nonzero = false;
    for _ in 0..20 {
        let (steering, throttle) = forward_tick(&graph, &mut activations, &obs);
        assert!(
            (-1.0..=1.0).contains(&steering),
            "steering out of [-1,1]: {}",
            steering
        );
        assert!(
            (0.0..=1.0).contains(&throttle),
            "throttle out of [0,1]: {}",
            throttle
        );
        if steering.abs() > 1e-6 || (throttle - 0.5).abs() > 1e-6 {
            saw_nonzero = true;
        }
    }
    assert!(
        saw_nonzero,
        "forward pass returned identical zero-activation output on every tick"
    );
}

// ── S2: Three-factor plasticity ──────────────────────────────────────────

/// Builds a small graph, writes synthetic activations, applies one tick of
/// plasticity, and asserts eligibility decays toward zero when `M = 0` and
/// `pre·post = 0`.
#[test]
fn eligibility_trace_decays_to_zero_with_m_zero() {
    let mut config = BrainInspiredConfig::default();
    config.obs_dim = 2;
    config.action_dim = 1;
    config.initial_hidden_neurons = 2;
    config.initial_edge_density = 1.0; // Fully connect so we have plenty of synapses.

    let mut rng = StdRng::seed_from_u64(21);
    let mut graph = BrainGraph::seed(&config, 1, &mut rng);

    // Seed eligibility with known non-zero values.
    for syn in graph.synapses.iter_mut() {
        syn.eligibility[0] = 1.0;
    }

    // Run 500 ticks with zero pre/post activations and M = 0 so eligibility
    // just decays by λ each tick without any Hebbian accumulation.
    let zeros = vec![0.0; graph.neurons.len()];
    for _ in 0..500 {
        let sample = CarLearnSample {
            env_id: 0,
            modulator: 0.0,
            episode_done: false,
            prev: &zeros,
            curr: &zeros,
        };
        apply_plasticity_tick(&mut graph, &[sample], config.lambda, config.eta, true);
    }

    // After 500 ticks at λ=0.992, 1.0 → 0.992^500 ≈ 0.018. Accept a generous
    // ceiling of 0.05 to stay above floating-point noise.
    for syn in graph.synapses.iter() {
        if syn.alive {
            assert!(
                syn.eligibility[0].abs() < 0.05,
                "eligibility did not decay: {}",
                syn.eligibility[0],
            );
        }
    }
}

/// Weight delta is linear in η for fixed (eligibility, M). Verify by doubling η.
#[test]
fn weight_update_magnitude_scales_with_eta() {
    let mut config = BrainInspiredConfig::default();
    config.obs_dim = 2;
    config.action_dim = 1;
    config.initial_hidden_neurons = 2;
    config.initial_edge_density = 0.8;

    let mut rng_a = StdRng::seed_from_u64(99);
    let mut graph_a = BrainGraph::seed(&config, 1, &mut rng_a);
    let mut rng_b = StdRng::seed_from_u64(99);
    let mut graph_b = BrainGraph::seed(&config, 1, &mut rng_b);

    // Seed identical eligibility on both graphs.
    for (a, b) in graph_a.synapses.iter_mut().zip(graph_b.synapses.iter_mut()) {
        a.eligibility[0] = 0.5;
        b.eligibility[0] = 0.5;
    }
    // Snapshot weights before update.
    let weights_before: Vec<f32> = graph_a.synapses.iter().map(|s| s.weight).collect();

    // Apply one tick with η and 2η. The activations are zero-filled, so the
    // eligibility update itself is a no-op (λ decay only) — but the weight
    // step uses the seeded eligibility, which is what we're testing.
    let zeros = vec![0.0; graph_a.neurons.len()];
    let sample = CarLearnSample {
        env_id: 0,
        modulator: 1.0,
        episode_done: false,
        prev: &zeros,
        curr: &zeros,
    };
    apply_plasticity_tick(&mut graph_a, &[sample], config.lambda, config.eta, true);
    apply_plasticity_tick(&mut graph_b, &[sample], config.lambda, 2.0 * config.eta, true);

    // Each aligned pair of live synapses: Δb / Δa ≈ 2 (within f32 rounding).
    for (i, (a, b)) in graph_a.synapses.iter().zip(graph_b.synapses.iter()).enumerate() {
        if !a.alive || !b.alive {
            continue;
        }
        let delta_a = a.weight - weights_before[i];
        let delta_b = b.weight - weights_before[i];
        if delta_a.abs() < 1e-8 {
            continue;
        }
        let ratio = delta_b / delta_a;
        assert!(
            (ratio - 2.0).abs() < 1e-3,
            "ratio Δb/Δa = {}, expected ≈ 2",
            ratio
        );
    }
}

/// Run forward + plasticity for many ticks with adversarial inputs and
/// assert no weight, eligibility, or activation goes NaN/Inf.
#[test]
fn plasticity_preserves_no_nan_no_inf_over_10k_ticks() {
    let config = BrainInspiredConfig::default();
    let mut rng = StdRng::seed_from_u64(314);
    let mut graph = BrainGraph::seed(&config, 1, &mut rng);
    let mut activations = NeuronActivations::default();

    for t in 0..10_000 {
        // Adversarial observation: oscillate between hard saturations.
        let obs: Vec<f32> = (0..config.obs_dim)
            .map(|i| if (t + i) % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let (s, th) = forward_tick(&graph, &mut activations, &obs);
        assert!(s.is_finite(), "steering non-finite at tick {}", t);
        assert!(th.is_finite(), "throttle non-finite at tick {}", t);

        // Modulator swings with hemispheric-sign flips.
        let m = if t % 100 < 50 { 1.0 } else { -1.0 };
        let sample = CarLearnSample {
            env_id: 0,
            modulator: m,
            episode_done: t % 500 == 499,
            prev: &activations.prev,
            curr: &activations.curr,
        };
        apply_plasticity_tick(&mut graph, &[sample], config.lambda, config.eta, true);
    }

    for syn in graph.synapses.iter() {
        assert!(syn.weight.is_finite(), "weight non-finite: {}", syn.weight);
        for &e in syn.eligibility.iter() {
            assert!(e.is_finite(), "eligibility non-finite: {}", e);
        }
    }
    for n in graph.neurons.iter() {
        assert!(n.bias.is_finite(), "bias non-finite");
        assert!(n.mean_rate.is_finite(), "mean_rate non-finite");
    }
}

/// After an episode terminal, that car's eligibility on every synapse must
/// be zero; other cars' eligibility is unaffected.
#[test]
fn terminal_episode_zeros_eligibility() {
    let config = BrainInspiredConfig::default();
    let mut rng = StdRng::seed_from_u64(77);
    let mut graph = BrainGraph::seed(&config, 3, &mut rng);

    // Seed distinguishable eligibility values per car.
    for syn in graph.synapses.iter_mut() {
        syn.eligibility[0] = 0.9;
        syn.eligibility[1] = 0.5;
        syn.eligibility[2] = 0.3;
    }

    let zeros = vec![0.0; graph.neurons.len()];
    let samples = vec![
        CarLearnSample {
            env_id: 0,
            modulator: 0.0,
            episode_done: true, // ← only car 0 terminates
            prev: &zeros,
            curr: &zeros,
        },
        CarLearnSample {
            env_id: 1,
            modulator: 0.0,
            episode_done: false,
            prev: &zeros,
            curr: &zeros,
        },
        CarLearnSample {
            env_id: 2,
            modulator: 0.0,
            episode_done: false,
            prev: &zeros,
            curr: &zeros,
        },
    ];
    apply_plasticity_tick(&mut graph, &samples, config.lambda, config.eta, true);

    for syn in graph.synapses.iter() {
        if !syn.alive {
            continue;
        }
        assert_eq!(syn.eligibility[0], 0.0, "car 0 eligibility not zeroed");
        // Cars 1 and 2: decayed by λ but still non-zero.
        let expected_1 = config.lambda * 0.5;
        let expected_2 = config.lambda * 0.3;
        assert!(
            (syn.eligibility[1] - expected_1).abs() < 1e-5,
            "car 1 eligibility drifted"
        );
        assert!(
            (syn.eligibility[2] - expected_2).abs() < 1e-5,
            "car 2 eligibility drifted"
        );
    }
}

// ── S3: Homeostasis ──────────────────────────────────────────────────────

/// Synaptic scaling pushes Σ|w_in| per non-input neuron toward target.
///
/// Seed a graph, hand-scale one neuron's incoming weights to 10× the target,
/// run scaling repeatedly, and assert the sum converges toward target.
#[test]
fn synaptic_scaling_brings_sum_to_target() {
    let config = BrainInspiredConfig::default();
    let mut rng = StdRng::seed_from_u64(555);
    let mut graph = BrainGraph::seed(&config, 1, &mut rng);

    // Pick a hidden neuron that has incoming synapses.
    let mut victim: Option<u32> = None;
    for (i, n) in graph.neurons.iter().enumerate() {
        if n.alive && matches!(n.role, NeuronRole::Hidden) && !n.incoming.is_empty() {
            victim = Some(i as u32);
            break;
        }
    }
    let victim = victim.expect("seed graph should have a hidden neuron with incoming edges");

    // Inflate its incoming weights to way above target.
    let incoming = graph.neurons[victim as usize].incoming.clone();
    for sid in incoming.iter() {
        let syn = &mut graph.synapses[*sid as usize];
        if syn.alive {
            syn.weight = 1.0 * syn.weight.signum().max(0.5_f32.signum());
            if syn.weight.abs() < 0.5 {
                syn.weight = 1.0;
            }
        }
    }
    // Force a known large sum: set every incoming weight to +5 so total ≈ 5×n.
    for sid in incoming.iter() {
        let syn = &mut graph.synapses[*sid as usize];
        if syn.alive {
            syn.weight = 5.0;
        }
    }
    let initial_sum: f32 = incoming
        .iter()
        .filter_map(|&sid| {
            let s = &graph.synapses[sid as usize];
            if s.alive { Some(s.weight.abs()) } else { None }
        })
        .sum();
    assert!(
        initial_sum > config.synaptic_scaling_target * 3.0,
        "setup: initial sum ({}) should be well above target ({})",
        initial_sum,
        config.synaptic_scaling_target
    );

    // Run many scaling passes.
    for _ in 0..200 {
        apply_synaptic_scaling(&mut graph, &config);
    }

    let final_sum: f32 = incoming
        .iter()
        .filter_map(|&sid| {
            let s = &graph.synapses[sid as usize];
            if s.alive { Some(s.weight.abs()) } else { None }
        })
        .sum();
    // Should have contracted toward target within 20%.
    let target = config.synaptic_scaling_target;
    assert!(
        (final_sum - target).abs() < target * 0.2,
        "final sum {} did not converge toward target {}",
        final_sum,
        target
    );
}

/// Intrinsic homeostat nudges bias toward a target rate band.
///
/// Drive a graph with activations that keep all hidden neurons at ~0
/// firing rate, then run many intrinsic updates and assert that at least
/// some hidden biases became positive (nudging neurons back above the lower
/// band).
#[test]
fn intrinsic_homeostat_moves_bias_toward_target_band() {
    let mut config = BrainInspiredConfig::default();
    // Make the effect faster for the test.
    config.intrinsic_bias_rate = 1e-2;
    let mut rng = StdRng::seed_from_u64(17);
    let mut graph = BrainGraph::seed(&config, 1, &mut rng);

    // Zero activations → mean_rate stays at 0 → below lower band → bias should drift up.
    let zeros = vec![0.0; graph.neurons.len()];
    let per_car: Vec<&[f32]> = vec![&zeros];

    for _ in 0..2000 {
        update_intrinsic_homeostat(&mut graph, &per_car, &config);
    }

    let mut positive_biases = 0u32;
    let mut hidden_count = 0u32;
    for n in &graph.neurons {
        if n.alive && matches!(n.role, NeuronRole::Hidden) {
            hidden_count += 1;
            if n.bias > 0.0 {
                positive_biases += 1;
            }
        }
    }
    assert!(hidden_count > 0);
    assert!(
        positive_biases as f32 / hidden_count as f32 > 0.5,
        "expected most hidden biases to drift positive; got {}/{}",
        positive_biases,
        hidden_count
    );
}

/// Idempotence: once a neuron's Σ|w_in| is exactly at target, scaling
/// should not mutate the weights.
#[test]
fn homeostasis_idempotent_at_steady_state() {
    let config = BrainInspiredConfig::default();
    let mut rng = StdRng::seed_from_u64(99);
    let mut graph = BrainGraph::seed(&config, 1, &mut rng);

    // Set every hidden neuron's incoming weights to exactly target / n per edge.
    for nid in 0..graph.neurons.len() {
        let incoming = graph.neurons[nid].incoming.clone();
        if incoming.is_empty() {
            continue;
        }
        let per_edge = config.synaptic_scaling_target / incoming.len() as f32;
        for sid in incoming {
            let syn = &mut graph.synapses[sid as usize];
            if syn.alive {
                syn.weight = per_edge;
            }
        }
    }

    let before: Vec<f32> = graph.synapses.iter().map(|s| s.weight).collect();
    apply_synaptic_scaling(&mut graph, &config);
    let after: Vec<f32> = graph.synapses.iter().map(|s| s.weight).collect();

    for (a, b) in before.iter().zip(after.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "weight drifted at steady state: {} → {}",
            a,
            b
        );
    }
}
