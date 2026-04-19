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
