//! Configuration for the brain-inspired learner.
//!
//! All hyperparameters live here so ablations can be run without recompiling
//! and all dials are visible in one place. Each dial is tagged either as
//! `RESEARCH-ANCHORED` (the value or a tight range is fixed by the seven-paper
//! research round, see `context/references/brain-inspired-learning/`) or `TUNE`
//! (no firm value exists; will need empirical tuning).
//!
//! See `context/notes/brain-v1-design.md` for the design rationale and
//! `context/plans/` for the implementation plan.

/// All brain-inspired learner dials.
///
/// Default values reflect the plan's starting point. Disable flags
/// (`enable_*`) allow each mechanism to be turned off independently for
/// ablations — the flags default on so the full v1 is active out of the box.
#[derive(Clone, Debug)]
pub struct BrainInspiredConfig {
    // ── Seed graph topology ────────────────────────────────────────────
    /// Number of observation dimensions (reserved input neurons).
    /// RESEARCH-ANCHORED — matches the 43-dim observation contract.
    pub obs_dim: usize,

    /// Number of action dimensions (reserved output neurons: steering, throttle).
    /// RESEARCH-ANCHORED — matches the 2-dim action contract.
    pub action_dim: usize,

    /// Initial count of plastic hidden neurons.
    /// RESEARCH-ANCHORED — `brain-v1-design.md` §Initialisation.
    pub initial_hidden_neurons: usize,

    /// Fraction of the possible directed edges (excluding self-loops and
    /// input-from-output) to materialise at seed time.
    /// RESEARCH-ANCHORED — ~10% per `brain-v1-design.md`.
    pub initial_edge_density: f32,

    /// Standard deviation of initial Gaussian synapse weights.
    /// RESEARCH-ANCHORED — σ ≈ 0.1 per `brain-v1-design.md`.
    pub initial_weight_sigma: f32,

    // ── Plasticity (S2) ────────────────────────────────────────────────
    /// Eligibility trace decay per tick. `e ← λ·e + pre·post`.
    /// RESEARCH-ANCHORED — λ=0.992 gives τ_e ≈ 2 s at 60 Hz, matching γ=0.995
    /// credit horizon. See `reward-design.md`.
    pub lambda: f32,

    /// Synaptic learning rate. `δw = η·M·e`.
    /// TUNE — research does not pin a value. Starting point; sweep in ablations.
    pub eta: f32,

    /// Per-car weight-update mode: `true` sums per-car updates (8× data rate),
    /// `false` averages them (safer if summing destabilises).
    /// TUNE — plan's default is sum; flipped to average as a safety fallback.
    pub sum_per_car_updates: bool,

    // ── Utility tracking (S4) ──────────────────────────────────────────
    /// Running-EMA rate for neuron utility.
    /// RESEARCH-ANCHORED — CBP §Rank 1 [CBP-UTIL].
    pub eta_utility: f32,

    /// Minimum tick-age a neuron must reach before it is replacement-eligible.
    /// RESEARCH-ANCHORED — m ≈ 1000 per CBP §Rank 1.
    pub maturity_ticks: u64,

    /// Fraction of hidden neurons replaced per structural event.
    /// TUNE — mid of CBP 1e-4..1e-3 range.
    pub replace_fraction: f32,

    // ── Structural cadence (S3/S4) ─────────────────────────────────────
    /// Run homeostasis + structural plasticity every N ticks.
    /// TUNE — too frequent destabilises; too rare slows adaptation.
    pub structural_cadence: u64,

    // ── Plateau detection (S4) ─────────────────────────────────────────
    /// Number of recent episodes used to detect a reward plateau.
    /// TUNE — research leaves this open.
    pub plateau_episode_window: usize,

    /// Relative delta below which mean reward is considered flat.
    /// TUNE — project-specific starting point.
    pub plateau_threshold: f32,

    // ── Synapse prune/sprout (S4) ──────────────────────────────────────
    /// Magnitude below which a synapse is pruned.
    /// TUNE.
    pub prune_weight_threshold: f32,

    /// Probability that a structural event triggers synapse sprouting.
    /// TUNE — roughly balances against prune rate to keep density stable.
    pub sprout_probability: f32,

    /// Number of candidate pairs considered per sprout event.
    pub sprout_candidates_per_event: usize,

    // ── Homeostasis (S3) ───────────────────────────────────────────────
    /// Target Σ|w_in| per non-input neuron. Synaptic scaling nudges toward this.
    /// TUNE — project-specific.
    pub synaptic_scaling_target: f32,

    /// Rate at which synaptic scaling corrects toward the target each event.
    /// TUNE — slow relative to plasticity.
    pub synaptic_scaling_rate: f32,

    /// Intrinsic firing rate target band (mean |tanh(z)|).
    /// TUNE.
    pub intrinsic_rate_band: (f32, f32),

    /// Per-tick rate at which bias adjusts toward the intrinsic target band.
    /// TUNE — slow relative to plasticity.
    pub intrinsic_bias_rate: f32,

    // ── Ablation flags ─────────────────────────────────────────────────
    /// Enable three-factor plasticity (S2).
    pub enable_plasticity: bool,
    /// Enable synaptic scaling + intrinsic excitability (S3).
    pub enable_homeostasis: bool,
    /// Enable structural plasticity — replacement / neurogenesis / sprout / prune (S4).
    pub enable_structural: bool,

    // ── Reproducibility ────────────────────────────────────────────────
    /// Optional fixed RNG seed. `None` = draw from `rand::rng()` at startup.
    pub rng_seed: Option<u64>,
}

impl Default for BrainInspiredConfig {
    fn default() -> Self {
        Self {
            obs_dim: 43,
            action_dim: 2,
            initial_hidden_neurons: 15,
            initial_edge_density: 0.10,
            initial_weight_sigma: 0.1,

            lambda: 0.992,
            eta: 1e-3,
            sum_per_car_updates: true,

            eta_utility: 0.99,
            maturity_ticks: 1_000,
            replace_fraction: 5e-4,

            structural_cadence: 128,

            plateau_episode_window: 50,
            plateau_threshold: 0.02,

            prune_weight_threshold: 0.01,
            sprout_probability: 0.10,
            sprout_candidates_per_event: 8,

            synaptic_scaling_target: 2.0,
            synaptic_scaling_rate: 0.05,
            intrinsic_rate_band: (0.10, 0.60),
            intrinsic_bias_rate: 1e-4,

            enable_plasticity: true,
            enable_homeostasis: true,
            enable_structural: true,

            rng_seed: None,
        }
    }
}
