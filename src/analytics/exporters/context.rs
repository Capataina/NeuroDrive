use crate::agent::observation::{ObservationConfig, OBSERVATION_DIM, NUM_RAYS, NUM_LOOKAHEAD_SAMPLES};
use crate::brain::ppo::PpoBrain;
use crate::game::car::{Car, TrainerConfig};
use crate::game::episode::EpisodeConfig;

/// Snapshot of the full run configuration captured from live Bevy resources.
/// Usable by both the analytics and profiling exporters.
pub struct RunContext {
    pub timestamp: u64,
    pub episode_count: usize,
    pub ppo_update_count: usize,
    // Environment
    pub car_count: usize,
    pub timeout_s: f32,
    pub rotation_speed: f32,
    pub thrust: f32,
    pub drag: f32,
    // Observations
    pub observation_dim: usize,
    pub ray_count: usize,
    pub lookahead_count: usize,
    pub lookahead_range: (f32, f32), // (near, far)
    // Reward
    pub velocity_reward_scale: f32,
    pub speed_reward_reference: f32,
    pub centreline_reward_coef: f32,
    pub centreline_reward_max_distance: f32,
    pub time_penalty_per_tick: f32,
    pub crash_penalty: f32,
    // PPO
    pub ppo_epochs: usize,
    pub clip_epsilon: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub max_steps: usize,
    pub min_update_steps: usize,
    pub samples_per_tick: usize,
    pub hidden_dim: usize,
    pub actor_lr: f32,
    pub critic_lr: f32,
}

impl RunContext {
    /// Captures a run context snapshot from live Bevy resources.
    ///
    /// Car physics parameters (rotation_speed, thrust, drag) are sourced from
    /// `Car::default()` since they are hardcoded in the default impl.
    pub fn capture(
        trainer_config: &TrainerConfig,
        episode_config: &EpisodeConfig,
        obs_config: &ObservationConfig,
        brain: &PpoBrain,
        episode_count: usize,
        ppo_update_count: usize,
    ) -> Self {
        let car_defaults = Car::default();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let lookahead_range = (
            obs_config.lookahead_distances.first().copied().unwrap_or(0.0),
            obs_config.lookahead_distances.last().copied().unwrap_or(0.0),
        );

        Self {
            timestamp,
            episode_count,
            ppo_update_count,
            car_count: trainer_config.num_envs,
            timeout_s: episode_config.timeout_s,
            rotation_speed: car_defaults.rotation_speed,
            thrust: car_defaults.thrust,
            drag: car_defaults.drag,
            observation_dim: OBSERVATION_DIM,
            ray_count: NUM_RAYS,
            lookahead_count: NUM_LOOKAHEAD_SAMPLES,
            lookahead_range,
            velocity_reward_scale: episode_config.velocity_reward_scale,
            speed_reward_reference: episode_config.speed_reward_reference,
            centreline_reward_coef: episode_config.centreline_reward_coef,
            centreline_reward_max_distance: episode_config.centreline_reward_max_distance,
            time_penalty_per_tick: episode_config.time_penalty_per_tick,
            crash_penalty: episode_config.crash_penalty,
            ppo_epochs: brain.config.ppo_epochs,
            clip_epsilon: brain.config.clip_epsilon,
            gamma: brain.config.gamma,
            gae_lambda: brain.config.gae_lambda,
            max_steps: brain.config.max_steps,
            min_update_steps: brain.config.min_update_steps,
            samples_per_tick: brain.config.samples_per_tick,
            hidden_dim: brain.config.actor_hidden_dim,
            actor_lr: brain.config.actor_lr,
            critic_lr: brain.config.critic_lr,
        }
    }

    /// Formats the run context as a Markdown section suitable for prepending to reports.
    pub fn to_markdown_header(&self) -> String {
        let datetime = {
            let secs = self.timestamp;
            // Simple UTC formatting without chrono dependency
            format!("Unix {secs}")
        };

        format!(
            "## Run Context\n\
             \n\
             | | |\n\
             |---|---|\n\
             | **Date** | {datetime} |\n\
             | **Episodes** | {} |\n\
             | **PPO Updates** | {} |\n\
             \n\
             ### Environment\n\
             \n\
             | Parameter | Value |\n\
             |-----------|-------|\n\
             | Cars | {} |\n\
             | Timeout | {:.0}s |\n\
             | Rotation speed | {:.1} rad/s |\n\
             | Thrust | {:.0} |\n\
             | Drag | {:.4} |\n\
             \n\
             ### Observations\n\
             \n\
             | Parameter | Value |\n\
             |-----------|-------|\n\
             | Dimension | {} |\n\
             | Rays | {} |\n\
             | Lookahead samples | {} ({:.0}–{:.0} units) |\n\
             \n\
             ### Reward\n\
             \n\
             | Parameter | Value |\n\
             |-----------|-------|\n\
             | Velocity projection | scale={:.1}, ref={:.0} |\n\
             | Centreline proximity | coef={:.2}, max_dist={:.0} |\n\
             | Time penalty | {:.4}/tick |\n\
             | Crash penalty | {:.1} |\n\
             \n\
             ### PPO\n\
             \n\
             | Parameter | Value |\n\
             |-----------|-------|\n\
             | Epochs | {} |\n\
             | Clip epsilon | {:.2} |\n\
             | Gamma | {:.3} |\n\
             | GAE lambda | {:.3} |\n\
             | Horizon | {} |\n\
             | Min update steps | {} |\n\
             | Samples/tick | {} |\n\
             | Hidden dim | {} |\n\
             | Actor LR | {:.0e} |\n\
             | Critic LR | {:.0e} |\n",
            self.episode_count,
            self.ppo_update_count,
            self.car_count,
            self.timeout_s,
            self.rotation_speed,
            self.thrust,
            self.drag,
            self.observation_dim,
            self.ray_count,
            self.lookahead_count,
            self.lookahead_range.0,
            self.lookahead_range.1,
            self.velocity_reward_scale,
            self.speed_reward_reference,
            self.centreline_reward_coef,
            self.centreline_reward_max_distance,
            self.time_penalty_per_tick,
            self.crash_penalty,
            self.ppo_epochs,
            self.clip_epsilon,
            self.gamma,
            self.gae_lambda,
            self.max_steps,
            self.min_update_steps,
            self.samples_per_tick,
            self.hidden_dim,
            self.actor_lr,
            self.critic_lr,
        )
    }
}
