#[derive(Clone, Debug, Default)]
pub struct RolloutBuffer {
    pub states: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub latent_actions: Vec<Vec<f32>>,
    pub safety_clamp_hits: Vec<[bool; 2]>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    pub dones: Vec<bool>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.latent_actions.clear();
        self.safety_clamp_hits.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
    }

    pub fn compute_gae(&self, next_value: f32, gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0; self.rewards.len()];
        let mut returns = vec![0.0; self.rewards.len()];
        let mut gae = 0.0;

        for t in (0..self.rewards.len()).rev() {
            let next_val = if t + 1 < self.rewards.len() {
                self.values[t + 1]
            } else {
                next_value
            };
            let mask = if self.dones[t] { 0.0 } else { 1.0 };

            let delta = self.rewards[t] + gamma * next_val * mask - self.values[t];
            gae = delta + gamma * lambda * mask * gae;

            advantages[t] = gae;
            returns[t] = gae + self.values[t];
        }

        // Normalize advantages
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let variance: f32 =
            advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = (variance + 1e-8).sqrt();

        for a in &mut advantages {
            *a = (*a - mean) / std;
        }

        (advantages, returns)
    }
}
