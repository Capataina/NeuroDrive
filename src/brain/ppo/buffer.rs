use std::collections::HashMap;

use bevy::prelude::*;

/// Trainer-wide rollout buffer that collects transitions from all environment
/// instances. Each transition is tagged with its source `env_id` so GAE can
/// be computed per-env without cross-env value leakage.
#[derive(Resource, Clone, Debug, Default)]
pub struct TrainerRolloutBuffer {
    pub states: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub latent_actions: Vec<Vec<f32>>,
    pub safety_clamp_hits: Vec<[bool; 2]>,
    pub old_log_probs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    pub dones: Vec<bool>,
    pub env_ids: Vec<u32>,
}

impl TrainerRolloutBuffer {
    pub fn push_pre_step(
        &mut self,
        env_id: u32,
        state: Vec<f32>,
        actions: Vec<f32>,
        latent_actions: Vec<f32>,
        safety_clamp_hits: [bool; 2],
        value: f32,
        log_prob: f32,
    ) {
        self.states.push(state);
        self.actions.push(actions);
        self.latent_actions.push(latent_actions);
        self.safety_clamp_hits.push(safety_clamp_hits);
        self.old_log_probs.push(log_prob);
        self.values.push(value);
        self.env_ids.push(env_id);
    }

    pub fn push_reward(&mut self, reward: f32, done: bool) {
        self.rewards.push(reward);
        self.dones.push(done);
    }

    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    pub fn pre_step_count(&self) -> usize {
        self.states.len()
    }

    pub fn pending_rewards(&self) -> usize {
        self.states.len().saturating_sub(self.rewards.len())
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.latent_actions.clear();
        self.safety_clamp_hits.clear();
        self.old_log_probs.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
        self.env_ids.clear();
    }

    pub fn is_aligned(&self) -> bool {
        let n = self.rewards.len();
        self.states.len() == n
            && self.actions.len() == n
            && self.latent_actions.len() == n
            && self.safety_clamp_hits.len() == n
            && self.old_log_probs.len() == n
            && self.values.len() == n
            && self.dones.len() == n
            && self.env_ids.len() == n
    }

    /// Computes GAE per-env to avoid cross-env value leakage in the interleaved
    /// buffer. Each env's transitions are grouped and GAE is computed within
    /// that group independently. Advantages are returned un-normalised;
    /// normalisation happens per-minibatch in the update loop.
    pub fn compute_gae_per_env(
        &self,
        bootstrap_values: &HashMap<u32, f32>,
        gamma: f32,
        lambda: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = self.rewards.len();
        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];

        // Group buffer indices by env_id, preserving insertion order
        let mut env_indices: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, &eid) in self.env_ids.iter().enumerate().take(n) {
            env_indices.entry(eid).or_default().push(i);
        }

        // Compute GAE independently within each env's transition sequence
        for (eid, indices) in &env_indices {
            let bootstrap = bootstrap_values.get(eid).copied().unwrap_or(0.0);
            let mut gae = 0.0;

            for (pos, &t) in indices.iter().enumerate().rev() {
                // Next value is either from the next transition of THIS env,
                // or the bootstrap value if this is the last one
                let next_val = if pos + 1 < indices.len() {
                    self.values[indices[pos + 1]]
                } else {
                    bootstrap
                };
                let mask = if self.dones[t] { 0.0 } else { 1.0 };

                let delta = self.rewards[t] + gamma * next_val * mask - self.values[t];
                gae = delta + gamma * lambda * mask * gae;

                advantages[t] = gae;
                returns[t] = gae + self.values[t];
            }
        }

        (advantages, returns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that per-env GAE with a single env produces the same result
    /// as the old flat GAE (regression test).
    #[test]
    fn single_env_gae_matches_flat_gae() {
        let mut buf = TrainerRolloutBuffer::default();
        let rewards = vec![1.0, 0.5, -1.0, 2.0];
        let values = vec![0.5, 0.3, 0.1, 0.8];
        let dones = vec![false, false, true, false];

        for i in 0..4 {
            buf.push_pre_step(0, vec![0.0], vec![0.0], vec![0.0], [false, false], values[i], 0.0);
            buf.push_reward(rewards[i], dones[i]);
        }

        let bootstrap = 0.4;
        let gamma = 0.99;
        let lambda = 0.95;

        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, bootstrap);
        let (adv_per_env, ret_per_env) = buf.compute_gae_per_env(&bootstraps, gamma, lambda);

        // Compute flat GAE manually with same logic
        let mut flat_adv = vec![0.0; 4];
        let mut flat_ret = vec![0.0; 4];
        let mut gae = 0.0;
        for t in (0..4).rev() {
            let next_val = if t + 1 < 4 { values[t + 1] } else { bootstrap };
            let mask = if dones[t] { 0.0 } else { 1.0 };
            let delta = rewards[t] + gamma * next_val * mask - values[t];
            gae = delta + gamma * lambda * mask * gae;
            flat_adv[t] = gae;
            flat_ret[t] = gae + values[t];
        }
        for i in 0..4 {
            assert!((adv_per_env[i] - flat_adv[i]).abs() < 1e-5, "adv mismatch at {i}");
            assert!((ret_per_env[i] - flat_ret[i]).abs() < 1e-5, "ret mismatch at {i}");
        }
    }

    /// Verify that per-env GAE does NOT leak values across envs.
    #[test]
    fn multi_env_gae_isolates_envs() {
        let mut buf = TrainerRolloutBuffer::default();

        // Interleave: env0, env1, env0, env1
        // env 0: reward 1.0 (not done), reward 2.0 (done)
        // env 1: reward -1.0 (not done), reward 0.5 (not done)
        let entries = [
            (0u32, 1.0, 0.5, false),
            (1u32, -1.0, 0.3, false),
            (0u32, 2.0, 0.8, true),
            (1u32, 0.5, 0.2, false),
        ];

        for &(eid, reward, value, done) in &entries {
            buf.push_pre_step(eid, vec![0.0], vec![0.0], vec![0.0], [false, false], value, 0.0);
            buf.push_reward(reward, done);
        }

        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, 0.0); // env 0 ended with done=true, bootstrap irrelevant
        bootstraps.insert(1, 0.6); // env 1 non-terminal, bootstrap = 0.6

        let gamma = 0.99;
        let lambda = 0.95;
        let (adv, ret) = buf.compute_gae_per_env(&bootstraps, gamma, lambda);

        // env 0 indices: [0, 2]
        // env 0 transition 2 (done=true): delta = 2.0 + 0 - 0.8 = 1.2, gae = 1.2
        // env 0 transition 0 (done=false): next_val = values[2] = 0.8
        //   delta = 1.0 + 0.99*0.8 - 0.5 = 1.292, gae = 1.292 + 0.99*0.95*1.2 = 2.4214
        let env0_ret_0 = ret[0];
        let env0_ret_2 = ret[2];

        // env 1 indices: [1, 3]
        // env 1 transition 3 (done=false): next_val = bootstrap = 0.6
        //   delta = 0.5 + 0.99*0.6 - 0.2 = 0.894, gae = 0.894
        // env 1 transition 1 (done=false): next_val = values[3] = 0.2
        //   delta = -1.0 + 0.99*0.2 - 0.3 = -1.102, gae = -1.102 + 0.99*0.95*0.894 = -0.26117
        let _env1_ret_1 = ret[1];
        let env1_ret_3 = ret[3];

        // Returns should be gae + value
        assert!((env0_ret_2 - (1.2 + 0.8)).abs() < 1e-4, "env0 t2 ret: {env0_ret_2}");
        assert!((env1_ret_3 - (0.894 + 0.2)).abs() < 1e-4, "env1 t3 ret: {env1_ret_3}");

        // Key check: env 0 t0's return should NOT depend on env 1 t1's value
        // If cross-env leakage existed, the return would be different
        let expected_env0_ret_0 = 2.4214 + 0.5;
        assert!(
            (env0_ret_0 - expected_env0_ret_0).abs() < 0.01,
            "env0 t0 ret: {env0_ret_0} expected ~{expected_env0_ret_0}"
        );

        // Advantages are un-normalised (per-chunk normalisation happens in update loop)
        assert!(adv.len() == 4);
    }
}
