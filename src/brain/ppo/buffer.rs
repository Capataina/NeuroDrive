use std::collections::HashMap;

use bevy::prelude::*;

/// Runtime scratch — grouped indices-by-env_id used inside `compute_gae_per_env`.
/// Held on the buffer so per-call allocations are avoided on the hot
/// prepare path; `std::mem::take` in `ppo_prepare_update` rotates it into
/// the frozen buffer, which then drops at the end of the update sequence.
#[derive(Clone, Debug, Default)]
pub struct EnvGrouping {
    /// Indexed by `env_id as usize`. Each inner `Vec` lists buffer positions
    /// where that env contributed a transition, in insertion order.
    buckets: Vec<Vec<usize>>,
}

/// Trainer-wide rollout buffer that collects transitions from all environment
/// instances. Each transition is tagged with its source `env_id` so GAE can
/// be computed per-env without cross-env value leakage.
///
/// `states`, `actions`, and `latent_actions` are stored as flat contiguous
/// `Vec<f32>` with stride `obs_dim` and `act_dim` respectively. This avoids
/// per-transition heap allocations and gives cache-friendly sequential access
/// during batched training.
#[derive(Resource, Clone, Debug)]
pub struct TrainerRolloutBuffer {
    pub states: Vec<f32>,          // flat, stride = obs_dim
    pub actions: Vec<f32>,         // flat, stride = act_dim
    pub latent_actions: Vec<f32>,  // flat, stride = act_dim
    pub obs_dim: usize,
    pub act_dim: usize,
    pub safety_clamp_hits: Vec<[bool; 2]>,
    pub old_log_probs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub values: Vec<f32>,
    pub dones: Vec<bool>,
    pub env_ids: Vec<u32>,
    /// Scratch for per-env index grouping during GAE; reused across calls.
    env_grouping: EnvGrouping,
}

impl Default for TrainerRolloutBuffer {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            latent_actions: Vec::new(),
            obs_dim: 43,
            act_dim: 2,
            safety_clamp_hits: Vec::new(),
            old_log_probs: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            dones: Vec::new(),
            env_ids: Vec::new(),
            env_grouping: EnvGrouping::default(),
        }
    }
}

impl TrainerRolloutBuffer {
    pub fn push_pre_step(
        &mut self,
        env_id: u32,
        state: &[f32],
        actions: &[f32],
        latent_actions: &[f32],
        safety_clamp_hits: [bool; 2],
        value: f32,
        log_prob: f32,
    ) {
        self.states.extend_from_slice(state);
        self.actions.extend_from_slice(actions);
        self.latent_actions.extend_from_slice(latent_actions);
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
        self.states.len() / self.obs_dim
    }

    pub fn pending_rewards(&self) -> usize {
        self.pre_step_count().saturating_sub(self.rewards.len())
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
        self.pre_step_count() == n
            && self.actions.len() / self.act_dim == n
            && self.latent_actions.len() / self.act_dim == n
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
    ///
    /// Uses `self.env_grouping` as reusable scratch — `env_id`s are dense
    /// small integers assigned by the trainer, so indexing a `Vec` by
    /// `env_id as usize` avoids the per-call `HashMap` allocation plus its
    /// non-deterministic iteration order.
    pub fn compute_gae_per_env(
        &mut self,
        bootstrap_values: &HashMap<u32, f32>,
        gamma: f32,
        lambda: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = self.rewards.len();
        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];

        if n == 0 {
            return (advantages, returns);
        }

        // Size buckets to `max(env_id) + 1`; reuse capacity across calls.
        let max_env_id = self.env_ids.iter().take(n).copied().max().unwrap_or(0) as usize;
        let needed = max_env_id + 1;
        let buckets = &mut self.env_grouping.buckets;
        if buckets.len() < needed {
            buckets.resize_with(needed, Vec::new);
        }
        for bucket in buckets.iter_mut().take(needed) {
            bucket.clear();
        }

        // Group buffer indices by env_id, preserving insertion order.
        for (i, &eid) in self.env_ids.iter().enumerate().take(n) {
            buckets[eid as usize].push(i);
        }

        // Compute GAE independently within each env's transition sequence.
        // Iteration order over envs is deterministic (ascending env_id).
        for (eid, indices) in buckets.iter().enumerate().take(needed) {
            if indices.is_empty() {
                continue;
            }
            let bootstrap = bootstrap_values.get(&(eid as u32)).copied().unwrap_or(0.0);
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
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
        let rewards = vec![1.0, 0.5, -1.0, 2.0];
        let values = vec![0.5, 0.3, 0.1, 0.8];
        let dones = vec![false, false, true, false];

        for i in 0..4 {
            buf.push_pre_step(0, &[0.0], &[0.0], &[0.0], [false, false], values[i], 0.0);
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
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };

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
            buf.push_pre_step(eid, &[0.0], &[0.0], &[0.0], [false, false], value, 0.0);
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

    // ── GAE extension tests (2026-04-18 test-suite expansion) ────────────

    #[test]
    fn gae_empty_buffer_returns_empty_vectors() {
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
        let bootstraps = HashMap::new();
        let (adv, ret) = buf.compute_gae_per_env(&bootstraps, 0.99, 0.95);
        assert!(adv.is_empty());
        assert!(ret.is_empty());
    }

    #[test]
    fn gae_returns_equal_advantages_plus_values() {
        // Foundational identity: return[i] == advantage[i] + value[i].
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
        let values = vec![0.5, 0.3, 0.1, 0.8];
        for i in 0..4 {
            buf.push_pre_step(0, &[0.0], &[0.0], &[0.0], [false, false], values[i], 0.0);
            buf.push_reward((i as f32) * 0.5, i == 3);
        }
        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, 0.0);
        let (adv, ret) = buf.compute_gae_per_env(&bootstraps, 0.99, 0.95);
        for i in 0..4 {
            assert!((ret[i] - (adv[i] + values[i])).abs() < 1e-4);
        }
    }

    #[test]
    fn gae_single_step_episode_advantage_is_reward_minus_value() {
        // When there is exactly one transition and it is terminal (done),
        // advantage = reward - value (bootstrap is masked out).
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
        buf.push_pre_step(0, &[0.0], &[0.0], &[0.0], [false, false], 2.0, 0.0);
        buf.push_reward(5.0, true); // done

        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, 999.0); // should be ignored because done=true
        let (adv, _ret) = buf.compute_gae_per_env(&bootstraps, 0.99, 0.95);
        // delta = 5.0 + gamma * 999.0 * 0 - 2.0 = 3.0. gae = 3.0.
        assert!((adv[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn gae_terminal_state_zero_bootstrap() {
        // When done=true mid-episode, the value from the next transition
        // must NOT propagate back past the terminal boundary.
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
        let values = vec![0.0, 0.0, 99.0]; // giant value after terminal — should not leak
        let rewards = vec![1.0, 1.0, 1.0];
        let dones = vec![false, true, false];
        for i in 0..3 {
            buf.push_pre_step(0, &[0.0], &[0.0], &[0.0], [false, false], values[i], 0.0);
            buf.push_reward(rewards[i], dones[i]);
        }
        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, 0.0);
        let (adv, _ret) = buf.compute_gae_per_env(&bootstraps, 0.99, 0.95);

        // The huge value at t=2 should not contribute to advantage at t=0 or t=1
        // (mask at t=1 is 0 because done=true).
        // If leakage existed, adv[0] would be much larger than ~1-2.
        assert!(adv[0].abs() < 10.0, "possible bootstrap leak at t=0: {}", adv[0]);
        assert!(adv[1].abs() < 10.0, "possible bootstrap leak at t=1: {}", adv[1]);
    }

    #[test]
    fn gae_env_grouping_capacity_preserved_across_calls() {
        // EnvGrouping should reuse its Vec<Vec<usize>> across calls to
        // compute_gae_per_env — the 2026-04-18 paper 2 finding.
        let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
        // First call with 4 transitions across 2 envs
        for (env_id, val) in [(0u32, 0.5), (1u32, 0.3), (0u32, 0.1), (1u32, 0.2)] {
            buf.push_pre_step(env_id, &[0.0], &[0.0], &[0.0], [false, false], val, 0.0);
            buf.push_reward(0.1, false);
        }
        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, 0.0);
        bootstraps.insert(1, 0.0);
        let _ = buf.compute_gae_per_env(&bootstraps, 0.99, 0.95);

        // After first call, env_grouping has 2 buckets with capacity.
        let cap_after_first: Vec<usize> =
            buf.env_grouping.buckets.iter().map(|v| v.capacity()).collect();
        assert_eq!(cap_after_first.len(), 2);
        assert!(cap_after_first.iter().all(|&c| c >= 2),
            "buckets should have capacity >=2 after 4 transitions across 2 envs");

        // Clear transitions and run again — capacity should be preserved.
        buf.clear();
        for (env_id, val) in [(0u32, 0.5), (1u32, 0.3), (0u32, 0.1), (1u32, 0.2)] {
            buf.push_pre_step(env_id, &[0.0], &[0.0], &[0.0], [false, false], val, 0.0);
            buf.push_reward(0.1, false);
        }
        let _ = buf.compute_gae_per_env(&bootstraps, 0.99, 0.95);

        let cap_after_second: Vec<usize> =
            buf.env_grouping.buckets.iter().map(|v| v.capacity()).collect();
        // The buckets are still at least as large as before (clear() preserves capacity).
        for (i, (&a, &b)) in cap_after_first.iter().zip(cap_after_second.iter()).enumerate() {
            assert!(b >= a, "bucket {} capacity shrank from {} to {}", i, a, b);
        }
    }

    #[test]
    fn gae_iteration_order_is_deterministic() {
        // With Vec<Vec<usize>> indexing by env_id, env iteration order is
        // strictly ascending — not HashMap-order-dependent. Run twice and
        // verify bit-identical output.
        let build = || -> TrainerRolloutBuffer {
            let mut buf = TrainerRolloutBuffer { obs_dim: 1, act_dim: 1, ..Default::default() };
            for (env_id, val) in [(2u32, 0.5), (0u32, 0.3), (1u32, 0.1), (2u32, 0.2), (0u32, 0.4)] {
                buf.push_pre_step(env_id, &[0.0], &[0.0], &[0.0], [false, false], val, 0.0);
                buf.push_reward(0.1, false);
            }
            buf
        };
        let mut bootstraps = HashMap::new();
        bootstraps.insert(0, 0.0);
        bootstraps.insert(1, 0.0);
        bootstraps.insert(2, 0.0);

        let mut buf1 = build();
        let mut buf2 = build();
        let (adv1, ret1) = buf1.compute_gae_per_env(&bootstraps, 0.99, 0.95);
        let (adv2, ret2) = buf2.compute_gae_per_env(&bootstraps, 0.99, 0.95);

        for i in 0..adv1.len() {
            assert!((adv1[i] - adv2[i]).abs() < 1e-9);
            assert!((ret1[i] - ret2[i]).abs() < 1e-9);
        }
    }
}
