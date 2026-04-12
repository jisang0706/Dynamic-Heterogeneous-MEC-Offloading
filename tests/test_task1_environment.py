from __future__ import annotations

import unittest

import numpy as np

from src.config import EnvironmentConfig
from src.environment import DynamicMECEnv


class Task1EnvironmentTests(unittest.TestCase):
    def test_reset_shapes_and_feature_ranges(self) -> None:
        config = EnvironmentConfig(num_agents=5, episode_length=10)
        env = DynamicMECEnv(config, seed=7)

        observation = env.reset()

        self.assertEqual(observation.device_obs.shape, (5, 14))
        self.assertEqual(observation.server_obs.shape, (3,))
        self.assertTrue(np.all(env.distances_m >= config.min_distance_m))
        self.assertTrue(np.all(env.distances_m <= config.max_distance_m))
        self.assertTrue(np.all(env.cpu_freqs_ghz > 0.0))
        self.assertTrue(np.isfinite(observation.device_obs).all())
        self.assertTrue(np.isfinite(observation.server_obs).all())

    def test_mobility_cpu_and_channel_update_each_step(self) -> None:
        config = EnvironmentConfig(num_agents=5, episode_length=10)
        env = DynamicMECEnv(config, seed=11)
        env.reset()

        prev_distances = env.distances_m.copy()
        prev_channel_ratios = env.channel_gain_ratios.copy()
        action = np.full((config.num_agents, config.num_tasks_per_step + 1), 0.5, dtype=np.float32)
        env.step(action)

        self.assertFalse(np.allclose(prev_distances, env.distances_m))
        # Raw channel gains are around 1e-10, so default allclose tolerances are too loose.
        self.assertFalse(np.allclose(prev_channel_ratios, env.channel_gain_ratios))
        self.assertTrue(np.all(env.distances_m >= config.min_distance_m))
        self.assertTrue(np.all(env.distances_m <= config.max_distance_m))

        for profile, freq in zip(env.device_profiles, env.cpu_freqs_ghz):
            self.assertGreaterEqual(freq, profile.cpu_range_ghz[0])
            self.assertLessEqual(freq, profile.cpu_range_ghz[1])

    def test_type_a_moves_faster_than_type_c_on_average(self) -> None:
        config = EnvironmentConfig(num_agents=15, episode_length=60)
        env = DynamicMECEnv(config, seed=5)
        env.reset()

        actions = np.zeros((config.num_agents, config.num_tasks_per_step + 1), dtype=np.float32)
        speed_history = []
        for _ in range(40):
            speed_history.append(env.speeds_m_s.copy())
            env.step(actions)

        speeds = np.asarray(speed_history, dtype=np.float32)
        a_indices = [idx for idx, profile in enumerate(env.device_profiles) if profile.type_name == "A"]
        c_indices = [idx for idx, profile in enumerate(env.device_profiles) if profile.type_name == "C"]
        self.assertTrue(a_indices)
        self.assertTrue(c_indices)
        self.assertGreater(float(speeds[:, a_indices].mean()), float(speeds[:, c_indices].mean()))

    def test_200_step_rollout_has_no_nan_or_inf(self) -> None:
        config = EnvironmentConfig(num_agents=5, episode_length=200)
        env = DynamicMECEnv(config, seed=17)
        observation = env.reset()
        rng = np.random.default_rng(17)

        self.assertTrue(np.isfinite(observation.device_obs).all())
        for step in range(config.episode_length):
            actions = rng.uniform(
                0.0,
                1.0,
                size=(config.num_agents, config.num_tasks_per_step + 1),
            ).astype(np.float32)
            observation, reward, done, info = env.step(actions)
            self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(np.isfinite(observation.device_obs).all())
            self.assertTrue(np.isfinite(observation.server_obs).all())
            self.assertTrue(np.isfinite(np.asarray(list(info.values()), dtype=np.float32)).all())
            if step < config.episode_length - 1:
                self.assertFalse(done)

        self.assertTrue(done)

    def test_li_style_intra_slot_ordering_reduces_small_task_delay(self) -> None:
        config = EnvironmentConfig(num_agents=1, episode_length=1, use_mobility=False, use_cpu_dynamics=False)
        env = DynamicMECEnv(config, seed=23)
        env.reset()
        env.local_queues[:] = 0.0
        env.edge_queue = 0.0
        env.prev_edge_queue = 0.0
        env.cpu_freqs_ghz[:] = np.asarray([1.0], dtype=np.float32)
        env.max_tx_powers_mw[:] = np.asarray([300.0], dtype=np.float32)
        env.channel_gains[:] = np.asarray([1.0], dtype=np.float32)
        env.task_matrix = np.asarray(
            [
                [
                    [1.0, 3.0, 10.0],
                    [1.0, 1.0, 10.0],
                    [1.0, 2.0, 10.0],
                ]
            ],
            dtype=np.float32,
        )

        no_offload_action = np.zeros((1, config.num_tasks_per_step + 1), dtype=np.float32)
        _, _, _, _ = env.step(no_offload_action)
        local_only_delays = env.last_reward_breakdown["task_completion_delay_s"][0]
        self.assertLess(local_only_delays[1], local_only_delays[2])
        self.assertLess(local_only_delays[2], local_only_delays[0])

        env.reset()
        env.local_queues[:] = 0.0
        env.edge_queue = 0.0
        env.prev_edge_queue = 0.0
        env.cpu_freqs_ghz[:] = np.asarray([3.0], dtype=np.float32)
        env.max_tx_powers_mw[:] = np.asarray([300.0], dtype=np.float32)
        env.channel_gains[:] = np.asarray([1.0], dtype=np.float32)
        env.task_matrix = np.asarray(
            [
                [
                    [3.0, 1.0, 10.0],
                    [1.0, 1.0, 10.0],
                    [2.0, 1.0, 10.0],
                ]
            ],
            dtype=np.float32,
        )

        full_offload_action = np.ones((1, config.num_tasks_per_step + 1), dtype=np.float32)
        _, _, _, _ = env.step(full_offload_action)
        edge_only_delays = env.last_reward_breakdown["task_completion_delay_s"][0]
        self.assertLess(edge_only_delays[1], edge_only_delays[2])
        self.assertLess(edge_only_delays[2], edge_only_delays[0])


if __name__ == "__main__":
    unittest.main()
