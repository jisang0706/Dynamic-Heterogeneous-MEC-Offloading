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
        prev_channels = env.channel_gains.copy()
        action = np.full((config.num_agents, config.num_tasks_per_step + 1), 0.5, dtype=np.float32)
        env.step(action)

        self.assertFalse(np.allclose(prev_distances, env.distances_m))
        self.assertFalse(np.allclose(prev_channels, env.channel_gains))
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


if __name__ == "__main__":
    unittest.main()
