from __future__ import annotations

import unittest

import numpy as np
import torch

from src.buffer import RolloutBuffer, Transition
from src.modules import RoleEncoder, TrajectoryEncoder, role_diversity_loss, role_identifiability_loss


class Task4RoleModuleTests(unittest.TestCase):
    def test_role_encoder_outputs_positive_sigma(self) -> None:
        encoder = RoleEncoder(obs_dim=16, role_dim=3, hidden_dim=12)
        obs = torch.randn(5, 16)

        mu, sigma = encoder(obs)

        self.assertEqual(tuple(mu.shape), (5, 3))
        self.assertEqual(tuple(sigma.shape), (5, 3))
        self.assertTrue(torch.all(sigma > 0.0))

    def test_trajectory_encoder_conditions_on_current_observation(self) -> None:
        encoder = TrajectoryEncoder(obs_dim=16, action_dim=4, role_dim=3, hidden_dim=64)
        trajectory = torch.randn(6, 20, 20)
        current_obs = torch.randn(6, 16)

        mu, sigma = encoder(trajectory, current_obs)

        self.assertEqual(tuple(mu.shape), (6, 3))
        self.assertEqual(tuple(sigma.shape), (6, 3))
        self.assertTrue(torch.all(sigma > 0.0))

    def test_role_identifiability_loss_is_zero_for_matching_gaussians(self) -> None:
        mu = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
        sigma = torch.tensor([[0.5, 0.7, 1.1]], dtype=torch.float32)

        loss = role_identifiability_loss(mu, sigma, mu, sigma)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_rollout_buffer_builds_agent_trajectory_windows(self) -> None:
        buffer = RolloutBuffer()
        num_agents = 2
        obs_dim = 16
        core_obs_dim = 14
        action_dim = 4
        role_dim = 3

        for step in range(3):
            actor_obs = np.full((num_agents, obs_dim), step, dtype=np.float32)
            core_obs = np.full((num_agents, core_obs_dim), step, dtype=np.float32)
            action = np.full((num_agents, action_dim), step + 0.5, dtype=np.float32)
            buffer.add(
                Transition(
                    actor_obs=actor_obs,
                    core_obs=core_obs,
                    server_info=np.array([0.1, 0.0, 1.0], dtype=np.float32),
                    role_mu=np.full((num_agents, role_dim), step, dtype=np.float32),
                    action=action,
                    log_prob=np.zeros(num_agents, dtype=np.float32),
                    reward=np.zeros(num_agents, dtype=np.float32),
                    done=False,
                )
            )

        batch = buffer.build_agent_trajectory_batch(
            window_size=2,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_scale=10.0,
        )

        self.assertEqual(tuple(batch["trajectory"].shape), (6, 2, 20))
        self.assertEqual(tuple(batch["current_obs"].shape), (6, 16))
        self.assertTrue(torch.allclose(batch["trajectory"][0], torch.zeros(2, 20)))
        expected_prev = torch.cat([torch.zeros(16), torch.full((4,), 0.05)], dim=0)
        self.assertTrue(torch.allclose(batch["trajectory"][2, -1], expected_prev))

    def test_role_diversity_loss_is_non_positive(self) -> None:
        role_mu = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

        loss = role_diversity_loss(role_mu)

        self.assertLessEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
