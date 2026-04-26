from __future__ import annotations

import unittest

import torch

from src.networks import RoleConditionedActor


class Task5ActorTests(unittest.TestCase):
    def test_role_conditioned_actor_outputs_li_style_action_distribution(self) -> None:
        actor = RoleConditionedActor(obs_dim=16, role_dim=3, action_dim=4, hidden_dim=128, use_role=True)
        obs = torch.randn(5, 16)
        role_mu = torch.randn(5, 3)

        mean, std = actor(obs, role_mu)

        self.assertEqual(tuple(mean.shape), (5, 4))
        self.assertEqual(tuple(std.shape), (5, 4))
        self.assertTrue(torch.all(mean >= 0.0))
        self.assertTrue(torch.all(mean <= 10.0))
        self.assertTrue(torch.all(std > 0.0))

    def test_actor_initialization_biases_power_mean_and_exploration(self) -> None:
        actor = RoleConditionedActor(
            obs_dim=16,
            role_dim=3,
            action_dim=4,
            hidden_dim=128,
            use_role=True,
            initial_action_std_env=0.25,
            initial_offloading_mean_env=0.65,
            initial_power_mean_env=0.8,
        )
        obs = torch.zeros(2, 16)
        role_mu = torch.zeros(2, 3)

        mean, std = actor(obs, role_mu)

        self.assertAlmostEqual(float(mean[:, 0].mean().item() / 10.0), 0.65, places=2)
        self.assertAlmostEqual(float(mean[:, 1].mean().item() / 10.0), 0.65, places=2)
        self.assertAlmostEqual(float(mean[:, 2].mean().item() / 10.0), 0.65, places=2)
        self.assertAlmostEqual(float(mean[:, -1].mean().item() / 10.0), 0.8, places=2)
        self.assertAlmostEqual(float(std[:, -1].mean().item() / 10.0), 0.25, places=3)

    def test_actor_sampling_matches_environment_interface(self) -> None:
        torch.manual_seed(7)
        actor = RoleConditionedActor(obs_dim=16, role_dim=3, action_dim=4, hidden_dim=128, use_role=True)
        obs = torch.randn(3, 16)
        role_mu = torch.randn(3, 3)

        action, env_action, log_prob = actor.sample_action(obs, role_mu)
        recomputed_log_prob, entropy, _, _ = actor.evaluate_actions(obs, action, role_mu)

        self.assertEqual(tuple(action.shape), (3, 4))
        self.assertEqual(tuple(env_action.shape), (3, 4))
        self.assertEqual(tuple(log_prob.shape), (3,))
        self.assertEqual(tuple(entropy.shape), (3,))
        self.assertTrue(torch.all(action >= 0.0))
        self.assertTrue(torch.all(action <= 10.0))
        self.assertTrue(torch.all(env_action >= 0.0))
        self.assertTrue(torch.all(env_action <= 1.0))
        self.assertTrue(torch.allclose(env_action, action / 10.0))
        self.assertTrue(torch.allclose(log_prob, recomputed_log_prob))

    def test_no_role_variant_uses_observation_only(self) -> None:
        actor = RoleConditionedActor(obs_dim=16, role_dim=3, action_dim=4, hidden_dim=128, use_role=False)
        obs = torch.randn(4, 16)

        mean, std = actor(obs)
        action, env_action, log_prob = actor.sample_action(obs)

        self.assertEqual(tuple(mean.shape), (4, 4))
        self.assertEqual(tuple(std.shape), (4, 4))
        self.assertEqual(tuple(action.shape), (4, 4))
        self.assertEqual(tuple(env_action.shape), (4, 4))
        self.assertEqual(tuple(log_prob.shape), (4,))

    def test_state_dependent_std_can_vary_with_observation(self) -> None:
        actor = RoleConditionedActor(
            obs_dim=16,
            role_dim=3,
            action_dim=4,
            hidden_dim=64,
            use_role=True,
            use_state_dependent_std=True,
            initial_action_std_env=0.20,
        )
        obs_a = torch.zeros(2, 16)
        obs_b = torch.ones(2, 16)
        role_mu = torch.zeros(2, 3)
        assert actor.std_head is not None
        with torch.no_grad():
            actor.std_head.weight[0, 0] = 0.5

        _, std_a = actor(obs_a, role_mu)
        _, std_b = actor(obs_b, role_mu)

        self.assertEqual(tuple(std_a.shape), (2, 4))
        self.assertTrue(torch.all(std_a > 0.0))
        self.assertNotAlmostEqual(float(std_a[0, 0].item()), float(std_b[0, 0].item()), places=6)


if __name__ == "__main__":
    unittest.main()
