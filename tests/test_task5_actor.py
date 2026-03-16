from __future__ import annotations

import unittest

import torch

from src.networks import RoleConditionedActor


class Task5ActorTests(unittest.TestCase):
    def test_role_conditioned_actor_outputs_li_style_action_distribution(self) -> None:
        actor = RoleConditionedActor(obs_dim=14, role_dim=3, action_dim=4, hidden_dim=128, use_role=True)
        obs = torch.randn(5, 14)
        role_mu = torch.randn(5, 3)

        mean, std = actor(obs, role_mu)

        self.assertEqual(tuple(mean.shape), (5, 4))
        self.assertEqual(tuple(std.shape), (5, 4))
        self.assertTrue(torch.all(mean >= 0.0))
        self.assertTrue(torch.all(mean <= 10.0))
        self.assertTrue(torch.all(std > 0.0))

    def test_actor_sampling_matches_environment_interface(self) -> None:
        torch.manual_seed(7)
        actor = RoleConditionedActor(obs_dim=14, role_dim=3, action_dim=4, hidden_dim=128, use_role=True)
        obs = torch.randn(3, 14)
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
        actor = RoleConditionedActor(obs_dim=14, role_dim=3, action_dim=4, hidden_dim=128, use_role=False)
        obs = torch.randn(4, 14)

        mean, std = actor(obs)
        action, env_action, log_prob = actor.sample_action(obs)

        self.assertEqual(tuple(mean.shape), (4, 4))
        self.assertEqual(tuple(std.shape), (4, 4))
        self.assertEqual(tuple(action.shape), (4, 4))
        self.assertEqual(tuple(env_action.shape), (4, 4))
        self.assertEqual(tuple(log_prob.shape), (4,))


if __name__ == "__main__":
    unittest.main()
