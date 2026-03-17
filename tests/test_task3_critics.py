from __future__ import annotations

import unittest

import torch

from src.modules import GraphBuilder
from src.networks import MLPCritic, PGCNCritic, SetCritic


class Task3CriticTests(unittest.TestCase):
    def test_mlp_critic_accepts_structured_inputs(self) -> None:
        critic = MLPCritic(obs_dim=14, num_agents=5, central_obs_dim=3, hidden_dim=200)
        device_obs = torch.randn(5, 14)
        server_obs = torch.tensor([0.1, 0.2, 1.0], dtype=torch.float32)

        value = critic(device_obs, server_obs)

        self.assertEqual(tuple(value.shape), (1, 1))

    def test_mlp_critic_accepts_batched_inputs(self) -> None:
        critic = MLPCritic(obs_dim=14, num_agents=5, central_obs_dim=3, hidden_dim=200)
        device_obs = torch.randn(4, 5, 14)
        server_obs = torch.randn(4, 3)

        value = critic(device_obs, server_obs)

        self.assertEqual(tuple(value.shape), (4, 1))

    def test_set_critic_uses_server_preserving_readout(self) -> None:
        critic = SetCritic(device_dim=14, server_dim=3, hidden_dim=64, head_hidden_dim=128)
        device_obs = torch.randn(3, 5, 14)
        server_obs = torch.randn(3, 3)

        value = critic(device_obs, server_obs)

        self.assertEqual(tuple(value.shape), (3, 1))
        self.assertEqual(critic.fc1.in_features, 128)

    def test_pgcn_critic_accepts_graph_batch_fallback(self) -> None:
        builder = GraphBuilder(num_devices=5, graph_type="star")
        graph = builder.build()
        critic = PGCNCritic(device_dim=14, server_dim=3, hidden_dim=64, head_hidden_dim=128, use_pyg=False)
        device_obs = torch.randn(5, 14)
        server_obs = torch.tensor([0.1, 0.0, 1.0], dtype=torch.float32)

        value = critic(device_obs, server_obs, graph=graph)

        self.assertEqual(tuple(value.shape), (1, 1))

    def test_pgcn_critic_accepts_pyg_data_when_available(self) -> None:
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric is not installed")

        builder = GraphBuilder(num_devices=3, graph_type="star_proximity", distance_threshold_m=150.0)
        graph = builder.build(
            positions=torch.tensor([[0.0, 0.0], [100.0, 0.0], [250.0, 0.0]], dtype=torch.float32)
        )
        critic = PGCNCritic(device_dim=14, server_dim=3, hidden_dim=64, head_hidden_dim=128, use_pyg=True)
        device_obs = torch.randn(3, 14)
        server_obs = torch.tensor([0.1, -0.1, 1.0], dtype=torch.float32)

        value = critic(device_obs, server_obs, graph=graph.to_pyg_data())

        self.assertEqual(tuple(value.shape), (1, 1))

    def test_pgcn_forward_supports_all_target_agent_counts(self) -> None:
        for num_agents in (5, 10, 15, 20):
            with self.subTest(num_agents=num_agents):
                critic = PGCNCritic(device_dim=14, server_dim=3, hidden_dim=64, head_hidden_dim=128, use_pyg=False)
                graph = GraphBuilder(num_devices=num_agents, graph_type="star").build()
                device_obs = torch.randn(2, num_agents, 14)
                server_obs = torch.randn(2, 3)

                value = critic(device_obs, server_obs, graph=graph)

                self.assertEqual(tuple(value.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
