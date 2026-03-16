from __future__ import annotations

import unittest

import torch

from src.modules import GraphBuilder


class Task2GraphBuilderTests(unittest.TestCase):
    def test_star_graph_has_static_bidirectional_server_edges(self) -> None:
        builder = GraphBuilder(num_devices=3, graph_type="star")
        device_obs = torch.arange(42, dtype=torch.float32).reshape(3, 14)
        server_obs = torch.tensor([0.2, 0.1, 1.0], dtype=torch.float32)

        graph = builder.build(device_obs=device_obs, server_obs=server_obs)

        self.assertEqual(tuple(graph.node_features.shape), (4, 14))
        self.assertEqual(tuple(graph.x.shape), (4, 14))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 6))
        self.assertEqual(graph.server_index, 3)
        expected_server_features = torch.tensor(
            [0.2, 0.1, 1.0] + [0.0] * 11,
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(graph.x[3], expected_server_features))

        expected_edges = {
            (0, 3), (1, 3), (2, 3),
            (3, 0), (3, 1), (3, 2),
        }
        actual_edges = {tuple(edge.tolist()) for edge in graph.edge_index.t()}
        self.assertEqual(actual_edges, expected_edges)

    def test_star_graph_topology_is_static_across_builds(self) -> None:
        builder = GraphBuilder(num_devices=4, graph_type="star")
        device_obs = torch.randn(4, 14)
        server_obs = torch.tensor([0.1, -0.2, 1.0], dtype=torch.float32)

        graph_a = builder.build(device_obs=device_obs, server_obs=server_obs)
        graph_b = builder.build(device_obs=device_obs + 1.0, server_obs=server_obs)

        self.assertTrue(torch.equal(graph_a.edge_index, graph_b.edge_index))
        self.assertTrue(torch.equal(graph_a.adjacency, graph_b.adjacency))

    def test_star_proximity_adds_device_edges_under_threshold(self) -> None:
        builder = GraphBuilder(num_devices=3, graph_type="star_proximity", distance_threshold_m=150.0)
        device_obs = torch.zeros(3, 14)
        server_obs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [300.0, 0.0],
            ],
            dtype=torch.float32,
        )

        graph = builder.build(device_obs=device_obs, server_obs=server_obs, positions=positions)

        expected_edges = {
            (0, 3), (1, 3), (2, 3),
            (3, 0), (3, 1), (3, 2),
            (0, 1), (1, 0),
        }
        actual_edges = {tuple(edge.tolist()) for edge in graph.edge_index.t()}
        self.assertEqual(actual_edges, expected_edges)
        self.assertIsNotNone(graph.pairwise_distances)
        self.assertLess(float(graph.pairwise_distances[0, 1]), 150.0)
        self.assertGreater(float(graph.pairwise_distances[0, 2]), 150.0)

    def test_star_proximity_requires_positions(self) -> None:
        builder = GraphBuilder(num_devices=2, graph_type="star_proximity")
        device_obs = torch.zeros(2, 14)
        server_obs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        with self.assertRaises(ValueError):
            builder.build(device_obs=device_obs, server_obs=server_obs)


if __name__ == "__main__":
    unittest.main()
