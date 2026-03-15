from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GraphBatch:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    edge_index: torch.Tensor


class GraphBuilder:
    def __init__(self, num_devices: int, graph_type: str = "star", distance_threshold_m: float = 150.0) -> None:
        self.num_devices = num_devices
        self.graph_type = graph_type
        self.distance_threshold_m = distance_threshold_m

    def build(
        self,
        device_obs: torch.Tensor,
        server_obs: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> GraphBatch:
        if device_obs.dim() != 2:
            raise ValueError(f"device_obs must be [num_devices, obs_dim], got {tuple(device_obs.shape)}")
        padded_server = self._pad_server_obs(server_obs, target_dim=device_obs.shape[-1])
        node_features = torch.cat([device_obs, padded_server.unsqueeze(0)], dim=0)

        adjacency = torch.eye(self.num_devices + 1, device=device_obs.device, dtype=device_obs.dtype)
        edges: list[tuple[int, int]] = []
        server_index = self.num_devices
        for device_index in range(self.num_devices):
            adjacency[device_index, server_index] = 1.0
            adjacency[server_index, device_index] = 1.0
            edges.append((device_index, server_index))
            edges.append((server_index, device_index))

        if self.graph_type == "star_proximity" and positions is not None:
            for src in range(self.num_devices):
                for dst in range(src + 1, self.num_devices):
                    distance = torch.linalg.norm(positions[src] - positions[dst]).item()
                    if distance < self.distance_threshold_m:
                        adjacency[src, dst] = 1.0
                        adjacency[dst, src] = 1.0
                        edges.append((src, dst))
                        edges.append((dst, src))

        edge_index = torch.as_tensor(edges, device=device_obs.device, dtype=torch.long).t().contiguous()
        return GraphBatch(node_features=node_features, adjacency=adjacency, edge_index=edge_index)

    @staticmethod
    def _pad_server_obs(server_obs: torch.Tensor, target_dim: int) -> torch.Tensor:
        if server_obs.dim() != 1:
            raise ValueError(f"server_obs must be [central_obs_dim], got {tuple(server_obs.shape)}")
        if server_obs.numel() > target_dim:
            raise ValueError("server_obs cannot be longer than the target node feature dimension")
        padding = torch.zeros(target_dim - server_obs.numel(), dtype=server_obs.dtype, device=server_obs.device)
        return torch.cat([server_obs, padding], dim=0)
