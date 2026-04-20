from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.modules.graph_builder import GraphBatch
from src.utils import orthogonal_init


def _load_pyg_layers() -> type[nn.Module]:
    from torch_geometric.nn import GCNConv

    return GCNConv


class DenseGraphConv(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain("relu"))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1)
        degree_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        normalized = degree_inv_sqrt.unsqueeze(-1) * adjacency * degree_inv_sqrt.unsqueeze(-2)
        return self.linear(normalized @ x)


class PGCNCritic(nn.Module):
    def __init__(
        self,
        device_dim: int = 14,
        server_dim: int = 3,
        hidden_dim: int = 64,
        head_hidden_dim: int = 128,
        use_pyg: bool | None = None,
    ) -> None:
        super().__init__()
        self.device_dim = device_dim
        self.server_dim = server_dim
        self.hidden_dim = hidden_dim

        self.device_encoder = orthogonal_init(nn.Linear(device_dim, hidden_dim), gain=nn.init.calculate_gain("relu"))
        self.server_encoder = orthogonal_init(nn.Linear(server_dim, hidden_dim), gain=nn.init.calculate_gain("relu"))

        if use_pyg is None:
            try:
                GCNConv = _load_pyg_layers()
                self.use_pyg = True
                self.conv1 = GCNConv(hidden_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, hidden_dim)
            except ModuleNotFoundError:
                self.use_pyg = False
                self.conv1 = DenseGraphConv(hidden_dim)
                self.conv2 = DenseGraphConv(hidden_dim)
                self.conv3 = DenseGraphConv(hidden_dim)
        elif use_pyg:
            GCNConv = _load_pyg_layers()
            self.use_pyg = True
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.use_pyg = False
            self.conv1 = DenseGraphConv(hidden_dim)
            self.conv2 = DenseGraphConv(hidden_dim)
            self.conv3 = DenseGraphConv(hidden_dim)

        self.fc1 = orthogonal_init(nn.Linear(hidden_dim * 3, head_hidden_dim), gain=nn.init.calculate_gain("relu"))
        self.fc2 = orthogonal_init(nn.Linear(head_hidden_dim, 1), gain=1.0)

    def forward(
        self,
        device_obs: torch.Tensor,
        server_obs: torch.Tensor,
        graph: GraphBatch | Any | None = None,
        edge_index: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if device_obs.dim() == 2:
            device_obs = device_obs.unsqueeze(0)
        if server_obs.dim() == 1:
            server_obs = server_obs.unsqueeze(0)
        if device_obs.shape[-1] != self.device_dim:
            raise ValueError(f"device_obs last dimension must be {self.device_dim}, got {device_obs.shape[-1]}")
        if server_obs.shape[-1] != self.server_dim:
            raise ValueError(f"server_obs last dimension must be {self.server_dim}, got {server_obs.shape[-1]}")

        edge_index, adjacency = self._extract_topology(
            graph=graph,
            edge_index=edge_index,
            adjacency=adjacency,
            num_nodes=device_obs.shape[1] + 1,
            device=device_obs.device,
            dtype=device_obs.dtype,
        )
        x = self._encode_nodes(device_obs, server_obs)

        if self.use_pyg and edge_index is not None:
            return self._forward_pyg(x=x, edge_index=edge_index)
        if adjacency is None:
            raise ValueError("adjacency is required for the dense fallback path.")
        return self._forward_dense(x=x, adjacency=adjacency)

    def _encode_nodes(self, device_obs: torch.Tensor, server_obs: torch.Tensor) -> torch.Tensor:
        device_emb = torch.relu(self.device_encoder(device_obs))
        server_emb = torch.relu(self.server_encoder(server_obs)).unsqueeze(1)
        return torch.cat([device_emb, server_emb], dim=1)

    def _forward_pyg(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes = x.shape[:2]
        hidden = x.reshape(batch_size * num_nodes, self.hidden_dim)
        hidden = torch.relu(self.conv1(hidden, edge_index))
        hidden = torch.relu(self.conv2(hidden, edge_index))
        hidden = torch.relu(self.conv3(hidden, edge_index))
        hidden = hidden.view(batch_size, num_nodes, self.hidden_dim)
        return self._head(hidden)

    def _forward_dense(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        hidden = torch.relu(self.conv1(x, adjacency))
        hidden = torch.relu(self.conv2(hidden, adjacency))
        hidden = torch.relu(self.conv3(hidden, adjacency))
        return self._head(hidden)

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        server_hidden = hidden[:, -1]
        device_hidden = hidden[:, :-1]
        device_hidden_mean = device_hidden.mean(dim=1)
        num_agents = device_hidden.shape[1]
        shared_server = server_hidden.unsqueeze(1).expand(-1, num_agents, -1)
        shared_device_mean = device_hidden_mean.unsqueeze(1).expand(-1, num_agents, -1)
        critic_input = torch.cat([device_hidden, shared_device_mean, shared_server], dim=-1)
        return self.fc2(torch.relu(self.fc1(critic_input))).squeeze(-1)

    @staticmethod
    def _extract_topology(
        graph: GraphBatch | Any | None,
        edge_index: torch.Tensor | None,
        adjacency: torch.Tensor | None,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if isinstance(graph, GraphBatch):
            extracted_edge_index = graph.edge_index.to(device=device)
            extracted_adjacency = graph.adjacency.to(device=device, dtype=dtype)
            return extracted_edge_index, extracted_adjacency

        if graph is not None and hasattr(graph, "edge_index"):
            extracted_edge_index = graph.edge_index.to(device=device)
            extracted_adjacency = getattr(graph, "adjacency", adjacency)
            if extracted_adjacency is not None:
                extracted_adjacency = extracted_adjacency.to(device=device, dtype=dtype)
            return extracted_edge_index, extracted_adjacency

        if adjacency is not None:
            return edge_index, adjacency.to(device=device, dtype=dtype)
        if edge_index is not None:
            return edge_index.to(device=device), PGCNCritic._dense_adjacency_from_edge_index(
                edge_index=edge_index.to(device=device),
                num_nodes=num_nodes,
                device=device,
                dtype=dtype,
            )
        raise ValueError("A graph topology must provide edge_index or adjacency.")

    @staticmethod
    def _dense_adjacency_from_edge_index(
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        adjacency = torch.eye(num_nodes, dtype=dtype, device=device)
        adjacency[edge_index[0], edge_index[1]] = 1.0
        return adjacency
