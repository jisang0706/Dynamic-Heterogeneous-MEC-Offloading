from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.modules.graph_builder import GraphBatch
from src.utils import orthogonal_init


def _load_pyg_layers() -> tuple[type[nn.Module], Any]:
    from torch_geometric.nn import GCNConv, global_mean_pool

    return GCNConv, global_mean_pool


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = orthogonal_init(nn.Linear(in_dim, out_dim), gain=nn.init.calculate_gain("relu"))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1)
        degree_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        normalized = degree_inv_sqrt.unsqueeze(-1) * adjacency * degree_inv_sqrt.unsqueeze(-2)
        return self.linear(normalized @ x)


class PGCNCritic(nn.Module):
    def __init__(
        self,
        node_dim: int = 14,
        gcn_hidden_dim: int = 64,
        head_hidden_dim: int = 128,
        use_pyg: bool | None = None,
    ) -> None:
        super().__init__()
        self._pyg_global_mean_pool = None

        if use_pyg is None:
            try:
                GCNConv, global_mean_pool = _load_pyg_layers()
                self.use_pyg = True
                self._pyg_global_mean_pool = global_mean_pool
                self.conv1 = GCNConv(node_dim, gcn_hidden_dim)
                self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
                self.conv3 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
            except ModuleNotFoundError:
                self.use_pyg = False
                self.conv1 = DenseGraphConv(node_dim, gcn_hidden_dim)
                self.conv2 = DenseGraphConv(gcn_hidden_dim, gcn_hidden_dim)
                self.conv3 = DenseGraphConv(gcn_hidden_dim, gcn_hidden_dim)
        elif use_pyg:
            GCNConv, global_mean_pool = _load_pyg_layers()
            self.use_pyg = True
            self._pyg_global_mean_pool = global_mean_pool
            self.conv1 = GCNConv(node_dim, gcn_hidden_dim)
            self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
            self.conv3 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        else:
            self.use_pyg = False
            self.conv1 = DenseGraphConv(node_dim, gcn_hidden_dim)
            self.conv2 = DenseGraphConv(gcn_hidden_dim, gcn_hidden_dim)
            self.conv3 = DenseGraphConv(gcn_hidden_dim, gcn_hidden_dim)

        self.fc1 = orthogonal_init(nn.Linear(gcn_hidden_dim, head_hidden_dim), gain=nn.init.calculate_gain("relu"))
        self.fc2 = orthogonal_init(nn.Linear(head_hidden_dim, 1), gain=1.0)

    def forward(
        self,
        graph: GraphBatch | Any | torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, edge_index, batch, adjacency = self._extract_inputs(
            graph=graph,
            edge_index=edge_index,
            batch=batch,
            adjacency=adjacency,
        )

        if self.use_pyg and edge_index is not None and x.dim() == 2:
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            hidden = torch.relu(self.conv1(x, edge_index))
            hidden = torch.relu(self.conv2(hidden, edge_index))
            hidden = torch.relu(self.conv3(hidden, edge_index))
            pooled = self._pyg_global_mean_pool(hidden, batch)
            return self.fc2(torch.relu(self.fc1(pooled)))

        if batch is not None and x.dim() == 2 and edge_index is not None and adjacency is None:
            return self._forward_batched_fallback(x=x, edge_index=edge_index, batch=batch)

        if adjacency is None:
            if edge_index is None:
                raise ValueError("Either edge_index or adjacency must be provided to PGCNCritic.")
            adjacency = self._dense_adjacency_from_edge_index(
                edge_index=edge_index,
                num_nodes=x.shape[-2],
                device=x.device,
                dtype=x.dtype,
            )

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        return self._forward_dense(x=x, adjacency=adjacency)

    def _extract_inputs(
        self,
        graph: GraphBatch | Any | torch.Tensor,
        edge_index: torch.Tensor | None,
        batch: torch.Tensor | None,
        adjacency: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(graph, GraphBatch):
            x = graph.x
            edge_index = graph.edge_index
            adjacency = graph.adjacency
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            return x, edge_index, batch, adjacency

        if hasattr(graph, "x") and hasattr(graph, "edge_index"):
            x = graph.x
            edge_index = graph.edge_index
            batch = getattr(graph, "batch", batch)
            adjacency = getattr(graph, "adjacency", adjacency)
            return x, edge_index, batch, adjacency

        if not isinstance(graph, torch.Tensor):
            raise TypeError("graph must be a GraphBatch, a PyG Data-like object, or a torch.Tensor.")

        return graph, edge_index, batch, adjacency

    def _forward_batched_fallback(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for graph_idx in range(int(batch.max().item()) + 1):
            node_ids = (batch == graph_idx).nonzero(as_tuple=False).flatten()
            sub_x = x[node_ids]
            mapping = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            mapping[node_ids] = torch.arange(node_ids.numel(), device=x.device)
            edge_mask = (mapping[edge_index[0]] >= 0) & (mapping[edge_index[1]] >= 0)
            local_edge_index = mapping[edge_index[:, edge_mask]]
            adjacency = self._dense_adjacency_from_edge_index(
                edge_index=local_edge_index,
                num_nodes=node_ids.numel(),
                device=x.device,
                dtype=x.dtype,
            )
            outputs.append(self._forward_dense(x=sub_x.unsqueeze(0), adjacency=adjacency.unsqueeze(0)).squeeze(0))
        return torch.stack(outputs, dim=0)

    def _forward_dense(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(x, adjacency))
        hidden = torch.relu(self.conv2(hidden, adjacency))
        hidden = torch.relu(self.conv3(hidden, adjacency))
        pooled = hidden.mean(dim=1)
        return self.fc2(torch.relu(self.fc1(pooled)))

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
