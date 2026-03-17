from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(slots=True)
class GraphBatch:
    adjacency: torch.Tensor
    edge_index: torch.Tensor
    pairwise_distances: torch.Tensor | None = None
    server_index: int = -1
    num_devices: int = 0

    @property
    def num_nodes(self) -> int:
        return self.num_devices + 1

    def to_pyg_data(self) -> object:
        try:
            from torch_geometric.data import Data
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch-geometric is required to convert GraphBatch to a PyG Data object."
            ) from exc

        return Data(
            edge_index=self.edge_index,
            adjacency=self.adjacency,
            pairwise_distances=self.pairwise_distances,
            server_index=self.server_index,
            num_devices=self.num_devices,
            num_nodes=self.num_nodes,
        )


class GraphBuilder:
    def __init__(self, num_devices: int, graph_type: str = "star", distance_threshold_m: float = 150.0) -> None:
        if num_devices <= 0:
            raise ValueError("num_devices must be positive")
        if graph_type not in {"star", "star_proximity"}:
            raise ValueError(f"Unsupported graph_type: {graph_type}")
        self.num_devices = num_devices
        self.graph_type = graph_type
        self.distance_threshold_m = distance_threshold_m
        self.server_index = num_devices
        self._star_edge_index = self._build_star_edge_index()
        self._star_adjacency = self._build_star_adjacency()

    def build(
        self,
        positions: torch.Tensor | None = None,
    ) -> GraphBatch:
        topology_device = self._star_adjacency.device if positions is None else positions.device
        adjacency = self._star_adjacency.to(device=topology_device).clone()
        edge_index = self._star_edge_index.to(device=topology_device).clone()
        pairwise_distances = None

        if self.graph_type == "star_proximity":
            pairwise_distances = self._pairwise_distances(positions=positions)
            proximity_edge_index = self._build_proximity_edge_index(pairwise_distances, device=adjacency.device)
            if proximity_edge_index.numel() > 0:
                edge_index = torch.cat([edge_index, proximity_edge_index], dim=1)
                adjacency = self._augment_adjacency_with_proximity(
                    adjacency=adjacency,
                    pairwise_distances=pairwise_distances,
                )

        return GraphBatch(
            adjacency=adjacency,
            edge_index=edge_index,
            pairwise_distances=pairwise_distances,
            server_index=self.server_index,
            num_devices=self.num_devices,
        )

    def _build_star_edge_index(self) -> torch.Tensor:
        src = list(range(self.num_devices)) + [self.server_index] * self.num_devices
        dst = [self.server_index] * self.num_devices + list(range(self.num_devices))
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_star_adjacency(self) -> torch.Tensor:
        adjacency = torch.eye(self.num_devices + 1, dtype=torch.float32)
        adjacency[: self.num_devices, self.server_index] = 1.0
        adjacency[self.server_index, : self.num_devices] = 1.0
        return adjacency

    def _pairwise_distances(self, positions: torch.Tensor | None) -> torch.Tensor:
        if positions is None:
            raise ValueError("positions must be provided when graph_type='star_proximity'")
        if positions.dim() != 2 or positions.shape != (self.num_devices, 2):
            raise ValueError(
                f"positions must have shape ({self.num_devices}, 2), got {tuple(positions.shape)}"
            )
        positions = positions.to(dtype=torch.float32)
        return torch.cdist(positions, positions)

    def _build_proximity_edge_index(self, pairwise_distances: torch.Tensor, device: torch.device) -> torch.Tensor:
        threshold_mask = (pairwise_distances < self.distance_threshold_m) & (~torch.eye(self.num_devices, dtype=torch.bool, device=device))
        proximity_edges = threshold_mask.nonzero(as_tuple=False).t().contiguous()
        if proximity_edges.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return proximity_edges.to(dtype=torch.long)

    def _augment_adjacency_with_proximity(self, adjacency: torch.Tensor, pairwise_distances: torch.Tensor) -> torch.Tensor:
        threshold_mask = pairwise_distances < self.distance_threshold_m
        adjacency[: self.num_devices, : self.num_devices] = torch.maximum(
            adjacency[: self.num_devices, : self.num_devices],
            threshold_mask.to(dtype=adjacency.dtype),
        )
        adjacency[: self.num_devices, : self.num_devices].fill_diagonal_(1.0)
        return adjacency


def to_pyg_batch(graphs: Sequence[GraphBatch]) -> object:
    try:
        from torch_geometric.data import Batch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch-geometric is required to convert GraphBatch sequences to a PyG Batch."
        ) from exc

    return Batch.from_data_list([graph.to_pyg_data() for graph in graphs])
