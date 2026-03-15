from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = orthogonal_init(nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        aggregated = adjacency @ x / degree
        return self.linear(aggregated)


class PGCNCritic(nn.Module):
    def __init__(self, node_dim: int = 14, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = SimpleGraphConv(node_dim, hidden_dim)
        self.conv2 = SimpleGraphConv(hidden_dim, hidden_dim)
        self.conv3 = SimpleGraphConv(hidden_dim, hidden_dim)
        self.head = orthogonal_init(
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        hidden = torch.relu(self.conv1(node_features, adjacency))
        hidden = torch.relu(self.conv2(hidden, adjacency))
        hidden = torch.relu(self.conv3(hidden, adjacency))
        pooled = hidden.mean(dim=1)
        return self.head(pooled)
