from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class DeterministicContextEncoder(nn.Module):
    def __init__(self, obs_dim: int = 14, context_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = orthogonal_init(
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, context_dim),
            )
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
