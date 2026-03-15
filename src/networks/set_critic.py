from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class SetCritic(nn.Module):
    def __init__(self, obs_dim: int = 14, central_obs_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.element_encoder = orthogonal_init(
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )
        self.head = orthogonal_init(
            nn.Sequential(
                nn.Linear(hidden_dim + central_obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        )

    def forward(self, device_obs: torch.Tensor, server_obs: torch.Tensor) -> torch.Tensor:
        if device_obs.dim() == 2:
            device_obs = device_obs.unsqueeze(0)
        if server_obs.dim() == 1:
            server_obs = server_obs.unsqueeze(0)
        encoded = self.element_encoder(device_obs)
        pooled = encoded.mean(dim=1)
        return self.head(torch.cat([pooled, server_obs], dim=-1))
