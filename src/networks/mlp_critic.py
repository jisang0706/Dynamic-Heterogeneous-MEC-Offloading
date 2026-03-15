from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int = 14, num_agents: int = 5, central_obs_dim: int = 3, hidden_dim: int = 200) -> None:
        super().__init__()
        input_dim = central_obs_dim + num_agents * obs_dim
        self.network = orthogonal_init(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        )

    def forward(self, device_obs: torch.Tensor, server_obs: torch.Tensor) -> torch.Tensor:
        if device_obs.dim() == 2:
            device_obs = device_obs.unsqueeze(0)
        if server_obs.dim() == 1:
            server_obs = server_obs.unsqueeze(0)
        flat = device_obs.flatten(start_dim=1)
        return self.network(torch.cat([server_obs, flat], dim=-1))
