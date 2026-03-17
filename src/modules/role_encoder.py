from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import orthogonal_init


class RoleEncoder(nn.Module):
    def __init__(self, obs_dim: int = 16, role_dim: int = 3, hidden_dim: int = 12) -> None:
        super().__init__()
        self.fc1 = orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=nn.init.calculate_gain("relu"))
        self.fc_mu = orthogonal_init(nn.Linear(hidden_dim, role_dim), gain=1.0)
        self.fc_sigma = orthogonal_init(nn.Linear(hidden_dim, role_dim), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        hidden = F.relu(self.fc1(obs))
        mu = self.fc_mu(hidden)
        sigma = torch.exp(self.fc_sigma(hidden).clamp(-5.0, 2.0))
        return mu, sigma
