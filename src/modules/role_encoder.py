from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import orthogonal_init


class RoleEncoder(nn.Module):
    def __init__(self, obs_dim: int = 14, role_dim: int = 3, hidden_dim: int = 12) -> None:
        super().__init__()
        self.backbone = orthogonal_init(
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
        )
        self.mu_head = orthogonal_init(nn.Linear(hidden_dim, role_dim))
        self.log_std_head = orthogonal_init(nn.Linear(hidden_dim, role_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        hidden = self.backbone(obs)
        mu = self.mu_head(hidden)
        std = F.softplus(torch.clamp(self.log_std_head(hidden), -5.0, 2.0)) + 1e-4
        return mu, std
