from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import orthogonal_init


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int = 14,
        action_dim: int = 4,
        role_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(obs_dim + action_dim, hidden_dim, batch_first=True)
        self.current_obs_proj = orthogonal_init(nn.Linear(obs_dim, hidden_dim))
        self.fusion = orthogonal_init(nn.Linear(hidden_dim * 2, hidden_dim))
        self.mu_head = orthogonal_init(nn.Linear(hidden_dim, role_dim))
        self.log_std_head = orthogonal_init(nn.Linear(hidden_dim, role_dim))

    def forward(self, trajectory: torch.Tensor, current_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        if current_obs.dim() == 1:
            current_obs = current_obs.unsqueeze(0)
        _, hidden = self.gru(trajectory)
        current = torch.tanh(self.current_obs_proj(current_obs))
        fused = torch.tanh(self.fusion(torch.cat([hidden[-1], current], dim=-1)))
        mu = self.mu_head(fused)
        std = F.softplus(torch.clamp(self.log_std_head(fused), -5.0, 2.0)) + 1e-4
        return mu, std
