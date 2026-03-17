from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        role_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(obs_dim + action_dim, hidden_dim, batch_first=True)
        fusion_dim = hidden_dim + obs_dim
        self.fc_mu = orthogonal_init(nn.Linear(fusion_dim, role_dim), gain=1.0)
        self.fc_sigma = orthogonal_init(nn.Linear(fusion_dim, role_dim), gain=1.0)

    def forward(self, trajectory: torch.Tensor, current_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        if current_obs.dim() == 1:
            current_obs = current_obs.unsqueeze(0)
        _, hidden = self.gru(trajectory)
        fused = torch.cat([hidden[-1], current_obs], dim=-1)
        mu = self.fc_mu(fused)
        sigma = torch.exp(self.fc_sigma(fused).clamp(-5.0, 2.0))
        return mu, sigma
