from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class RoleConditionedActor(nn.Module):
    def __init__(self, obs_dim: int = 14, role_dim: int = 3, action_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = orthogonal_init(nn.Linear(obs_dim + role_dim, hidden_dim))
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.tanh = nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor, role_mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if role_mu.dim() == 1:
            role_mu = role_mu.unsqueeze(0)
        hidden = self.tanh(self.fc1(torch.cat([obs, role_mu], dim=-1)))
        hidden = self.tanh(self.fc2(hidden))
        mean = self.tanh(self.fc3(hidden)) * 5.0 + 5.0
        std = self.log_std.exp().expand_as(mean)
        return mean, std
