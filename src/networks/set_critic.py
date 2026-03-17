from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class SetCritic(nn.Module):
    def __init__(
        self,
        device_dim: int = 14,
        server_dim: int = 3,
        hidden_dim: int = 64,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.device_dim = device_dim
        self.server_dim = server_dim

        relu_gain = nn.init.calculate_gain("relu")
        self.device_encoder = orthogonal_init(nn.Linear(device_dim, hidden_dim), gain=relu_gain)
        self.server_encoder = orthogonal_init(nn.Linear(server_dim, hidden_dim), gain=relu_gain)
        self.fc1 = orthogonal_init(nn.Linear(hidden_dim * 2, head_hidden_dim), gain=relu_gain)
        self.fc2 = orthogonal_init(nn.Linear(head_hidden_dim, 1), gain=1.0)

    def forward(self, device_obs: torch.Tensor, server_obs: torch.Tensor) -> torch.Tensor:
        if device_obs.dim() == 2:
            device_obs = device_obs.unsqueeze(0)
        if server_obs.dim() == 1:
            server_obs = server_obs.unsqueeze(0)
        if device_obs.shape[-1] != self.device_dim:
            raise ValueError(f"device_obs last dimension must be {self.device_dim}, got {device_obs.shape[-1]}")
        if server_obs.shape[-1] != self.server_dim:
            raise ValueError(f"server_obs last dimension must be {self.server_dim}, got {server_obs.shape[-1]}")

        device_emb = torch.relu(self.device_encoder(device_obs))
        pooled_devices = device_emb.mean(dim=1)
        server_emb = torch.relu(self.server_encoder(server_obs))
        critic_input = torch.cat([server_emb, pooled_devices], dim=-1)
        return self.fc2(torch.relu(self.fc1(critic_input)))
