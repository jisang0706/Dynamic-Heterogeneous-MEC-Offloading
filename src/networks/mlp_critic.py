from __future__ import annotations

import torch
from torch import nn

from src.utils import orthogonal_init


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int = 14, num_agents: int = 5, central_obs_dim: int = 3, hidden_dim: int = 200) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.central_obs_dim = central_obs_dim
        self.input_dim = central_obs_dim + num_agents * obs_dim

        self.fc1 = orthogonal_init(nn.Linear(self.input_dim, hidden_dim), gain=nn.init.calculate_gain("tanh"))
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain("tanh"))
        self.value_fc1 = orthogonal_init(nn.Linear(hidden_dim + obs_dim, hidden_dim), gain=nn.init.calculate_gain("tanh"))
        self.value_fc2 = orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)
        self.activation = nn.Tanh()

    def build_input(self, device_obs: torch.Tensor, server_obs: torch.Tensor) -> torch.Tensor:
        if device_obs.dim() == 2:
            device_obs = device_obs.unsqueeze(0)
        if server_obs.dim() == 1:
            server_obs = server_obs.unsqueeze(0)
        if device_obs.shape[1] != self.num_agents or device_obs.shape[2] != self.obs_dim:
            raise ValueError(
                f"device_obs must have shape [batch, {self.num_agents}, {self.obs_dim}], got {tuple(device_obs.shape)}"
            )
        if server_obs.shape[1] != self.central_obs_dim:
            raise ValueError(
                f"server_obs must have shape [batch, {self.central_obs_dim}], got {tuple(server_obs.shape)}"
            )
        flat = device_obs.flatten(start_dim=1)
        return torch.cat([server_obs, flat], dim=-1)

    def forward(self, device_obs: torch.Tensor, server_obs: torch.Tensor | None = None) -> torch.Tensor:
        if server_obs is None:
            raise ValueError("server_obs must be provided for structured per-agent value prediction.")
        inputs = self.build_input(device_obs=device_obs, server_obs=server_obs)
        hidden = self.activation(self.fc1(inputs))
        hidden = self.activation(self.fc2(hidden))
        if device_obs.dim() == 2:
            device_obs = device_obs.unsqueeze(0)
        shared_context = hidden.unsqueeze(1).expand(-1, self.num_agents, -1)
        per_agent_input = torch.cat([shared_context, device_obs], dim=-1)
        value_hidden = self.activation(self.value_fc1(per_agent_input))
        return self.value_fc2(value_hidden).squeeze(-1)
