from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from src.utils import orthogonal_init


class RoleConditionedActor(nn.Module):
    def __init__(
        self,
        obs_dim: int = 14,
        role_dim: int = 3,
        action_dim: int = 4,
        hidden_dim: int = 128,
        use_role: bool = True,
    ) -> None:
        super().__init__()
        self.use_role = use_role
        self.action_dim = action_dim
        self.role_dim = role_dim
        tanh_gain = nn.init.calculate_gain("tanh")
        input_dim = obs_dim + role_dim if use_role else obs_dim
        self.fc1 = orthogonal_init(nn.Linear(input_dim, hidden_dim), gain=tanh_gain)
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=tanh_gain)
        self.fc3 = orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.tanh = nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _prepare_inputs(self, obs: torch.Tensor, role_mu: torch.Tensor | None) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if not self.use_role:
            return obs
        if role_mu is None:
            raise ValueError("role_mu must be provided when use_role=True.")
        if role_mu.dim() == 1:
            role_mu = role_mu.unsqueeze(0)
        return torch.cat([obs, role_mu], dim=-1)

    def forward(self, obs: torch.Tensor, role_mu: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.tanh(self.fc1(self._prepare_inputs(obs, role_mu)))
        hidden = self.tanh(self.fc2(hidden))
        mean = self.tanh(self.fc3(hidden)) * 5.0 + 5.0
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def distribution(self, obs: torch.Tensor, role_mu: torch.Tensor | None = None) -> Normal:
        mean, std = self.forward(obs, role_mu)
        return Normal(mean, std)

    def action_to_env(self, action: torch.Tensor) -> torch.Tensor:
        return action / 10.0

    def sample_action(
        self,
        obs: torch.Tensor,
        role_mu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs, role_mu)
        action = torch.clamp(distribution.sample(), 0.0, 10.0)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        env_action = self.action_to_env(action)
        return action, env_action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        role_mu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs, role_mu)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        mean, std = distribution.mean, distribution.stddev
        return log_prob, entropy, mean, std
