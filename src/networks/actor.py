from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from src.utils import orthogonal_init


class RoleConditionedActor(nn.Module):
    def __init__(
        self,
        obs_dim: int = 16,
        role_dim: int = 3,
        action_dim: int = 4,
        hidden_dim: int = 128,
        use_role: bool = True,
        initial_action_std_env: float = 0.25,
        initial_offloading_mean_env: float = 0.65,
        initial_power_mean_env: float = 0.8,
    ) -> None:
        super().__init__()
        self.use_role = use_role
        self.action_dim = action_dim
        self.role_dim = role_dim
        if initial_action_std_env <= 0.0:
            raise ValueError("initial_action_std_env must be positive.")
        if not 0.0 < initial_offloading_mean_env < 1.0:
            raise ValueError("initial_offloading_mean_env must be strictly between 0 and 1.")
        if not 0.0 < initial_power_mean_env < 1.0:
            raise ValueError("initial_power_mean_env must be strictly between 0 and 1.")
        tanh_gain = nn.init.calculate_gain("tanh")
        input_dim = obs_dim + role_dim if use_role else obs_dim
        self.fc1 = orthogonal_init(nn.Linear(input_dim, hidden_dim), gain=tanh_gain)
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=tanh_gain)
        self.fc3 = orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.tanh = nn.Tanh()
        native_std = max(initial_action_std_env * 10.0, 1e-4)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(native_std), dtype=torch.float32))
        with torch.no_grad():
            self.fc3.bias.zero_()
            self.fc3.bias[:-1] = self._env_mean_to_native_bias(initial_offloading_mean_env)
            self.fc3.bias[-1] = self._env_mean_to_native_bias(initial_power_mean_env)

    @staticmethod
    def _env_mean_to_native_bias(env_mean: float) -> float:
        centered = max(min(env_mean * 2.0 - 1.0, 0.999), -0.999)
        return math.atanh(centered)

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


class MultiAgentRoleConditionedActor(nn.Module):
    def __init__(
        self,
        num_agents: int,
        actor_type: str = "shared",
        obs_dim: int = 16,
        role_dim: int = 3,
        action_dim: int = 4,
        hidden_dim: int = 128,
        use_role: bool = True,
        initial_action_std_env: float = 0.25,
        initial_offloading_mean_env: float = 0.65,
        initial_power_mean_env: float = 0.8,
    ) -> None:
        super().__init__()
        if actor_type not in {"shared", "individual"}:
            raise ValueError(f"Unsupported actor_type: {actor_type}")
        self.num_agents = num_agents
        self.actor_type = actor_type
        self.use_role = use_role
        self.obs_dim = obs_dim
        self.role_dim = role_dim
        self.action_dim = action_dim
        self.shared_obs_dim = obs_dim + num_agents if actor_type == "shared" else obs_dim

        if actor_type == "shared":
            self.shared_actor = RoleConditionedActor(
                obs_dim=self.shared_obs_dim,
                role_dim=role_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                use_role=use_role,
                initial_action_std_env=initial_action_std_env,
                initial_offloading_mean_env=initial_offloading_mean_env,
                initial_power_mean_env=initial_power_mean_env,
            )
            self.actors = None
        else:
            self.shared_actor = None
            self.actors = nn.ModuleList(
                [
                    RoleConditionedActor(
                        obs_dim=obs_dim,
                        role_dim=role_dim,
                        action_dim=action_dim,
                        hidden_dim=hidden_dim,
                        use_role=use_role,
                        initial_action_std_env=initial_action_std_env,
                        initial_offloading_mean_env=initial_offloading_mean_env,
                        initial_power_mean_env=initial_power_mean_env,
                    )
                    for _ in range(num_agents)
                ]
            )

    def _append_agent_identity(self, obs: torch.Tensor) -> torch.Tensor:
        if self.actor_type != "shared":
            return obs
        if obs.dim() == 2:
            agent_ids = torch.eye(self.num_agents, dtype=obs.dtype, device=obs.device)
            return torch.cat([obs, agent_ids], dim=-1)
        if obs.dim() == 3:
            batch_size = obs.shape[0]
            agent_ids = torch.eye(self.num_agents, dtype=obs.dtype, device=obs.device)
            agent_ids = agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([obs, agent_ids], dim=-1)
        raise ValueError(f"Unsupported observation rank for shared actor: {obs.dim()}")

    def forward(self, obs: torch.Tensor, role_mu: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.actor_type == "shared":
            assert self.shared_actor is not None
            return self.shared_actor(self._append_agent_identity(obs), role_mu)

        squeeze_batch = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
            squeeze_batch = True
        if self.use_role and role_mu is None:
            raise ValueError("role_mu must be provided when use_role=True.")
        if self.use_role and role_mu.dim() == 2:
            role_mu = role_mu.unsqueeze(0)
        if obs.shape[1] != self.num_agents:
            raise ValueError(f"obs second dimension must match num_agents={self.num_agents}")

        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        assert self.actors is not None
        for agent_idx, actor in enumerate(self.actors):
            agent_role = None if role_mu is None else role_mu[:, agent_idx]
            mean, std = actor(obs[:, agent_idx], agent_role)
            means.append(mean)
            stds.append(std)
        stacked_mean = torch.stack(means, dim=1)
        stacked_std = torch.stack(stds, dim=1)
        if squeeze_batch:
            stacked_mean = stacked_mean.squeeze(0)
            stacked_std = stacked_std.squeeze(0)
        return stacked_mean, stacked_std

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
        return log_prob, entropy, distribution.mean, distribution.stddev
