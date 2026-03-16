from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F

from src.train import PPOTrainer
from src.utils import orthogonal_init


@dataclass(slots=True)
class DeterministicContextBaselineSpec:
    name: str = "Deterministic Context Encoder"
    description: str = "Deterministic MLP encoder baseline without Gaussian posterior or L_I."


class DeterministicContextEncoder(nn.Module):
    def __init__(self, obs_dim: int = 14, context_dim: int = 3, hidden_dim: int = 12) -> None:
        super().__init__()
        self.fc1 = orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=nn.init.calculate_gain("relu"))
        self.fc_context = orthogonal_init(nn.Linear(hidden_dim, context_dim), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.fc_context(F.relu(self.fc1(obs)))


class DeterministicContextTrainer(PPOTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        stale_role_encoder = self.role_encoder
        self.context_encoder = DeterministicContextEncoder(
            obs_dim=config.environment.observation_dim,
            context_dim=config.model.role_dim,
            hidden_dim=config.model.role_hidden_dim,
        ).to(self.device)
        self.role_encoder = self.context_encoder
        self.trajectory_encoder = None
        if stale_role_encoder is not None:
            stale_role_encoder.cpu()
            del stale_role_encoder
        self.actor_modules = [self.actor, self.context_encoder]
        actor_parameters = list(chain.from_iterable(module.parameters() for module in self.actor_modules))
        self.actor_optimizer = self.torch.optim.Adam(actor_parameters, lr=config.training.learning_rate)

    def _actor_role_posterior(self, device_obs):
        flat_obs = device_obs.reshape(-1, device_obs.shape[-1])
        context = self.context_encoder(flat_obs)
        context_shape = (*device_obs.shape[:-1], self.config.model.role_dim)
        return context.reshape(context_shape), None
