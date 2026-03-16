from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if idx < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
    return module


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == self.mean.shape:
            array = np.expand_dims(array, axis=0)
        if array.ndim == 0:
            array = array.reshape(1)
        batch_mean = array.mean(axis=0)
        batch_var = array.var(axis=0)
        batch_count = array.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


class ObservationScaler:
    def __init__(self, shape: tuple[int, ...], clip_range: float = 5.0, epsilon: float = 1e-8) -> None:
        self.rms = RunningMeanStd(shape=shape)
        self.clip_range = clip_range
        self.epsilon = epsilon

    def update(self, values: np.ndarray) -> None:
        self.rms.update(values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        normalized = (array - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(normalized, -self.clip_range, self.clip_range).astype(np.float32)

    def update_and_transform(self, values: np.ndarray) -> np.ndarray:
        self.update(values)
        return self.transform(values)

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "rms": self.rms.state_dict(),
            "clip_range": self.clip_range,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.clip_range = float(state["clip_range"])
        self.epsilon = float(state["epsilon"])
        self.rms.load_state_dict(state["rms"])


class RewardScaler:
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())
        self.running_return = 0.0

    def reset(self) -> None:
        self.running_return = 0.0

    def scale(self, reward: float) -> float:
        reward_value = float(reward)
        self.running_return = self.running_return * self.gamma + reward_value
        self.return_rms.update(np.asarray([self.running_return], dtype=np.float32))
        scaled_reward = reward_value / float(np.sqrt(self.return_rms.var + self.epsilon))
        return float(scaled_reward)

    def state_dict(self) -> dict[str, float | dict[str, np.ndarray | float]]:
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "running_return": self.running_return,
            "return_rms": self.return_rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, float | dict[str, np.ndarray | float]]) -> None:
        self.gamma = float(state["gamma"])
        self.epsilon = float(state["epsilon"])
        self.running_return = float(state["running_return"])
        self.return_rms.load_state_dict(state["return_rms"])


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: float | torch.Tensor = 0.0,
    normalize_advantages: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.dim() != 1 or values.dim() != 1 or dones.dim() != 1:
        raise ValueError("rewards, values, and dones must be 1D tensors.")
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must share the same shape.")

    advantages = torch.zeros_like(rewards)
    last_advantage = rewards.new_tensor(0.0)
    next_value = rewards.new_tensor(last_value)

    for step_idx in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[step_idx]
        delta = rewards[step_idx] + gamma * next_value * non_terminal - values[step_idx]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[step_idx] = last_advantage
        next_value = values[step_idx]

    returns = advantages + values
    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    return advantages, returns
