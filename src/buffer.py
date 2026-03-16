from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.utils import compute_gae


@dataclass(slots=True)
class Transition:
    device_obs: np.ndarray
    server_obs: np.ndarray
    role_mu: np.ndarray
    action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    done: bool
    positions: np.ndarray | None = None
    joint_reward: float | None = None
    scaled_joint_reward: float | None = None
    value: float | None = None


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def extend(self, transitions: list[Transition]) -> None:
        self.transitions.extend(transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def mean_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([transition.reward.mean() for transition in self.transitions]))

    def mean_joint_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([self._joint_reward(transition) for transition in self.transitions]))

    def mean_scaled_joint_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([self._scaled_joint_reward(transition) for transition in self.transitions]))

    def build_agent_trajectory_batch(
        self,
        window_size: int,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if action_scale <= 0.0:
            raise ValueError("action_scale must be positive.")
        if not self.transitions:
            raise ValueError("RolloutBuffer is empty.")

        num_steps = len(self.transitions)
        num_agents = self.transitions[0].device_obs.shape[0]
        traj_feature_dim = obs_dim + action_dim

        trajectories = np.zeros((num_steps * num_agents, window_size, traj_feature_dim), dtype=np.float32)
        current_obs = np.zeros((num_steps * num_agents, obs_dim), dtype=np.float32)
        role_mu = np.zeros((num_steps * num_agents, self.transitions[0].role_mu.shape[-1]), dtype=np.float32)
        actions = np.zeros((num_steps * num_agents, action_dim), dtype=np.float32)

        for step_idx, transition in enumerate(self.transitions):
            for agent_idx in range(num_agents):
                sample_idx = step_idx * num_agents + agent_idx
                current_obs[sample_idx] = transition.device_obs[agent_idx]
                role_mu[sample_idx] = transition.role_mu[agent_idx]
                actions[sample_idx] = transition.action[agent_idx]

                history_start = max(0, step_idx - window_size)
                history = []
                for hist_idx in range(history_start, step_idx):
                    hist_transition = self.transitions[hist_idx]
                    history.append(
                        np.concatenate(
                            [
                                hist_transition.device_obs[agent_idx],
                                hist_transition.action[agent_idx] / action_scale,
                            ],
                            axis=-1,
                        )
                    )
                if history:
                    history_array = np.asarray(history, dtype=np.float32)
                    trajectories[sample_idx, -len(history_array):] = history_array

        return {
            "trajectory": torch.as_tensor(trajectories, dtype=torch.float32, device=device),
            "current_obs": torch.as_tensor(current_obs, dtype=torch.float32, device=device),
            "role_mu": torch.as_tensor(role_mu, dtype=torch.float32, device=device),
            "action": torch.as_tensor(actions, dtype=torch.float32, device=device),
        }

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
        device: torch.device | str = "cpu",
        normalize_advantages: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not self.transitions:
            raise ValueError("RolloutBuffer is empty.")

        rewards = torch.as_tensor(
            [self._scaled_joint_reward(transition) for transition in self.transitions],
            dtype=torch.float32,
            device=device,
        )
        values = torch.as_tensor(
            [self._value(transition) for transition in self.transitions],
            dtype=torch.float32,
            device=device,
        )
        dones = torch.as_tensor(
            [float(transition.done) for transition in self.transitions],
            dtype=torch.float32,
            device=device,
        )
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            last_value=last_value,
            normalize_advantages=normalize_advantages,
        )
        return {
            "joint_reward": rewards,
            "value": values,
            "advantage": advantages,
            "return": returns,
        }

    def build_step_batch(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        tensors = self.as_tensors(device=device)
        num_steps, num_agents = tensors["device_obs"].shape[:2]
        step_index = torch.arange(num_steps, device=device, dtype=torch.long)
        agent_index = torch.arange(num_agents, device=device, dtype=torch.long)
        flat_step_index = step_index.unsqueeze(1).expand(num_steps, num_agents).reshape(-1)
        flat_agent_index = agent_index.unsqueeze(0).expand(num_steps, num_agents).reshape(-1)
        tensors["step_index"] = flat_step_index
        tensors["agent_index"] = flat_agent_index
        return tensors

    def as_tensors(self, device: torch.device | str = "cpu") -> dict[str, Any]:
        if not self.transitions:
            raise ValueError("RolloutBuffer is empty.")
        num_agents = self.transitions[0].device_obs.shape[0]
        position_shape = (num_agents, 2)
        stacked = {
            "device_obs": np.stack([item.device_obs for item in self.transitions]),
            "server_obs": np.stack([item.server_obs for item in self.transitions]),
            "positions": np.stack([self._positions(item, position_shape=position_shape) for item in self.transitions]),
            "role_mu": np.stack([item.role_mu for item in self.transitions]),
            "action": np.stack([item.action for item in self.transitions]),
            "log_prob": np.stack([item.log_prob for item in self.transitions]),
            "reward": np.stack([item.reward for item in self.transitions]),
            "joint_reward": np.asarray([self._joint_reward(item) for item in self.transitions], dtype=np.float32),
            "scaled_joint_reward": np.asarray(
                [self._scaled_joint_reward(item) for item in self.transitions],
                dtype=np.float32,
            ),
            "value": np.asarray([self._value(item) for item in self.transitions], dtype=np.float32),
            "done": np.asarray([item.done for item in self.transitions], dtype=np.float32),
        }
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in stacked.items()}

    @staticmethod
    def _joint_reward(transition: Transition) -> float:
        if transition.joint_reward is not None:
            return float(transition.joint_reward)
        return float(np.sum(transition.reward))

    @staticmethod
    def _value(transition: Transition) -> float:
        if transition.value is not None:
            return float(transition.value)
        return 0.0

    @staticmethod
    def _scaled_joint_reward(transition: Transition) -> float:
        if transition.scaled_joint_reward is not None:
            return float(transition.scaled_joint_reward)
        return RolloutBuffer._joint_reward(transition)

    @staticmethod
    def _positions(transition: Transition, position_shape: tuple[int, int]) -> np.ndarray:
        if transition.positions is None:
            return np.zeros(position_shape, dtype=np.float32)
        return np.asarray(transition.positions, dtype=np.float32)
