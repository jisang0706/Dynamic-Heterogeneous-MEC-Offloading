from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.utils import compute_gae


@dataclass(slots=True)
class Transition:
    actor_obs: np.ndarray
    core_obs: np.ndarray
    server_info: np.ndarray
    role_mu: np.ndarray
    action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    done: bool
    scaled_reward: np.ndarray | float | None = None
    positions: np.ndarray | None = None
    joint_reward: float | None = None
    scaled_joint_reward: float | None = None
    value: np.ndarray | float | None = None
    timeout_ratio: float | None = None
    taskwise_delay_gap: np.ndarray | None = None
    shared_congestion_price: float | None = None

    @property
    def device_obs(self) -> np.ndarray:
        return self.core_obs

    @property
    def server_obs(self) -> np.ndarray:
        return self.server_info


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
        return float(
            np.mean(
                [
                    float(np.asarray(self._scaled_reward(transition), dtype=np.float32).sum())
                    for transition in self.transitions
                ]
            )
        )

    def mean_shared_congestion_price(self) -> float:
        if not self.transitions:
            return 0.0
        prices = [
            float(transition.shared_congestion_price)
            for transition in self.transitions
            if transition.shared_congestion_price is not None
        ]
        if not prices:
            return 0.0
        return float(np.mean(prices))

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
        num_agents = self.transitions[0].core_obs.shape[0]
        traj_feature_dim = obs_dim + action_dim

        trajectories = np.zeros((num_steps * num_agents, window_size, traj_feature_dim), dtype=np.float32)
        current_obs = np.zeros((num_steps * num_agents, obs_dim), dtype=np.float32)
        role_mu = np.zeros((num_steps * num_agents, self.transitions[0].role_mu.shape[-1]), dtype=np.float32)
        actions = np.zeros((num_steps * num_agents, action_dim), dtype=np.float32)

        for step_idx, transition in enumerate(self.transitions):
            for agent_idx in range(num_agents):
                sample_idx = step_idx * num_agents + agent_idx
                current_obs[sample_idx] = transition.actor_obs[agent_idx]
                role_mu[sample_idx] = transition.role_mu[agent_idx]
                actions[sample_idx] = transition.action[agent_idx]

                history_start = max(0, step_idx - window_size)
                history = []
                for hist_idx in range(history_start, step_idx):
                    hist_transition = self.transitions[hist_idx]
                    history.append(
                        np.concatenate(
                            [
                                hist_transition.actor_obs[agent_idx],
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
            np.stack([self._scaled_reward(transition) for transition in self.transitions]),
            dtype=torch.float32,
            device=device,
        )
        values = torch.as_tensor(
            np.stack([self._value(transition) for transition in self.transitions]),
            dtype=torch.float32,
            device=device,
        )
        dones = torch.as_tensor(
            [float(transition.done) for transition in self.transitions],
            dtype=torch.float32,
            device=device,
        )
        if rewards.dim() == 1:
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
                last_value=last_value,
                normalize_advantages=normalize_advantages,
            )
        else:
            if values.shape != rewards.shape:
                raise ValueError(f"values shape {tuple(values.shape)} must match rewards shape {tuple(rewards.shape)}")
            if isinstance(last_value, torch.Tensor):
                last_value_tensor = last_value.to(device=device, dtype=torch.float32)
            else:
                last_value_tensor = torch.as_tensor(last_value, dtype=torch.float32, device=device)
            if last_value_tensor.dim() == 0:
                last_value_tensor = last_value_tensor.expand(rewards.shape[1])
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            for agent_idx in range(rewards.shape[1]):
                agent_advantage, agent_return = compute_gae(
                    rewards=rewards[:, agent_idx],
                    values=values[:, agent_idx],
                    dones=dones,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    last_value=last_value_tensor[agent_idx],
                    normalize_advantages=normalize_advantages,
                )
                advantages[:, agent_idx] = agent_advantage
                returns[:, agent_idx] = agent_return
        return {
            "joint_reward": rewards,
            "value": values,
            "advantage": advantages,
            "return": returns,
        }

    def build_step_batch(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        tensors = self.as_tensors(device=device)
        num_steps, num_agents = tensors["core_obs"].shape[:2]
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
        num_agents = self.transitions[0].core_obs.shape[0]
        position_shape = (num_agents, 2)
        stacked = {
            "actor_obs": np.stack([item.actor_obs for item in self.transitions]),
            "core_obs": np.stack([item.core_obs for item in self.transitions]),
            "server_info": np.stack([item.server_info for item in self.transitions]),
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
            "value": np.stack([self._value(item) for item in self.transitions]).astype(np.float32),
            "timeout_ratio": np.asarray([self._timeout_ratio(item) for item in self.transitions], dtype=np.float32),
            "taskwise_delay_gap": np.stack([self._taskwise_delay_gap(item) for item in self.transitions]).astype(np.float32),
            "done": np.asarray([item.done for item in self.transitions], dtype=np.float32),
        }
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in stacked.items()}

    @staticmethod
    def _joint_reward(transition: Transition) -> float:
        if transition.joint_reward is not None:
            return float(transition.joint_reward)
        return float(np.sum(transition.reward))

    @staticmethod
    def _value(transition: Transition) -> np.ndarray:
        if transition.value is not None:
            value = np.asarray(transition.value, dtype=np.float32)
            reward = np.asarray(transition.reward, dtype=np.float32)
            if value.ndim == 0 and reward.ndim > 0:
                return np.full_like(reward, float(value), dtype=np.float32)
            return value
        return np.zeros_like(np.asarray(transition.reward, dtype=np.float32), dtype=np.float32)

    @staticmethod
    def _scaled_joint_reward(transition: Transition) -> float:
        if transition.scaled_joint_reward is not None:
            return float(transition.scaled_joint_reward)
        return RolloutBuffer._joint_reward(transition)

    @staticmethod
    def _scaled_reward(transition: Transition) -> np.ndarray:
        if transition.scaled_reward is not None:
            scaled_reward = np.asarray(transition.scaled_reward, dtype=np.float32)
            reward = np.asarray(transition.reward, dtype=np.float32)
            if scaled_reward.ndim == 0 and reward.ndim > 0:
                return np.full_like(
                    reward,
                    float(scaled_reward) / max(reward.shape[0], 1),
                    dtype=np.float32,
                )
            return scaled_reward
        reward = np.asarray(transition.reward, dtype=np.float32)
        if transition.scaled_joint_reward is not None and reward.ndim > 0:
            return np.full_like(
                reward,
                float(transition.scaled_joint_reward) / max(reward.shape[0], 1),
                dtype=np.float32,
            )
        return reward

    @staticmethod
    def _timeout_ratio(transition: Transition) -> float:
        if transition.timeout_ratio is not None:
            return float(transition.timeout_ratio)
        return 0.0

    @staticmethod
    def _positions(transition: Transition, position_shape: tuple[int, int]) -> np.ndarray:
        if transition.positions is None:
            return np.zeros(position_shape, dtype=np.float32)
        return np.asarray(transition.positions, dtype=np.float32)

    @staticmethod
    def _taskwise_delay_gap(transition: Transition) -> np.ndarray:
        if transition.taskwise_delay_gap is None:
            reward = np.asarray(transition.reward, dtype=np.float32)
            task_count = max(int(np.asarray(transition.action, dtype=np.float32).shape[-1]) - 1, 0)
            return np.zeros((reward.shape[0], task_count), dtype=np.float32)
        return np.asarray(transition.taskwise_delay_gap, dtype=np.float32)
