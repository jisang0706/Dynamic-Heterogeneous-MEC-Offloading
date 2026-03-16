from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class Transition:
    device_obs: np.ndarray
    server_obs: np.ndarray
    role_mu: np.ndarray
    action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    done: bool


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def mean_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([transition.reward.mean() for transition in self.transitions]))

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

    def as_tensors(self, device: torch.device | str = "cpu") -> dict[str, Any]:
        if not self.transitions:
            raise ValueError("RolloutBuffer is empty.")
        stacked = {
            "device_obs": np.stack([item.device_obs for item in self.transitions]),
            "server_obs": np.stack([item.server_obs for item in self.transitions]),
            "role_mu": np.stack([item.role_mu for item in self.transitions]),
            "action": np.stack([item.action for item in self.transitions]),
            "log_prob": np.stack([item.log_prob for item in self.transitions]),
            "reward": np.stack([item.reward for item in self.transitions]),
            "done": np.asarray([item.done for item in self.transitions], dtype=np.float32),
        }
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in stacked.items()}
