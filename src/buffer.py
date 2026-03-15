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
