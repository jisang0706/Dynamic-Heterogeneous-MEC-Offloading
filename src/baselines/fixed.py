from __future__ import annotations

import numpy as np


def local_only_actions(num_agents: int, num_tasks: int = 3) -> np.ndarray:
    actions = np.zeros((num_agents, num_tasks + 1), dtype=np.float32)
    return actions


def edge_only_actions(num_agents: int, num_tasks: int = 3) -> np.ndarray:
    actions = np.ones((num_agents, num_tasks + 1), dtype=np.float32)
    return actions


def random_actions(num_agents: int, num_tasks: int = 3, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(num_agents, num_tasks + 1)).astype(np.float32)
