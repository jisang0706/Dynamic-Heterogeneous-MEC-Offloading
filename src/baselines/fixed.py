from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment import DynamicMECEnv
from src.baselines.qag import queue_aware_greedy_actions


def local_only_actions(num_agents: int, num_tasks: int = 3) -> np.ndarray:
    return np.zeros((num_agents, num_tasks + 1), dtype=np.float32)


def edge_only_actions(num_agents: int, num_tasks: int = 3) -> np.ndarray:
    return np.ones((num_agents, num_tasks + 1), dtype=np.float32)


def random_actions(num_agents: int, num_tasks: int = 3, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(num_agents, num_tasks + 1)).astype(np.float32)


@dataclass(slots=True)
class FixedPolicySummary:
    policy_name: str
    episodes: int
    mean_episode_joint_reward: float
    mean_step_joint_reward: float
    mean_step_device_reward: float


def run_fixed_policy_baseline(config, policy_name: str, num_episodes: int | None = None) -> FixedPolicySummary:
    normalized_name = policy_name.upper()
    if normalized_name not in {"LOCAL_ONLY", "EDGE_ONLY", "RANDOM", "QAG"}:
        raise ValueError(f"Unsupported fixed baseline: {policy_name}")

    episode_count = config.training.total_episodes if num_episodes is None else num_episodes
    env = DynamicMECEnv(config.environment, seed=config.seed)
    rng_seed = config.seed
    episode_joint_rewards: list[float] = []
    step_joint_rewards: list[float] = []
    step_device_rewards: list[float] = []

    for episode_idx in range(episode_count):
        env.reset()
        episode_reward = 0.0
        random_rng = None if normalized_name != "RANDOM" else np.random.default_rng(rng_seed + episode_idx)
        for _ in range(config.environment.episode_length):
            if normalized_name == "LOCAL_ONLY":
                action = local_only_actions(config.environment.num_agents, config.environment.num_tasks_per_step)
            elif normalized_name == "EDGE_ONLY":
                action = edge_only_actions(config.environment.num_agents, config.environment.num_tasks_per_step)
            elif normalized_name == "QAG":
                action = queue_aware_greedy_actions(env)
            else:
                assert random_rng is not None
                action = random_rng.uniform(
                    0.0,
                    1.0,
                    size=(config.environment.num_agents, config.environment.num_tasks_per_step + 1),
                ).astype(np.float32)
            _, reward, done, _ = env.step(action)
            joint_reward = float(reward.sum())
            episode_reward += joint_reward
            step_joint_rewards.append(joint_reward)
            step_device_rewards.append(float(reward.mean()))
            if done:
                break
        episode_joint_rewards.append(episode_reward)

    return FixedPolicySummary(
        policy_name=normalized_name,
        episodes=episode_count,
        mean_episode_joint_reward=float(np.mean(episode_joint_rewards)) if episode_joint_rewards else 0.0,
        mean_step_joint_reward=float(np.mean(step_joint_rewards)) if step_joint_rewards else 0.0,
        mean_step_device_reward=float(np.mean(step_device_rewards)) if step_device_rewards else 0.0,
    )
