from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import ExperimentConfig
from src.environment import DynamicMECEnv
from src.networks import MultiAgentRoleConditionedActor
from src.utils import ObservationScaler, RewardScaler, compute_gae, orthogonal_init, set_seed


@dataclass(slots=True)
class IPPOBaselineSpec:
    name: str = "IPPO"
    description: str = "Independent PPO baseline with local critics and no centralized information."


@dataclass(slots=True)
class IPPOTransition:
    device_obs: np.ndarray
    action: np.ndarray
    log_prob: np.ndarray
    reward: np.ndarray
    scaled_reward: np.ndarray
    value: np.ndarray
    done: bool


@dataclass(slots=True)
class IPPOSmokeSummary:
    steps: int
    mean_reward: float
    mean_joint_reward: float
    last_value: float
    critic_type: str
    last_l_i_loss: float | None = None


@dataclass(slots=True)
class IPPOUpdateSummary:
    steps: int
    mean_joint_reward: float
    mean_scaled_joint_reward: float
    actor_loss: float
    critic_loss: float
    entropy: float
    l_i_loss: float | None = None


@dataclass(slots=True)
class IPPOTrainingRunSummary:
    episodes: int
    updates: int
    mean_episode_joint_reward: float
    critic_type: str
    episode_log_path: str | None = None
    update_log_path: str | None = None
    last_checkpoint_path: str | None = None
    last_actor_loss: float | None = None
    last_critic_loss: float | None = None
    last_entropy: float | None = None
    last_l_i_loss: float | None = None


class IPPORolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[IPPOTransition] = []

    def add(self, transition: IPPOTransition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def mean_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([transition.reward.mean() for transition in self.transitions]))

    def mean_joint_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([transition.reward.sum() for transition in self.transitions]))

    def mean_scaled_joint_reward(self) -> float:
        if not self.transitions:
            return 0.0
        return float(np.mean([transition.scaled_reward.sum() for transition in self.transitions]))

    def as_tensors(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        if not self.transitions:
            raise ValueError("IPPORolloutBuffer is empty.")
        stacked = {
            "device_obs": np.stack([item.device_obs for item in self.transitions]),
            "action": np.stack([item.action for item in self.transitions]),
            "log_prob": np.stack([item.log_prob for item in self.transitions]),
            "reward": np.stack([item.reward for item in self.transitions]),
            "scaled_reward": np.stack([item.scaled_reward for item in self.transitions]),
            "value": np.stack([item.value for item in self.transitions]),
            "done": np.asarray([item.done for item in self.transitions], dtype=np.float32),
        }
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in stacked.items()}

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: torch.Tensor | np.ndarray,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        batch = self.as_tensors(device=device)
        num_agents = batch["reward"].shape[1]
        advantages = torch.zeros_like(batch["reward"])
        returns = torch.zeros_like(batch["reward"])
        for agent_idx in range(num_agents):
            agent_advantage, agent_return = compute_gae(
                rewards=batch["scaled_reward"][:, agent_idx],
                values=batch["value"][:, agent_idx],
                dones=batch["done"],
                gamma=gamma,
                gae_lambda=gae_lambda,
                last_value=last_value[agent_idx],
                normalize_advantages=True,
            )
            advantages[:, agent_idx] = agent_advantage
            returns[:, agent_idx] = agent_return
        return {
            "advantage": advantages,
            "return": returns,
        }


class IndependentValueCritic(nn.Module):
    def __init__(self, obs_dim: int = 14, hidden_dim: int = 200) -> None:
        super().__init__()
        tanh_gain = nn.init.calculate_gain("tanh")
        self.fc1 = orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=tanh_gain)
        self.fc2 = orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=tanh_gain)
        self.fc3 = orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)
        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        hidden = self.activation(self.fc1(obs))
        hidden = self.activation(self.fc2(hidden))
        return self.fc3(hidden).squeeze(-1)


class IPPOTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.seed)
        self.config.ensure_output_dirs()

        self.env = DynamicMECEnv(config.environment, seed=config.seed)
        self.actor = MultiAgentRoleConditionedActor(
            num_agents=config.environment.num_agents,
            actor_type="individual",
            obs_dim=config.environment.observation_dim,
            role_dim=config.model.role_dim,
            action_dim=config.model.action_dim,
            hidden_dim=config.model.actor_hidden_dim,
            use_role=False,
        ).to(self.device)
        critic_hidden_dim = max(config.model.critic_hidden_dim, 200)
        self.critics = nn.ModuleList(
            [IndependentValueCritic(obs_dim=config.environment.observation_dim, hidden_dim=critic_hidden_dim) for _ in range(config.environment.num_agents)]
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.training.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.training.learning_rate)

        self.device_obs_scaler = ObservationScaler(shape=(config.environment.observation_dim,)) if config.training.use_obs_scaling else None
        self.reward_scalers = (
            [RewardScaler(gamma=config.training.gamma) for _ in range(config.environment.num_agents)]
            if config.training.use_reward_scaling
            else None
        )
        self.episode_history: list[dict[str, float | int | None]] = []
        self.update_history: list[dict[str, float | int | None]] = []
        self.last_rollout_episode_lengths: list[int] = []
        self.episode_log_path = None
        self.update_log_path = None
        self.last_checkpoint_path = None
        if config.training.run_mode == "train":
            self.episode_log_path = config.output_root / "logs" / "episode_history_ippo.jsonl"
            self.update_log_path = config.output_root / "logs" / "update_history_ippo.jsonl"
            self.episode_log_path.write_text("", encoding="utf-8")
            self.update_log_path.write_text("", encoding="utf-8")

    def _scale_device_obs(self, device_obs: torch.Tensor, update_stats: bool) -> torch.Tensor:
        if self.device_obs_scaler is None:
            return device_obs
        obs_np = device_obs.detach().cpu().numpy()
        if update_stats:
            scaled = self.device_obs_scaler.update_and_transform(obs_np)
        else:
            scaled = self.device_obs_scaler.transform(obs_np)
        return torch.from_numpy(scaled).float().to(self.device)

    def _append_json_record(self, path: Path, payload: dict[str, float | int | None]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _save_checkpoint(self, episodes_completed: int, update_index: int, suffix: str | None = None) -> str:
        checkpoint_name = (
            f"ippo_checkpoint_{suffix}.pt"
            if suffix is not None
            else f"ippo_checkpoint_ep{episodes_completed:04d}_u{update_index:04d}.pt"
        )
        checkpoint_path = self.config.output_root / "models" / checkpoint_name
        torch.save(
            {
                "config": self.config.to_dict(),
                "episodes_completed": episodes_completed,
                "update_index": update_index,
                "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "device_obs_scaler": None if self.device_obs_scaler is None else self.device_obs_scaler.state_dict(),
                "reward_scalers": None if self.reward_scalers is None else [scaler.state_dict() for scaler in self.reward_scalers],
                "episode_history": self.episode_history,
                "update_history": self.update_history,
            },
            checkpoint_path,
        )
        self.last_checkpoint_path = str(checkpoint_path)
        return self.last_checkpoint_path

    def _critic_values(self, device_obs: torch.Tensor) -> torch.Tensor:
        values = [critic(device_obs[:, agent_idx]) if device_obs.dim() == 3 else critic(device_obs[agent_idx]) for agent_idx, critic in enumerate(self.critics)]
        return torch.stack(values, dim=-1)

    def collect_rollouts(self, num_episodes: int, max_steps: int | None = None):
        buffer = IPPORolloutBuffer()
        episode_joint_rewards: list[float] = []
        episode_lengths: list[int] = []
        steps_remaining = max_steps
        last_value = torch.zeros(self.config.environment.num_agents, dtype=torch.float32, device=self.device)
        budget_truncated = False

        for episode_idx in range(num_episodes):
            if steps_remaining is not None and steps_remaining <= 0:
                break
            observation = self.env.reset()
            episode_joint_reward = 0.0
            episode_length = 0
            if self.reward_scalers is not None:
                for scaler in self.reward_scalers:
                    scaler.reset()
            step_limit = self.config.environment.episode_length
            if steps_remaining is not None:
                step_limit = min(step_limit, steps_remaining)

            for _ in range(step_limit):
                device_obs = torch.from_numpy(observation.device_obs).float().to(self.device)
                device_obs = self._scale_device_obs(device_obs, update_stats=True)
                with torch.no_grad():
                    action, env_action, log_prob = self.actor.sample_action(device_obs)
                    value = self._critic_values(device_obs)

                next_observation, reward, done, _ = self.env.step(env_action.cpu().numpy())
                if self.reward_scalers is None:
                    scaled_reward = reward.astype(np.float32)
                else:
                    scaled_reward = np.asarray(
                        [self.reward_scalers[agent_idx].scale(float(reward[agent_idx])) for agent_idx in range(self.config.environment.num_agents)],
                        dtype=np.float32,
                    )
                episode_joint_reward += float(reward.sum())
                episode_length += 1
                buffer.add(
                    IPPOTransition(
                        device_obs=device_obs.cpu().numpy(),
                        action=action.cpu().numpy(),
                        log_prob=log_prob.cpu().numpy(),
                        reward=reward.copy(),
                        scaled_reward=scaled_reward,
                        value=value.squeeze(0).cpu().numpy() if value.dim() == 2 and value.shape[0] == 1 else value.cpu().numpy(),
                        done=done,
                    )
                )
                observation = next_observation
                if steps_remaining is not None:
                    steps_remaining -= 1
                    if steps_remaining == 0 and not done:
                        budget_truncated = True
                        break
                if done:
                    break

            if budget_truncated:
                next_device_obs = torch.from_numpy(observation.device_obs).float().to(self.device)
                next_device_obs = self._scale_device_obs(next_device_obs, update_stats=False)
                with torch.no_grad():
                    last_value = self._critic_values(next_device_obs).squeeze(0)
            else:
                last_value = torch.zeros(self.config.environment.num_agents, dtype=torch.float32, device=self.device)

            episode_joint_rewards.append(episode_joint_reward)
            episode_lengths.append(episode_length)
            if budget_truncated:
                break

        self.last_rollout_episode_lengths = episode_lengths
        return buffer, last_value, episode_joint_rewards

    def update(self, buffer: IPPORolloutBuffer, last_value: torch.Tensor) -> IPPOUpdateSummary:
        batch = buffer.as_tensors(device=self.device)
        gae_batch = buffer.compute_returns_and_advantages(
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
            last_value=last_value,
            device=self.device,
        )
        num_steps = batch["device_obs"].shape[0]
        mini_batch_size = min(self.config.training.batch_size, num_steps)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.config.training.ppo_epochs):
            permutation = torch.randperm(num_steps, device=self.device)
            for start_idx in range(0, num_steps, mini_batch_size):
                step_indices = permutation[start_idx : start_idx + mini_batch_size]
                device_obs = batch["device_obs"][step_indices]
                action = batch["action"][step_indices]
                old_log_prob = batch["log_prob"][step_indices]
                advantage = gae_batch["advantage"][step_indices]
                returns = gae_batch["return"][step_indices]

                log_prob, entropy, _, _ = self.actor.evaluate_actions(device_obs, action)
                ratio = torch.exp(log_prob - old_log_prob)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.training.ppo_clip,
                    1.0 + self.config.training.ppo_clip,
                ) * advantage
                actor_loss = -torch.minimum(unclipped, clipped).mean() - self.config.training.entropy_coeff * entropy.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.training.gradient_clip)
                self.actor_optimizer.step()

                value_pred = self._critic_values(device_obs)
                critic_loss = torch.stack(
                    [
                        torch.nn.functional.mse_loss(value_pred[:, agent_idx], returns[:, agent_idx])
                        for agent_idx in range(self.config.environment.num_agents)
                    ]
                ).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.training.gradient_clip)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.mean().item()))

        return IPPOUpdateSummary(
            steps=len(buffer),
            mean_joint_reward=buffer.mean_joint_reward(),
            mean_scaled_joint_reward=buffer.mean_scaled_joint_reward(),
            actor_loss=sum(actor_losses) / max(len(actor_losses), 1),
            critic_loss=sum(critic_losses) / max(len(critic_losses), 1),
            entropy=sum(entropies) / max(len(entropies), 1),
            l_i_loss=None,
        )

    def run_smoke_rollout(self) -> IPPOSmokeSummary:
        buffer, last_value, _ = self.collect_rollouts(num_episodes=1, max_steps=self.config.training.smoke_steps)
        return IPPOSmokeSummary(
            steps=len(buffer),
            mean_reward=buffer.mean_reward(),
            mean_joint_reward=buffer.mean_joint_reward(),
            last_value=float(last_value.mean().item()),
            critic_type="ippo",
            last_l_i_loss=None,
        )

    def train(self) -> IPPOTrainingRunSummary:
        updates = 0
        episodes_completed = 0
        last_update: IPPOUpdateSummary | None = None
        episode_joint_rewards: list[float] = []

        episodes_remaining = self.config.training.total_episodes
        while episodes_remaining > 0:
            rollout_episodes = min(self.config.training.update_every_episodes, episodes_remaining)
            buffer, last_value, batch_episode_rewards = self.collect_rollouts(rollout_episodes)
            episode_joint_rewards.extend(batch_episode_rewards)
            for episode_offset, joint_reward in enumerate(batch_episode_rewards):
                record = {
                    "episode": episodes_completed + episode_offset + 1,
                    "joint_reward": joint_reward,
                    "steps": self.last_rollout_episode_lengths[episode_offset],
                }
                self.episode_history.append(record)
                if self.episode_log_path is not None:
                    self._append_json_record(self.episode_log_path, record)
            episodes_completed += len(batch_episode_rewards)
            last_update = self.update(buffer, last_value)
            updates += 1
            update_record = {
                "update": updates,
                "episodes_completed": episodes_completed,
                "steps": last_update.steps,
                "mean_joint_reward": last_update.mean_joint_reward,
                "mean_scaled_joint_reward": last_update.mean_scaled_joint_reward,
                "actor_loss": last_update.actor_loss,
                "critic_loss": last_update.critic_loss,
                "entropy": last_update.entropy,
                "l_i_loss": None,
            }
            self.update_history.append(update_record)
            if self.update_log_path is not None:
                self._append_json_record(self.update_log_path, update_record)
            if (
                self.config.training.save_every_episodes > 0
                and episodes_completed > 0
                and episodes_completed % self.config.training.save_every_episodes == 0
            ):
                self._save_checkpoint(episodes_completed, updates)
            episodes_remaining -= rollout_episodes

        if episodes_completed > 0:
            self._save_checkpoint(episodes_completed, updates, suffix="final")

        return IPPOTrainingRunSummary(
            episodes=self.config.training.total_episodes,
            updates=updates,
            mean_episode_joint_reward=float(np.mean(episode_joint_rewards)) if episode_joint_rewards else 0.0,
            critic_type="ippo",
            episode_log_path=None if self.episode_log_path is None else str(self.episode_log_path),
            update_log_path=None if self.update_log_path is None else str(self.update_log_path),
            last_checkpoint_path=self.last_checkpoint_path,
            last_actor_loss=None if last_update is None else last_update.actor_loss,
            last_critic_loss=None if last_update is None else last_update.critic_loss,
            last_entropy=None if last_update is None else last_update.entropy,
            last_l_i_loss=None,
        )
