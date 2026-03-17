from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import chain
from typing import Any

from src.config import ExperimentConfig, build_config_from_args


@dataclass(slots=True)
class SmokeRunSummary:
    steps: int
    mean_reward: float
    mean_joint_reward: float
    last_value: float
    critic_type: str
    last_l_i_loss: float | None = None
    last_l_var_loss: float | None = None


@dataclass(slots=True)
class TrainingUpdateSummary:
    steps: int
    mean_joint_reward: float
    mean_scaled_joint_reward: float
    actor_loss: float
    critic_loss: float
    entropy: float
    l_i_loss: float | None = None
    l_var_loss: float | None = None
    role_mu_var_per_dim: list[float] | None = None
    role_sigma_mean_per_dim: list[float] | None = None
    near_zero_sigma_fraction: float | None = None


@dataclass(slots=True)
class TrainingRunSummary:
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
    last_l_var_loss: float | None = None


def _load_training_components() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install dependencies with `pip install -r requirements.txt` "
            "before running training."
        ) from exc

    from src.buffer import RolloutBuffer, Transition
    from src.environment import DynamicMECEnv
    from src.modules import (
        GraphBuilder,
        RoleEncoder,
        TrajectoryEncoder,
        role_diversity_loss,
        role_identifiability_loss,
        role_posterior_diagnostics,
        role_variance_floor_loss,
        to_pyg_batch,
    )
    from src.networks import MLPCritic, MultiAgentRoleConditionedActor, PGCNCritic, SetCritic
    from src.utils import ObservationScaler, RewardScaler, set_seed

    return {
        "ObservationScaler": ObservationScaler,
        "torch": torch,
        "RewardScaler": RewardScaler,
        "RolloutBuffer": RolloutBuffer,
        "Transition": Transition,
        "DynamicMECEnv": DynamicMECEnv,
        "GraphBuilder": GraphBuilder,
        "RoleEncoder": RoleEncoder,
        "TrajectoryEncoder": TrajectoryEncoder,
        "role_diversity_loss": role_diversity_loss,
        "MLPCritic": MLPCritic,
        "MultiAgentRoleConditionedActor": MultiAgentRoleConditionedActor,
        "PGCNCritic": PGCNCritic,
        "SetCritic": SetCritic,
        "role_identifiability_loss": role_identifiability_loss,
        "role_posterior_diagnostics": role_posterior_diagnostics,
        "role_variance_floor_loss": role_variance_floor_loss,
        "set_seed": set_seed,
        "to_pyg_batch": to_pyg_batch,
    }


def build_critic(config: ExperimentConfig, components: dict[str, Any]) -> Any:
    PGCNCritic = components["PGCNCritic"]
    MLPCritic = components["MLPCritic"]
    SetCritic = components["SetCritic"]
    if config.model.critic_type == "pgcn":
        return PGCNCritic(
            device_dim=config.environment.observation_dim,
            server_dim=config.environment.central_observation_dim,
        )
    if config.model.critic_type == "mlp":
        return MLPCritic(
            obs_dim=config.environment.observation_dim,
            num_agents=config.environment.num_agents,
            central_obs_dim=config.environment.central_observation_dim,
        )
    return SetCritic(
        device_dim=config.environment.observation_dim,
        server_dim=config.environment.central_observation_dim,
    )


class PPOTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.components = _load_training_components()
        self.torch = self.components["torch"]
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.components["set_seed"](config.seed)
        self.config.ensure_output_dirs()

        DynamicMECEnv = self.components["DynamicMECEnv"]
        GraphBuilder = self.components["GraphBuilder"]
        ObservationScaler = self.components["ObservationScaler"]
        RoleEncoder = self.components["RoleEncoder"]
        RewardScaler = self.components["RewardScaler"]
        TrajectoryEncoder = self.components["TrajectoryEncoder"]
        MultiAgentRoleConditionedActor = self.components["MultiAgentRoleConditionedActor"]

        self.env = DynamicMECEnv(config.environment, seed=config.seed)
        self.graph_builder = GraphBuilder(
            num_devices=config.environment.num_agents,
            graph_type=config.environment.graph_type,
            distance_threshold_m=config.environment.distance_threshold_m,
        )
        self.role_encoder = None
        self.trajectory_encoder = None
        if config.model.use_role:
            self.role_encoder = RoleEncoder(
                obs_dim=config.environment.actor_observation_dim,
                role_dim=config.model.role_dim,
                hidden_dim=config.model.role_hidden_dim,
            ).to(self.device)
            if config.model.use_l_i:
                self.trajectory_encoder = TrajectoryEncoder(
                    obs_dim=config.environment.actor_observation_dim,
                    action_dim=config.model.action_dim,
                    role_dim=config.model.role_dim,
                    hidden_dim=config.model.trajectory_hidden_dim,
                ).to(self.device)

        self.actor = MultiAgentRoleConditionedActor(
            num_agents=config.environment.num_agents,
            actor_type=config.model.actor_type,
            obs_dim=config.environment.actor_observation_dim,
            role_dim=config.model.role_dim,
            action_dim=config.model.action_dim,
            hidden_dim=config.model.actor_hidden_dim,
            use_role=config.model.use_role,
        ).to(self.device)
        self.critic = build_critic(config, self.components).to(self.device)
        self.role_diversity_loss = self.components["role_diversity_loss"]
        self.role_identifiability_loss = self.components["role_identifiability_loss"]
        self.role_posterior_diagnostics = self.components["role_posterior_diagnostics"]
        self.role_variance_floor_loss = self.components["role_variance_floor_loss"]

        actor_modules = [self.actor]
        if self.role_encoder is not None:
            actor_modules.append(self.role_encoder)
        if self.trajectory_encoder is not None:
            actor_modules.append(self.trajectory_encoder)
        self.actor_modules = actor_modules
        actor_parameters = list(chain.from_iterable(module.parameters() for module in actor_modules))
        self.actor_optimizer = self.torch.optim.Adam(actor_parameters, lr=config.training.learning_rate)
        self.critic_optimizer = self.torch.optim.Adam(self.critic.parameters(), lr=config.training.learning_rate)
        self.device_obs_scaler = None
        self.server_obs_scaler = None
        if config.training.use_obs_scaling:
            self.device_obs_scaler = ObservationScaler(shape=(config.environment.observation_dim,))
            self.server_obs_scaler = ObservationScaler(shape=(config.environment.central_observation_dim,))
        self.reward_scaler = RewardScaler(gamma=config.training.gamma) if config.training.use_reward_scaling else None
        self.episode_history: list[dict[str, Any]] = []
        self.update_history: list[dict[str, Any]] = []
        self.last_rollout_episode_lengths: list[int] = []
        self.episode_log_path = None
        self.update_log_path = None
        self.last_checkpoint_path = None
        if config.training.run_mode == "train":
            self.episode_log_path = config.output_root / "logs" / "episode_history.jsonl"
            self.update_log_path = config.output_root / "logs" / "update_history.jsonl"
            self.episode_log_path.write_text("", encoding="utf-8")
            self.update_log_path.write_text("", encoding="utf-8")

    def _actor_role_posterior(self, actor_obs: Any) -> tuple[Any, Any | None]:
        if self.role_encoder is None:
            shape = (*actor_obs.shape[:-1], self.config.model.role_dim)
            zeros = self.torch.zeros(shape, dtype=actor_obs.dtype, device=actor_obs.device)
            return zeros, None

        flat_obs = actor_obs.reshape(-1, actor_obs.shape[-1])
        role_mu, role_sigma = self.role_encoder(flat_obs)
        role_shape = (*actor_obs.shape[:-1], self.config.model.role_dim)
        return role_mu.reshape(role_shape), role_sigma.reshape(role_shape)

    def _scale_device_obs(self, device_obs: Any, update_stats: bool) -> Any:
        if self.device_obs_scaler is None:
            return device_obs
        if update_stats:
            scaled = self.device_obs_scaler.update_and_transform(device_obs.detach().cpu().numpy())
        else:
            scaled = self.device_obs_scaler.transform(device_obs.detach().cpu().numpy())
        return self.torch.from_numpy(scaled).float().to(self.device)

    def _scale_server_obs(self, server_obs: Any, update_stats: bool) -> Any:
        if self.server_obs_scaler is None:
            return server_obs
        server_np = server_obs.detach().cpu().numpy()
        server_batch = server_np if server_np.ndim > 1 else server_np.reshape(1, -1)
        if update_stats:
            scaled = self.server_obs_scaler.update_and_transform(server_batch)
        else:
            scaled = self.server_obs_scaler.transform(server_batch)
        if server_obs.dim() == 1:
            scaled = scaled[0]
        return self.torch.from_numpy(scaled).float().to(self.device)

    def _build_actor_observation(self, core_obs: Any, server_info: Any) -> Any:
        queue_broadcast = server_info[..., : self.config.environment.actor_queue_broadcast_dim]
        if core_obs.dim() == 2:
            expanded_queue = queue_broadcast.unsqueeze(0).expand(core_obs.shape[0], -1)
            return self.torch.cat([core_obs, expanded_queue], dim=-1)
        expanded_queue = queue_broadcast.unsqueeze(1).expand(-1, core_obs.shape[1], -1)
        return self.torch.cat([core_obs, expanded_queue], dim=-1)

    def _prepare_model_observation(self, observation: Any, update_stats: bool) -> tuple[Any, Any, Any]:
        core_obs = self.torch.from_numpy(observation.device_obs).float().to(self.device)
        server_info = self.torch.from_numpy(observation.server_obs).float().to(self.device)
        core_obs = self._scale_device_obs(core_obs, update_stats)
        server_info = self._scale_server_obs(server_info, update_stats)
        actor_obs = self._build_actor_observation(core_obs, server_info)
        return actor_obs, core_obs, server_info

    @staticmethod
    def _append_json_record(path: Any, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _record_episode(self, episode_index: int, joint_reward: float, steps: int) -> None:
        record = {
            "episode": episode_index,
            "joint_reward": joint_reward,
            "steps": steps,
        }
        self.episode_history.append(record)
        if self.episode_log_path is not None:
            self._append_json_record(self.episode_log_path, record)

    def _record_update(self, update_index: int, episodes_completed: int, summary: TrainingUpdateSummary) -> None:
        record = {
            "update": update_index,
            "episodes_completed": episodes_completed,
            "steps": summary.steps,
            "mean_joint_reward": summary.mean_joint_reward,
            "mean_scaled_joint_reward": summary.mean_scaled_joint_reward,
            "actor_loss": summary.actor_loss,
            "critic_loss": summary.critic_loss,
            "entropy": summary.entropy,
            "l_i_loss": summary.l_i_loss,
            "l_var_loss": summary.l_var_loss,
            "role_mu_var_per_dim": summary.role_mu_var_per_dim,
            "role_sigma_mean_per_dim": summary.role_sigma_mean_per_dim,
            "near_zero_sigma_fraction": summary.near_zero_sigma_fraction,
        }
        self.update_history.append(record)
        if self.update_log_path is not None:
            self._append_json_record(self.update_log_path, record)

    def _save_checkpoint(self, episodes_completed: int, update_index: int, suffix: str | None = None) -> str:
        checkpoint_name = (
            f"checkpoint_{suffix}.pt" if suffix is not None else f"checkpoint_ep{episodes_completed:04d}_u{update_index:04d}.pt"
        )
        checkpoint_path = self.config.output_root / "models" / checkpoint_name
        checkpoint = {
            "config": self.config.to_dict(),
            "episodes_completed": episodes_completed,
            "update_index": update_index,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "role_encoder": None if self.role_encoder is None else self.role_encoder.state_dict(),
            "trajectory_encoder": None if self.trajectory_encoder is None else self.trajectory_encoder.state_dict(),
            "device_obs_scaler": None if self.device_obs_scaler is None else self.device_obs_scaler.state_dict(),
            "server_obs_scaler": None if self.server_obs_scaler is None else self.server_obs_scaler.state_dict(),
            "reward_scaler": None if self.reward_scaler is None else self.reward_scaler.state_dict(),
            "episode_history": self.episode_history,
            "update_history": self.update_history,
        }
        self.torch.save(checkpoint, checkpoint_path)
        self.last_checkpoint_path = str(checkpoint_path)
        return self.last_checkpoint_path

    def _critic_values(self, device_obs: Any, server_obs: Any, positions: Any) -> Any:
        if self.config.model.critic_type == "pgcn":
            if device_obs.dim() == 2:
                graph = self.graph_builder.build(positions=positions)
                return self.critic(device_obs, server_obs, graph=graph).squeeze(-1)

            graphs = [
                self.graph_builder.build(positions=positions[step_idx])
                for step_idx in range(device_obs.shape[0])
            ]
            if self.critic.use_pyg:
                batch_graph = self.components["to_pyg_batch"](graphs).to(self.device)
                return self.critic(device_obs, server_obs, graph=batch_graph).squeeze(-1)

            stacked_adjacency = self.torch.stack([graph.adjacency for graph in graphs], dim=0)
            return self.critic(device_obs, server_obs, adjacency=stacked_adjacency).squeeze(-1)

        return self.critic(device_obs, server_obs).squeeze(-1)

    def _compute_l_i_loss(self, trajectory_batch: dict[str, Any], step_indices: Any | None = None) -> Any | None:
        if self.role_encoder is None or self.trajectory_encoder is None:
            return None

        num_agents = self.config.environment.num_agents
        if step_indices is None:
            current_obs = trajectory_batch["current_obs"]
            trajectory = trajectory_batch["trajectory"]
        else:
            agent_offsets = self.torch.arange(num_agents, device=self.device)
            flat_indices = (step_indices.unsqueeze(1) * num_agents + agent_offsets.unsqueeze(0)).reshape(-1)
            current_obs = trajectory_batch["current_obs"][flat_indices]
            trajectory = trajectory_batch["trajectory"][flat_indices]

        role_mu, role_sigma = self.role_encoder(current_obs)
        traj_mu, traj_sigma = self.trajectory_encoder(trajectory, current_obs)
        return self.role_identifiability_loss(
            role_mu=role_mu,
            role_std=role_sigma,
            traj_mu=traj_mu,
            traj_std=traj_sigma,
        )

    def _compute_role_diagnostics(self, role_mu: Any, role_sigma: Any | None) -> dict[str, Any] | None:
        if role_sigma is None:
            return None
        diagnostics = self.role_posterior_diagnostics(
            role_mu=role_mu,
            role_std=role_sigma,
            sigma_floor=self.config.training.sigma_floor,
        )
        return {
            "role_mu_var_per_dim": diagnostics["role_mu_var_per_dim"].detach().cpu(),
            "role_sigma_mean_per_dim": diagnostics["role_sigma_mean_per_dim"].detach().cpu(),
            "near_zero_sigma_fraction": float(diagnostics["near_zero_sigma_fraction"].item()),
        }

    def collect_rollouts(
        self,
        num_episodes: int,
        max_steps: int | None = None,
    ) -> tuple[Any, float, list[float]]:
        RolloutBuffer = self.components["RolloutBuffer"]
        Transition = self.components["Transition"]
        buffer = RolloutBuffer()
        episode_joint_rewards: list[float] = []
        episode_lengths: list[int] = []
        last_value = 0.0
        steps_remaining = max_steps
        budget_truncated = False

        for episode_idx in range(num_episodes):
            if steps_remaining is not None and steps_remaining <= 0:
                break
            observation = self.env.reset()
            episode_joint_reward = 0.0
            episode_length = 0
            done = False
            if self.reward_scaler is not None:
                self.reward_scaler.reset()
            step_limit = self.config.environment.episode_length
            if steps_remaining is not None:
                step_limit = min(step_limit, steps_remaining)

            for _ in range(step_limit):
                positions_np = self.env.positions.copy()
                actor_obs, core_obs, server_info = self._prepare_model_observation(observation, update_stats=True)
                positions = self.torch.from_numpy(positions_np).float().to(self.device)

                with self.torch.no_grad():
                    role_mu, _ = self._actor_role_posterior(actor_obs)
                    action, env_action, log_prob = self.actor.sample_action(
                        actor_obs,
                        role_mu if self.config.model.use_role else None,
                    )
                    value = float(self._critic_values(core_obs, server_info, positions).item())

                next_observation, reward, done, _ = self.env.step(env_action.cpu().numpy())
                joint_reward = float(reward.sum())
                scaled_joint_reward = joint_reward
                if self.reward_scaler is not None:
                    scaled_joint_reward = self.reward_scaler.scale(joint_reward)
                episode_joint_reward += joint_reward
                episode_length += 1
                buffer.add(
                    Transition(
                        actor_obs=actor_obs.cpu().numpy(),
                        core_obs=core_obs.cpu().numpy(),
                        server_info=server_info.cpu().numpy(),
                        positions=positions_np.copy(),
                        role_mu=role_mu.cpu().numpy(),
                        action=action.cpu().numpy(),
                        log_prob=log_prob.cpu().numpy(),
                        reward=reward.copy(),
                        joint_reward=joint_reward,
                        scaled_joint_reward=scaled_joint_reward,
                        value=value,
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
                _, core_obs, server_info = self._prepare_model_observation(observation, update_stats=False)
                positions = self.torch.from_numpy(self.env.positions.copy()).float().to(self.device)
                with self.torch.no_grad():
                    last_value = float(self._critic_values(core_obs, server_info, positions).item())
            else:
                last_value = 0.0

            episode_joint_rewards.append(episode_joint_reward)
            episode_lengths.append(episode_length)
            if budget_truncated:
                break

        self.last_rollout_episode_lengths = episode_lengths
        return buffer, last_value, episode_joint_rewards

    def update(self, buffer: Any, last_value: float = 0.0) -> TrainingUpdateSummary:
        if len(buffer) == 0:
            raise ValueError("RolloutBuffer is empty.")

        batch = buffer.as_tensors(device=self.device)
        gae_batch = buffer.compute_returns_and_advantages(
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
            last_value=last_value,
            device=self.device,
            normalize_advantages=True,
        )
        trajectory_batch = None
        if self.config.model.use_role and self.config.model.use_l_i:
            trajectory_batch = buffer.build_agent_trajectory_batch(
                window_size=self.config.training.trajectory_window,
                obs_dim=self.config.environment.actor_observation_dim,
                action_dim=self.config.model.action_dim,
                action_scale=self.config.training.trajectory_action_scale,
                device=self.device,
            )

        num_steps = batch["actor_obs"].shape[0]
        mini_batch_size = min(self.config.training.batch_size, num_steps)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        l_i_losses: list[float] = []
        l_var_losses: list[float] = []
        role_mu_var_history: list[Any] = []
        role_sigma_mean_history: list[Any] = []
        near_zero_sigma_history: list[float] = []

        for _ in range(self.config.training.ppo_epochs):
            permutation = self.torch.randperm(num_steps, device=self.device)
            for start_idx in range(0, num_steps, mini_batch_size):
                step_indices = permutation[start_idx : start_idx + mini_batch_size]
                actor_obs = batch["actor_obs"][step_indices]
                core_obs = batch["core_obs"][step_indices]
                server_info = batch["server_info"][step_indices]
                positions = batch["positions"][step_indices]
                action = batch["action"][step_indices]
                old_log_prob = batch["log_prob"][step_indices]
                advantage = gae_batch["advantage"][step_indices]
                returns = gae_batch["return"][step_indices]

                role_mu, role_sigma = self._actor_role_posterior(actor_obs)
                log_prob, entropy, _, _ = self.actor.evaluate_actions(
                    actor_obs,
                    action,
                    role_mu if self.config.model.use_role else None,
                )
                ratio = self.torch.exp(log_prob - old_log_prob)
                expanded_advantage = advantage.unsqueeze(-1).expand_as(ratio)
                unclipped = ratio * expanded_advantage
                clipped = self.torch.clamp(
                    ratio,
                    1.0 - self.config.training.ppo_clip,
                    1.0 + self.config.training.ppo_clip,
                ) * expanded_advantage
                policy_loss = -self.torch.minimum(unclipped, clipped).mean()
                entropy_bonus = entropy.mean()

                l_i_loss = actor_obs.new_tensor(0.0)
                if trajectory_batch is not None:
                    computed_l_i = self._compute_l_i_loss(trajectory_batch, step_indices=step_indices)
                    if computed_l_i is not None:
                        l_i_loss = computed_l_i
                        l_i_losses.append(float(l_i_loss.item()))
                l_var_loss = actor_obs.new_tensor(0.0)
                diagnostics = self._compute_role_diagnostics(role_mu, role_sigma)
                if role_sigma is not None:
                    l_var_loss = self.role_variance_floor_loss(role_sigma, self.config.training.sigma_floor)
                    l_var_losses.append(float(l_var_loss.item()))
                if diagnostics is not None:
                    role_mu_var_history.append(diagnostics["role_mu_var_per_dim"])
                    role_sigma_mean_history.append(diagnostics["role_sigma_mean_per_dim"])
                    near_zero_sigma_history.append(diagnostics["near_zero_sigma_fraction"])
                l_d_loss = actor_obs.new_tensor(0.0)
                if self.config.model.use_role and self.config.model.use_l_d_simple:
                    flat_role_mu = role_mu.reshape(-1, role_mu.shape[-1])
                    l_d_loss = self.role_diversity_loss(flat_role_mu)

                actor_loss = (
                    policy_loss
                    - self.config.training.entropy_coeff * entropy_bonus
                    + self.config.training.l_i_coeff * l_i_loss
                    + self.config.training.lambda_var * l_var_loss
                    + self.config.training.l_d_coeff * l_d_loss
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.torch.nn.utils.clip_grad_norm_(
                    list(chain.from_iterable(module.parameters() for module in self.actor_modules)),
                    self.config.training.gradient_clip,
                )
                self.actor_optimizer.step()

                value_pred = self._critic_values(core_obs, server_info, positions)
                critic_loss = self.torch.nn.functional.mse_loss(value_pred, returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.training.gradient_clip)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy_bonus.item()))

        return TrainingUpdateSummary(
            steps=len(buffer),
            mean_joint_reward=buffer.mean_joint_reward(),
            mean_scaled_joint_reward=buffer.mean_scaled_joint_reward(),
            actor_loss=sum(actor_losses) / max(len(actor_losses), 1),
            critic_loss=sum(critic_losses) / max(len(critic_losses), 1),
            entropy=sum(entropies) / max(len(entropies), 1),
            l_i_loss=(sum(l_i_losses) / len(l_i_losses)) if l_i_losses else None,
            l_var_loss=(sum(l_var_losses) / len(l_var_losses)) if l_var_losses else None,
            role_mu_var_per_dim=(
                self.torch.stack(role_mu_var_history, dim=0).mean(dim=0).tolist() if role_mu_var_history else None
            ),
            role_sigma_mean_per_dim=(
                self.torch.stack(role_sigma_mean_history, dim=0).mean(dim=0).tolist() if role_sigma_mean_history else None
            ),
            near_zero_sigma_fraction=(
                sum(near_zero_sigma_history) / len(near_zero_sigma_history) if near_zero_sigma_history else None
            ),
        )

    def run_smoke_rollout(self) -> SmokeRunSummary:
        buffer, _, _ = self.collect_rollouts(num_episodes=1, max_steps=self.config.training.smoke_steps)
        last_transition_value = 0.0 if len(buffer) == 0 else float(buffer.transitions[-1].value or 0.0)
        last_l_i_loss = None
        last_l_var_loss = None
        if self.config.model.use_role and self.config.model.use_l_i and len(buffer) > 0:
            trajectory_batch = buffer.build_agent_trajectory_batch(
                window_size=self.config.training.trajectory_window,
                obs_dim=self.config.environment.actor_observation_dim,
                action_dim=self.config.model.action_dim,
                action_scale=self.config.training.trajectory_action_scale,
                device=self.device,
            )
            computed_l_i = self._compute_l_i_loss(trajectory_batch)
            if computed_l_i is not None:
                last_l_i_loss = float(computed_l_i.item())
        if self.config.model.use_role and self.role_encoder is not None and len(buffer) > 0:
            batch = buffer.as_tensors(device=self.device)
            flat_actor_obs = batch["actor_obs"].reshape(-1, batch["actor_obs"].shape[-1])
            with self.torch.no_grad():
                _, role_sigma = self.role_encoder(flat_actor_obs)
                if role_sigma is not None:
                    last_l_var_loss = float(
                        self.role_variance_floor_loss(role_sigma, self.config.training.sigma_floor).item()
                    )

        return SmokeRunSummary(
            steps=len(buffer),
            mean_reward=buffer.mean_reward(),
            mean_joint_reward=buffer.mean_joint_reward(),
            last_value=last_transition_value,
            critic_type=self.config.model.critic_type,
            last_l_i_loss=last_l_i_loss,
            last_l_var_loss=last_l_var_loss,
        )

    def train(self) -> TrainingRunSummary:
        updates = 0
        last_update: TrainingUpdateSummary | None = None
        episode_joint_rewards: list[float] = []
        episodes_completed = 0

        episodes_remaining = self.config.training.total_episodes
        while episodes_remaining > 0:
            rollout_episodes = min(self.config.training.update_every_episodes, episodes_remaining)
            buffer, last_value, batch_episode_rewards = self.collect_rollouts(num_episodes=rollout_episodes)
            episode_joint_rewards.extend(batch_episode_rewards)
            for episode_offset, joint_reward in enumerate(batch_episode_rewards):
                episode_number = episodes_completed + episode_offset + 1
                episode_steps = self.last_rollout_episode_lengths[episode_offset]
                self._record_episode(episode_number, joint_reward, episode_steps)
            episodes_completed += len(batch_episode_rewards)
            last_update = self.update(buffer, last_value=last_value)
            updates += 1
            self._record_update(updates, episodes_completed, last_update)
            should_save_periodic = (
                self.config.training.save_every_episodes > 0
                and episodes_completed > 0
                and episodes_completed % self.config.training.save_every_episodes == 0
            )
            if should_save_periodic:
                self._save_checkpoint(episodes_completed, updates)
            episodes_remaining -= rollout_episodes

        if episodes_completed > 0:
            self._save_checkpoint(episodes_completed, updates, suffix="final")

        return TrainingRunSummary(
            episodes=self.config.training.total_episodes,
            updates=updates,
            mean_episode_joint_reward=sum(episode_joint_rewards) / max(len(episode_joint_rewards), 1),
            critic_type=self.config.model.critic_type,
            episode_log_path=None if self.episode_log_path is None else str(self.episode_log_path),
            update_log_path=None if self.update_log_path is None else str(self.update_log_path),
            last_checkpoint_path=self.last_checkpoint_path,
            last_actor_loss=None if last_update is None else last_update.actor_loss,
            last_critic_loss=None if last_update is None else last_update.critic_loss,
            last_entropy=None if last_update is None else last_update.entropy,
            last_l_i_loss=None if last_update is None else last_update.l_i_loss,
            last_l_var_loss=None if last_update is None else last_update.l_var_loss,
        )


def run_smoke_rollout(config: ExperimentConfig) -> SmokeRunSummary:
    from src.baselines import (
        DeterministicContextTrainer,
        IPPOTrainer,
        apply_experiment_variant,
        build_li_original_command,
        li_original_available,
        li_original_missing_files,
        run_fixed_policy_baseline,
        run_maddpg_baseline,
    )

    resolved_config, variant = apply_experiment_variant(config, config.training.variant_id)
    if variant is None or variant.runner_kind == "ppo":
        trainer = PPOTrainer(resolved_config)
        return trainer.run_smoke_rollout()
    if variant.runner_kind == "det_context":
        trainer = DeterministicContextTrainer(resolved_config)
        return trainer.run_smoke_rollout()
    if variant.runner_kind == "ippo":
        trainer = IPPOTrainer(resolved_config)
        summary = trainer.run_smoke_rollout()
        return SmokeRunSummary(
            steps=summary.steps,
            mean_reward=summary.mean_reward,
            mean_joint_reward=summary.mean_joint_reward,
            last_value=summary.last_value,
            critic_type=summary.critic_type,
            last_l_i_loss=summary.last_l_i_loss,
            last_l_var_loss=None,
        )
    if variant.runner_kind == "fixed":
        fixed_summary = run_fixed_policy_baseline(resolved_config, variant.variant_id, num_episodes=1)
        return SmokeRunSummary(
            steps=resolved_config.environment.episode_length,
            mean_reward=fixed_summary.mean_step_device_reward,
            mean_joint_reward=fixed_summary.mean_step_joint_reward,
            last_value=0.0,
            critic_type=variant.variant_id.lower(),
            last_l_i_loss=None,
            last_l_var_loss=None,
        )
    if variant.runner_kind == "external":
        if not li_original_available():
            missing = ", ".join(str(path) for path in li_original_missing_files())
            raise SystemExit(f"B0 requires the original li_code repository files. Missing: {missing}")
        command = " ".join(build_li_original_command())
        raise SystemExit(f"B0 should be run through the original Li code path: {command}")
    run_maddpg_baseline()
    raise SystemExit("Unreachable baseline branch.")


def run_training(config: ExperimentConfig) -> TrainingRunSummary:
    from src.baselines import (
        DeterministicContextTrainer,
        IPPOTrainer,
        apply_experiment_variant,
        build_li_original_command,
        li_original_available,
        li_original_missing_files,
        run_fixed_policy_baseline,
        run_maddpg_baseline,
    )

    resolved_config, variant = apply_experiment_variant(config, config.training.variant_id)
    if variant is None or variant.runner_kind == "ppo":
        trainer = PPOTrainer(resolved_config)
        return trainer.train()
    if variant.runner_kind == "det_context":
        trainer = DeterministicContextTrainer(resolved_config)
        return trainer.train()
    if variant.runner_kind == "ippo":
        trainer = IPPOTrainer(resolved_config)
        summary = trainer.train()
        return TrainingRunSummary(
            episodes=summary.episodes,
            updates=summary.updates,
            mean_episode_joint_reward=summary.mean_episode_joint_reward,
            critic_type=summary.critic_type,
            episode_log_path=summary.episode_log_path,
            update_log_path=summary.update_log_path,
            last_checkpoint_path=summary.last_checkpoint_path,
            last_actor_loss=summary.last_actor_loss,
            last_critic_loss=summary.last_critic_loss,
            last_entropy=summary.last_entropy,
            last_l_i_loss=summary.last_l_i_loss,
            last_l_var_loss=None,
        )
    if variant.runner_kind == "fixed":
        fixed_summary = run_fixed_policy_baseline(resolved_config, variant.variant_id)
        return TrainingRunSummary(
            episodes=fixed_summary.episodes,
            updates=0,
            mean_episode_joint_reward=fixed_summary.mean_episode_joint_reward,
            critic_type=variant.variant_id.lower(),
            episode_log_path=None,
            update_log_path=None,
            last_checkpoint_path=None,
            last_actor_loss=None,
            last_critic_loss=None,
            last_entropy=None,
            last_l_i_loss=None,
            last_l_var_loss=None,
        )
    if variant.runner_kind == "external":
        if not li_original_available():
            missing = ", ".join(str(path) for path in li_original_missing_files())
            raise SystemExit(f"B0 requires the original li_code repository files. Missing: {missing}")
        command = " ".join(build_li_original_command())
        raise SystemExit(f"B0 should be run through the original Li code path: {command}")
    run_maddpg_baseline()
    raise SystemExit("Unreachable baseline branch.")


def main() -> None:
    config = build_config_from_args()
    if config.training.run_mode == "train":
        summary = run_training(config)
        print(
            f"train_run critic={summary.critic_type} episodes={summary.episodes} updates={summary.updates} "
            f"mean_episode_joint_reward={summary.mean_episode_joint_reward:.4f} "
            f"last_actor_loss={summary.last_actor_loss if summary.last_actor_loss is not None else 'n/a'} "
            f"last_critic_loss={summary.last_critic_loss if summary.last_critic_loss is not None else 'n/a'} "
            f"last_l_i_loss={summary.last_l_i_loss if summary.last_l_i_loss is not None else 'n/a'} "
            f"last_l_var_loss={summary.last_l_var_loss if summary.last_l_var_loss is not None else 'n/a'} "
            f"checkpoint={summary.last_checkpoint_path if summary.last_checkpoint_path is not None else 'n/a'}"
        )
        return

    summary = run_smoke_rollout(config)
    print(
        f"smoke_run critic={summary.critic_type} steps={summary.steps} "
        f"mean_reward={summary.mean_reward:.4f} mean_joint_reward={summary.mean_joint_reward:.4f} "
        f"last_value={summary.last_value:.4f} "
        f"last_l_i_loss={summary.last_l_i_loss if summary.last_l_i_loss is not None else 'n/a'} "
        f"last_l_var_loss={summary.last_l_var_loss if summary.last_l_var_loss is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
