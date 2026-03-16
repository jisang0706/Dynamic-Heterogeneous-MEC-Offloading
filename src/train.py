from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import ExperimentConfig, build_config_from_args


@dataclass(slots=True)
class SmokeRunSummary:
    steps: int
    mean_reward: float
    last_value: float
    critic_type: str
    last_l_i_loss: float | None = None


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
    from src.modules import GraphBuilder, RoleEncoder, TrajectoryEncoder, role_identifiability_loss
    from src.networks import MLPCritic, PGCNCritic, RoleConditionedActor, SetCritic
    from src.utils import set_seed

    return {
        "torch": torch,
        "RolloutBuffer": RolloutBuffer,
        "Transition": Transition,
        "DynamicMECEnv": DynamicMECEnv,
        "GraphBuilder": GraphBuilder,
        "RoleEncoder": RoleEncoder,
        "TrajectoryEncoder": TrajectoryEncoder,
        "MLPCritic": MLPCritic,
        "PGCNCritic": PGCNCritic,
        "RoleConditionedActor": RoleConditionedActor,
        "SetCritic": SetCritic,
        "role_identifiability_loss": role_identifiability_loss,
        "set_seed": set_seed,
    }


def build_critic(config: ExperimentConfig, components: dict[str, Any]) -> Any:
    PGCNCritic = components["PGCNCritic"]
    MLPCritic = components["MLPCritic"]
    SetCritic = components["SetCritic"]
    if config.model.critic_type == "pgcn":
        return PGCNCritic(node_dim=config.environment.observation_dim)
    if config.model.critic_type == "mlp":
        return MLPCritic(
            obs_dim=config.environment.observation_dim,
            num_agents=config.environment.num_agents,
            central_obs_dim=config.environment.central_observation_dim,
        )
    return SetCritic(
        obs_dim=config.environment.observation_dim,
        central_obs_dim=config.environment.central_observation_dim,
    )


def run_smoke_rollout(config: ExperimentConfig) -> SmokeRunSummary:
    components = _load_training_components()
    torch = components["torch"]
    set_seed = components["set_seed"]
    DynamicMECEnv = components["DynamicMECEnv"]
    GraphBuilder = components["GraphBuilder"]
    RoleEncoder = components["RoleEncoder"]
    TrajectoryEncoder = components["TrajectoryEncoder"]
    RoleConditionedActor = components["RoleConditionedActor"]
    RolloutBuffer = components["RolloutBuffer"]
    Transition = components["Transition"]
    role_identifiability_loss = components["role_identifiability_loss"]

    set_seed(config.seed)
    config.ensure_output_dirs()

    env = DynamicMECEnv(config.environment, seed=config.seed)
    graph_builder = GraphBuilder(
        num_devices=config.environment.num_agents,
        graph_type=config.environment.graph_type,
        distance_threshold_m=config.environment.distance_threshold_m,
    )
    role_encoder = RoleEncoder(
        obs_dim=config.environment.observation_dim,
        role_dim=config.model.role_dim,
        hidden_dim=config.model.role_hidden_dim,
    )
    actor = RoleConditionedActor(
        obs_dim=config.environment.observation_dim,
        role_dim=config.model.role_dim,
        action_dim=config.model.action_dim,
        hidden_dim=config.model.actor_hidden_dim,
    )
    trajectory_encoder = TrajectoryEncoder(
        obs_dim=config.environment.observation_dim,
        action_dim=config.model.action_dim,
        role_dim=config.model.role_dim,
        hidden_dim=config.model.trajectory_hidden_dim,
    )
    critic = build_critic(config, components)
    buffer = RolloutBuffer()

    observation = env.reset()
    last_value = 0.0
    last_l_i_loss: float | None = None

    for _ in range(min(config.training.smoke_steps, config.environment.episode_length)):
        device_obs = torch.from_numpy(observation.device_obs).float()
        server_obs = torch.from_numpy(observation.server_obs).float()

        with torch.no_grad():
            role_mu, _ = role_encoder(device_obs)
            mean, std = actor(device_obs, role_mu)
            distribution = torch.distributions.Normal(mean, std)
            sampled_action = torch.clamp(distribution.sample(), 0.0, 10.0)
            env_action = sampled_action / 10.0
            log_prob = distribution.log_prob(sampled_action).sum(dim=-1)

            if config.model.critic_type == "pgcn":
                graph = graph_builder.build(
                    device_obs=device_obs,
                    server_obs=server_obs,
                    positions=torch.from_numpy(env.positions).float(),
                )
                last_value = float(critic(graph).mean().item())
            else:
                last_value = float(critic(device_obs, server_obs).mean().item())

        next_observation, reward, done, _ = env.step(env_action.cpu().numpy())
        buffer.add(
            Transition(
                device_obs=observation.device_obs.copy(),
                server_obs=observation.server_obs.copy(),
                role_mu=role_mu.cpu().numpy(),
                action=sampled_action.cpu().numpy(),
                log_prob=log_prob.cpu().numpy(),
                reward=reward.copy(),
                done=done,
            )
        )
        observation = next_observation
        if done:
            break

    if config.model.use_l_i and len(buffer) > 0:
        with torch.no_grad():
            trajectory_batch = buffer.build_agent_trajectory_batch(
                window_size=config.training.trajectory_window,
                obs_dim=config.environment.observation_dim,
                action_dim=config.model.action_dim,
                action_scale=config.training.trajectory_action_scale,
                device="cpu",
            )
            role_mu_batch, role_sigma_batch = role_encoder(trajectory_batch["current_obs"])
            traj_mu_batch, traj_sigma_batch = trajectory_encoder(
                trajectory_batch["trajectory"],
                trajectory_batch["current_obs"],
            )
            last_l_i_loss = float(
                role_identifiability_loss(
                    role_mu=role_mu_batch,
                    role_std=role_sigma_batch,
                    traj_mu=traj_mu_batch,
                    traj_std=traj_sigma_batch,
                ).item()
            )

    return SmokeRunSummary(
        steps=len(buffer),
        mean_reward=buffer.mean_reward(),
        last_value=last_value,
        critic_type=config.model.critic_type,
        last_l_i_loss=last_l_i_loss,
    )


def main() -> None:
    config = build_config_from_args()
    summary = run_smoke_rollout(config)
    print(
        f"smoke_run critic={summary.critic_type} steps={summary.steps} "
        f"mean_reward={summary.mean_reward:.4f} last_value={summary.last_value:.4f} "
        f"last_l_i_loss={summary.last_l_i_loss if summary.last_l_i_loss is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
