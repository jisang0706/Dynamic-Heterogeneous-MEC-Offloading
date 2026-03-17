from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baselines import (
    DeterministicContextTrainer,
    IPPOTrainer,
    apply_experiment_variant,
    edge_only_actions,
    get_experiment_variant,
    local_only_actions,
    queue_aware_greedy_actions,
)
from src.buffer import RolloutBuffer, Transition
from src.config import ExperimentConfig, TrainingConfig, _str_to_bool, build_config_from_dict
from src.environment import DynamicMECEnv
from src.modules.role_loss import diagonal_gaussian_kl
from src.train import PPOTrainer


@dataclass(slots=True)
class LoadedPolicy:
    config: ExperimentConfig
    runner_kind: str
    variant_id: str | None
    label: str
    checkpoint_path: Path | None
    env: DynamicMECEnv
    device: torch.device
    actor: Any | None = None
    role_encoder: Any | None = None
    trajectory_encoder: Any | None = None
    device_obs_scaler: Any | None = None
    server_obs_scaler: Any | None = None
    use_role: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MEC offloading checkpoints and baselines")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--trace-label", type=str, default=None)
    parser.add_argument("--deterministic-policy", type=_str_to_bool, default=True)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--graph-type", choices=("star", "star_proximity"), default="star")
    parser.add_argument("--distance-threshold-m", type=float, default=150.0)
    parser.add_argument("--use-mobility", type=_str_to_bool, default=True)
    parser.add_argument("--use-cpu-dynamics", type=_str_to_bool, default=True)
    return parser


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=True) + "\n")


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _diagonal_gaussian_nll(sample: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.pow(2).clamp_min(1e-12)
    quadratic = (sample - mu).pow(2) / variance
    log_det = 2.0 * torch.log(std.clamp_min(1e-12))
    return 0.5 * (quadratic + log_det + math.log(2.0 * math.pi)).sum(dim=-1)


def _pairwise_distance_correlation(role_mu: np.ndarray | None, action_env: np.ndarray) -> float | None:
    if role_mu is None or role_mu.shape[0] < 2:
        return None
    role_distance = np.linalg.norm(role_mu[:, None, :] - role_mu[None, :, :], axis=-1)
    action_distance = np.linalg.norm(action_env[:, None, :] - action_env[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(role_distance, dtype=bool), k=1)
    role_values = role_distance[mask]
    action_values = action_distance[mask]
    if role_values.size == 0:
        return None
    if np.allclose(role_values.std(), 0.0) or np.allclose(action_values.std(), 0.0):
        return None
    correlation = np.corrcoef(role_values, action_values)[0, 1]
    if np.isnan(correlation):
        return None
    return float(correlation)


def _distance_regime(distance_m: float) -> str:
    if distance_m < 100.0:
        return "near"
    if distance_m < 175.0:
        return "mid"
    return "far"


def _cpu_regime(cpu_ghz: float) -> str:
    if cpu_ghz < 2.0:
        return "low"
    if cpu_ghz < 2.5:
        return "mid"
    return "high"


def build_operational_mode_summary(device_records: list[dict[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for record in device_records:
        key = (_distance_regime(record["distance_m"]), _cpu_regime(record["cpu_ghz"]))
        grouped.setdefault(key, []).append(record)

    summary: list[dict[str, Any]] = []
    for distance_regime, cpu_regime in (
        ("near", "low"),
        ("near", "mid"),
        ("near", "high"),
        ("mid", "low"),
        ("mid", "mid"),
        ("mid", "high"),
        ("far", "low"),
        ("far", "mid"),
        ("far", "high"),
    ):
        records = grouped.get((distance_regime, cpu_regime), [])
        summary.append(
            {
                "distance_regime": distance_regime,
                "cpu_regime": cpu_regime,
                "count": len(records),
                "avg_offloading_ratio": _mean_or_none([item["offloading_ratio"] for item in records]),
                "avg_power_ratio": _mean_or_none([item["power_ratio"] for item in records]),
                "avg_timeout_ratio": _mean_or_none([item["timeout_ratio"] for item in records]),
            }
        )
    return summary


def summarize_evaluation(
    policy: LoadedPolicy,
    step_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    device_records: list[dict[str, float]],
    results_dir: Path,
    trace_path: Path,
) -> dict[str, Any]:
    metrics = {
        "mean_episode_joint_reward": float(np.mean([item["joint_reward"] for item in episode_records])) if episode_records else 0.0,
        "std_episode_joint_reward": float(np.std([item["joint_reward"] for item in episode_records])) if episode_records else 0.0,
        "mean_step_joint_reward": float(np.mean([item["joint_reward"] for item in step_records])) if step_records else 0.0,
        "mean_step_device_reward": float(np.mean([item["mean_device_reward"] for item in step_records])) if step_records else 0.0,
        "mean_timeout_ratio": float(np.mean([item["timeout_ratio"] for item in step_records])) if step_records else 0.0,
        "mean_edge_queue": float(np.mean([item["edge_queue"] for item in step_records])) if step_records else 0.0,
        "mean_local_queue": float(np.mean([item["mean_local_queue"] for item in step_records])) if step_records else 0.0,
        "mean_task_processing_cost": float(np.mean([item["mean_task_processing_cost"] for item in step_records])) if step_records else 0.0,
        "mean_task_completion_delay_s": float(np.mean([item["mean_task_completion_delay_s"] for item in step_records])) if step_records else 0.0,
        "mean_distance_m": float(np.mean([item["mean_distance_m"] for item in step_records])) if step_records else 0.0,
        "mean_cpu_ghz": float(np.mean([item["mean_cpu_ghz"] for item in step_records])) if step_records else 0.0,
        "mean_offloading_ratio": float(np.mean([item["mean_offloading_ratio"] for item in step_records])) if step_records else 0.0,
        "mean_power_ratio": float(np.mean([item["mean_power_ratio"] for item in step_records])) if step_records else 0.0,
    }

    optional_metric_names = (
        "mean_role_kl",
        "mean_role_nll",
        "mean_role_action_distance_correlation",
        "mean_role_std",
        "mean_role_variance",
        "mean_near_zero_sigma_fraction",
    )
    for metric_name in optional_metric_names:
        values = [float(item[metric_name]) for item in episode_records if item.get(metric_name) is not None]
        metrics[metric_name] = _mean_or_none(values)

    variant = get_experiment_variant(policy.variant_id)
    return {
        "label": policy.label,
        "variant_id": policy.variant_id,
        "variant_name": None if variant is None else variant.name,
        "runner_kind": policy.runner_kind,
        "checkpoint": None if policy.checkpoint_path is None else str(policy.checkpoint_path),
        "episodes": len(episode_records),
        "num_agents": policy.config.environment.num_agents,
        "deterministic_policy": True,
        "metrics": metrics,
        "operational_mode_bins": build_operational_mode_summary(device_records),
        "episode_metrics": episode_records,
        "trace_path": str(trace_path),
        "results_dir": str(results_dir),
        "config": policy.config.to_dict(),
    }


def _build_eval_config(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, training=replace(config.training, run_mode="smoke"))


def _load_trainer_checkpoint(trainer: Any, checkpoint: dict[str, Any], runner_kind: str) -> None:
    trainer.actor.load_state_dict(checkpoint["actor"])
    trainer.actor.eval()
    if runner_kind == "ippo":
        if checkpoint.get("critics") is not None:
            trainer.critics.load_state_dict(checkpoint["critics"])
            trainer.critics.eval()
        if checkpoint.get("device_obs_scaler") is not None and trainer.device_obs_scaler is not None:
            trainer.device_obs_scaler.load_state_dict(checkpoint["device_obs_scaler"])
        if checkpoint.get("server_obs_scaler") is not None and getattr(trainer, "server_obs_scaler", None) is not None:
            trainer.server_obs_scaler.load_state_dict(checkpoint["server_obs_scaler"])
        return

    if checkpoint.get("critic") is not None:
        trainer.critic.load_state_dict(checkpoint["critic"])
        trainer.critic.eval()
    if checkpoint.get("role_encoder") is not None and trainer.role_encoder is not None:
        trainer.role_encoder.load_state_dict(checkpoint["role_encoder"])
        trainer.role_encoder.eval()
    if checkpoint.get("trajectory_encoder") is not None and trainer.trajectory_encoder is not None:
        trainer.trajectory_encoder.load_state_dict(checkpoint["trajectory_encoder"])
        trainer.trajectory_encoder.eval()
    if checkpoint.get("device_obs_scaler") is not None and trainer.device_obs_scaler is not None:
        trainer.device_obs_scaler.load_state_dict(checkpoint["device_obs_scaler"])
    if checkpoint.get("server_obs_scaler") is not None and trainer.server_obs_scaler is not None:
        trainer.server_obs_scaler.load_state_dict(checkpoint["server_obs_scaler"])


def _torch_load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_checkpoint_policy(checkpoint_path: Path) -> LoadedPolicy:
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    base_config = build_config_from_dict(checkpoint["config"])
    resolved_config, variant = apply_experiment_variant(base_config, base_config.training.variant_id)
    eval_config = _build_eval_config(resolved_config)

    if "critics" in checkpoint:
        runner_kind = "ippo"
        trainer = IPPOTrainer(eval_config)
    elif variant is not None and variant.runner_kind == "det_context":
        runner_kind = "det_context"
        trainer = DeterministicContextTrainer(eval_config)
    else:
        runner_kind = "ppo"
        trainer = PPOTrainer(eval_config)

    _load_trainer_checkpoint(trainer, checkpoint, runner_kind=runner_kind)
    label = checkpoint_path.stem
    return LoadedPolicy(
        config=eval_config,
        runner_kind=runner_kind,
        variant_id=eval_config.training.variant_id,
        label=label,
        checkpoint_path=checkpoint_path,
        env=trainer.env,
        device=trainer.device,
        actor=trainer.actor,
        role_encoder=getattr(trainer, "role_encoder", None),
        trajectory_encoder=getattr(trainer, "trajectory_encoder", None),
        device_obs_scaler=getattr(trainer, "device_obs_scaler", None),
        server_obs_scaler=getattr(trainer, "server_obs_scaler", None),
        use_role=bool(eval_config.model.use_role),
    )


def _build_fixed_policy(args: argparse.Namespace) -> LoadedPolicy:
    base_config = ExperimentConfig(seed=args.seed, output_root=args.output_root)
    environment = replace(
        base_config.environment,
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        graph_type=args.graph_type,
        distance_threshold_m=args.distance_threshold_m,
        use_mobility=args.use_mobility,
        use_cpu_dynamics=args.use_cpu_dynamics,
    )
    training = replace(base_config.training, run_mode="smoke", variant_id=args.variant_id)
    resolved_config, variant = apply_experiment_variant(replace(base_config, environment=environment, training=training), args.variant_id)
    if variant is None or variant.runner_kind != "fixed":
        raise SystemExit("Checkpoint-free evaluation currently supports fixed baselines such as LOCAL_ONLY, EDGE_ONLY, RANDOM, and QAG.")
    return LoadedPolicy(
        config=resolved_config,
        runner_kind="fixed",
        variant_id=variant.variant_id,
        label=variant.variant_id.lower(),
        checkpoint_path=None,
        env=DynamicMECEnv(resolved_config.environment, seed=resolved_config.seed),
        device=torch.device("cpu"),
        use_role=False,
    )


def load_policy_from_args(args: argparse.Namespace) -> LoadedPolicy:
    if args.checkpoint is not None:
        return _load_checkpoint_policy(args.checkpoint)
    return _build_fixed_policy(args)


def _scale_device_obs(policy: LoadedPolicy, device_obs: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(device_obs).float().to(policy.device)
    if policy.device_obs_scaler is not None:
        scaled = policy.device_obs_scaler.transform(device_obs)
        tensor = torch.from_numpy(scaled).float().to(policy.device)
    return tensor


def _scale_server_obs(policy: LoadedPolicy, server_obs: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(server_obs).float().to(policy.device)
    if policy.server_obs_scaler is not None:
        scaled = policy.server_obs_scaler.transform(server_obs.reshape(1, -1))[0]
        tensor = torch.from_numpy(scaled).float().to(policy.device)
    return tensor


def _build_actor_observation(policy: LoadedPolicy, core_obs: torch.Tensor, server_info: torch.Tensor) -> torch.Tensor:
    queue_broadcast = server_info[..., : policy.config.environment.actor_queue_broadcast_dim]
    if core_obs.dim() == 2:
        expanded_queue = queue_broadcast.unsqueeze(0).expand(core_obs.shape[0], -1)
        return torch.cat([core_obs, expanded_queue], dim=-1)
    expanded_queue = queue_broadcast.unsqueeze(1).expand(-1, core_obs.shape[1], -1)
    return torch.cat([core_obs, expanded_queue], dim=-1)


def _select_fixed_action(policy: LoadedPolicy, episode_idx: int, step_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    num_agents = policy.config.environment.num_agents
    num_tasks = policy.config.environment.num_tasks_per_step
    normalized_name = (policy.variant_id or "RANDOM").upper()
    if normalized_name == "LOCAL_ONLY":
        env_action = local_only_actions(num_agents, num_tasks)
    elif normalized_name == "EDGE_ONLY":
        env_action = edge_only_actions(num_agents, num_tasks)
    elif normalized_name == "QAG":
        env_action = queue_aware_greedy_actions(policy.env)
    else:
        seed = policy.config.seed + episode_idx * policy.config.environment.episode_length + step_idx
        env_action = np.random.default_rng(seed).uniform(0.0, 1.0, size=(num_agents, num_tasks + 1)).astype(np.float32)
    action = env_action * 10.0
    return env_action, action, np.zeros(num_agents, dtype=np.float32), None


def _select_learned_action(
    policy: LoadedPolicy,
    observation: Any,
    deterministic_policy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with torch.no_grad():
        core_obs = _scale_device_obs(policy, observation.device_obs)
        server_info = _scale_server_obs(policy, observation.server_obs)
        actor_obs = _build_actor_observation(policy, core_obs, server_info)

        role_mu = None
        role_sigma = None
        if policy.runner_kind != "ippo" and policy.role_encoder is not None:
            role_mu, role_sigma = policy.role_encoder(actor_obs)
        elif policy.use_role:
            zeros = torch.zeros(
                (policy.config.environment.num_agents, policy.config.model.role_dim),
                dtype=actor_obs.dtype,
                device=policy.device,
            )
            role_mu = zeros

        actor_role = role_mu if policy.use_role else None
        if deterministic_policy:
            mean, _ = policy.actor(actor_obs, actor_role)
            action = torch.clamp(mean, 0.0, 10.0)
            log_prob, _, _, _ = policy.actor.evaluate_actions(actor_obs, action, actor_role)
        else:
            action, _, log_prob = policy.actor.sample_action(actor_obs, actor_role)
        env_action = policy.actor.action_to_env(action)

    return (
        actor_obs.cpu().numpy(),
        core_obs.cpu().numpy(),
        server_info.cpu().numpy(),
        action.cpu().numpy(),
        env_action.cpu().numpy(),
        log_prob.cpu().numpy(),
        None if role_mu is None else role_mu.cpu().numpy(),
        None if role_sigma is None else role_sigma.cpu().numpy(),
    )


def _episode_role_metrics(policy: LoadedPolicy, buffer: RolloutBuffer) -> dict[str, float | None]:
    if policy.role_encoder is None or policy.trajectory_encoder is None or len(buffer) == 0:
        return {
            "mean_role_kl": None,
            "mean_role_nll": None,
        }

    batch = buffer.build_agent_trajectory_batch(
        window_size=policy.config.training.trajectory_window,
        obs_dim=policy.config.environment.actor_observation_dim,
        action_dim=policy.config.model.action_dim,
        action_scale=policy.config.training.trajectory_action_scale,
        device=policy.device,
    )
    with torch.no_grad():
        role_mu, role_std = policy.role_encoder(batch["current_obs"])
        traj_mu, traj_std = policy.trajectory_encoder(batch["trajectory"], batch["current_obs"])
        kl = diagonal_gaussian_kl(role_mu, role_std, traj_mu, traj_std)
        nll = _diagonal_gaussian_nll(role_mu, traj_mu, traj_std)
    return {
        "mean_role_kl": float(kl.mean().item()),
        "mean_role_nll": float(nll.mean().item()),
    }


def evaluate_policy(policy: LoadedPolicy, episodes: int, deterministic_policy: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = policy.env
    step_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    device_records: list[dict[str, float]] = []
    role_std_means: list[float] = []
    role_var_means: list[float] = []
    near_zero_sigma_fractions: list[float] = []
    role_action_corrs: list[float] = []

    for episode_idx in range(episodes):
        observation = env.reset()
        episode_joint_reward = 0.0
        episode_timeout_ratios: list[float] = []
        episode_costs: list[float] = []
        episode_edge_queues: list[float] = []
        episode_role_action_corrs: list[float] = []
        episode_near_zero_sigma_fractions: list[float] = []
        rollout_buffer = RolloutBuffer()

        for step_idx in range(policy.config.environment.episode_length):
            positions = env.positions.copy()
            if policy.runner_kind == "fixed":
                env_action, action_native, log_prob, role_sigma = _select_fixed_action(policy, episode_idx, step_idx)
                role_mu = np.zeros((policy.config.environment.num_agents, policy.config.model.role_dim), dtype=np.float32)
                scaled_core_obs = observation.device_obs.copy()
                scaled_server_info = observation.server_obs.copy()
                scaled_actor_obs = np.concatenate(
                    [
                        scaled_core_obs,
                        np.broadcast_to(
                            scaled_server_info[: policy.config.environment.actor_queue_broadcast_dim],
                            (policy.config.environment.num_agents, policy.config.environment.actor_queue_broadcast_dim),
                        ),
                    ],
                    axis=-1,
                ).astype(np.float32)
            else:
                (
                    scaled_actor_obs,
                    scaled_core_obs,
                    scaled_server_info,
                    action_native,
                    env_action,
                    log_prob,
                    role_mu,
                    role_sigma,
                ) = _select_learned_action(policy, observation, deterministic_policy=deterministic_policy)
                if role_mu is None:
                    role_mu = np.zeros((policy.config.environment.num_agents, policy.config.model.role_dim), dtype=np.float32)

            next_observation, reward, done, info = env.step(env_action)
            joint_reward = float(reward.sum())
            episode_joint_reward += joint_reward

            task_cost = env.last_reward_breakdown["task_normalized_cost"]
            task_delay = env.last_reward_breakdown["task_completion_delay_s"]
            timeout_mask = env.last_reward_breakdown["task_timeout_mask"]
            per_agent_timeout = timeout_mask.mean(axis=1).astype(np.float32)
            per_agent_offloading = env_action[:, :-1].mean(axis=1).astype(np.float32)

            if role_sigma is not None:
                role_std_means.append(float(np.mean(role_sigma)))
                role_var_means.append(float(np.mean(np.square(role_sigma))))
                near_zero_fraction = float(np.mean(role_sigma < policy.config.training.sigma_floor))
                near_zero_sigma_fractions.append(near_zero_fraction)
                episode_near_zero_sigma_fractions.append(near_zero_fraction)
            role_action_corr = _pairwise_distance_correlation(role_mu if role_mu is not None else None, env_action)
            if role_action_corr is not None:
                role_action_corrs.append(role_action_corr)
                episode_role_action_corrs.append(role_action_corr)

            rollout_buffer.add(
                Transition(
                    actor_obs=np.asarray(scaled_actor_obs, dtype=np.float32),
                    core_obs=np.asarray(scaled_core_obs, dtype=np.float32),
                    server_info=np.asarray(scaled_server_info, dtype=np.float32),
                    role_mu=np.asarray(role_mu, dtype=np.float32),
                    action=np.asarray(action_native, dtype=np.float32),
                    log_prob=np.asarray(log_prob, dtype=np.float32),
                    reward=reward.copy(),
                    done=done,
                    positions=positions,
                    joint_reward=joint_reward,
                )
            )

            step_records.append(
                {
                    "episode": episode_idx + 1,
                    "step": step_idx + 1,
                    "joint_reward": joint_reward,
                    "mean_device_reward": float(reward.mean()),
                    "timeout_ratio": float(info["timeout_ratio"]),
                    "edge_queue": float(info["edge_queue"]),
                    "delta_edge_queue": float(info["delta_edge_queue"]),
                    "mean_local_queue": float(env.local_queues.mean()),
                    "mean_task_processing_cost": float(task_cost.mean()),
                    "mean_task_completion_delay_s": float(task_delay.mean()),
                    "mean_distance_m": float(info["mean_distance_m"]),
                    "mean_cpu_ghz": float(env.cpu_freqs_ghz.mean()),
                    "mean_offloading_ratio": float(env_action[:, :-1].mean()),
                    "mean_power_ratio": float(env_action[:, -1].mean()),
                    "device_rewards": reward.astype(np.float32),
                    "device_distances_m": env.distances_m.astype(np.float32),
                    "device_cpu_ghz": env.cpu_freqs_ghz.astype(np.float32),
                    "device_local_queues": env.local_queues.astype(np.float32),
                    "device_timeout_ratio": per_agent_timeout,
                    "device_offloading_ratio": per_agent_offloading,
                    "power_ratio": env_action[:, -1].astype(np.float32),
                    "offloading_ratio_tasks": env_action[:, :-1].astype(np.float32),
                    "role_mu": None if role_mu is None else np.asarray(role_mu, dtype=np.float32),
                    "role_sigma": None if role_sigma is None else np.asarray(role_sigma, dtype=np.float32),
                }
            )

            for agent_idx in range(policy.config.environment.num_agents):
                device_records.append(
                    {
                        "distance_m": float(env.distances_m[agent_idx]),
                        "cpu_ghz": float(env.cpu_freqs_ghz[agent_idx]),
                        "offloading_ratio": float(per_agent_offloading[agent_idx]),
                        "power_ratio": float(env_action[agent_idx, -1]),
                        "timeout_ratio": float(per_agent_timeout[agent_idx]),
                    }
                )

            episode_timeout_ratios.append(float(info["timeout_ratio"]))
            episode_costs.append(float(task_cost.mean()))
            episode_edge_queues.append(float(info["edge_queue"]))

            observation = next_observation
            if done:
                break

        role_metrics = _episode_role_metrics(policy, rollout_buffer)
        episode_records.append(
            {
                "episode": episode_idx + 1,
                "joint_reward": episode_joint_reward,
                "steps": len([record for record in step_records if record["episode"] == episode_idx + 1]),
                "mean_timeout_ratio": float(np.mean(episode_timeout_ratios)) if episode_timeout_ratios else 0.0,
                "mean_task_processing_cost": float(np.mean(episode_costs)) if episode_costs else 0.0,
                "mean_edge_queue": float(np.mean(episode_edge_queues)) if episode_edge_queues else 0.0,
                "mean_role_action_distance_correlation": _mean_or_none(episode_role_action_corrs),
                "mean_near_zero_sigma_fraction": _mean_or_none(episode_near_zero_sigma_fractions),
                **role_metrics,
            }
        )

    results_dir = policy.config.output_root / "results"
    summary = summarize_evaluation(
        policy=policy,
        step_records=step_records,
        episode_records=episode_records,
        device_records=device_records,
        results_dir=results_dir,
        trace_path=results_dir / "unused.jsonl",
    )
    summary["metrics"]["mean_role_action_distance_correlation"] = _mean_or_none(role_action_corrs)
    summary["metrics"]["mean_role_std"] = _mean_or_none(role_std_means)
    summary["metrics"]["mean_role_variance"] = _mean_or_none(role_var_means)
    summary["metrics"]["mean_near_zero_sigma_fraction"] = _mean_or_none(near_zero_sigma_fractions)
    return summary, step_records


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    policy = load_policy_from_args(args)
    results_dir = policy.config.output_root / "results" if args.results_dir is None else args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    label = policy.label if args.trace_label is None else args.trace_label
    trace_path = results_dir / f"evaluation_{label}_trace.jsonl"
    summary_path = results_dir / f"evaluation_{label}_summary.json"

    summary, step_records = evaluate_policy(policy, episodes=args.episodes, deterministic_policy=args.deterministic_policy)
    summary["trace_path"] = str(trace_path)
    summary["results_dir"] = str(results_dir)
    summary["deterministic_policy"] = bool(args.deterministic_policy)

    _write_jsonl(trace_path, step_records)
    _write_json(summary_path, summary)
    return summary, summary_path, trace_path


def main() -> None:
    args = build_parser().parse_args()
    summary, summary_path, trace_path = run_evaluation(args)
    metrics = summary["metrics"]
    variant_label = summary["variant_id"] if summary["variant_id"] is not None else summary["label"]
    print(
        f"evaluation variant={variant_label} episodes={summary['episodes']} "
        f"mean_episode_joint_reward={metrics['mean_episode_joint_reward']:.4f} "
        f"timeout_ratio={metrics['mean_timeout_ratio']:.4f} "
        f"mean_task_cost={metrics['mean_task_processing_cost']:.4f} "
        f"summary={summary_path} trace={trace_path}"
    )


if __name__ == "__main__":
    main()
