from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass(slots=True)
class EnvironmentConfig:
    num_agents: int = 5
    episode_length: int = 200
    num_tasks_per_step: int = 3
    use_mobility: bool = True
    use_cpu_dynamics: bool = True
    graph_type: str = "star"
    distance_threshold_m: float = 150.0
    min_distance_m: float = 50.0
    max_distance_m: float = 250.0
    dt: float = 0.5
    sigma_v: float = 0.5
    path_loss_exp: float = 3.76
    path_loss_kappa_linear: float = 10 ** (-128.1 / 10)
    total_bandwidth_hz: float = 10e6
    noise_density_dbm_hz: float = -174.0
    server_cpu_ghz: float = 25.0
    max_cpu_ghz: float = 3.0
    max_tx_power_mw: float = 300.0
    task_size_range_mb: tuple[float, float] = (0.16, 1.6)
    task_density_range_gcycles_per_mb: tuple[float, float] = (0.2, 2.0)
    task_deadline_range_s: tuple[float, float] = (0.2, 1.0)
    reward_timeout_penalty: float = 5000.0
    reward_scale: float = 1000.0
    delay_weight: float = 0.5
    energy_weight: float = 0.5
    local_energy_coeff_j_per_gcycle: float = 0.05
    min_rate_bps: float = 1.0
    queue_clip_max: float = 20.0
    observation_dim: int = 14
    central_observation_dim: int = 3

    @property
    def noise_density_w_hz(self) -> float:
        return 10 ** ((self.noise_density_dbm_hz - 30.0) / 10.0)

    @property
    def max_task_work_gcycles(self) -> float:
        return self.task_size_range_mb[1] * self.task_density_range_gcycles_per_mb[1]

    @property
    def local_energy_reference_j(self) -> float:
        return self.local_energy_coeff_j_per_gcycle * self.max_task_work_gcycles

    @property
    def tx_energy_reference_j(self) -> float:
        return (self.max_tx_power_mw / 1000.0) * self.dt


@dataclass(slots=True)
class ModelConfig:
    critic_type: str = "pgcn"
    use_role: bool = True
    use_l_i: bool = True
    actor_type: str = "shared"
    role_dim: int = 3
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    role_hidden_dim: int = 12
    trajectory_hidden_dim: int = 64
    action_dim: int = 4


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 4e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.1
    entropy_coeff: float = 0.01
    l_i_coeff: float = 1e-4
    gradient_clip: float = 2.0
    update_every_episodes: int = 4
    ppo_epochs: int = 4
    batch_size: int = 800
    total_episodes: int = 4000
    smoke_steps: int = 8
    trajectory_window: int = 20
    trajectory_action_scale: float = 10.0


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 42
    output_root: Path = Path(".")
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def ensure_output_dirs(self) -> None:
        for directory in ("logs", "models", "results"):
            (self.output_root / directory).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic heterogeneous MEC offloading scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("."))

    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--graph-type", choices=("star", "star_proximity"), default="star")
    parser.add_argument("--distance-threshold-m", type=float, default=150.0)
    parser.add_argument("--use-mobility", type=_str_to_bool, default=True)
    parser.add_argument("--use-cpu-dynamics", type=_str_to_bool, default=True)

    parser.add_argument("--critic-type", choices=("pgcn", "mlp", "set"), default="pgcn")
    parser.add_argument("--use-role", type=_str_to_bool, default=True)
    parser.add_argument("--use-l-i", type=_str_to_bool, default=True)
    parser.add_argument("--actor-type", choices=("shared", "individual"), default="shared")
    parser.add_argument("--role-dim", type=int, default=3)

    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--trajectory-window", type=int, default=20)
    parser.add_argument("--trajectory-action-scale", type=float, default=10.0)
    return parser


def build_config_from_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    args = build_parser().parse_args(argv)
    env = EnvironmentConfig(
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        graph_type=args.graph_type,
        distance_threshold_m=args.distance_threshold_m,
        use_mobility=args.use_mobility,
        use_cpu_dynamics=args.use_cpu_dynamics,
    )
    model = ModelConfig(
        critic_type=args.critic_type,
        use_role=args.use_role,
        use_l_i=args.use_l_i,
        actor_type=args.actor_type,
        role_dim=args.role_dim,
    )
    training = TrainingConfig(
        learning_rate=args.learning_rate,
        smoke_steps=args.smoke_steps,
        trajectory_window=args.trajectory_window,
        trajectory_action_scale=args.trajectory_action_scale,
    )
    return ExperimentConfig(
        seed=args.seed,
        output_root=args.output_root,
        environment=env,
        model=model,
        training=training,
    )
