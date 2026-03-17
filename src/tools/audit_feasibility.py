from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EnvironmentConfig, ExperimentConfig, _str_to_bool
from src.environment import DynamicMECEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit optimistic best-case task feasibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--use-mobility", type=_str_to_bool, default=True)
    parser.add_argument("--use-cpu-dynamics", type=_str_to_bool, default=True)
    parser.add_argument("--delay-mode", choices=("li_original", "bestcase_slack"), default="bestcase_slack")
    parser.add_argument("--u-slack", type=float, default=1.5)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--resets-per-seed", type=int, default=3)
    return parser


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


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


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    environment = EnvironmentConfig(
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        use_mobility=args.use_mobility,
        use_cpu_dynamics=args.use_cpu_dynamics,
        delay_mode=args.delay_mode,
        u_slack=args.u_slack,
    )
    return ExperimentConfig(
        seed=args.seed,
        output_root=args.output_root,
        run_feasibility_audit=True,
        environment=environment,
    )


def run_feasibility_audit(config: ExperimentConfig, num_seeds: int = 5, resets_per_seed: int = 3) -> dict[str, Any]:
    config.ensure_output_dirs()
    seeds = [config.seed + offset for offset in range(num_seeds)]
    ratio_values: list[float] = []
    threshold_values: list[float] = []
    best_values: list[float] = []
    infeasible_count = 0
    total_tasks = 0
    per_type: dict[str, dict[str, list[float] | int]] = {}

    for seed in seeds:
        env = DynamicMECEnv(config.environment, seed=seed)
        for _ in range(resets_per_seed):
            env.reset()
            components = env.compute_best_case_delay_components()
            d_best = components["d_best_s"]
            d_c = env.task_deadlines_s
            infeasible_mask = d_c + 1e-8 < d_best
            ratios = d_c / np.maximum(d_best, 1e-8)

            infeasible_count += int(infeasible_mask.sum())
            total_tasks += int(d_c.size)
            ratio_values.extend(ratios.reshape(-1).tolist())
            threshold_values.extend(d_c.reshape(-1).tolist())
            best_values.extend(d_best.reshape(-1).tolist())

            for device_idx, profile in enumerate(env.device_profiles):
                entry = per_type.setdefault(
                    profile.type_name,
                    {
                        "count": 0,
                        "infeasible_count": 0,
                        "ratio_values": [],
                        "threshold_values": [],
                        "best_values": [],
                    },
                )
                entry["count"] += int(d_c.shape[1])
                entry["infeasible_count"] += int(infeasible_mask[device_idx].sum())
                entry["ratio_values"].extend(ratios[device_idx].tolist())
                entry["threshold_values"].extend(d_c[device_idx].tolist())
                entry["best_values"].extend(d_best[device_idx].tolist())

    per_type_summary = {}
    for type_name, entry in per_type.items():
        per_type_summary[type_name] = {
            "count": int(entry["count"]),
            "infeasible_count": int(entry["infeasible_count"]),
            "infeasible_rate": float(entry["infeasible_count"]) / max(int(entry["count"]), 1),
            "ratio_summary": _summary_stats(entry["ratio_values"]),
            "threshold_summary": _summary_stats(entry["threshold_values"]),
            "best_delay_summary": _summary_stats(entry["best_values"]),
        }

    summary = {
        "seed": config.seed,
        "num_seeds": num_seeds,
        "resets_per_seed": resets_per_seed,
        "delay_mode": config.environment.delay_mode,
        "u_slack": config.environment.u_slack,
        "total_tasks": total_tasks,
        "infeasible_count": infeasible_count,
        "infeasible_rate": float(infeasible_count) / max(total_tasks, 1),
        "ratio_summary": _summary_stats(ratio_values),
        "threshold_summary": _summary_stats(threshold_values),
        "best_delay_summary": _summary_stats(best_values),
        "per_type": per_type_summary,
    }
    output_path = config.output_root / "results" / "feasibility_audit_summary.json"
    output_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    summary["output_path"] = str(output_path)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    config = build_experiment_config(args)
    summary = run_feasibility_audit(config, num_seeds=args.num_seeds, resets_per_seed=args.resets_per_seed)
    print(
        f"feasibility_audit delay_mode={summary['delay_mode']} total_tasks={summary['total_tasks']} "
        f"infeasible_count={summary['infeasible_count']} infeasible_rate={summary['infeasible_rate']:.6f} "
        f"ratio_mean={summary['ratio_summary']['mean']:.4f} output={summary['output_path']}"
    )


if __name__ == "__main__":
    main()
