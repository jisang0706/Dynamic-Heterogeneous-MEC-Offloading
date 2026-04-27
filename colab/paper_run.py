from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import get_experiment_variant
from src.config import ProtocolStageSpec, get_protocol_stage


DEFAULT_STAGE_TRAIN_EPISODES = {
    "smoke": 100,
    "core": 4000,
    "scale": 4000,
}

DEFAULT_STAGE_EVAL_EPISODES = {
    "smoke": 20,
    "core": 50,
    "scale": 50,
}

DEFAULT_CHECKPOINT_MILESTONES = {
    "smoke": (100,),
    "core": (100, 200, 400, 600, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600),
    "scale": (100, 200, 400, 600, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600),
}

DEFAULT_SEED_POOL = (42, 43, 44, 45, 46)


@dataclass(slots=True)
class CheckpointInfo:
    path: Path
    episodes_completed: int
    update_index: int
    selection_rule: str


@dataclass(slots=True)
class RunSpec:
    stage_id: str
    variant_id: str
    runner_kind: str
    num_agents: int
    seed: int
    episode_length: int
    train_episodes: int
    eval_episodes: int
    output_root: Path


@dataclass(frozen=True, slots=True)
class ScaleRunProfile:
    profile_id: str
    resource_scaling_mode: str
    total_bandwidth_hz: float
    server_cpu_ghz: float
    u_slack: float
    initial_offloading_mean_env: float
    initial_power_mean_env: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab paper-grade training runner with auto-resume.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "core", "scale", "all"), default="core")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--num-agents", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--delay-mode", choices=("li_original", "bestcase_slack"), default="bestcase_slack")
    parser.add_argument("--update-every-episodes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.07)
    parser.add_argument("--entropy-coeff", type=float, default=2e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--local-reward-weight", type=float, default=0.6)
    parser.add_argument("--shared-congestion-delta-coeff", type=float, default=50.0)
    parser.add_argument("--shared-congestion-queue-coeff", type=float, default=10.0)
    parser.add_argument("--shared-congestion-delta-reference", type=float, default=10.0)
    parser.add_argument("--shared-congestion-queue-reference", type=float, default=20.0)
    parser.add_argument("--l-i-coeff", type=float, default=5e-5)
    parser.add_argument("--l-i-warmup-updates", type=int, default=100)
    parser.add_argument("--monotonic-offloading-coeff", type=float, default=1e-3)
    parser.add_argument("--monotonic-offloading-coeff-final", type=float, default=5e-4)
    parser.add_argument("--monotonic-decay-start-fraction", type=float, default=0.6)
    parser.add_argument("--monotonic-decay-end-fraction", type=float, default=1.0)
    parser.add_argument("--monotonic-queue-reference", type=float, default=20.0)
    parser.add_argument("--lambda-var", type=float, default=1e-5)
    parser.add_argument("--sigma-floor", type=float, default=0.05)
    parser.add_argument("--initial-action-std-env", type=float, default=0.15)
    parser.add_argument("--initial-offloading-mean-env", type=float, default=0.70)
    parser.add_argument("--initial-power-mean-env", type=float, default=0.8)
    parser.add_argument(
        "--large-scale-profile",
        choices=("default", "matched_resource", "paper_scale_v1", "paper_scale_v2"),
        default="paper_scale_v2",
    )
    parser.add_argument("--use-obs-scaling", choices=("true", "false"), default="true")
    parser.add_argument("--use-reward-scaling", choices=("true", "false"), default="true")
    parser.add_argument("--resource-scaling-mode", choices=("fixed", "linear_after_threshold"), default="linear_after_threshold")
    parser.add_argument("--resource-scaling-base-agents", type=int, default=5)
    parser.add_argument("--resource-scaling-start-agents", type=int, default=10)
    parser.add_argument("--save-every-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-selection-mode", choices=("final_only", "milestone_best"), default="milestone_best")
    parser.add_argument("--checkpoint-milestones", nargs="*", type=int, default=None)
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    return parser


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _torch_load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def resolve_scale_run_profile(spec: RunSpec, args: argparse.Namespace) -> ScaleRunProfile:
    # Keep the environment definition shared across methods and only use mild
    # profile tuning to avoid the worst timeout-saturation regime at larger M.
    base_profile = ScaleRunProfile(
        profile_id="matched_resource" if args.large_scale_profile == "matched_resource" else "default",
        resource_scaling_mode=args.resource_scaling_mode,
        total_bandwidth_hz=10e6,
        server_cpu_ghz=25.0,
        u_slack=1.8,
        initial_offloading_mean_env=args.initial_offloading_mean_env,
        initial_power_mean_env=args.initial_power_mean_env,
    )
    if args.large_scale_profile in {"default", "matched_resource"}:
        return base_profile

    if args.large_scale_profile == "paper_scale_v1":
        tuned_profiles = {
            10: ScaleRunProfile(profile_id="paper_scale_v1", resource_scaling_mode="fixed", total_bandwidth_hz=20e6, server_cpu_ghz=60.0, u_slack=2.2, initial_offloading_mean_env=0.60, initial_power_mean_env=args.initial_power_mean_env),
            15: ScaleRunProfile(profile_id="paper_scale_v1", resource_scaling_mode="fixed", total_bandwidth_hz=30e6, server_cpu_ghz=90.0, u_slack=2.4, initial_offloading_mean_env=0.58, initial_power_mean_env=args.initial_power_mean_env),
            20: ScaleRunProfile(profile_id="paper_scale_v1", resource_scaling_mode="fixed", total_bandwidth_hz=40e6, server_cpu_ghz=120.0, u_slack=2.6, initial_offloading_mean_env=0.55, initial_power_mean_env=args.initial_power_mean_env),
        }
        return tuned_profiles.get(spec.num_agents, base_profile)

    tuned_profiles = {
        10: ScaleRunProfile(profile_id="paper_scale_v2", resource_scaling_mode="fixed", total_bandwidth_hz=20e6, server_cpu_ghz=70.0, u_slack=2.4, initial_offloading_mean_env=0.55, initial_power_mean_env=0.90),
        15: ScaleRunProfile(profile_id="paper_scale_v2", resource_scaling_mode="fixed", total_bandwidth_hz=30e6, server_cpu_ghz=105.0, u_slack=2.6, initial_offloading_mean_env=0.52, initial_power_mean_env=0.92),
        20: ScaleRunProfile(profile_id="paper_scale_v2", resource_scaling_mode="fixed", total_bandwidth_hz=40e6, server_cpu_ghz=140.0, u_slack=2.8, initial_offloading_mean_env=0.50, initial_power_mean_env=0.95),
    }
    return tuned_profiles.get(spec.num_agents, base_profile)


def _checkpoint_priority(path: Path) -> int:
    if path.stem.endswith("final"):
        return 2
    if path.stem.endswith("latest"):
        return 1
    return 0


def _checkpoint_selection_rule(path: Path) -> str:
    if path.stem.endswith("final"):
        return "final_checkpoint"
    if path.stem.endswith("latest"):
        return "latest_checkpoint"
    return "periodic_checkpoint"


def _checkpoint_prefix(runner_kind: str) -> str:
    return "ippo_checkpoint" if runner_kind == "ippo" else "checkpoint"


def discover_checkpoint_candidates(output_root: Path, runner_kind: str) -> list[CheckpointInfo]:
    models_dir = output_root / "models"
    if not models_dir.exists():
        return []

    prefix = _checkpoint_prefix(runner_kind)
    candidate_paths: list[Path] = []
    for suffix in ("final", "latest"):
        path = models_dir / f"{prefix}_{suffix}.pt"
        if path.exists():
            candidate_paths.append(path)
    candidate_paths.extend(sorted(models_dir.glob(f"{prefix}_ep*_u*.pt")))
    if not candidate_paths:
        return []

    seen_paths: set[Path] = set()
    candidates: list[CheckpointInfo] = []
    for checkpoint_path in candidate_paths:
        if checkpoint_path in seen_paths:
            continue
        seen_paths.add(checkpoint_path)
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        candidates.append(
            CheckpointInfo(
                path=checkpoint_path,
                episodes_completed=int(checkpoint.get("episodes_completed", 0)),
                update_index=int(checkpoint.get("update_index", 0)),
                selection_rule=_checkpoint_selection_rule(checkpoint_path),
            )
        )
    return candidates


def discover_best_checkpoint(output_root: Path, runner_kind: str) -> CheckpointInfo | None:
    candidates = discover_checkpoint_candidates(output_root, runner_kind)
    if not candidates:
        return None

    best: CheckpointInfo | None = None
    for info in candidates:
        if best is None:
            best = info
            continue
        best_key = (best.episodes_completed, best.update_index, _checkpoint_priority(best.path))
        info_key = (info.episodes_completed, info.update_index, _checkpoint_priority(info.path))
        if info_key > best_key:
            best = info
    return best


def expected_final_checkpoint(output_root: Path, runner_kind: str) -> Path:
    return output_root / "models" / f"{_checkpoint_prefix(runner_kind)}_final.pt"


def expected_selected_summary(output_root: Path) -> tuple[Path, Path]:
    results_dir = output_root / "results"
    return (
        results_dir / "evaluation_selected_summary.json",
        results_dir / "evaluation_selected_trace.jsonl",
    )


def default_seeds(stage_spec: ProtocolStageSpec) -> list[int]:
    seed_count = stage_spec.recommended_seed_count[0]
    return list(DEFAULT_SEED_POOL[:seed_count])


def resolve_checkpoint_targets(spec: RunSpec, args: argparse.Namespace) -> list[int]:
    configured = args.checkpoint_milestones
    if configured:
        milestones = [int(item) for item in configured]
    else:
        milestones = list(DEFAULT_CHECKPOINT_MILESTONES.get(spec.stage_id, ()))
    filtered = [episode for episode in milestones if 0 < episode < spec.train_episodes]
    filtered.append(spec.train_episodes)
    return sorted(set(filtered))


def select_checkpoint_candidates(candidates: list[CheckpointInfo], target_episodes: list[int]) -> list[CheckpointInfo]:
    if not candidates:
        return []

    best_by_episode: dict[int, CheckpointInfo] = {}
    for info in candidates:
        current = best_by_episode.get(info.episodes_completed)
        if current is None:
            best_by_episode[info.episodes_completed] = info
            continue
        current_key = (current.update_index, _checkpoint_priority(current.path))
        info_key = (info.update_index, _checkpoint_priority(info.path))
        if info_key > current_key:
            best_by_episode[info.episodes_completed] = info

    available_episodes = sorted(best_by_episode.keys())
    selected: list[CheckpointInfo] = []
    seen_paths: set[Path] = set()
    for target in target_episodes:
        eligible = [episode for episode in available_episodes if episode <= target]
        chosen_episode = eligible[-1] if eligible else available_episodes[0]
        chosen = best_by_episode[chosen_episode]
        if chosen.path in seen_paths:
            continue
        selected.append(chosen)
        seen_paths.add(chosen.path)
    return selected


def _summary_sort_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    metrics = summary["metrics"]
    return (
        float(metrics["mean_timeout_ratio"]),
        float(metrics["mean_task_processing_cost"]),
        -float(metrics["mean_episode_joint_reward"]),
    )


def _candidate_trace_label(checkpoint: CheckpointInfo) -> str:
    return checkpoint.path.stem


def _candidate_summary_path(results_dir: Path, checkpoint: CheckpointInfo) -> Path:
    return results_dir / f"evaluation_{_candidate_trace_label(checkpoint)}_summary.json"


def _candidate_trace_path(results_dir: Path, checkpoint: CheckpointInfo) -> Path:
    return results_dir / f"evaluation_{_candidate_trace_label(checkpoint)}_trace.jsonl"


def _write_selected_outputs(
    output_root: Path,
    summary_path: Path,
    trace_path: Path,
    checkpoint: CheckpointInfo,
    target_episodes: list[int],
) -> tuple[Path, Path]:
    selected_summary_path, selected_trace_path = expected_selected_summary(output_root)
    selected_summary = _load_summary(summary_path)
    selected_summary["trace_path"] = str(selected_trace_path)
    selected_summary["results_dir"] = str(output_root / "results")
    selected_summary["runner_selection"] = {
        "mode": "milestone_best",
        "target_episodes": target_episodes,
        "selected_checkpoint_path": str(checkpoint.path),
        "selected_checkpoint_selection_rule": checkpoint.selection_rule,
        "selected_checkpoint_episodes_completed": checkpoint.episodes_completed,
        "selected_checkpoint_update_index": checkpoint.update_index,
        "source_summary_path": str(summary_path),
        "source_trace_path": str(trace_path),
    }
    selected_summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(selected_summary_path, selected_summary)
    shutil.copy2(trace_path, selected_trace_path)
    return selected_summary_path, selected_trace_path


def _selected_summary_is_fresh(
    summary_path: Path,
    checkpoints: list[CheckpointInfo],
    target_episodes: list[int],
) -> bool:
    if not summary_path.exists() or not checkpoints:
        return False
    try:
        summary = _load_summary(summary_path)
    except json.JSONDecodeError:
        return False
    selection = summary.get("runner_selection", {})
    if selection.get("mode") != "milestone_best":
        return False
    if list(selection.get("target_episodes", [])) != target_episodes:
        return False
    latest_checkpoint_mtime = max(checkpoint.path.stat().st_mtime for checkpoint in checkpoints)
    return summary_path.stat().st_mtime >= latest_checkpoint_mtime


def stage_run_specs(stage_id: str, args: argparse.Namespace) -> tuple[ProtocolStageSpec, list[RunSpec]]:
    stage_spec = get_protocol_stage(stage_id)
    if stage_spec is None:
        raise SystemExit(f"Unknown protocol stage: {stage_id}")

    variants = args.variants or list(stage_spec.recommended_methods)
    agent_counts = args.num_agents or list(stage_spec.recommended_num_agents)
    seeds = args.seeds or default_seeds(stage_spec)
    train_episodes = args.train_episodes or DEFAULT_STAGE_TRAIN_EPISODES[stage_id]
    eval_episodes = args.eval_episodes or DEFAULT_STAGE_EVAL_EPISODES[stage_id]

    specs: list[RunSpec] = []
    stage_root = args.workspace_root / stage_id
    for num_agents in agent_counts:
        for seed in seeds:
            for variant_name in variants:
                variant = get_experiment_variant(variant_name)
                if variant is None:
                    raise SystemExit(f"Unknown variant id: {variant_name}")
                if variant.runner_kind in {"external", "unsupported"}:
                    raise SystemExit(f"Variant {variant.variant_id} is not supported by the Colab paper runner.")
                output_root = stage_root / f"m{num_agents:02d}" / f"seed_{seed}" / variant.variant_id.lower()
                specs.append(
                    RunSpec(
                        stage_id=stage_id,
                        variant_id=variant.variant_id,
                        runner_kind=variant.runner_kind,
                        num_agents=num_agents,
                        seed=seed,
                        episode_length=args.episode_length,
                        train_episodes=train_episodes,
                        eval_episodes=eval_episodes,
                        output_root=output_root,
                    )
                )
    return stage_spec, specs


def _run_command(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _train_command(spec: RunSpec, args: argparse.Namespace, resume_from: Path | None) -> list[str]:
    scale_profile = resolve_scale_run_profile(spec, args)
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--output-root",
        str(spec.output_root),
        "--run-mode",
        "train",
        "--variant-id",
        spec.variant_id,
        "--seed",
        str(spec.seed),
        "--delay-mode",
        args.delay_mode,
        "--num-agents",
        str(spec.num_agents),
        "--episode-length",
        str(spec.episode_length),
        "--total-episodes",
        str(spec.train_episodes),
        "--update-every-episodes",
        str(args.update_every_episodes),
        "--batch-size",
        str(args.batch_size),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--actor-learning-rate",
        str(args.actor_learning_rate),
        "--ppo-clip",
        str(args.ppo_clip),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--local-reward-weight",
        str(args.local_reward_weight),
        "--shared-congestion-delta-coeff",
        str(args.shared_congestion_delta_coeff),
        "--shared-congestion-queue-coeff",
        str(args.shared_congestion_queue_coeff),
        "--shared-congestion-delta-reference",
        str(args.shared_congestion_delta_reference),
        "--shared-congestion-queue-reference",
        str(args.shared_congestion_queue_reference),
        "--gradient-clip",
        str(args.gradient_clip),
        "--l-i-coeff",
        str(args.l_i_coeff),
        "--l-i-warmup-updates",
        str(args.l_i_warmup_updates),
        "--monotonic-offloading-coeff",
        str(args.monotonic_offloading_coeff),
        "--monotonic-offloading-coeff-final",
        str(args.monotonic_offloading_coeff_final),
        "--monotonic-decay-start-fraction",
        str(args.monotonic_decay_start_fraction),
        "--monotonic-decay-end-fraction",
        str(args.monotonic_decay_end_fraction),
        "--monotonic-queue-reference",
        str(args.monotonic_queue_reference),
        "--lambda-var",
        str(args.lambda_var),
        "--sigma-floor",
        str(args.sigma_floor),
        "--u-slack",
        str(scale_profile.u_slack),
        "--total-bandwidth-hz",
        str(scale_profile.total_bandwidth_hz),
        "--server-cpu-ghz",
        str(scale_profile.server_cpu_ghz),
        "--initial-action-std-env",
        str(args.initial_action_std_env),
        "--initial-offloading-mean-env",
        str(scale_profile.initial_offloading_mean_env),
        "--initial-power-mean-env",
        str(scale_profile.initial_power_mean_env),
        "--use-obs-scaling",
        args.use_obs_scaling,
        "--use-reward-scaling",
        args.use_reward_scaling,
        "--resource-scaling-mode",
        scale_profile.resource_scaling_mode,
        "--resource-scaling-base-agents",
        str(args.resource_scaling_base_agents),
        "--resource-scaling-start-agents",
        str(args.resource_scaling_start_agents),
        "--save-every-episodes",
        str(args.save_every_episodes),
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    return command


def _evaluate_command(
    spec: RunSpec,
    args: argparse.Namespace,
    checkpoint_path: Path | None,
    checkpoint_selection_rule: str,
    results_dir: Path | None = None,
    trace_label: str | None = None,
) -> list[str]:
    scale_profile = resolve_scale_run_profile(spec, args)
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--output-root",
        str(spec.output_root),
        "--episodes",
        str(spec.eval_episodes),
        "--delay-mode",
        args.delay_mode,
        "--protocol-stage",
        spec.stage_id,
        "--checkpoint-selection-rule",
        checkpoint_selection_rule,
        "--u-slack",
        str(scale_profile.u_slack),
        "--total-bandwidth-hz",
        str(scale_profile.total_bandwidth_hz),
        "--server-cpu-ghz",
        str(scale_profile.server_cpu_ghz),
        "--resource-scaling-mode",
        scale_profile.resource_scaling_mode,
        "--resource-scaling-base-agents",
        str(args.resource_scaling_base_agents),
        "--resource-scaling-start-agents",
        str(args.resource_scaling_start_agents),
    ]
    if checkpoint_path is not None:
        command.extend(["--checkpoint", str(checkpoint_path)])
    else:
        command.extend(
            [
                "--variant-id",
                spec.variant_id,
                "--seed",
                str(spec.seed),
                "--num-agents",
                str(spec.num_agents),
                "--episode-length",
                str(spec.episode_length),
            ]
        )
    if results_dir is not None:
        command.extend(["--results-dir", str(results_dir)])
    if trace_label is not None:
        command.extend(["--trace-label", trace_label])
    return command


def discover_summary_file(spec: RunSpec) -> Path | None:
    results_dir = spec.output_root / "results"
    if not results_dir.exists():
        return None
    expected_selected = results_dir / "evaluation_selected_summary.json"
    if expected_selected.exists():
        return expected_selected
    if spec.runner_kind == "fixed":
        expected = results_dir / f"evaluation_{spec.variant_id.lower()}_summary.json"
        if expected.exists():
            return expected
    expected_final = results_dir / "evaluation_checkpoint_final_summary.json"
    if expected_final.exists():
        return expected_final
    candidates = sorted(results_dir.glob("evaluation_*_summary.json"))
    return None if not candidates else candidates[-1]


def ensure_training(spec: RunSpec, args: argparse.Namespace, repo_root: Path) -> tuple[Path | None, str, CheckpointInfo | None]:
    if spec.runner_kind == "fixed":
        return None, "fixed_policy", None

    best_checkpoint = discover_best_checkpoint(spec.output_root, spec.runner_kind)
    final_checkpoint = expected_final_checkpoint(spec.output_root, spec.runner_kind)
    completed_episodes = 0 if best_checkpoint is None else best_checkpoint.episodes_completed
    if (
        not args.force_train
        and best_checkpoint is not None
        and completed_episodes >= spec.train_episodes
        and final_checkpoint.exists()
    ):
        print(
            f"[skip-train] {spec.stage_id} {spec.variant_id} m={spec.num_agents} seed={spec.seed} "
            f"already reached {completed_episodes} episodes."
        )
        return final_checkpoint, "final_checkpoint", best_checkpoint

    resume_from = None if args.force_train or best_checkpoint is None else best_checkpoint.path
    _run_command(_train_command(spec, args, resume_from=resume_from), cwd=repo_root)
    refreshed_checkpoint = discover_best_checkpoint(spec.output_root, spec.runner_kind)
    if refreshed_checkpoint is None:
        raise SystemExit(f"Training completed but no checkpoint was found for {spec.output_root}")
    resolved_checkpoint = final_checkpoint if final_checkpoint.exists() else refreshed_checkpoint.path
    resolved_rule = "final_checkpoint" if final_checkpoint.exists() else refreshed_checkpoint.selection_rule
    return resolved_checkpoint, resolved_rule, refreshed_checkpoint


def ensure_evaluation(
    spec: RunSpec,
    args: argparse.Namespace,
    repo_root: Path,
    checkpoint_path: Path | None,
    checkpoint_selection_rule: str,
) -> tuple[Path, Path | None, str]:
    if spec.runner_kind != "fixed" and args.checkpoint_selection_mode == "milestone_best":
        all_candidates = discover_checkpoint_candidates(spec.output_root, spec.runner_kind)
        target_episodes = resolve_checkpoint_targets(spec, args)
        selected_candidates = select_checkpoint_candidates(all_candidates, target_episodes)
        if not selected_candidates:
            raise SystemExit(f"No candidate checkpoints available for milestone selection in {spec.output_root}")

        selected_summary_path, _ = expected_selected_summary(spec.output_root)
        if _selected_summary_is_fresh(selected_summary_path, selected_candidates, target_episodes) and not args.force_evaluate:
            print(f"[skip-eval] {spec.stage_id} {spec.variant_id} m={spec.num_agents} seed={spec.seed}")
            selected_summary = _load_summary(selected_summary_path)
            selection = selected_summary["runner_selection"]
            return (
                selected_summary_path,
                Path(selection["selected_checkpoint_path"]),
                str(selection["selected_checkpoint_selection_rule"]),
            )

        sweep_dir = spec.output_root / "results" / "checkpoint_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        evaluated: list[tuple[CheckpointInfo, Path, Path, dict[str, Any]]] = []
        for candidate in selected_candidates:
            label = _candidate_trace_label(candidate)
            candidate_summary_path = _candidate_summary_path(sweep_dir, candidate)
            summary_is_fresh = (
                candidate_summary_path.exists()
                and candidate_summary_path.stat().st_mtime >= candidate.path.stat().st_mtime
            )
            if not summary_is_fresh or args.force_evaluate:
                _run_command(
                    _evaluate_command(
                        spec,
                        args,
                        checkpoint_path=candidate.path,
                        checkpoint_selection_rule=candidate.selection_rule,
                        results_dir=sweep_dir,
                        trace_label=label,
                    ),
                    cwd=repo_root,
                )
            summary = _load_summary(candidate_summary_path)
            evaluated.append((candidate, candidate_summary_path, _candidate_trace_path(sweep_dir, candidate), summary))

        best_candidate, best_summary_path, best_trace_path, _ = min(
            evaluated,
            key=lambda item: _summary_sort_key(item[3]),
        )
        canonical_summary_path, _ = _write_selected_outputs(
            spec.output_root,
            best_summary_path,
            best_trace_path,
            best_candidate,
            target_episodes,
        )
        return canonical_summary_path, best_candidate.path, best_candidate.selection_rule

    summary_path = discover_summary_file(spec)
    if spec.runner_kind == "fixed":
        summary_is_fresh = summary_path is not None and summary_path.exists()
    else:
        summary_is_fresh = (
            summary_path is not None
            and checkpoint_path is not None
            and summary_path.stat().st_mtime >= checkpoint_path.stat().st_mtime
        )
    if summary_is_fresh and not args.force_evaluate:
        print(f"[skip-eval] {spec.stage_id} {spec.variant_id} m={spec.num_agents} seed={spec.seed}")
        return summary_path, checkpoint_path, checkpoint_selection_rule

    _run_command(
        _evaluate_command(
            spec,
            args,
            checkpoint_path=checkpoint_path,
            checkpoint_selection_rule=checkpoint_selection_rule,
        ),
        cwd=repo_root,
    )
    refreshed_summary = discover_summary_file(spec)
    if refreshed_summary is None:
        raise SystemExit(f"Evaluation completed but no summary file was found for {spec.output_root}")
    return refreshed_summary, checkpoint_path, checkpoint_selection_rule


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_stage_visualizations(stage_root: Path, stage_id: str, repo_root: Path, summary_files: list[Path]) -> list[str]:
    summary_files = sorted(path for path in summary_files if path.exists())
    if not summary_files:
        return []

    generated_roots: list[str] = []
    overall_root = stage_root / "compare" / "overall"
    overall_plots = overall_root / "plots"
    overall_root.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            sys.executable,
            "-m",
            "src.visualize",
            "--output-root",
            str(overall_root),
            "--results-dir",
            str(stage_root),
            "--plots-dir",
            str(overall_plots),
            "--protocol-stage",
            stage_id,
            "--summary-files",
            *[str(path) for path in summary_files],
        ],
        cwd=repo_root,
    )
    generated_roots.append(str(overall_plots))

    grouped: dict[int, list[Path]] = {}
    for summary_path in summary_files:
        summary = _load_summary(summary_path)
        grouped.setdefault(int(summary["num_agents"]), []).append(summary_path)

    for num_agents, paths in sorted(grouped.items()):
        compare_root = stage_root / "compare" / f"m{num_agents:02d}"
        plots_dir = compare_root / "plots"
        compare_root.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                sys.executable,
                "-m",
                "src.visualize",
                "--output-root",
                str(compare_root),
                "--results-dir",
                str(stage_root),
                "--plots-dir",
                str(plots_dir),
                "--protocol-stage",
                stage_id,
                "--summary-files",
                *[str(path) for path in paths],
            ],
            cwd=repo_root,
        )
        generated_roots.append(str(plots_dir))

    return generated_roots


def run_stage(stage_id: str, args: argparse.Namespace, repo_root: Path) -> None:
    stage_spec, run_specs = stage_run_specs(stage_id, args)
    stage_root = args.workspace_root / stage_id
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "stage": stage_id,
        "stage_spec": _json_ready(asdict(stage_spec)),
        "train_episodes": run_specs[0].train_episodes if run_specs else 0,
        "eval_episodes": run_specs[0].eval_episodes if run_specs else 0,
        "runs": [],
        "generated_plot_roots": [],
    }
    manifest_path = stage_root / "paper_run_manifest.json"

    for spec in run_specs:
        print(
            f"[run] stage={spec.stage_id} variant={spec.variant_id} "
            f"m={spec.num_agents} seed={spec.seed} root={spec.output_root}"
        )
        scale_profile = resolve_scale_run_profile(spec, args)
        checkpoint_path, checkpoint_rule, checkpoint_info = ensure_training(spec, args, repo_root)
        summary_path, selected_checkpoint_path, selected_checkpoint_rule = ensure_evaluation(
            spec,
            args,
            repo_root,
            checkpoint_path,
            checkpoint_rule,
        )
        selected_checkpoint_info = None
        if selected_checkpoint_path is not None:
            selected_checkpoint_payload = _torch_load_checkpoint(Path(selected_checkpoint_path))
            selected_checkpoint_info = CheckpointInfo(
                path=Path(selected_checkpoint_path),
                episodes_completed=int(selected_checkpoint_payload.get("episodes_completed", 0)),
                update_index=int(selected_checkpoint_payload.get("update_index", 0)),
                selection_rule=selected_checkpoint_rule,
            )
        manifest["runs"].append(
            {
                "stage": spec.stage_id,
                "variant_id": spec.variant_id,
                "runner_kind": spec.runner_kind,
                "num_agents": spec.num_agents,
                "seed": spec.seed,
                "train_episodes": spec.train_episodes,
                "eval_episodes": spec.eval_episodes,
                "output_root": spec.output_root,
                "checkpoint_selection_mode": args.checkpoint_selection_mode,
                "checkpoint_path": selected_checkpoint_path,
                "checkpoint_selection_rule": selected_checkpoint_rule,
                "checkpoint_episodes_completed": None
                if selected_checkpoint_info is None
                else selected_checkpoint_info.episodes_completed,
                "checkpoint_update_index": None
                if selected_checkpoint_info is None
                else selected_checkpoint_info.update_index,
                "summary_path": summary_path,
                "scale_profile": _json_ready(asdict(scale_profile)),
            }
        )
        _write_json(manifest_path, manifest)

    if not args.skip_visualize:
        manifest["generated_plot_roots"] = generate_stage_visualizations(
            stage_root,
            stage_id,
            repo_root,
            [Path(run["summary_path"]) for run in manifest["runs"]],
        )
        _write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    repo_root = REPO_ROOT

    stage_ids = ["smoke", "core", "scale"] if args.stage == "all" else [args.stage]
    for stage_id in stage_ids:
        run_stage(stage_id, args, repo_root)


if __name__ == "__main__":
    main()
