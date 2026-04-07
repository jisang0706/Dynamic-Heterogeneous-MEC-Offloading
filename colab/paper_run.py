from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--update-every-episodes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--l-i-coeff", type=float, default=1e-4)
    parser.add_argument("--lambda-var", type=float, default=1e-5)
    parser.add_argument("--sigma-floor", type=float, default=0.05)
    parser.add_argument("--initial-action-std-env", type=float, default=0.25)
    parser.add_argument("--initial-offloading-mean-env", type=float, default=0.65)
    parser.add_argument("--initial-power-mean-env", type=float, default=0.8)
    parser.add_argument("--use-obs-scaling", choices=("true", "false"), default="true")
    parser.add_argument("--use-reward-scaling", choices=("true", "false"), default="true")
    parser.add_argument("--save-every-episodes", type=int, default=100)
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


def discover_best_checkpoint(output_root: Path, runner_kind: str) -> CheckpointInfo | None:
    models_dir = output_root / "models"
    if not models_dir.exists():
        return None

    prefix = _checkpoint_prefix(runner_kind)
    candidate_paths: list[Path] = []
    for suffix in ("final", "latest"):
        path = models_dir / f"{prefix}_{suffix}.pt"
        if path.exists():
            candidate_paths.append(path)
    if not candidate_paths:
        candidate_paths.extend(sorted(models_dir.glob(f"{prefix}_ep*_u*.pt")))
    if not candidate_paths:
        return None

    best: CheckpointInfo | None = None
    for checkpoint_path in candidate_paths:
        checkpoint = _torch_load_checkpoint(checkpoint_path)
        info = CheckpointInfo(
            path=checkpoint_path,
            episodes_completed=int(checkpoint.get("episodes_completed", 0)),
            update_index=int(checkpoint.get("update_index", 0)),
            selection_rule=_checkpoint_selection_rule(checkpoint_path),
        )
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


def default_seeds(stage_spec: ProtocolStageSpec) -> list[int]:
    seed_count = stage_spec.recommended_seed_count[0]
    return list(DEFAULT_SEED_POOL[:seed_count])


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
        "--ppo-clip",
        str(args.ppo_clip),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--gradient-clip",
        str(args.gradient_clip),
        "--l-i-coeff",
        str(args.l_i_coeff),
        "--lambda-var",
        str(args.lambda_var),
        "--sigma-floor",
        str(args.sigma_floor),
        "--initial-action-std-env",
        str(args.initial_action_std_env),
        "--initial-offloading-mean-env",
        str(args.initial_offloading_mean_env),
        "--initial-power-mean-env",
        str(args.initial_power_mean_env),
        "--use-obs-scaling",
        args.use_obs_scaling,
        "--use-reward-scaling",
        args.use_reward_scaling,
        "--save-every-episodes",
        str(args.save_every_episodes),
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    return command


def _evaluate_command(
    spec: RunSpec,
    checkpoint_path: Path | None,
    checkpoint_selection_rule: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--output-root",
        str(spec.output_root),
        "--episodes",
        str(spec.eval_episodes),
        "--protocol-stage",
        spec.stage_id,
        "--checkpoint-selection-rule",
        checkpoint_selection_rule,
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
    return command


def discover_summary_file(spec: RunSpec) -> Path | None:
    results_dir = spec.output_root / "results"
    if not results_dir.exists():
        return None
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
) -> Path:
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
        return summary_path

    _run_command(
        _evaluate_command(spec, checkpoint_path=checkpoint_path, checkpoint_selection_rule=checkpoint_selection_rule),
        cwd=repo_root,
    )
    refreshed_summary = discover_summary_file(spec)
    if refreshed_summary is None:
        raise SystemExit(f"Evaluation completed but no summary file was found for {spec.output_root}")
    return refreshed_summary


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_stage_visualizations(stage_root: Path, stage_id: str, repo_root: Path) -> list[str]:
    summary_files = sorted(stage_root.rglob("evaluation_*_summary.json"))
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
        checkpoint_path, checkpoint_rule, checkpoint_info = ensure_training(spec, args, repo_root)
        summary_path = ensure_evaluation(spec, args, repo_root, checkpoint_path, checkpoint_rule)
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
                "checkpoint_path": checkpoint_path,
                "checkpoint_selection_rule": checkpoint_rule,
                "checkpoint_episodes_completed": None if checkpoint_info is None else checkpoint_info.episodes_completed,
                "checkpoint_update_index": None if checkpoint_info is None else checkpoint_info.update_index,
                "summary_path": summary_path,
            }
        )
        _write_json(manifest_path, manifest)

    if not args.skip_visualize:
        manifest["generated_plot_roots"] = generate_stage_visualizations(stage_root, stage_id, repo_root)
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
