from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DISPLAY_NAME_MAP = {
    "A1": "RC-P-GCN-MAPPO",
    "A2": "RC-P-GCN-MAPPO (No L_I)",
    "A3": "RC-P-GCN-MAPPO (Static)",
    "A4": "RC-P-GCN-MAPPO (Star+Prox)",
    "A5_ROLE2": "RC-P-GCN-MAPPO (Role=2)",
    "A5_ROLE5": "RC-P-GCN-MAPPO (Role=5)",
    "A6A": "RC-P-GCN-MAPPO (Mobility)",
    "A6B": "RC-P-GCN-MAPPO (CPU Var)",
    "A7_100": "RC-P-GCN-MAPPO (D=100)",
    "A7_200": "RC-P-GCN-MAPPO (D=200)",
    "A8": "RC-P-GCN-MAPPO (+L_D)",
    "B0": "Li et al. Exact",
    "B1": "Li-Arch MAPPO",
    "B2": "Shared-MLP-MAPPO",
    "B3": "P-GCN-MAPPO",
    "B4": "RC-MAPPO",
    "B5": "DC-P-GCN-MAPPO",
    "B6": "Set-MAPPO",
    "B7": "IPPO",
    "B8": "MADDPG",
    "LOCAL_ONLY": "Local Only",
    "EDGE_ONLY": "Edge Only",
    "RANDOM": "Random",
    "QAG": "QAG",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize evaluation results and learning curves")
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--summary-files", type=Path, nargs="*", default=None)
    parser.add_argument("--trace-file", type=Path, default=None)
    parser.add_argument("--timeline-agent", type=int, default=0)
    parser.add_argument("--protocol-stage", choices=("smoke", "core", "scale"), default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_numeric_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    abs_value = abs(float(value))
    if abs_value >= 1e5:
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_numeric_with_std(mean: float | None, std: float | None) -> str:
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None or not np.isfinite(std) or abs(float(std)) < 1e-12:
        return _format_numeric_value(mean)
    return f"{_format_numeric_value(mean)}\n±{_format_numeric_value(std)}"


def _annotate_bars(
    axis: Any,
    bars: Any,
    values: list[float | None],
    *,
    stds: list[float | None] | None = None,
    fontsize: int = 7,
) -> None:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return
    value_span = max(finite_values) - min(finite_values)
    offset = max(value_span * 0.02, max(abs(value) for value in finite_values) * 0.015, 0.01)

    for index, (bar, value) in enumerate(zip(bars, values)):
        if value is None or not np.isfinite(value):
            continue
        std_value = None if stds is None else stds[index]
        label = _format_numeric_with_std(value, std_value)
        x = bar.get_x() + bar.get_width() / 2.0
        height = float(bar.get_height())
        if height >= 0.0:
            y = height + offset
            va = "bottom"
        else:
            y = height - offset
            va = "top"
        axis.text(x, y, label, ha="center", va=va, fontsize=fontsize)


def _annotate_heatmap(axis: Any, heatmap: np.ndarray, image: Any, fontsize: int = 8) -> None:
    finite_values = heatmap[np.isfinite(heatmap)]
    if finite_values.size == 0:
        return
    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    midpoint = (value_min + value_max) / 2.0

    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            value = heatmap[row, col]
            if not np.isfinite(value):
                label = "n/a"
                text_color = "black"
            else:
                label = _format_numeric_value(float(value))
                normalized = image.norm(float(value))
                text_color = "white" if normalized < 0.45 else "black"
                if value_max == value_min:
                    text_color = "white" if float(value) <= midpoint else "black"
            axis.text(col, row, label, ha="center", va="center", color=text_color, fontsize=fontsize)


def _display_label(raw_label: str | None) -> str:
    if raw_label is None:
        return "run"
    return DISPLAY_NAME_MAP.get(raw_label, raw_label)


def _summary_display_label(summary: dict[str, Any], fallback_index: int) -> str:
    raw_label = summary.get("variant_id") or summary.get("label", f"run_{fallback_index}")
    return _display_label(raw_label)


def discover_summary_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("evaluation_*_summary.json"))


def discover_trace_file(results_dir: Path) -> Path | None:
    traces = sorted(results_dir.glob("evaluation_*_trace.jsonl"))
    return None if not traces else traces[0]


def load_evaluation_summaries(results_dir: Path, summary_files: list[Path] | None = None) -> list[dict[str, Any]]:
    resolved_files = discover_summary_files(results_dir) if summary_files is None else summary_files
    return [_load_json(path) for path in resolved_files]


def load_learning_history(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return _load_jsonl(log_path)


def filter_summaries_by_protocol_stage(summaries: list[dict[str, Any]], protocol_stage: str | None) -> list[dict[str, Any]]:
    if protocol_stage is None:
        return summaries
    return [summary for summary in summaries if summary.get("protocol", {}).get("stage") == protocol_stage]


def aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str | None], list[dict[str, Any]]] = {}
    for summary in summaries:
        raw_label = summary.get("variant_id") or summary.get("label", "run")
        num_agents = int(summary.get("num_agents", 0))
        protocol_stage = summary.get("protocol", {}).get("stage")
        grouped.setdefault((raw_label, num_agents, protocol_stage), []).append(summary)

    metric_names = (
        "mean_episode_joint_reward",
        "mean_timeout_ratio",
        "mean_task_processing_cost",
        "mean_edge_queue",
        "mean_local_queue",
        "mean_role_kl",
        "mean_role_std",
        "mean_near_zero_sigma_fraction",
    )
    aggregated: list[dict[str, Any]] = []
    for (raw_label, num_agents, protocol_stage), items in sorted(grouped.items()):
        seeds = [int(item.get("seed", item.get("config", {}).get("seed", -1))) for item in items]
        checkpoint_rules = sorted(
            {
                str(item.get("protocol", {}).get("checkpoint_selection_rule"))
                for item in items
                if item.get("protocol", {}).get("checkpoint_selection_rule") is not None
            }
        )
        metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            values = [
                float(item["metrics"][metric_name])
                for item in items
                if item.get("metrics", {}).get(metric_name) is not None
            ]
            metrics[metric_name] = {
                "mean": None if not values else float(np.mean(values)),
                "std": None if not values else float(np.std(values)),
            }
        aggregated.append(
            {
                "label": raw_label,
                "display_label": _display_label(raw_label),
                "variant_id": items[0].get("variant_id"),
                "num_agents": num_agents,
                "protocol_stage": protocol_stage,
                "num_runs": len(items),
                "seeds": seeds,
                "checkpoint_selection_rules": checkpoint_rules,
                "metrics": metrics,
            }
        )
    return aggregated


def write_seed_aggregation_report(summaries: list[dict[str, Any]], results_dir: Path) -> Path | None:
    if not summaries:
        return None
    report = {"aggregated_runs": aggregate_seed_summaries(summaries)}
    path = results_dir / "protocol_seed_aggregation.json"
    path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def plot_learning_curves(output_root: Path, plots_dir: Path) -> Path | None:
    episode_logs = [
        ("ppo", output_root / "logs" / "episode_history.jsonl"),
        ("ippo", output_root / "logs" / "episode_history_ippo.jsonl"),
    ]
    update_logs = [
        ("ppo", output_root / "logs" / "update_history.jsonl"),
        ("ippo", output_root / "logs" / "update_history_ippo.jsonl"),
    ]

    if not any(path.exists() for _, path in episode_logs + update_logs):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, path in episode_logs:
        history = load_learning_history(path)
        if history:
            axes[0].plot(
                [item["episode"] for item in history],
                [item["joint_reward"] for item in history],
                label=label,
            )
    axes[0].set_title("Episode Joint Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Joint Reward [a.u.]")
    if axes[0].lines:
        axes[0].legend()

    for label, path in update_logs:
        history = load_learning_history(path)
        if history:
            axes[1].plot(
                [item["update"] for item in history],
                [item["mean_joint_reward"] for item in history],
                label=f"{label}: reward",
            )
            if "critic_loss" in history[0]:
                axes[1].plot(
                    [item["update"] for item in history],
                    [item["critic_loss"] for item in history],
                    linestyle="--",
                    label=f"{label}: critic_loss",
                )
    axes[1].set_title("Update Metrics")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Mixed Units")
    if axes[1].lines:
        axes[1].legend()

    fig.tight_layout()
    output_path = plots_dir / "learning_curves.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_summary_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    if not summaries:
        return None

    labels = [_summary_display_label(summary, idx) for idx, summary in enumerate(summaries)]
    reward = [summary["metrics"]["mean_episode_joint_reward"] for summary in summaries]
    timeout = [summary["metrics"]["mean_timeout_ratio"] for summary in summaries]
    cost = [summary["metrics"]["mean_task_processing_cost"] for summary in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    reward_bars = axes[0].bar(labels, reward, color="#c44e52")
    axes[0].set_title("Mean Episode Joint Reward")
    axes[0].set_ylabel("Reward [a.u.]")
    axes[0].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[0], reward_bars, reward)

    timeout_bars = axes[1].bar(labels, timeout, color="#4c72b0")
    axes[1].set_title("Mean Timeout Ratio")
    axes[1].set_ylabel("Ratio [-]")
    axes[1].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[1], timeout_bars, timeout)

    cost_bars = axes[2].bar(labels, cost, color="#55a868")
    axes[2].set_title("Mean Task Cost")
    axes[2].set_ylabel("Normalized Cost [-]")
    axes[2].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[2], cost_bars, cost)

    fig.tight_layout()
    output_path = plots_dir / "evaluation_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_seed_aggregation_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    aggregated = aggregate_seed_summaries(summaries)
    if not aggregated:
        return None

    labels = [f"{item['display_label']}@M{item['num_agents']}" for item in aggregated]
    reward_mean = [item["metrics"]["mean_episode_joint_reward"]["mean"] or 0.0 for item in aggregated]
    reward_std = [item["metrics"]["mean_episode_joint_reward"]["std"] or 0.0 for item in aggregated]
    timeout_mean = [item["metrics"]["mean_timeout_ratio"]["mean"] or 0.0 for item in aggregated]
    timeout_std = [item["metrics"]["mean_timeout_ratio"]["std"] or 0.0 for item in aggregated]
    queue_mean = [item["metrics"]["mean_edge_queue"]["mean"] or 0.0 for item in aggregated]
    queue_std = [item["metrics"]["mean_edge_queue"]["std"] or 0.0 for item in aggregated]

    x = np.arange(len(aggregated))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    reward_bars = axes[0].bar(x, reward_mean, yerr=reward_std, color="#c44e52", capsize=4)
    axes[0].set_title("Reward Mean ± Std")
    axes[0].set_ylabel("Reward [a.u.]")
    axes[0].set_xticks(x, labels, rotation=45)
    _annotate_bars(axes[0], reward_bars, reward_mean, stds=reward_std)

    timeout_bars = axes[1].bar(x, timeout_mean, yerr=timeout_std, color="#4c72b0", capsize=4)
    axes[1].set_title("Timeout Mean ± Std")
    axes[1].set_ylabel("Ratio [-]")
    axes[1].set_xticks(x, labels, rotation=45)
    _annotate_bars(axes[1], timeout_bars, timeout_mean, stds=timeout_std)

    queue_bars = axes[2].bar(x, queue_mean, yerr=queue_std, color="#55a868", capsize=4)
    axes[2].set_title("Edge Queue Mean ± Std")
    axes[2].set_ylabel("Queue [Gcycles]")
    axes[2].set_xticks(x, labels, rotation=45)
    _annotate_bars(axes[2], queue_bars, queue_mean, stds=queue_std)

    fig.tight_layout()
    output_path = plots_dir / "seed_aggregation_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _trace_array(records: list[dict[str, Any]], key: str, agent_index: int) -> np.ndarray:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            values.append(float("nan"))
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            values.append(float(array[agent_index]))
        else:
            values.append(float(array[agent_index].mean()))
    return np.asarray(values, dtype=np.float32)


def plot_role_transition_timeline(trace_records: list[dict[str, Any]], plots_dir: Path, agent_index: int = 0) -> Path | None:
    if not trace_records:
        return None
    episode_one = [record for record in trace_records if int(record["episode"]) == 1]
    if not episode_one:
        return None

    steps = np.asarray([record["step"] for record in episode_one], dtype=np.int32)
    distance = _trace_array(episode_one, "device_distances_m", agent_index)
    cpu = _trace_array(episode_one, "device_cpu_ghz", agent_index)
    offloading = _trace_array(episode_one, "device_offloading_ratio", agent_index)
    power = _trace_array(episode_one, "power_ratio", agent_index)

    first_role = next((record.get("role_mu") for record in episode_one if record.get("role_mu") is not None), None)
    if first_role is not None:
        role_dim = np.asarray(first_role, dtype=np.float32).shape[-1]
        role_series = [
            np.asarray([np.asarray(record["role_mu"], dtype=np.float32)[agent_index, dim] for record in episode_one], dtype=np.float32)
            for dim in range(role_dim)
        ]
    else:
        role_series = []

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(steps, distance, color="#4c72b0")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title(f"Role Transition Timeline: agent {agent_index}")

    axes[1].plot(steps, cpu, color="#dd8452")
    axes[1].set_ylabel("CPU (GHz)")

    for dim, series in enumerate(role_series):
        axes[2].plot(steps, series, label=f"role[{dim}]")
    axes[2].set_ylabel("role_mu [-]")
    if role_series:
        axes[2].legend(loc="upper right")

    axes[3].plot(steps, offloading, label="offloading", color="#55a868")
    axes[3].plot(steps, power, label="power", color="#c44e52")
    axes[3].set_ylabel("Action Ratio [-]")
    axes[3].set_xlabel("Step")
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    output_path = plots_dir / "role_transition_timeline.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_operational_mode_summary(summary: dict[str, Any], plots_dir: Path) -> Path | None:
    bins = summary.get("operational_mode_bins", [])
    if not bins:
        return None

    distance_labels = ["near", "mid", "far"]
    cpu_labels = ["low", "mid", "high"]
    metric_keys = ("avg_offloading_ratio", "avg_power_ratio", "avg_timeout_ratio")
    metric_titles = ("Offloading Ratio", "Power Ratio", "Timeout Ratio")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric_key, title in zip(axes, metric_keys, metric_titles):
        heatmap = np.full((len(cpu_labels), len(distance_labels)), np.nan, dtype=np.float32)
        for item in bins:
            row = cpu_labels.index(item["cpu_regime"])
            col = distance_labels.index(item["distance_regime"])
            value = item.get(metric_key)
            heatmap[row, col] = np.nan if value is None else float(value)
        image = axis.imshow(heatmap, cmap="viridis", aspect="auto")
        _annotate_heatmap(axis, heatmap, image)
        axis.set_xticks(range(len(distance_labels)), distance_labels)
        axis.set_yticks(range(len(cpu_labels)), cpu_labels)
        axis.set_title(title)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Ratio [-]")

    fig.tight_layout()
    output_path = plots_dir / "operational_mode_heatmaps.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_identifiability_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    filtered = [summary for summary in summaries if summary["metrics"].get("mean_role_kl") is not None]
    if not filtered:
        return None

    labels = [_summary_display_label(summary, idx) for idx, summary in enumerate(filtered)]
    kls = [summary["metrics"]["mean_role_kl"] for summary in filtered]
    nlls = [summary["metrics"]["mean_role_nll"] for summary in filtered]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    kl_bars = axes[0].bar(labels, kls, color="#8172b3")
    axes[0].set_title("Role KL")
    axes[0].set_ylabel("KL [nats]")
    axes[0].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[0], kl_bars, kls)

    nll_bars = axes[1].bar(labels, nlls, color="#937860")
    axes[1].set_title("Role NLL")
    axes[1].set_ylabel("NLL [nats]")
    axes[1].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[1], nll_bars, nlls)

    fig.tight_layout()
    output_path = plots_dir / "identifiability_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_posterior_uncertainty_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    filtered = [
        summary
        for summary in summaries
        if summary["metrics"].get("mean_role_std") is not None or summary["metrics"].get("mean_role_variance") is not None
    ]
    if not filtered:
        return None

    labels = [_summary_display_label(summary, idx) for idx, summary in enumerate(filtered)]
    mean_std = [
        0.0 if summary["metrics"].get("mean_role_std") is None else float(summary["metrics"]["mean_role_std"])
        for summary in filtered
    ]
    mean_var = [
        0.0 if summary["metrics"].get("mean_role_variance") is None else float(summary["metrics"]["mean_role_variance"])
        for summary in filtered
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    std_bars = axes[0].bar(labels, mean_std, color="#4c72b0")
    axes[0].set_title("Role Posterior Std")
    axes[0].set_ylabel("Std [-]")
    axes[0].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[0], std_bars, mean_std)

    var_bars = axes[1].bar(labels, mean_var, color="#dd8452")
    axes[1].set_title("Role Posterior Variance")
    axes[1].set_ylabel("Variance [-]")
    axes[1].tick_params(axis="x", rotation=45)
    _annotate_bars(axes[1], var_bars, mean_var)

    fig.tight_layout()
    output_path = plots_dir / "posterior_uncertainty_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c1_decomposition(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    target_variants = {"A1", "A3", "A6A", "A6B"}
    selected = [summary for summary in summaries if summary.get("variant_id") in target_variants]
    if len(selected) < 2:
        return None

    labels = [_display_label(summary["variant_id"]) for summary in selected]
    rewards = [summary["metrics"]["mean_episode_joint_reward"] for summary in selected]
    timeout = [summary["metrics"]["mean_timeout_ratio"] for summary in selected]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    reward_bars = axes[0].bar(labels, rewards, color="#64b5cd")
    axes[0].set_title("C1 Reward Decomposition")
    axes[0].set_ylabel("Reward [a.u.]")
    _annotate_bars(axes[0], reward_bars, rewards)
    timeout_bars = axes[1].bar(labels, timeout, color="#da8bc3")
    axes[1].set_title("C1 Timeout Decomposition")
    axes[1].set_ylabel("Ratio [-]")
    _annotate_bars(axes[1], timeout_bars, timeout)
    fig.tight_layout()

    output_path = plots_dir / "c1_decomposition.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_plots(
    output_root: Path,
    results_dir: Path,
    plots_dir: Path,
    summary_files: list[Path] | None = None,
    trace_file: Path | None = None,
    timeline_agent: int = 0,
    protocol_stage: str | None = None,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    summaries = filter_summaries_by_protocol_stage(
        load_evaluation_summaries(results_dir, summary_files=summary_files),
        protocol_stage=protocol_stage,
    )
    resolved_trace = discover_trace_file(results_dir) if trace_file is None else trace_file

    generated = []
    write_seed_aggregation_report(summaries, results_dir)
    for plot_path in (
        plot_learning_curves(output_root, plots_dir),
        plot_summary_comparison(summaries, plots_dir),
        plot_seed_aggregation_comparison(summaries, plots_dir),
        plot_identifiability_comparison(summaries, plots_dir),
        plot_posterior_uncertainty_comparison(summaries, plots_dir),
        plot_c1_decomposition(summaries, plots_dir),
    ):
        if plot_path is not None:
            generated.append(plot_path)

    if summaries:
        operational_plot = plot_operational_mode_summary(summaries[0], plots_dir)
        if operational_plot is not None:
            generated.append(operational_plot)

    if resolved_trace is not None and resolved_trace.exists():
        timeline_plot = plot_role_transition_timeline(_load_jsonl(resolved_trace), plots_dir, agent_index=timeline_agent)
        if timeline_plot is not None:
            generated.append(timeline_plot)

    return generated


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root
    results_dir = output_root / "results" if args.results_dir is None else args.results_dir
    plots_dir = results_dir / "plots" if args.plots_dir is None else args.plots_dir
    generated = generate_plots(
        output_root=output_root,
        results_dir=results_dir,
        plots_dir=plots_dir,
        summary_files=args.summary_files,
        trace_file=args.trace_file,
        timeline_agent=args.timeline_agent,
        protocol_stage=args.protocol_stage,
    )
    print(f"generated_plots={len(generated)} plots_dir={plots_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
