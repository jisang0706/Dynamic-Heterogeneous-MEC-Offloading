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
    "A9": "P-GCN-MAPPO",
    "A9_NOROLE": "P-GCN-MAPPO (Indiv, No Role)",
    "B0": "Li et al. Exact",
    "B1": "MAPPO",
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

PAPER_VARIANT_GROUPS = (
    ("A9", "A9_NOROLE"),
    ("B1",),
    ("QAG",),
)
PAPER_LABEL_MAP = {
    "A9": "P-GCN-MAPPO",
    "A9_NOROLE": "P-GCN-MAPPO",
    "A1": "P-GCN-MAPPO",
    "B3": "P-GCN-MAPPO",
    "B1": "MAPPO",
    "QAG": "QAG",
}
PAPER_COLOR_MAP = {
    "A9": "#4c72b0",
    "A9_NOROLE": "#4c72b0",
    "A1": "#4c72b0",
    "B3": "#4c72b0",
    "B1": "#55a868",
    "QAG": "#c44e52",
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
    parser.add_argument("--paper-only", action="store_true")
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
    max_magnitude = max(abs(value) for value in finite_values)
    offset = max(value_span * 0.02, max_magnitude * 0.015, 0.01)
    lower_bound, upper_bound = axis.get_ylim()

    for index, (bar, value) in enumerate(zip(bars, values)):
        if value is None or not np.isfinite(value):
            continue
        std_value = None if stds is None else stds[index]
        label = _format_numeric_with_std(value, std_value)
        x = bar.get_x() + bar.get_width() / 2.0
        height = float(bar.get_height())
        inside_margin = max(abs(height) * 0.08, offset * 1.25)
        can_place_inside = abs(height) >= max(max_magnitude * 0.08, 0.05)
        if can_place_inside and height >= 0.0:
            y = height - inside_margin
            va = "top"
            color = "white"
        elif can_place_inside and height < 0.0:
            y = height + inside_margin
            va = "bottom"
            color = "white"
        elif height >= 0.0:
            y = height + offset
            va = "bottom"
            color = "black"
            upper_bound = max(upper_bound, y + offset)
        else:
            y = height - offset
            va = "top"
            color = "black"
            lower_bound = min(lower_bound, y - offset)
        axis.text(x, y, label, ha="center", va=va, fontsize=fontsize, color=color)

    current_low, current_high = axis.get_ylim()
    if lower_bound < current_low or upper_bound > current_high:
        span = max(current_high - current_low, offset * 4.0)
        pad = max(span * 0.04, offset)
        axis.set_ylim(min(lower_bound - pad, current_low), max(upper_bound + pad, current_high))


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


def _plot_label(raw_label: str | None, *, num_agents: int | None = None) -> str:
    label = _display_label(raw_label)
    label = label.replace("-MAPPO", "\nMAPPO")
    label = label.replace(" MAPPO", "\nMAPPO")
    label = label.replace(" (", "\n(")
    if num_agents is not None:
        label = f"{label}\n(M={num_agents})"
    return label


def _summary_display_label(summary: dict[str, Any], fallback_index: int) -> str:
    raw_label = summary.get("variant_id") or summary.get("label", f"run_{fallback_index}")
    return _plot_label(raw_label)


def _variant_identifier(item: dict[str, Any]) -> str:
    raw_variant = item.get("variant_id")
    if raw_variant is not None:
        return str(raw_variant)
    raw_label = item.get("label", "run")
    return str(raw_label)


def _paper_display_label(item: dict[str, Any]) -> str:
    variant_id = _variant_identifier(item)
    return PAPER_LABEL_MAP.get(variant_id, _display_label(variant_id))


def _paper_color(item: dict[str, Any]) -> str:
    return PAPER_COLOR_MAP.get(_variant_identifier(item), "#4c72b0")


def _is_paper_variant(variant_id: str) -> bool:
    return any(variant_id in group for group in PAPER_VARIANT_GROUPS)


def _looks_like_paper_selection(aggregated: list[dict[str, Any]]) -> bool:
    if not aggregated:
        return False
    variant_ids = {_variant_identifier(item) for item in aggregated}
    if not all(_is_paper_variant(variant_id) for variant_id in variant_ids):
        return False
    return (
        any(variant_id in PAPER_VARIANT_GROUPS[0] for variant_id in variant_ids)
        and any(variant_id in PAPER_VARIANT_GROUPS[1] for variant_id in variant_ids)
        and any(variant_id in PAPER_VARIANT_GROUPS[2] for variant_id in variant_ids)
    )


def _select_paper_variants(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for variant_group in PAPER_VARIANT_GROUPS:
        group_candidates = [item for item in aggregated if _variant_identifier(item) in variant_group]
        if not group_candidates:
            continue
        group_candidates.sort(
            key=lambda item: (
                variant_group.index(_variant_identifier(item)),
                0 if int(item.get("num_agents", -1)) == 5 else 1,
                0 if item.get("protocol_stage") == "core" else 1,
                -int(item.get("num_runs", 0)),
            )
        )
        selected.append(group_candidates[0])
    return selected


def _comparison_width(num_items: int, *, min_width: float = 10.0, per_item: float = 1.8, max_width: float = 28.0) -> float:
    return float(min(max(min_width, num_items * per_item), max_width))


def _comparison_tick_fontsize(num_items: int) -> int:
    if num_items <= 5:
        return 9
    if num_items <= 8:
        return 8
    if num_items <= 12:
        return 7
    return 6


def _comparison_annotation_fontsize(num_items: int) -> int:
    if num_items <= 5:
        return 8
    if num_items <= 8:
        return 7
    if num_items <= 12:
        return 6
    return 5


def _set_bar_ticks(axis: Any, x: np.ndarray, labels: list[str], *, num_items: int) -> None:
    axis.set_xticks(x, labels)
    axis.tick_params(axis="x", labelsize=_comparison_tick_fontsize(num_items))
    axis.margins(x=0.02)


def _draw_wave_break(axis: Any, *, at_top: bool) -> None:
    baseline = 1.0 if at_top else 0.0
    amplitude = 0.010
    width = 0.055
    centers = (0.06, 0.94)
    t = np.linspace(0.0, 1.0, 80)
    for center in centers:
        x = center + (t - 0.5) * width
        y = baseline + amplitude * np.sin(2.0 * np.pi * t)
        axis.plot(
            x,
            y,
            transform=axis.transAxes,
            color="black",
            linewidth=1.2,
            solid_capstyle="round",
            clip_on=False,
        )


def _plot_metric_bars(
    axis: Any,
    x: np.ndarray,
    labels: list[str],
    means: list[float],
    stds: list[float],
    colors: list[str],
    *,
    title: str,
    ylabel: str,
    annotation_fontsize: int,
) -> Any:
    bars = axis.bar(x, means, yerr=stds, color=colors, capsize=4)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    _set_bar_ticks(axis, x, labels, num_items=len(labels))
    axis.margins(y=0.08)
    _annotate_bars(axis, bars, means, stds=stds, fontsize=annotation_fontsize)
    return bars


def _broken_axis_spec(
    variant_ids: list[str],
    means: list[float],
    stds: list[float],
) -> tuple[int, tuple[float, float], tuple[float, float]] | None:
    if "QAG" not in variant_ids:
        return None
    qag_index = variant_ids.index("QAG")
    qag_mean = float(means[qag_index])
    qag_std = float(stds[qag_index])
    other_bounds = [
        float(mean + std)
        for index, (mean, std) in enumerate(zip(means, stds))
        if index != qag_index
    ]
    other_lows = [
        float(mean - std)
        for index, (mean, std) in enumerate(zip(means, stds))
        if index != qag_index
    ]
    if not other_bounds or not other_lows:
        return None

    max_other = max(other_bounds)
    qag_lower = qag_mean - qag_std
    if qag_mean <= max_other * 1.6 or qag_lower <= max_other * 1.15:
        return None

    lower_span = max_other - min(other_lows)
    lower_pad = max(lower_span * 0.12, max(abs(value) for value in means) * 0.02, 0.05)
    bottom_upper = max_other + lower_pad
    top_lower = bottom_upper + (qag_lower - bottom_upper) * 0.55
    if top_lower <= bottom_upper:
        return None

    bottom_lower = min(0.0, min(other_lows) - lower_pad)
    top_upper = qag_mean + qag_std + max(lower_pad, abs(qag_mean) * 0.05)
    return qag_index, (bottom_lower, bottom_upper), (top_lower, top_upper)


def _plot_broken_metric_bars(
    fig: Any,
    cell: Any,
    x: np.ndarray,
    labels: list[str],
    means: list[float],
    stds: list[float],
    colors: list[str],
    *,
    title: str,
    ylabel: str,
    annotation_fontsize: int,
    variant_ids: list[str],
) -> None:
    broken_axis = _broken_axis_spec(variant_ids, means, stds)
    if broken_axis is None:
        axis = fig.add_subplot(cell)
        _plot_metric_bars(
            axis,
            x,
            labels,
            means,
            stds,
            colors,
            title=title,
            ylabel=ylabel,
            annotation_fontsize=annotation_fontsize,
        )
        return

    qag_index, bottom_ylim, top_ylim = broken_axis
    subgrid = cell.subgridspec(2, 1, height_ratios=(1.0, 1.9), hspace=0.05)
    axis_top = fig.add_subplot(subgrid[0, 0])
    axis_bottom = fig.add_subplot(subgrid[1, 0], sharex=axis_top)

    bars_top = axis_top.bar(x, means, yerr=stds, color=colors, capsize=4)
    bars_bottom = axis_bottom.bar(x, means, yerr=stds, color=colors, capsize=4)

    axis_top.set_ylim(*top_ylim)
    axis_bottom.set_ylim(*bottom_ylim)
    axis_top.set_title(title)
    axis_bottom.set_ylabel(ylabel)
    _set_bar_ticks(axis_bottom, x, labels, num_items=len(labels))
    axis_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    axis_top.spines["bottom"].set_visible(False)
    axis_bottom.spines["top"].set_visible(False)
    axis_top.margins(y=0.03)
    axis_bottom.margins(y=0.08)
    _draw_wave_break(axis_top, at_top=False)
    _draw_wave_break(axis_bottom, at_top=True)

    top_values: list[float | None] = [None] * len(means)
    bottom_values: list[float | None] = [float(value) for value in means]
    top_stds: list[float | None] = [None] * len(stds)
    bottom_stds: list[float | None] = [float(std) for std in stds]
    top_values[qag_index] = float(means[qag_index])
    top_stds[qag_index] = float(stds[qag_index])
    bottom_values[qag_index] = None
    bottom_stds[qag_index] = None

    _annotate_bars(axis_top, bars_top, top_values, stds=top_stds, fontsize=annotation_fontsize)
    _annotate_bars(axis_bottom, bars_bottom, bottom_values, stds=bottom_stds, fontsize=annotation_fontsize)


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
        "mean_role_nll",
        "mean_role_std",
        "mean_role_variance",
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
    aggregated = aggregate_seed_summaries(summaries)
    if not aggregated:
        return None

    use_paper_style = _looks_like_paper_selection(aggregated)
    selected = _select_paper_variants(aggregated) if use_paper_style else aggregated
    labels = [
        _paper_display_label(item) if use_paper_style else _plot_label(item["label"], num_agents=item["num_agents"])
        for item in selected
    ]
    reward_mean = [item["metrics"]["mean_episode_joint_reward"]["mean"] or 0.0 for item in selected]
    reward_std = [item["metrics"]["mean_episode_joint_reward"]["std"] or 0.0 for item in selected]
    timeout_mean = [item["metrics"]["mean_timeout_ratio"]["mean"] or 0.0 for item in selected]
    timeout_std = [item["metrics"]["mean_timeout_ratio"]["std"] or 0.0 for item in selected]
    cost_mean = [item["metrics"]["mean_task_processing_cost"]["mean"] or 0.0 for item in selected]
    cost_std = [item["metrics"]["mean_task_processing_cost"]["std"] or 0.0 for item in selected]
    variant_ids = [_variant_identifier(item) for item in selected]

    x = np.arange(len(selected))
    num_items = len(selected)
    annotation_fontsize = _comparison_annotation_fontsize(num_items)
    reward_colors = [_paper_color(item) for item in selected] if use_paper_style else ["#c44e52"] * num_items
    timeout_colors = [_paper_color(item) for item in selected] if use_paper_style else ["#4c72b0"] * num_items
    cost_colors = [_paper_color(item) for item in selected] if use_paper_style else ["#55a868"] * num_items

    fig = plt.figure(figsize=(_comparison_width(num_items, min_width=14.0), 5.2))
    grid = fig.add_gridspec(1, 3, wspace=0.28)
    axis_reward = fig.add_subplot(grid[0, 0])
    _plot_metric_bars(
        axis_reward,
        x,
        labels,
        reward_mean,
        reward_std,
        reward_colors,
        title="Mean Episode Joint Reward",
        ylabel="Reward [a.u.]",
        annotation_fontsize=annotation_fontsize,
    )

    axis_timeout = fig.add_subplot(grid[0, 1])
    _plot_metric_bars(
        axis_timeout,
        x,
        labels,
        timeout_mean,
        timeout_std,
        timeout_colors,
        title="Mean Timeout Ratio",
        ylabel="Ratio [-]",
        annotation_fontsize=annotation_fontsize,
    )

    _plot_broken_metric_bars(
        fig,
        grid[0, 2],
        x,
        labels,
        cost_mean,
        cost_std,
        cost_colors,
        title="Mean Task Cost",
        ylabel="Normalized Cost [-]",
        annotation_fontsize=annotation_fontsize,
        variant_ids=variant_ids,
    )

    fig.tight_layout()
    output_path = plots_dir / "evaluation_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_seed_aggregation_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    aggregated = aggregate_seed_summaries(summaries)
    if not aggregated:
        return None

    use_paper_style = _looks_like_paper_selection(aggregated)
    selected = _select_paper_variants(aggregated) if use_paper_style else aggregated
    labels = [
        _paper_display_label(item) if use_paper_style else _plot_label(item["label"], num_agents=item["num_agents"])
        for item in selected
    ]
    reward_mean = [item["metrics"]["mean_episode_joint_reward"]["mean"] or 0.0 for item in selected]
    reward_std = [item["metrics"]["mean_episode_joint_reward"]["std"] or 0.0 for item in selected]
    timeout_mean = [item["metrics"]["mean_timeout_ratio"]["mean"] or 0.0 for item in selected]
    timeout_std = [item["metrics"]["mean_timeout_ratio"]["std"] or 0.0 for item in selected]
    queue_mean = [item["metrics"]["mean_edge_queue"]["mean"] or 0.0 for item in selected]
    queue_std = [item["metrics"]["mean_edge_queue"]["std"] or 0.0 for item in selected]

    x = np.arange(len(selected))
    num_items = len(selected)
    annotation_fontsize = _comparison_annotation_fontsize(num_items)
    reward_colors = [_paper_color(item) for item in selected] if use_paper_style else ["#c44e52"] * num_items
    timeout_colors = [_paper_color(item) for item in selected] if use_paper_style else ["#4c72b0"] * num_items
    queue_colors = [_paper_color(item) for item in selected] if use_paper_style else ["#55a868"] * num_items
    fig, axes = plt.subplots(1, 3, figsize=(_comparison_width(num_items, min_width=14.0), 5.2))
    _plot_metric_bars(
        axes[0],
        x,
        labels,
        reward_mean,
        reward_std,
        reward_colors,
        title="Reward Mean ± Std",
        ylabel="Reward [a.u.]",
        annotation_fontsize=annotation_fontsize,
    )
    _plot_metric_bars(
        axes[1],
        x,
        labels,
        timeout_mean,
        timeout_std,
        timeout_colors,
        title="Timeout Mean ± Std",
        ylabel="Ratio [-]",
        annotation_fontsize=annotation_fontsize,
    )
    _plot_metric_bars(
        axes[2],
        x,
        labels,
        queue_mean,
        queue_std,
        queue_colors,
        title="Edge Queue Mean ± Std",
        ylabel="Queue [Gcycles]",
        annotation_fontsize=annotation_fontsize,
    )

    fig.tight_layout()
    output_path = plots_dir / "seed_aggregation_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_paper_main_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    aggregated = aggregate_seed_summaries(summaries)
    selected = _select_paper_variants(aggregated)
    if len(selected) < 2:
        return None

    labels = [_paper_display_label(item) for item in selected]
    colors = [_paper_color(item) for item in selected]
    variant_ids = [_variant_identifier(item) for item in selected]
    x = np.arange(len(selected))
    num_items = len(selected)
    annotation_fontsize = _comparison_annotation_fontsize(num_items)

    reward_mean = [item["metrics"]["mean_episode_joint_reward"]["mean"] or 0.0 for item in selected]
    reward_std = [item["metrics"]["mean_episode_joint_reward"]["std"] or 0.0 for item in selected]
    timeout_mean = [item["metrics"]["mean_timeout_ratio"]["mean"] or 0.0 for item in selected]
    timeout_std = [item["metrics"]["mean_timeout_ratio"]["std"] or 0.0 for item in selected]
    queue_mean = [item["metrics"]["mean_edge_queue"]["mean"] or 0.0 for item in selected]
    queue_std = [item["metrics"]["mean_edge_queue"]["std"] or 0.0 for item in selected]
    cost_mean = [item["metrics"]["mean_task_processing_cost"]["mean"] or 0.0 for item in selected]
    cost_std = [item["metrics"]["mean_task_processing_cost"]["std"] or 0.0 for item in selected]

    fig = plt.figure(figsize=(11.5, 8.5))
    grid = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    axis_reward = fig.add_subplot(grid[0, 0])
    _plot_metric_bars(
        axis_reward,
        x,
        labels,
        reward_mean,
        reward_std,
        colors,
        title="Mean Episode Joint Reward",
        ylabel="Reward [a.u.]",
        annotation_fontsize=annotation_fontsize,
    )

    axis_timeout = fig.add_subplot(grid[0, 1])
    _plot_metric_bars(
        axis_timeout,
        x,
        labels,
        timeout_mean,
        timeout_std,
        colors,
        title="Mean Timeout Ratio",
        ylabel="Ratio [-]",
        annotation_fontsize=annotation_fontsize,
    )

    axis_queue = fig.add_subplot(grid[1, 0])
    _plot_metric_bars(
        axis_queue,
        x,
        labels,
        queue_mean,
        queue_std,
        colors,
        title="Mean Edge Queue",
        ylabel="Queue [Gcycles]",
        annotation_fontsize=annotation_fontsize,
    )

    _plot_broken_metric_bars(
        fig,
        grid[1, 1],
        x,
        labels,
        cost_mean,
        cost_std,
        colors,
        title="Mean Task Cost",
        ylabel="Normalized Cost [-]",
        annotation_fontsize=annotation_fontsize,
        variant_ids=variant_ids,
    )

    output_path = plots_dir / "paper_main_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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


def plot_operational_mode_context_summary(summary: dict[str, Any], plots_dir: Path) -> Path | None:
    bins = summary.get("operational_mode_bins", [])
    if not bins:
        return None

    distance_labels = ["near", "mid", "far"]
    cpu_labels = ["low", "mid", "high"]
    metric_specs = (
        ("count", "Sample Count", False),
        ("avg_deadline_s", "Mean Deadline (s)", True),
        ("avg_best_case_delay_s", "Mean Best-Case Delay (s)", True),
        ("avg_deadline_to_bestcase_ratio", "Mean d_c / d_best", True),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for axis, (metric_key, title, use_float) in zip(axes.flat, metric_specs):
        heatmap = np.full((len(cpu_labels), len(distance_labels)), np.nan, dtype=np.float32)
        for item in bins:
            row = cpu_labels.index(item["cpu_regime"])
            col = distance_labels.index(item["distance_regime"])
            value = item.get(metric_key)
            if value is None:
                continue
            heatmap[row, col] = float(value)
        image = axis.imshow(heatmap, cmap="viridis", aspect="auto")
        for row in range(heatmap.shape[0]):
            for col in range(heatmap.shape[1]):
                value = heatmap[row, col]
                if not np.isfinite(value):
                    label = "n/a"
                    color = "black"
                elif use_float:
                    label = _format_numeric_value(float(value))
                    color = "white" if image.norm(float(value)) < 0.45 else "black"
                else:
                    label = str(int(round(float(value))))
                    color = "white" if image.norm(float(value)) < 0.45 else "black"
                axis.text(col, row, label, ha="center", va="center", color=color, fontsize=8)
        axis.set_xticks(range(len(distance_labels)), distance_labels)
        axis.set_yticks(range(len(cpu_labels)), cpu_labels)
        axis.set_title(title)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Count [-]" if metric_key == "count" else "Value [-]")

    fig.tight_layout()
    output_path = plots_dir / "operational_mode_context_heatmaps.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_identifiability_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    aggregated = aggregate_seed_summaries(summaries)
    filtered = [item for item in aggregated if item["metrics"].get("mean_role_kl", {}).get("mean") is not None]
    if not filtered:
        return None

    labels = [_plot_label(item["label"], num_agents=item["num_agents"]) for item in filtered]
    kls = [item["metrics"]["mean_role_kl"]["mean"] or 0.0 for item in filtered]
    kl_std = [item["metrics"]["mean_role_kl"]["std"] or 0.0 for item in filtered]
    nlls = [item["metrics"]["mean_role_nll"]["mean"] or 0.0 for item in filtered]
    nll_std = [item["metrics"]["mean_role_nll"]["std"] or 0.0 for item in filtered]

    x = np.arange(len(filtered))
    num_items = len(filtered)
    annotation_fontsize = _comparison_annotation_fontsize(num_items)
    fig, axes = plt.subplots(1, 2, figsize=(_comparison_width(num_items, min_width=11.0), 4.8))
    kl_bars = axes[0].bar(x, kls, yerr=kl_std, color="#8172b3", capsize=4)
    axes[0].set_title("Role KL")
    axes[0].set_ylabel("KL [nats]")
    _set_bar_ticks(axes[0], x, labels, num_items=num_items)
    axes[0].margins(y=0.08)
    _annotate_bars(axes[0], kl_bars, kls, stds=kl_std, fontsize=annotation_fontsize)

    nll_bars = axes[1].bar(x, nlls, yerr=nll_std, color="#937860", capsize=4)
    axes[1].set_title("Role NLL")
    axes[1].set_ylabel("NLL [nats]")
    _set_bar_ticks(axes[1], x, labels, num_items=num_items)
    axes[1].margins(y=0.08)
    _annotate_bars(axes[1], nll_bars, nlls, stds=nll_std, fontsize=annotation_fontsize)

    fig.tight_layout()
    output_path = plots_dir / "identifiability_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_posterior_uncertainty_comparison(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    aggregated = aggregate_seed_summaries(summaries)
    filtered = [
        item
        for item in aggregated
        if item["metrics"].get("mean_role_std", {}).get("mean") is not None
        or item["metrics"].get("mean_role_variance", {}).get("mean") is not None
    ]
    if not filtered:
        return None

    labels = [_plot_label(item["label"], num_agents=item["num_agents"]) for item in filtered]
    mean_std = [
        0.0 if item["metrics"].get("mean_role_std", {}).get("mean") is None else float(item["metrics"]["mean_role_std"]["mean"])
        for item in filtered
    ]
    std_std = [
        0.0 if item["metrics"].get("mean_role_std", {}).get("std") is None else float(item["metrics"]["mean_role_std"]["std"])
        for item in filtered
    ]
    mean_var = [
        0.0 if item["metrics"].get("mean_role_variance", {}).get("mean") is None else float(item["metrics"]["mean_role_variance"]["mean"])
        for item in filtered
    ]
    var_std = [
        0.0 if item["metrics"].get("mean_role_variance", {}).get("std") is None else float(item["metrics"]["mean_role_variance"]["std"])
        for item in filtered
    ]

    x = np.arange(len(filtered))
    num_items = len(filtered)
    annotation_fontsize = _comparison_annotation_fontsize(num_items)
    fig, axes = plt.subplots(1, 2, figsize=(_comparison_width(num_items, min_width=11.0), 4.8))
    std_bars = axes[0].bar(x, mean_std, yerr=std_std, color="#4c72b0", capsize=4)
    axes[0].set_title("Role Posterior Std")
    axes[0].set_ylabel("Std [-]")
    _set_bar_ticks(axes[0], x, labels, num_items=num_items)
    axes[0].margins(y=0.08)
    _annotate_bars(axes[0], std_bars, mean_std, stds=std_std, fontsize=annotation_fontsize)

    var_bars = axes[1].bar(x, mean_var, yerr=var_std, color="#dd8452", capsize=4)
    axes[1].set_title("Role Posterior Variance")
    axes[1].set_ylabel("Variance [-]")
    _set_bar_ticks(axes[1], x, labels, num_items=num_items)
    axes[1].margins(y=0.08)
    _annotate_bars(axes[1], var_bars, mean_var, stds=var_std, fontsize=annotation_fontsize)

    fig.tight_layout()
    output_path = plots_dir / "posterior_uncertainty_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c1_decomposition(summaries: list[dict[str, Any]], plots_dir: Path) -> Path | None:
    target_variants = {"A1", "A3", "A6A", "A6B"}
    aggregated = aggregate_seed_summaries(summaries)
    selected = [item for item in aggregated if item.get("variant_id") in target_variants]
    if len(selected) < 2:
        return None

    labels = [_plot_label(item["variant_id"], num_agents=item["num_agents"]) for item in selected]
    rewards = [item["metrics"]["mean_episode_joint_reward"]["mean"] or 0.0 for item in selected]
    reward_std = [item["metrics"]["mean_episode_joint_reward"]["std"] or 0.0 for item in selected]
    timeout = [item["metrics"]["mean_timeout_ratio"]["mean"] or 0.0 for item in selected]
    timeout_std = [item["metrics"]["mean_timeout_ratio"]["std"] or 0.0 for item in selected]

    x = np.arange(len(selected))
    num_items = len(selected)
    annotation_fontsize = _comparison_annotation_fontsize(num_items)
    fig, axes = plt.subplots(1, 2, figsize=(_comparison_width(num_items, min_width=10.0, max_width=18.0), 4.2))
    reward_bars = axes[0].bar(x, rewards, yerr=reward_std, color="#64b5cd", capsize=4)
    axes[0].set_title("C1 Reward Decomposition")
    axes[0].set_ylabel("Reward [a.u.]")
    _set_bar_ticks(axes[0], x, labels, num_items=num_items)
    _annotate_bars(axes[0], reward_bars, rewards, stds=reward_std, fontsize=annotation_fontsize)
    timeout_bars = axes[1].bar(x, timeout, yerr=timeout_std, color="#da8bc3", capsize=4)
    axes[1].set_title("C1 Timeout Decomposition")
    axes[1].set_ylabel("Ratio [-]")
    _set_bar_ticks(axes[1], x, labels, num_items=num_items)
    _annotate_bars(axes[1], timeout_bars, timeout, stds=timeout_std, fontsize=annotation_fontsize)
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
    paper_only: bool = False,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    summaries = filter_summaries_by_protocol_stage(
        load_evaluation_summaries(results_dir, summary_files=summary_files),
        protocol_stage=protocol_stage,
    )
    resolved_trace = discover_trace_file(results_dir) if trace_file is None else trace_file

    generated = []
    write_seed_aggregation_report(summaries, results_dir)
    if paper_only:
        paper_plot = plot_paper_main_comparison(summaries, plots_dir)
        if paper_plot is not None:
            generated.append(paper_plot)
        return generated

    for plot_path in (
        plot_learning_curves(output_root, plots_dir),
        plot_summary_comparison(summaries, plots_dir),
        plot_seed_aggregation_comparison(summaries, plots_dir),
        plot_paper_main_comparison(summaries, plots_dir),
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
        operational_context_plot = plot_operational_mode_context_summary(summaries[0], plots_dir)
        if operational_context_plot is not None:
            generated.append(operational_context_plot)

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
        paper_only=args.paper_only,
    )
    print(f"generated_plots={len(generated)} plots_dir={plots_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
