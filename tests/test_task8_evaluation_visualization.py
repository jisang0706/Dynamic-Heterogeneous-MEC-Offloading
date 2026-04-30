from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate import build_operational_mode_summary
from src.visualize import generate_plots


class Task8EvaluationVisualizationTests(unittest.TestCase):
    def test_operational_mode_summary_groups_device_records(self) -> None:
        records = [
            {
                "distance_m": 90.0,
                "cpu_ghz": 2.7,
                "offloading_ratio": 0.8,
                "power_ratio": 0.7,
                "timeout_ratio": 0.1,
                "deadline_s": 0.9,
                "best_case_delay_s": 0.5,
                "deadline_to_bestcase_ratio": 1.8,
            },
            {
                "distance_m": 92.0,
                "cpu_ghz": 2.6,
                "offloading_ratio": 0.6,
                "power_ratio": 0.5,
                "timeout_ratio": 0.2,
                "deadline_s": 0.8,
                "best_case_delay_s": 0.4,
                "deadline_to_bestcase_ratio": 2.0,
            },
            {
                "distance_m": 180.0,
                "cpu_ghz": 1.8,
                "offloading_ratio": 0.3,
                "power_ratio": 0.4,
                "timeout_ratio": 0.5,
                "deadline_s": 1.1,
                "best_case_delay_s": 0.7,
                "deadline_to_bestcase_ratio": 1.57,
            },
        ]

        summary = build_operational_mode_summary(records)

        self.assertEqual(len(summary), 9)
        near_high = next(item for item in summary if item["distance_regime"] == "near" and item["cpu_regime"] == "high")
        self.assertEqual(near_high["count"], 2)
        self.assertAlmostEqual(near_high["avg_offloading_ratio"], 0.7)
        self.assertAlmostEqual(near_high["avg_deadline_to_bestcase_ratio"], 1.9)
        far_low = next(item for item in summary if item["distance_regime"] == "far" and item["cpu_regime"] == "low")
        self.assertEqual(far_low["count"], 1)
        self.assertAlmostEqual(far_low["avg_timeout_ratio"], 0.5)

    def test_generate_plots_from_synthetic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            logs_dir = root / "logs"
            results_dir.mkdir()
            logs_dir.mkdir()

            summary_payload = {
                "label": "checkpoint_final",
                "variant_id": "A1",
                "metrics": {
                    "mean_episode_joint_reward": -123.0,
                    "mean_timeout_ratio": 0.2,
                    "mean_task_processing_cost": 1.4,
                    "mean_role_kl": 0.3,
                    "mean_role_nll": 1.1,
                    "mean_role_std": 0.4,
                    "mean_role_variance": 0.16,
                },
                "operational_mode_bins": [
                    {
                        "distance_regime": distance_regime,
                        "cpu_regime": cpu_regime,
                        "count": 1,
                        "avg_offloading_ratio": 0.5,
                        "avg_power_ratio": 0.6,
                        "avg_timeout_ratio": 0.2,
                        "avg_deadline_s": 0.8,
                        "avg_best_case_delay_s": 0.5,
                        "avg_deadline_to_bestcase_ratio": 1.6,
                    }
                    for distance_regime in ("near", "mid", "far")
                    for cpu_regime in ("low", "mid", "high")
                ],
            }
            (results_dir / "evaluation_checkpoint_final_summary.json").write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )

            trace_records = [
                {
                    "episode": 1,
                    "step": step + 1,
                    "device_distances_m": [100.0 + step, 120.0 + step],
                    "device_cpu_ghz": [2.5, 2.0],
                    "device_offloading_ratio": [0.3 + 0.01 * step, 0.4],
                    "power_ratio": [0.5, 0.6],
                    "role_mu": [[0.1 * step, 0.2, 0.3], [0.2, 0.1, 0.4]],
                }
                for step in range(5)
            ]
            with (results_dir / "evaluation_checkpoint_final_trace.jsonl").open("w", encoding="utf-8") as handle:
                for record in trace_records:
                    handle.write(json.dumps(record) + "\n")

            with (logs_dir / "episode_history.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode": 1, "joint_reward": -100.0, "steps": 5}) + "\n")
            with (logs_dir / "update_history.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "update": 1,
                            "mean_joint_reward": -100.0,
                            "critic_loss": 1.5,
                        }
                    )
                    + "\n"
                )

            generated = generate_plots(output_root=root, results_dir=results_dir, plots_dir=results_dir / "plots")

            self.assertGreaterEqual(len(generated), 5)
            for path in generated:
                self.assertTrue(path.exists(), msg=str(path))

    def test_generate_plots_writes_paper_main_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            results_dir.mkdir()

            paper_variants = (
                ("A9_NOROLE", -130.0, 0.22, 2.1, 1.4),
                ("B1", -150.0, 0.28, 2.8, 1.8),
                ("QAG", -5000.0, 0.65, 8.5, 12.0),
            )
            for variant_id, reward, timeout, queue, cost in paper_variants:
                (results_dir / f"evaluation_{variant_id.lower()}_summary.json").write_text(
                    json.dumps(
                        {
                            "label": variant_id.lower(),
                            "variant_id": variant_id,
                            "seed": 42,
                            "num_agents": 5,
                            "metrics": {
                                "mean_episode_joint_reward": reward,
                                "mean_timeout_ratio": timeout,
                                "mean_task_processing_cost": cost,
                                "mean_edge_queue": queue,
                                "mean_local_queue": 1.0,
                                "mean_role_kl": None,
                                "mean_role_nll": None,
                                "mean_role_std": None,
                                "mean_role_variance": None,
                                "mean_near_zero_sigma_fraction": None,
                            },
                            "operational_mode_bins": [],
                            "protocol": {
                                "stage": "core",
                                "checkpoint_selection_rule": "final_checkpoint",
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            generated = generate_plots(
                output_root=root,
                results_dir=results_dir,
                plots_dir=results_dir / "plots",
                protocol_stage="core",
            )

            self.assertIn(results_dir / "plots" / "paper_main_comparison.png", generated)


if __name__ == "__main__":
    unittest.main()
