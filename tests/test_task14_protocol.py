from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import get_protocol_stage
from src.visualize import aggregate_seed_summaries, generate_plots


class Task14ProtocolTests(unittest.TestCase):
    def test_protocol_stage_registry_matches_plan(self) -> None:
        smoke = get_protocol_stage("smoke")
        core = get_protocol_stage("core")
        scale = get_protocol_stage("scale")

        self.assertIsNotNone(smoke)
        self.assertEqual(smoke.recommended_num_agents, (5,))
        self.assertIn("QAG", smoke.recommended_methods)

        self.assertIsNotNone(core)
        self.assertEqual(core.recommended_num_agents, (5, 10))
        self.assertEqual(core.recommended_seed_count, (3, 3))
        self.assertIn("A9_NOROLE", core.recommended_methods)

        self.assertIsNotNone(scale)
        self.assertEqual(scale.recommended_num_agents, (15, 20))
        self.assertEqual(scale.recommended_seed_count, (3, 5))

    def test_seed_aggregation_groups_by_variant_and_agent_count(self) -> None:
        summaries = [
            {
                "variant_id": "B3",
                "label": "checkpoint_seed1",
                "num_agents": 5,
                "seed": 1,
                "metrics": {
                    "mean_episode_joint_reward": -100.0,
                    "mean_timeout_ratio": 0.2,
                    "mean_task_processing_cost": 1.0,
                    "mean_edge_queue": 2.0,
                    "mean_local_queue": 1.5,
                    "mean_role_kl": None,
                    "mean_role_std": None,
                    "mean_near_zero_sigma_fraction": None,
                },
                "protocol": {"stage": "core", "checkpoint_selection_rule": "final_checkpoint"},
            },
            {
                "variant_id": "B3",
                "label": "checkpoint_seed2",
                "num_agents": 5,
                "seed": 2,
                "metrics": {
                    "mean_episode_joint_reward": -120.0,
                    "mean_timeout_ratio": 0.3,
                    "mean_task_processing_cost": 1.2,
                    "mean_edge_queue": 2.5,
                    "mean_local_queue": 1.7,
                    "mean_role_kl": None,
                    "mean_role_std": None,
                    "mean_near_zero_sigma_fraction": None,
                },
                "protocol": {"stage": "core", "checkpoint_selection_rule": "final_checkpoint"},
            },
        ]

        aggregated = aggregate_seed_summaries(summaries)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["label"], "B3")
        self.assertEqual(aggregated[0]["num_runs"], 2)
        self.assertAlmostEqual(aggregated[0]["metrics"]["mean_episode_joint_reward"]["mean"], -110.0)

    def test_generate_plots_writes_protocol_seed_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            plots_dir = results_dir / "plots"
            results_dir.mkdir()

            for seed, reward in ((1, -100.0), (2, -120.0)):
                summary_payload = {
                    "label": f"checkpoint_seed{seed}",
                    "variant_id": "B3",
                    "seed": seed,
                    "num_agents": 5,
                    "metrics": {
                        "mean_episode_joint_reward": reward,
                        "mean_timeout_ratio": 0.2 + 0.05 * seed,
                        "mean_task_processing_cost": 1.0 + 0.1 * seed,
                        "mean_edge_queue": 2.0 + 0.1 * seed,
                        "mean_local_queue": 1.0 + 0.1 * seed,
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
                (results_dir / f"evaluation_b3_seed{seed}_summary.json").write_text(
                    json.dumps(summary_payload),
                    encoding="utf-8",
                )

            generated = generate_plots(
                output_root=root,
                results_dir=results_dir,
                plots_dir=plots_dir,
                protocol_stage="core",
            )

            self.assertTrue((results_dir / "protocol_seed_aggregation.json").exists())
            self.assertIn(plots_dir / "seed_aggregation_comparison.png", generated)

    def test_generate_plots_uses_manifest_selected_summaries_for_manual_stage_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_root = Path(temp_dir) / "core"
            compare_root = stage_root / "compare" / "overall"
            plots_dir = compare_root / "plots"
            run_results_dir = stage_root / "m05" / "seed_42" / "a1" / "results"
            sweep_dir = run_results_dir / "checkpoint_sweep"
            run_results_dir.mkdir(parents=True)
            sweep_dir.mkdir(parents=True)

            selected_summary_path = run_results_dir / "evaluation_selected_summary.json"
            selected_summary_path.write_text(
                json.dumps(
                    {
                        "label": "checkpoint_final",
                        "variant_id": "A1",
                        "seed": 42,
                        "num_agents": 5,
                        "metrics": {
                            "mean_episode_joint_reward": -100.0,
                            "mean_timeout_ratio": 0.2,
                            "mean_task_processing_cost": 1.0,
                            "mean_edge_queue": 2.0,
                            "mean_local_queue": 1.0,
                            "mean_role_kl": None,
                            "mean_role_nll": None,
                            "mean_role_std": None,
                            "mean_role_variance": None,
                            "mean_near_zero_sigma_fraction": None,
                        },
                        "operational_mode_bins": [],
                        "results_dir": str(run_results_dir),
                        "protocol": {
                            "stage": "core",
                            "checkpoint_selection_rule": "periodic_checkpoint",
                        },
                        "runner_selection": {
                            "mode": "milestone_best",
                            "selected_checkpoint_path": "checkpoint_ep0200_u0050.pt",
                        },
                    }
                ),
                encoding="utf-8",
            )

            # This sweep summary should be ignored for manual reruns because the manifest
            # points to the selected summary produced by the automatic paper runner.
            (sweep_dir / "evaluation_checkpoint_ep0100_u0025_summary.json").write_text(
                json.dumps(
                    {
                        "label": "checkpoint_ep0100_u0025",
                        "variant_id": "A1",
                        "seed": 42,
                        "num_agents": 5,
                        "metrics": {
                            "mean_episode_joint_reward": -999.0,
                            "mean_timeout_ratio": 0.9,
                            "mean_task_processing_cost": 9.0,
                            "mean_edge_queue": 9.0,
                            "mean_local_queue": 9.0,
                            "mean_role_kl": None,
                            "mean_role_nll": None,
                            "mean_role_std": None,
                            "mean_role_variance": None,
                            "mean_near_zero_sigma_fraction": None,
                        },
                        "operational_mode_bins": [],
                        "results_dir": str(sweep_dir),
                        "protocol": {
                            "stage": "core",
                            "checkpoint_selection_rule": "periodic_checkpoint",
                        },
                    }
                ),
                encoding="utf-8",
            )

            (stage_root / "paper_run_manifest.json").write_text(
                json.dumps(
                    {
                        "stage": "core",
                        "runs": [
                            {
                                "summary_path": str(selected_summary_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            generate_plots(
                output_root=compare_root,
                results_dir=stage_root,
                plots_dir=plots_dir,
                protocol_stage="core",
            )

            report = json.loads((stage_root / "protocol_seed_aggregation.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["aggregated_runs"]), 1)
            self.assertAlmostEqual(report["aggregated_runs"][0]["metrics"]["mean_episode_joint_reward"]["mean"], -100.0)


if __name__ == "__main__":
    unittest.main()
