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


if __name__ == "__main__":
    unittest.main()
