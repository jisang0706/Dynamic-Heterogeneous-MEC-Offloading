from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import EnvironmentConfig, ExperimentConfig
from src.environment import DynamicMECEnv
from src.tools.audit_feasibility import run_feasibility_audit


class Task9FeasibilityTests(unittest.TestCase):
    def test_bestcase_slack_deadlines_match_optimistic_bound(self) -> None:
        config = EnvironmentConfig(num_agents=5, episode_length=4, delay_mode="bestcase_slack", u_slack=1.5)
        env = DynamicMECEnv(config, seed=21)
        env.reset()

        components = env.compute_best_case_delay_components()
        expected_deadlines = config.u_slack * components["d_best_s"]

        np.testing.assert_allclose(env.task_deadlines_s, expected_deadlines, rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(env.task_deadlines_s + 1e-8 >= components["d_best_s"]))

    def test_li_original_delay_mode_keeps_deadlines_in_configured_range(self) -> None:
        config = EnvironmentConfig(num_agents=5, episode_length=4, delay_mode="li_original")
        env = DynamicMECEnv(config, seed=33)
        env.reset()

        self.assertTrue(np.all(env.task_deadlines_s >= config.task_deadline_range_s[0]))
        self.assertTrue(np.all(env.task_deadlines_s <= config.task_deadline_range_s[1]))

    def test_feasibility_audit_reports_zero_infeasible_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                seed=17,
                output_root=Path(temp_dir),
                run_feasibility_audit=True,
                environment=EnvironmentConfig(num_agents=10, episode_length=4, delay_mode="bestcase_slack", u_slack=1.5),
            )

            summary = run_feasibility_audit(config, num_seeds=2, resets_per_seed=2)

            self.assertEqual(summary["infeasible_count"], 0)
            self.assertEqual(summary["infeasible_rate"], 0.0)
            self.assertIn("A", summary["per_type"])
            self.assertIn("B", summary["per_type"])
            self.assertIn("C", summary["per_type"])
            self.assertTrue((Path(temp_dir) / "results" / "feasibility_audit_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
