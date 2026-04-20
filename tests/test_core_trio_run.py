from __future__ import annotations

import tempfile
import unittest

from colab import core_trio_run


class CoreTrioRunTests(unittest.TestCase):
    def test_lock_trio_args_sets_expected_stage_variants_and_agent_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = core_trio_run.build_parser().parse_args(["--workspace-root", tmp_dir])

        locked = core_trio_run.lock_trio_args(args)
        self.assertEqual(locked.stage, "core")
        self.assertEqual(locked.variants, ["A1", "B1", "QAG"])
        self.assertEqual(locked.num_agents, [5, 10])

    def test_lock_trio_args_rejects_variant_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = core_trio_run.build_parser().parse_args(
                ["--workspace-root", tmp_dir, "--variants", "A1"]
            )

        with self.assertRaises(SystemExit):
            core_trio_run.lock_trio_args(args)

    def test_lock_trio_args_keeps_core_training_budget_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = core_trio_run.build_parser().parse_args(["--workspace-root", tmp_dir])

        locked = core_trio_run.lock_trio_args(args)
        self.assertIsNone(locked.train_episodes)
        self.assertIsNone(locked.eval_episodes)


if __name__ == "__main__":
    unittest.main()
