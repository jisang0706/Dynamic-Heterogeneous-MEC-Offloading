from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from colab import paper_run


class PaperRunTests(unittest.TestCase):
    def test_parser_uses_stabilized_training_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = paper_run.build_parser().parse_args(["--workspace-root", tmp_dir])

        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.ppo_clip, 0.05)
        self.assertEqual(args.entropy_coeff, 1e-3)
        self.assertEqual(args.gradient_clip, 1.0)
        self.assertEqual(args.initial_action_std_env, 0.10)
        self.assertEqual(args.initial_offloading_mean_env, 0.75)
        self.assertEqual(args.initial_power_mean_env, 0.8)
        self.assertEqual(args.large_scale_profile, "paper_scale_v1")
        self.assertEqual(args.use_obs_scaling, "false")
        self.assertEqual(args.use_reward_scaling, "true")
        self.assertEqual(args.resource_scaling_mode, "linear_after_threshold")
        self.assertEqual(args.resource_scaling_base_agents, 5)
        self.assertEqual(args.resource_scaling_start_agents, 10)
        self.assertEqual(args.checkpoint_selection_mode, "milestone_best")

    def test_scale_profile_keeps_m5_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = paper_run.build_parser().parse_args(["--workspace-root", tmp_dir])
            spec = paper_run.RunSpec(
                stage_id="core",
                variant_id="A1",
                runner_kind="ppo",
                num_agents=5,
                seed=42,
                episode_length=200,
                train_episodes=4000,
                eval_episodes=50,
                output_root=Path(tmp_dir) / "core" / "m05" / "seed_42" / "a1",
            )
            profile = paper_run.resolve_scale_run_profile(spec, args)

        self.assertEqual(profile.resource_scaling_mode, "linear_after_threshold")
        self.assertEqual(profile.total_bandwidth_hz, 10e6)
        self.assertEqual(profile.server_cpu_ghz, 25.0)
        self.assertEqual(profile.u_slack, 1.5)
        self.assertEqual(profile.initial_offloading_mean_env, 0.75)

    def test_scale_profile_tunes_m10_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = paper_run.build_parser().parse_args(["--workspace-root", tmp_dir])
            spec = paper_run.RunSpec(
                stage_id="core",
                variant_id="A1",
                runner_kind="ppo",
                num_agents=10,
                seed=42,
                episode_length=200,
                train_episodes=4000,
                eval_episodes=50,
                output_root=Path(tmp_dir) / "core" / "m10" / "seed_42" / "a1",
            )
            profile = paper_run.resolve_scale_run_profile(spec, args)

        self.assertEqual(profile.resource_scaling_mode, "fixed")
        self.assertEqual(profile.total_bandwidth_hz, 20e6)
        self.assertEqual(profile.server_cpu_ghz, 60.0)
        self.assertEqual(profile.u_slack, 1.9)
        self.assertEqual(profile.initial_offloading_mean_env, 0.60)

    def test_resolve_checkpoint_targets_includes_stage_milestones_and_final_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = paper_run.build_parser().parse_args(["--workspace-root", tmp_dir])
            spec = paper_run.RunSpec(
                stage_id="core",
                variant_id="A1",
                runner_kind="ppo",
                num_agents=5,
                seed=42,
                episode_length=200,
                train_episodes=800,
                eval_episodes=50,
                output_root=Path(tmp_dir) / "core" / "m05" / "seed_42" / "a1",
            )

        self.assertEqual(paper_run.resolve_checkpoint_targets(spec, args), [100, 200, 400, 800])

    def test_select_checkpoint_candidates_prefers_final_checkpoint_at_same_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            models_dir = output_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            checkpoints = {
                "checkpoint_ep0100_u0025.pt": {"episodes_completed": 100, "update_index": 25},
                "checkpoint_ep0400_u0100.pt": {"episodes_completed": 400, "update_index": 100},
                "checkpoint_latest.pt": {"episodes_completed": 4000, "update_index": 1000},
                "checkpoint_final.pt": {"episodes_completed": 4000, "update_index": 1000},
            }
            for filename, payload in checkpoints.items():
                torch.save(payload, models_dir / filename)

            candidates = paper_run.discover_checkpoint_candidates(output_root, "ppo")
            selected = paper_run.select_checkpoint_candidates(candidates, [100, 4000])

        self.assertEqual([info.path.name for info in selected], ["checkpoint_ep0100_u0025.pt", "checkpoint_final.pt"])


if __name__ == "__main__":
    unittest.main()
