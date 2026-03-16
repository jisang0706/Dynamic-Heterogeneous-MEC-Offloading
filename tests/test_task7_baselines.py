from __future__ import annotations

import unittest

from src.baselines import (
    DeterministicContextTrainer,
    IPPOTrainer,
    apply_experiment_variant,
    li_original_available,
    li_original_missing_files,
    run_fixed_policy_baseline,
)
from src.config import EnvironmentConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.train import SmokeRunSummary, TrainingRunSummary, run_smoke_rollout, run_training


class Task7BaselineTests(unittest.TestCase):
    def test_variant_registry_applies_expected_baseline_and_ablation_settings(self) -> None:
        config = ExperimentConfig()

        b3_config, b3_variant = apply_experiment_variant(config, "B3")
        a4_config, a4_variant = apply_experiment_variant(config, "A4")

        self.assertIsNotNone(b3_variant)
        self.assertEqual(b3_config.model.critic_type, "pgcn")
        self.assertFalse(b3_config.model.use_role)
        self.assertEqual(b3_config.model.actor_type, "shared")

        self.assertIsNotNone(a4_variant)
        self.assertEqual(a4_config.environment.graph_type, "star_proximity")
        self.assertTrue(a4_config.model.use_role)
        self.assertTrue(a4_config.model.use_l_i)

    def test_fixed_policy_baseline_returns_summary(self) -> None:
        config = ExperimentConfig(
            environment=EnvironmentConfig(num_agents=5, episode_length=2),
            training=TrainingConfig(total_episodes=1),
        )

        summary = run_fixed_policy_baseline(config, "LOCAL_ONLY", num_episodes=1)

        self.assertEqual(summary.policy_name, "LOCAL_ONLY")
        self.assertEqual(summary.episodes, 1)

    def test_li_original_reports_missing_files_in_current_workspace(self) -> None:
        self.assertFalse(li_original_available())
        self.assertGreater(len(li_original_missing_files()), 0)

    def test_deterministic_context_trainer_runs_smoke_rollout(self) -> None:
        config = ExperimentConfig(
            seed=5,
            environment=EnvironmentConfig(num_agents=5, episode_length=4, graph_type="star"),
            model=ModelConfig(critic_type="pgcn", use_role=True, use_l_i=False, actor_type="shared", role_dim=3),
            training=TrainingConfig(run_mode="smoke", smoke_steps=2),
        )

        trainer = DeterministicContextTrainer(config)
        summary = trainer.run_smoke_rollout()

        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.critic_type, "pgcn")

    def test_ippo_trainer_runs_smoke_rollout(self) -> None:
        config = ExperimentConfig(
            seed=9,
            environment=EnvironmentConfig(num_agents=5, episode_length=4, graph_type="star"),
            model=ModelConfig(critic_type="mlp", use_role=False, use_l_i=False, actor_type="individual", actor_hidden_dim=200),
            training=TrainingConfig(run_mode="smoke", smoke_steps=2),
        )

        trainer = IPPOTrainer(config)
        summary = trainer.run_smoke_rollout()

        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.critic_type, "ippo")

    def test_variant_entrypoints_return_public_summary_types_for_ippo(self) -> None:
        smoke_config = ExperimentConfig(
            seed=11,
            environment=EnvironmentConfig(num_agents=5, episode_length=4, graph_type="star"),
            model=ModelConfig(critic_type="mlp", use_role=False, use_l_i=False, actor_type="individual", actor_hidden_dim=200),
            training=TrainingConfig(run_mode="smoke", smoke_steps=2, variant_id="B7"),
        )
        train_config = ExperimentConfig(
            seed=12,
            environment=EnvironmentConfig(num_agents=5, episode_length=2, graph_type="star"),
            model=ModelConfig(critic_type="mlp", use_role=False, use_l_i=False, actor_type="individual", actor_hidden_dim=200),
            training=TrainingConfig(
                run_mode="train",
                variant_id="B7",
                total_episodes=1,
                update_every_episodes=1,
                batch_size=2,
                ppo_epochs=1,
            ),
        )

        smoke_summary = run_smoke_rollout(smoke_config)
        train_summary = run_training(train_config)

        self.assertIsInstance(smoke_summary, SmokeRunSummary)
        self.assertEqual(smoke_summary.critic_type, "ippo")
        self.assertIsInstance(train_summary, TrainingRunSummary)
        self.assertEqual(train_summary.critic_type, "ippo")


if __name__ == "__main__":
    unittest.main()
