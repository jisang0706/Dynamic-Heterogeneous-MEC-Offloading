from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.buffer import RolloutBuffer, Transition
from src.config import EnvironmentConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.networks import MultiAgentRoleConditionedActor
from src.train import PPOTrainer
from src.utils import ObservationScaler, RewardScaler, compute_gae


class Task6TrainingPipelineTests(unittest.TestCase):
    def test_compute_gae_resets_across_done_boundaries(self) -> None:
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        values = torch.zeros(3, dtype=torch.float32)
        dones = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=1.0,
            gae_lambda=1.0,
            last_value=0.0,
            normalize_advantages=False,
        )

        expected = torch.tensor([3.0, 2.0, 3.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_multi_agent_actor_supports_shared_and_individual_modes(self) -> None:
        obs = torch.randn(4, 5, 16)
        role_mu = torch.randn(4, 5, 3)

        shared_actor = MultiAgentRoleConditionedActor(
            num_agents=5,
            actor_type="shared",
            obs_dim=16,
            role_dim=3,
            action_dim=4,
            hidden_dim=64,
            use_role=True,
        )
        individual_actor = MultiAgentRoleConditionedActor(
            num_agents=5,
            actor_type="individual",
            obs_dim=16,
            role_dim=3,
            action_dim=4,
            hidden_dim=64,
            use_role=True,
        )

        shared_mean, shared_std = shared_actor(obs, role_mu)
        individual_mean, individual_std = individual_actor(obs, role_mu)

        self.assertEqual(tuple(shared_mean.shape), (4, 5, 4))
        self.assertEqual(tuple(shared_std.shape), (4, 5, 4))
        self.assertEqual(tuple(individual_mean.shape), (4, 5, 4))
        self.assertEqual(tuple(individual_std.shape), (4, 5, 4))

    def test_shared_actor_can_differentiate_agents_with_identity_conditioning(self) -> None:
        actor = MultiAgentRoleConditionedActor(
            num_agents=5,
            actor_type="shared",
            obs_dim=16,
            role_dim=3,
            action_dim=4,
            hidden_dim=64,
            use_role=False,
        )
        obs = torch.zeros(5, 16)

        mean, _ = actor(obs)

        self.assertFalse(torch.allclose(mean[0], mean[1]))

    def test_individual_actor_can_use_global_context_summary(self) -> None:
        actor = MultiAgentRoleConditionedActor(
            num_agents=3,
            actor_type="individual",
            obs_dim=16,
            role_dim=3,
            action_dim=4,
            hidden_dim=64,
            use_role=False,
            context_pooling="mean_max_min",
        )
        baseline_obs = torch.zeros(3, 16)
        shifted_obs = torch.zeros(3, 16)
        shifted_obs[1, 0] = 1.0

        baseline_mean, _ = actor(baseline_obs)
        shifted_mean, _ = actor(shifted_obs)

        self.assertFalse(torch.allclose(baseline_mean[0], shifted_mean[0]))

    def test_prepare_model_observation_adds_delay_aware_actor_features(self) -> None:
        config = ExperimentConfig(
            seed=5,
            environment=EnvironmentConfig(
                num_agents=5,
                episode_length=2,
                graph_type="star",
                use_delay_aware_actor_features=True,
            ),
            model=ModelConfig(
                critic_type="mlp",
                use_role=False,
                use_l_i=False,
                actor_type="individual",
                actor_context_pooling="mean_max_min",
            ),
            training=TrainingConfig(run_mode="train", total_episodes=1, update_every_episodes=1),
        )
        trainer = PPOTrainer(config)
        observation = trainer.env.reset()

        actor_obs, core_obs, server_info = trainer._prepare_model_observation(observation, update_stats=True)

        self.assertEqual(tuple(core_obs.shape), (5, 14))
        self.assertEqual(tuple(server_info.shape), (3,))
        self.assertEqual(tuple(actor_obs.shape), (5, 19))
        self.assertIsNotNone(trainer.actor_obs_scaler)
        self.assertTrue(torch.isfinite(actor_obs).all())

    def test_rollout_buffer_computes_agentwise_returns_from_stored_transitions(self) -> None:
        buffer = RolloutBuffer()
        for reward_value, done in ((1.0, False), (2.0, True)):
            reward = np.full(2, reward_value, dtype=np.float32)
            buffer.add(
                Transition(
                    actor_obs=np.zeros((2, 16), dtype=np.float32),
                    core_obs=np.zeros((2, 14), dtype=np.float32),
                    server_info=np.zeros(3, dtype=np.float32),
                    role_mu=np.zeros((2, 3), dtype=np.float32),
                    action=np.zeros((2, 4), dtype=np.float32),
                    log_prob=np.zeros(2, dtype=np.float32),
                    reward=reward,
                    done=done,
                    value=0.0,
                )
            )

        gae_batch = buffer.compute_returns_and_advantages(
            gamma=1.0,
            gae_lambda=1.0,
            last_value=0.0,
            normalize_advantages=False,
        )

        expected_returns = torch.tensor([[3.0, 3.0], [2.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(gae_batch["return"], expected_returns))

    def test_observation_and_reward_scalers_produce_finite_outputs(self) -> None:
        obs_scaler = ObservationScaler(shape=(3,))
        reward_scaler = RewardScaler(gamma=0.99)

        scaled_obs = obs_scaler.update_and_transform(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
        scaled_reward = reward_scaler.scale(5.0)

        self.assertEqual(scaled_obs.shape, (1, 3))
        self.assertTrue(np.isfinite(scaled_obs).all())
        self.assertTrue(np.isfinite(scaled_reward))

    def test_collect_rollouts_respects_global_max_step_budget(self) -> None:
        config = ExperimentConfig(
            seed=11,
            environment=EnvironmentConfig(num_agents=5, episode_length=2, graph_type="star"),
            model=ModelConfig(critic_type="mlp", use_role=False, use_l_i=False, actor_type="shared"),
            training=TrainingConfig(run_mode="train", total_episodes=2, update_every_episodes=2),
        )
        trainer = PPOTrainer(config)

        buffer, last_value, episode_rewards = trainer.collect_rollouts(num_episodes=2, max_steps=3)

        self.assertEqual(len(buffer), 3)
        self.assertEqual(len(episode_rewards), 2)
        self.assertTrue(np.isfinite(last_value.detach().cpu().numpy()).all())

    def test_tiny_ppo_update_produces_finite_losses(self) -> None:
        config = ExperimentConfig(
            seed=7,
            environment=EnvironmentConfig(
                num_agents=5,
                episode_length=4,
                graph_type="star",
            ),
            model=ModelConfig(
                critic_type="mlp",
                use_role=True,
                use_l_i=True,
                actor_type="shared",
                role_dim=3,
                action_dim=4,
            ),
            training=TrainingConfig(
                run_mode="train",
                learning_rate=4e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ppo_clip=0.1,
                entropy_coeff=0.01,
                l_i_coeff=1e-4,
                lambda_var=1e-5,
                sigma_floor=0.05,
                gradient_clip=2.0,
                update_every_episodes=1,
                ppo_epochs=1,
                batch_size=2,
                total_episodes=1,
                smoke_steps=2,
                trajectory_window=2,
                trajectory_action_scale=10.0,
            ),
        )
        trainer = PPOTrainer(config)

        buffer, last_value, episode_rewards = trainer.collect_rollouts(num_episodes=1, max_steps=2)
        update = trainer.update(buffer, last_value=last_value)

        self.assertEqual(len(episode_rewards), 1)
        self.assertGreater(update.steps, 0)
        self.assertTrue(np.isfinite(update.actor_loss))
        self.assertTrue(np.isfinite(update.critic_loss))
        self.assertTrue(np.isfinite(update.entropy))
        self.assertTrue(update.l_i_loss is None or np.isfinite(update.l_i_loss))
        self.assertTrue(update.effective_l_i_coeff is None or np.isfinite(update.effective_l_i_coeff))
        self.assertTrue(update.l_var_loss is None or np.isfinite(update.l_var_loss))
        if update.role_mu_var_per_dim is not None:
            self.assertEqual(len(update.role_mu_var_per_dim), 3)
        if update.role_sigma_mean_per_dim is not None:
            self.assertEqual(len(update.role_sigma_mean_per_dim), 3)
        self.assertEqual(len(update.policy_mean_env_mean_per_dim or []), 4)
        self.assertEqual(len(update.policy_mean_env_std_per_dim or []), 4)
        self.assertEqual(len(update.policy_std_env_mean_per_dim or []), 4)
        self.assertEqual(len(update.sampled_env_action_mean_per_dim or []), 4)
        self.assertEqual(len(update.sampled_env_action_std_per_dim or []), 4)
        self.assertTrue(
            update.sampled_env_action_near_zero_fraction is None
            or np.isfinite(update.sampled_env_action_near_zero_fraction)
        )
        self.assertTrue(
            update.sampled_env_action_near_one_fraction is None
            or np.isfinite(update.sampled_env_action_near_one_fraction)
        )
        self.assertTrue(update.non_timeout_task_fraction is None or np.isfinite(update.non_timeout_task_fraction))
        self.assertEqual(len(update.policy_log_std_mean_per_dim or []), 4)
        self.assertTrue(
            update.near_zero_sigma_fraction is None or np.isfinite(update.near_zero_sigma_fraction)
        )

    def test_trainer_resume_restores_checkpoint_state_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = ExperimentConfig(
                seed=17,
                output_root=root,
                environment=EnvironmentConfig(num_agents=5, episode_length=2, graph_type="star"),
                model=ModelConfig(critic_type="mlp", use_role=False, use_l_i=False, actor_type="shared"),
                training=TrainingConfig(run_mode="train", total_episodes=2, update_every_episodes=1),
            )
            trainer = PPOTrainer(config)
            trainer.episode_history = [{"episode": 1, "joint_reward": -10.0, "steps": 2}]
            trainer.update_history = [
                {
                    "update": 1,
                    "episodes_completed": 1,
                    "steps": 2,
                    "mean_joint_reward": -10.0,
                    "mean_scaled_joint_reward": -1.0,
                    "actor_loss": 0.1,
                    "critic_loss": 0.2,
                    "entropy": 0.3,
                    "l_i_loss": None,
                    "l_var_loss": None,
                    "role_mu_var_per_dim": None,
                    "role_sigma_mean_per_dim": None,
                    "near_zero_sigma_fraction": None,
                }
            ]
            trainer.episodes_completed = 1
            trainer.updates_completed = 1
            checkpoint_path = trainer._save_checkpoint(1, 1, suffix="latest")

            resumed_config = ExperimentConfig(
                seed=17,
                output_root=root,
                environment=EnvironmentConfig(num_agents=5, episode_length=2, graph_type="star"),
                model=ModelConfig(critic_type="mlp", use_role=False, use_l_i=False, actor_type="shared"),
                training=TrainingConfig(
                    run_mode="train",
                    total_episodes=2,
                    update_every_episodes=1,
                    resume_from=checkpoint_path,
                ),
            )
            resumed_trainer = PPOTrainer(resumed_config)

            self.assertEqual(resumed_trainer.episodes_completed, 1)
            self.assertEqual(resumed_trainer.updates_completed, 1)
            self.assertEqual(len(resumed_trainer.episode_history), 1)
            self.assertEqual(len(resumed_trainer.update_history), 1)
            self.assertEqual(
                resumed_trainer.episode_log_path.read_text(encoding="utf-8").count("\n"),
                1,
            )
            self.assertEqual(
                resumed_trainer.update_log_path.read_text(encoding="utf-8").count("\n"),
                1,
            )


if __name__ == "__main__":
    unittest.main()
