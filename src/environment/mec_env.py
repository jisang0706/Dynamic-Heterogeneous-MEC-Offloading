from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import EnvironmentConfig
from src.environment.cpu_dynamics import CPUDynamics
from src.environment.device_types import DeviceProfile, build_device_profiles
from src.environment.mobility import GaussMarkovMobility, MobilityState


@dataclass(slots=True)
class ObservationBundle:
    device_obs: np.ndarray
    server_obs: np.ndarray


class DynamicMECEnv:
    """Dynamic heterogeneous MEC environment for TASK 1."""

    def __init__(self, config: EnvironmentConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.mobility = GaussMarkovMobility(
            sigma_v=config.sigma_v,
            dt=config.dt,
            min_distance_m=config.min_distance_m,
            max_distance_m=config.max_distance_m,
            rng=self.rng,
        )
        self.cpu_dynamics = CPUDynamics(rng=self.rng)
        self.step_count = 0
        self.device_profiles: list[DeviceProfile] = []
        self.mobility_states: list[MobilityState] = []
        self.cpu_freqs_ghz = np.empty(0, dtype=np.float32)
        self.max_tx_powers_mw = np.empty(0, dtype=np.float32)
        self.local_queues = np.empty(0, dtype=np.float32)
        self.channel_gains = np.empty(0, dtype=np.float32)
        self.channel_gain_ratios = np.empty(0, dtype=np.float32)
        self.edge_queue = 0.0
        self.prev_edge_queue = 0.0
        self.task_matrix = np.empty((0, config.num_tasks_per_step, 3), dtype=np.float32)
        self.last_reward_breakdown: dict[str, np.ndarray] = {}

    @property
    def positions(self) -> np.ndarray:
        if not self.mobility_states:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([state.position for state in self.mobility_states]).astype(np.float32)

    @property
    def distances_m(self) -> np.ndarray:
        if not self.mobility_states:
            return np.empty(0, dtype=np.float32)
        return np.asarray([state.distance_m for state in self.mobility_states], dtype=np.float32)

    @property
    def speeds_m_s(self) -> np.ndarray:
        if not self.mobility_states:
            return np.empty(0, dtype=np.float32)
        return np.asarray([np.linalg.norm(state.velocity) for state in self.mobility_states], dtype=np.float32)

    @property
    def task_data_size_mb(self) -> np.ndarray:
        return self.task_matrix[:, :, 0]

    @property
    def task_density_gcycles_per_mb(self) -> np.ndarray:
        return self.task_matrix[:, :, 1]

    @property
    def task_deadlines_s(self) -> np.ndarray:
        return self.task_matrix[:, :, 2]

    def reset(self) -> ObservationBundle:
        self.step_count = 0
        self.edge_queue = 0.0
        self.prev_edge_queue = 0.0
        self.local_queues = np.zeros(self.config.num_agents, dtype=np.float32)

        self.device_profiles = build_device_profiles(self.config.num_agents, self.rng)
        self.mobility_states = [
            self.mobility.initialize(profile.init_distance_m, profile.speed_m_s)
            for profile in self.device_profiles
        ]
        self.cpu_freqs_ghz = np.asarray([profile.cpu_ghz for profile in self.device_profiles], dtype=np.float32)
        self.max_tx_powers_mw = np.asarray([profile.max_tx_power_mw for profile in self.device_profiles], dtype=np.float32)
        self._update_channel_state()
        self.task_matrix = self._sample_tasks()
        self.last_reward_breakdown = {}
        return self._build_observation()

    def step(self, actions: np.ndarray) -> tuple[ObservationBundle, np.ndarray, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.config.num_agents, self.config.num_tasks_per_step + 1)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected actions with shape {expected_shape}, got {actions.shape}")

        clipped = np.clip(actions, 0.0, 1.0)
        offload = clipped[:, : self.config.num_tasks_per_step]
        power_ratio = clipped[:, -1]
        evaluation = self._evaluate_slot(offload=offload, power_ratio=power_ratio)
        self.local_queues = evaluation["next_local_queues"]
        self.prev_edge_queue = self.edge_queue
        self.edge_queue = float(evaluation["next_edge_queue"])
        reward = evaluation["device_rewards"]
        timeout_ratio = float(evaluation["timeout_mask"].mean())
        self.last_reward_breakdown = {
            "task_completion_delay_s": evaluation["task_completion_delay_s"],
            "task_normalized_cost": evaluation["task_normalized_cost"],
            "task_timeout_mask": evaluation["timeout_mask"].astype(np.float32),
        }

        self._advance_dynamics()
        self._update_channel_state()
        self.task_matrix = self._sample_tasks()
        self.step_count += 1
        done = self.step_count >= self.config.episode_length
        info = {
            "edge_queue": float(self.edge_queue),
            "delta_edge_queue": float(self.edge_queue - self.prev_edge_queue),
            "mean_distance_m": float(self.distances_m.mean()),
            "timeout_ratio": timeout_ratio,
            "mean_reward": float(reward.mean()),
            "mean_channel_gain": float(self.channel_gains.mean()),
        }
        return self._build_observation(), reward, done, info

    def _advance_dynamics(self) -> None:
        next_states: list[MobilityState] = []
        next_freqs: list[float] = []
        for profile, state, freq in zip(self.device_profiles, self.mobility_states, self.cpu_freqs_ghz):
            if self.config.use_mobility:
                state = self.mobility.step(state, alpha=profile.alpha)
            if self.config.use_cpu_dynamics:
                freq = self.cpu_dynamics.step(freq, *profile.cpu_range_ghz)
            next_states.append(state)
            next_freqs.append(freq)
        self.mobility_states = next_states
        self.cpu_freqs_ghz = np.asarray(next_freqs, dtype=np.float32)

    def _sample_tasks(self) -> np.ndarray:
        shape = (self.config.num_agents, self.config.num_tasks_per_step)
        data_size_mb = self.rng.uniform(*self.config.task_size_range_mb, size=shape)
        comp_density = self.rng.uniform(*self.config.task_density_range_gcycles_per_mb, size=shape)
        if self.config.delay_mode == "li_original":
            deadline = self.rng.uniform(*self.config.task_deadline_range_s, size=shape)
        else:
            deadline = self.compute_best_case_delay_components(
                data_size_mb=data_size_mb,
                comp_density_gcycles_per_mb=comp_density,
            )["delay_threshold_s"]
        return np.stack((data_size_mb, comp_density, deadline), axis=-1).astype(np.float32)

    def compute_best_case_delay_components(
        self,
        data_size_mb: np.ndarray | None = None,
        comp_density_gcycles_per_mb: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if data_size_mb is None:
            data_size_mb = self.task_data_size_mb
        if comp_density_gcycles_per_mb is None:
            comp_density_gcycles_per_mb = self.task_density_gcycles_per_mb
        if not self.device_profiles or self.channel_gains.size == 0 or self.max_tx_powers_mw.size == 0:
            raise ValueError("Device profiles, channel gains, and max transmit powers must be initialized before computing best-case delays.")

        data_size_mb = np.asarray(data_size_mb, dtype=np.float32)
        comp_density_gcycles_per_mb = np.asarray(comp_density_gcycles_per_mb, dtype=np.float32)
        task_work_gcycles = data_size_mb * comp_density_gcycles_per_mb

        f_max_type_ghz = np.asarray([profile.cpu_range_ghz[1] for profile in self.device_profiles], dtype=np.float32)
        d_local_best_s = task_work_gcycles / np.maximum(f_max_type_ghz[:, None], 1e-6)

        device_bandwidth_hz = self.config.effective_total_bandwidth_hz / float(self.config.num_agents)
        noise_power_w = self.config.noise_density_w_hz * device_bandwidth_hz
        tx_power_w = self.max_tx_powers_mw / 1000.0
        snr = (tx_power_w * self.channel_gains) / max(noise_power_w, 1e-12)
        beta_max_bps = device_bandwidth_hz * np.log2(1.0 + np.maximum(snr, 0.0))
        beta_max_bps = np.maximum(beta_max_bps, self.config.min_rate_bps)

        tx_delay_best_s = (data_size_mb * 1e6) / beta_max_bps[:, None]
        edge_compute_best_s = task_work_gcycles / max(self.config.effective_server_cpu_ghz, 1e-6)
        d_edge_best_s = tx_delay_best_s + edge_compute_best_s
        d_best_s = np.minimum(d_local_best_s, d_edge_best_s)
        delay_threshold_s = self.config.u_slack * d_best_s

        return {
            "task_work_gcycles": task_work_gcycles.astype(np.float32),
            "beta_max_bps": beta_max_bps.astype(np.float32),
            "d_local_best_s": d_local_best_s.astype(np.float32),
            "d_edge_best_s": d_edge_best_s.astype(np.float32),
            "d_best_s": d_best_s.astype(np.float32),
            "delay_threshold_s": delay_threshold_s.astype(np.float32),
        }

    def compute_actor_relative_features(self) -> np.ndarray:
        if self.config.actor_relative_feature_dim <= 0:
            return np.empty((self.config.num_agents, 0), dtype=np.float32)
        if self.config.actor_relative_feature_dim != 3:
            raise ValueError("compute_actor_relative_features currently supports actor_relative_feature_dim=3 only.")
        if self.local_queues.size == 0 or self.task_matrix.size == 0:
            return np.zeros((self.config.num_agents, self.config.actor_relative_feature_dim), dtype=np.float32)

        task_work_gcycles = self.task_data_size_mb * self.task_density_gcycles_per_mb
        mean_task_work_gcycles = task_work_gcycles.mean(axis=1)
        mean_task_bits = self.task_data_size_mb.mean(axis=1) * 1e6
        local_cpu_ghz = np.maximum(self.cpu_freqs_ghz, 1e-6)
        server_cpu_ghz = max(self.config.effective_server_cpu_ghz, 1e-6)

        local_queue_delay_s = self.local_queues / local_cpu_ghz
        edge_queue_delay_s = np.full(
            self.config.num_agents,
            self.edge_queue / server_cpu_ghz,
            dtype=np.float32,
        )

        device_bandwidth_hz = self.config.effective_total_bandwidth_hz / float(self.config.num_agents)
        noise_power_w = self.config.noise_density_w_hz * device_bandwidth_hz
        tx_power_w = self.max_tx_powers_mw / 1000.0
        snr = (tx_power_w * self.channel_gains) / max(noise_power_w, 1e-12)
        beta_max_bps = device_bandwidth_hz * np.log2(1.0 + np.maximum(snr, 0.0))
        beta_max_bps = np.maximum(beta_max_bps, self.config.min_rate_bps)

        mean_local_task_delay_s = mean_task_work_gcycles / local_cpu_ghz
        mean_edge_task_delay_s = (mean_task_bits / beta_max_bps) + (mean_task_work_gcycles / server_cpu_ghz)
        local_edge_delay_gap_s = (
            local_queue_delay_s + mean_local_task_delay_s
        ) - (
            edge_queue_delay_s + mean_edge_task_delay_s
        )

        return np.stack(
            (
                local_queue_delay_s,
                edge_queue_delay_s,
                local_edge_delay_gap_s,
            ),
            axis=-1,
        ).astype(np.float32)

    def _build_observation(self) -> ObservationBundle:
        device_obs = []
        for index in range(self.config.num_agents):
            state = self.mobility_states[index]
            cpu_freq = float(self.cpu_freqs_ghz[index])
            max_tx_power = float(self.max_tx_powers_mw[index])
            channel_ratio = float(self.channel_gain_ratios[index])
            flat_tasks = self.task_matrix[index].reshape(-1)
            task_features = np.asarray(
                [
                    flat_tasks[0] / self.config.task_size_range_mb[1],
                    flat_tasks[1] / self.config.task_density_range_gcycles_per_mb[1],
                    self._scale_deadline_observation(float(flat_tasks[2])),
                    flat_tasks[3] / self.config.task_size_range_mb[1],
                    flat_tasks[4] / self.config.task_density_range_gcycles_per_mb[1],
                    self._scale_deadline_observation(float(flat_tasks[5])),
                    flat_tasks[6] / self.config.task_size_range_mb[1],
                    flat_tasks[7] / self.config.task_density_range_gcycles_per_mb[1],
                    self._scale_deadline_observation(float(flat_tasks[8])),
                ],
                dtype=np.float32,
            )
            obs = np.concatenate(
                [
                    np.asarray(
                        [
                            self._scale_queue_observation(float(self.local_queues[index])),
                            self._scale_channel_observation(channel_ratio),
                        ],
                        dtype=np.float32,
                    ),
                    task_features,
                    np.asarray(
                        [
                            state.distance_m / self.config.max_distance_m,
                            cpu_freq / self.config.max_cpu_ghz,
                            max_tx_power / self.config.max_tx_power_mw,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            device_obs.append(obs)

        delta_edge_queue = self.edge_queue - self.prev_edge_queue
        server_obs = np.asarray(
            [
                self._scale_queue_observation(float(self.edge_queue)),
                self._scale_signed_queue_delta(delta_edge_queue),
                self.config.effective_server_cpu_ghz / 25.0,
            ],
            dtype=np.float32,
        )
        return ObservationBundle(
            device_obs=np.stack(device_obs).astype(np.float32),
            server_obs=server_obs,
        )

    def _evaluate_slot(self, offload: np.ndarray, power_ratio: np.ndarray) -> dict[str, np.ndarray | float]:
        task_work = self.task_data_size_mb * self.task_density_gcycles_per_mb
        local_work = task_work * (1.0 - offload)
        edge_work = task_work * offload
        offloaded_bits = self.task_data_size_mb * offload * 1e6

        tx_power_w = (power_ratio * self.max_tx_powers_mw) / 1000.0
        device_bandwidth_hz = self.config.effective_total_bandwidth_hz / float(self.config.num_agents)
        noise_power_w = self.config.noise_density_w_hz * device_bandwidth_hz
        snr = (tx_power_w * self.channel_gains) / max(noise_power_w, 1e-12)
        tx_rate_bps = device_bandwidth_hz * np.log2(1.0 + np.maximum(snr, 0.0))
        tx_rate_bps = np.maximum(tx_rate_bps, self.config.min_rate_bps)

        task_completion_delay_s = np.zeros_like(task_work, dtype=np.float32)
        task_normalized_cost = np.zeros_like(task_work, dtype=np.float32)
        task_lateness_clipped = np.zeros_like(task_work, dtype=np.float32)
        timeout_mask = np.zeros_like(task_work, dtype=bool)
        transmission_finish_s = np.zeros_like(task_work, dtype=np.float32)
        local_completion_s = np.zeros_like(task_work, dtype=np.float32)
        tx_energy_j = np.zeros_like(task_work, dtype=np.float32)

        for device_idx in range(self.config.num_agents):
            local_backlog = float(self.local_queues[device_idx])
            local_cumulative = 0.0
            tx_cumulative = 0.0
            rate_bps = float(tx_rate_bps[device_idx])
            power_w = float(tx_power_w[device_idx])
            cpu_freq = max(float(self.cpu_freqs_ghz[device_idx]), 1e-6)

            local_order = np.argsort(local_work[device_idx], kind="stable")
            for task_idx in local_order:
                local_piece = float(local_work[device_idx, task_idx])
                if local_piece > 0.0:
                    local_cumulative += local_piece
                    local_completion_s[device_idx, task_idx] = (local_backlog + local_cumulative) / cpu_freq

            tx_order = np.argsort(offloaded_bits[device_idx], kind="stable")
            for task_idx in tx_order:
                edge_piece = float(edge_work[device_idx, task_idx])
                if edge_piece > 0.0:
                    tx_duration_s = float(offloaded_bits[device_idx, task_idx]) / rate_bps
                    tx_cumulative += tx_duration_s
                    transmission_finish_s[device_idx, task_idx] = tx_cumulative
                    tx_energy_j[device_idx, task_idx] = power_w * tx_duration_s

        edge_completion_s, next_edge_queue = self._compute_edge_schedule(
            edge_work=edge_work,
            arrival_times_s=transmission_finish_s,
        )
        local_energy_j = local_work * self.config.local_energy_coeff_j_per_gcycle

        for device_idx in range(self.config.num_agents):
            for task_idx in range(self.config.num_tasks_per_step):
                local_delay = float(local_completion_s[device_idx, task_idx])
                edge_delay = float(edge_completion_s[device_idx, task_idx])

                if local_work[device_idx, task_idx] <= 0.0:
                    completion_delay = edge_delay
                elif edge_work[device_idx, task_idx] <= 0.0:
                    completion_delay = local_delay
                else:
                    completion_delay = max(local_delay, edge_delay)

                deadline = max(float(self.task_deadlines_s[device_idx, task_idx]), 1e-6)
                delay_norm = completion_delay / deadline
                lateness = max(0.0, delay_norm - 1.0)
                clipped_lateness = min(lateness, self.config.reward_lateness_clip)
                energy_norm = self._normalize_energy(
                    local_energy_j=float(local_energy_j[device_idx, task_idx]),
                    tx_energy_j=float(tx_energy_j[device_idx, task_idx]),
                )
                normalized_cost = (
                    self.config.delay_weight * delay_norm
                    + self.config.energy_weight * energy_norm
                )

                task_completion_delay_s[device_idx, task_idx] = completion_delay
                task_normalized_cost[device_idx, task_idx] = normalized_cost
                task_lateness_clipped[device_idx, task_idx] = clipped_lateness
                timeout_mask[device_idx, task_idx] = completion_delay > deadline

        device_rewards = np.zeros(self.config.num_agents, dtype=np.float32)
        for device_idx in range(self.config.num_agents):
            timeout_penalties = timeout_mask[device_idx].sum() * self.config.reward_timeout_penalty
            lateness_penalties = task_lateness_clipped[device_idx].sum() * self.config.reward_lateness_penalty
            non_timeout_cost = task_normalized_cost[device_idx][~timeout_mask[device_idx]].sum() * self.config.reward_scale
            device_rewards[device_idx] = -float(timeout_penalties + lateness_penalties + non_timeout_cost)

        next_local_queues = np.maximum(
            0.0,
            self.local_queues + local_work.sum(axis=1) - self.cpu_freqs_ghz * self.config.dt,
        ).astype(np.float32)
        return {
            "device_rewards": device_rewards,
            "next_local_queues": next_local_queues,
            "next_edge_queue": next_edge_queue,
            "task_completion_delay_s": task_completion_delay_s.astype(np.float32),
            "task_normalized_cost": task_normalized_cost.astype(np.float32),
            "task_lateness_clipped": task_lateness_clipped.astype(np.float32),
            "timeout_mask": timeout_mask,
        }

    def _compute_edge_schedule(
        self,
        edge_work: np.ndarray,
        arrival_times_s: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        edge_completion_s = np.zeros_like(edge_work, dtype=np.float32)
        server_freq = max(self.config.effective_server_cpu_ghz, 1e-6)
        edge_available_s = self.edge_queue / server_freq
        arrivals: list[tuple[float, int, int, float]] = []
        remaining_edge_work = max(0.0, self.edge_queue - server_freq * self.config.dt)

        for device_idx in range(self.config.num_agents):
            for task_idx in range(self.config.num_tasks_per_step):
                work = float(edge_work[device_idx, task_idx])
                if work > 0.0:
                    arrivals.append((float(arrival_times_s[device_idx, task_idx]), device_idx, task_idx, work))

        arrivals.sort(key=lambda item: (item[0], item[1], item[2]))
        for arrival_time_s, device_idx, task_idx, work in arrivals:
            service_start_s = max(arrival_time_s, edge_available_s)
            service_end_s = service_start_s + work / server_freq
            edge_completion_s[device_idx, task_idx] = service_end_s
            edge_available_s = service_end_s
            processed_by_slot_end = 0.0
            if service_start_s < self.config.dt:
                processed_by_slot_end = min(work, max(0.0, self.config.dt - service_start_s) * server_freq)
            remaining_edge_work += work - processed_by_slot_end

        return edge_completion_s, float(max(0.0, remaining_edge_work))

    def _normalize_energy(self, local_energy_j: float, tx_energy_j: float) -> float:
        local_norm = local_energy_j / max(self.config.local_energy_reference_j, 1e-6)
        tx_norm = tx_energy_j / max(self.config.tx_energy_reference_j, 1e-6)
        return float(0.5 * (local_norm + tx_norm))

    def _scale_queue_observation(self, queue_value: float) -> float:
        queue_reference = max(self.config.queue_clip_max * 5.0, 100.0)
        scaled = np.log1p(max(queue_value, 0.0)) / np.log1p(queue_reference)
        return float(np.clip(scaled, 0.0, 2.0))

    def _scale_signed_queue_delta(self, delta_queue: float) -> float:
        delta_reference = max(self.config.queue_clip_max, 20.0)
        scaled = np.sign(delta_queue) * np.log1p(abs(delta_queue)) / np.log1p(delta_reference)
        return float(np.clip(scaled, -2.0, 2.0))

    def _channel_ratio_reference(self) -> float:
        device_bandwidth_hz = self.config.effective_total_bandwidth_hz / float(self.config.num_agents)
        noise_power_w = self.config.noise_density_w_hz * device_bandwidth_hz
        min_distance_km = max(self.config.min_distance_m / 1000.0, 1e-6)
        best_large_scale_gain = self.config.path_loss_kappa_linear * (min_distance_km ** (-self.config.path_loss_exp))
        return max(best_large_scale_gain / max(noise_power_w, 1e-12), 1.0)

    def _scale_channel_observation(self, channel_ratio: float) -> float:
        reference = self._channel_ratio_reference()
        scaled = np.log10(1.0 + max(channel_ratio, 0.0)) / np.log10(1.0 + reference)
        return float(np.clip(scaled, 0.0, 2.0))

    def _scale_deadline_observation(self, deadline_s: float) -> float:
        deadline_reference_s = max(self.config.task_deadline_range_s[1], 5.0)
        scaled = np.log1p(max(deadline_s, 0.0)) / np.log1p(deadline_reference_s)
        return float(np.clip(scaled, 0.0, 2.0))

    def _update_channel_state(self) -> None:
        channel_gains = []
        channel_gain_ratios = []
        device_bandwidth_hz = self.config.effective_total_bandwidth_hz / float(self.config.num_agents)
        noise_power_w = self.config.noise_density_w_hz * device_bandwidth_hz

        for distance_m in self.distances_m:
            distance_km = max(float(distance_m) / 1000.0, 1e-6)
            large_scale_gain = self.config.path_loss_kappa_linear * (distance_km ** (-self.config.path_loss_exp))
            fading = float(self.rng.exponential(1.0))
            channel_gain = large_scale_gain * fading
            channel_gains.append(channel_gain)
            channel_gain_ratios.append(channel_gain / max(noise_power_w, 1e-12))

        self.channel_gains = np.asarray(channel_gains, dtype=np.float32)
        self.channel_gain_ratios = np.asarray(channel_gain_ratios, dtype=np.float32)
