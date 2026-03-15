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
    """Environment scaffold with dynamic observations and placeholder queue mechanics."""

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
        self.edge_queue = 0.0
        self.prev_edge_queue = 0.0
        self.task_matrix = np.empty((0, config.num_tasks_per_step, 3), dtype=np.float32)

    @property
    def positions(self) -> np.ndarray:
        if not self.mobility_states:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([state.position for state in self.mobility_states]).astype(np.float32)

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
        self.task_matrix = self._sample_tasks()
        return self._build_observation()

    def step(self, actions: np.ndarray) -> tuple[ObservationBundle, np.ndarray, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.config.num_agents, 4)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected actions with shape {expected_shape}, got {actions.shape}")

        clipped = np.clip(actions, 0.0, 1.0)
        offload = clipped[:, : self.config.num_tasks_per_step]
        power_ratio = clipped[:, -1]

        task_work = self.task_matrix[:, :, 0] * self.task_matrix[:, :, 1]
        local_work = np.sum(task_work * (1.0 - offload), axis=1)
        edge_work = np.sum(task_work * offload, axis=1)

        local_service = self.cpu_freqs_ghz * self.config.dt
        self.local_queues = np.maximum(0.0, self.local_queues + local_work - local_service)

        self.prev_edge_queue = self.edge_queue
        edge_arrivals = float(edge_work.sum())
        edge_service = self.config.server_cpu_ghz * self.config.dt
        self.edge_queue = max(0.0, self.edge_queue + edge_arrivals - edge_service)

        transmission_delay = edge_work / np.maximum(power_ratio * 3.0 + 1e-3, 1e-3)
        local_delay = local_work / np.maximum(self.cpu_freqs_ghz, 1e-3)
        estimated_delay = local_delay + transmission_delay
        deadline = self.task_matrix[:, :, 2].mean(axis=1)
        timeout = estimated_delay > deadline

        reward = -(
            0.5 * self.local_queues
            + 0.1 * edge_work
            + 0.05 * power_ratio
        )
        reward = reward.astype(np.float32)
        reward[timeout] -= 5.0

        self._advance_dynamics()
        self.task_matrix = self._sample_tasks()
        self.step_count += 1
        done = self.step_count >= self.config.episode_length
        info = {
            "edge_queue": float(self.edge_queue),
            "delta_edge_queue": float(self.edge_queue - self.prev_edge_queue),
            "mean_distance_m": float(np.mean([state.distance_m for state in self.mobility_states])),
            "timeout_ratio": float(np.mean(timeout.astype(np.float32))),
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
        data_size_mb = self.rng.uniform(0.16, 1.6, size=shape)
        comp_density = self.rng.uniform(0.2, 2.0, size=shape)
        deadline = self.rng.uniform(0.2, 1.0, size=shape)
        return np.stack((data_size_mb, comp_density, deadline), axis=-1).astype(np.float32)

    def _build_observation(self) -> ObservationBundle:
        device_obs = []
        for index in range(self.config.num_agents):
            state = self.mobility_states[index]
            cpu_freq = float(self.cpu_freqs_ghz[index])
            max_tx_power = float(self.max_tx_powers_mw[index])
            channel_ratio = self._channel_ratio(state.distance_m)
            flat_tasks = self.task_matrix[index].reshape(-1)
            task_features = np.asarray(
                [
                    flat_tasks[0] / 1.6,
                    flat_tasks[1] / 2.0,
                    flat_tasks[2] / 1.0,
                    flat_tasks[3] / 1.6,
                    flat_tasks[4] / 2.0,
                    flat_tasks[5] / 1.0,
                    flat_tasks[6] / 1.6,
                    flat_tasks[7] / 2.0,
                    flat_tasks[8] / 1.0,
                ],
                dtype=np.float32,
            )
            obs = np.concatenate(
                [
                    np.asarray(
                        [
                            np.clip(self.local_queues[index], 0.0, 20.0) / 10.0,
                            np.clip(channel_ratio, 0.0, 20.0) / 10.0,
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
                np.clip(self.edge_queue, 0.0, 20.0) / 10.0,
                np.clip(delta_edge_queue, -10.0, 10.0) / 10.0,
                self.config.server_cpu_ghz / 25.0,
            ],
            dtype=np.float32,
        )
        return ObservationBundle(
            device_obs=np.stack(device_obs).astype(np.float32),
            server_obs=server_obs,
        )

    def _channel_ratio(self, distance_m: float) -> float:
        path_loss = self.config.path_loss_kappa_linear * (distance_m ** (-self.config.path_loss_exp))
        fading = float(self.rng.exponential(1.0))
        bandwidth_hz = 10e6
        noise_density_w_hz = 10 ** ((-174.0 - 30.0) / 10.0)
        noise_power = noise_density_w_hz * bandwidth_hz
        return float((path_loss * fading) / max(noise_power, 1e-12))
