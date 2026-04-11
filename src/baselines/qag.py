from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment import DynamicMECEnv


@dataclass(slots=True)
class QAGBaselineSpec:
    name: str = "Queue-Aware Greedy"
    description: str = "Queue-aware non-learning heuristic using current queues, channel, CPU, and task thresholds."


def _max_tx_rate_bps(env: DynamicMECEnv) -> np.ndarray:
    device_bandwidth_hz = env.config.effective_total_bandwidth_hz / float(env.config.num_agents)
    noise_power_w = env.config.noise_density_w_hz * device_bandwidth_hz
    tx_power_w = env.max_tx_powers_mw / 1000.0
    snr = (tx_power_w * env.channel_gains) / max(noise_power_w, 1e-12)
    return np.maximum(device_bandwidth_hz * np.log2(1.0 + np.maximum(snr, 0.0)), env.config.min_rate_bps).astype(np.float32)


def queue_aware_greedy_actions(env: DynamicMECEnv) -> np.ndarray:
    num_agents = env.config.num_agents
    num_tasks = env.config.num_tasks_per_step
    action = np.zeros((num_agents, num_tasks + 1), dtype=np.float32)

    beta_max_bps = _max_tx_rate_bps(env)
    task_work_gcycles = env.task_data_size_mb * env.task_density_gcycles_per_mb
    edge_queue = float(env.edge_queue)
    server_freq = max(env.config.effective_server_cpu_ghz, 1e-6)

    for agent_idx in range(num_agents):
        local_queue = float(env.local_queues[agent_idx])
        cpu_freq = max(float(env.cpu_freqs_ghz[agent_idx]), 1e-6)
        rate_bps = max(float(beta_max_bps[agent_idx]), env.config.min_rate_bps)

        for task_idx in range(num_tasks):
            data_size_mb = float(env.task_data_size_mb[agent_idx, task_idx])
            task_work = float(task_work_gcycles[agent_idx, task_idx])
            deadline = float(env.task_deadlines_s[agent_idx, task_idx])

            d_local_full = (local_queue + task_work) / cpu_freq
            d_edge_full = (data_size_mb * 1e6) / rate_bps + (edge_queue + task_work) / server_freq

            if d_local_full <= deadline and d_edge_full > deadline:
                theta = 0.0
            elif d_edge_full <= deadline and d_local_full > deadline:
                theta = 1.0
            else:
                theta = np.clip(
                    0.5 + (d_local_full - d_edge_full) / (2.0 * (d_local_full + d_edge_full + 1e-8)),
                    0.0,
                    1.0,
                )
            action[agent_idx, task_idx] = float(theta)

        action[agent_idx, -1] = 1.0 if np.any(action[agent_idx, :-1] > 0.05) else 0.0

    return action.astype(np.float32)
