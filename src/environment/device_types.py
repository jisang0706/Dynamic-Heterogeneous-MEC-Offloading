from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeviceTypeSpec:
    name: str
    cpu_range_ghz: tuple[float, float]
    init_distance_range_m: tuple[float, float]
    tx_power_range_mw: tuple[float, float]
    speed_m_s: float
    alpha: float


@dataclass(slots=True)
class DeviceProfile:
    type_name: str
    cpu_range_ghz: tuple[float, float]
    cpu_ghz: float
    max_tx_power_mw: float
    init_distance_m: float
    speed_m_s: float
    alpha: float


DEVICE_TYPES: dict[str, DeviceTypeSpec] = {
    "A": DeviceTypeSpec("A", (2.5, 3.0), (80.0, 130.0), (200.0, 300.0), 3.0, 0.6),
    "B": DeviceTypeSpec("B", (2.0, 2.5), (100.0, 170.0), (150.0, 250.0), 1.5, 0.6),
    "C": DeviceTypeSpec("C", (1.5, 2.0), (120.0, 200.0), (100.0, 200.0), 0.8, 0.6),
}

AGENT_TYPE_DISTRIBUTIONS: dict[int, tuple[str, ...]] = {
    5: ("A", "A", "B", "B", "C"),
    10: ("A", "A", "A", "B", "B", "B", "B", "C", "C", "C"),
    15: ("A",) * 5 + ("B",) * 5 + ("C",) * 5,
    20: ("A",) * 7 + ("B",) * 7 + ("C",) * 6,
}


def get_type_sequence(num_agents: int) -> list[DeviceTypeSpec]:
    if num_agents in AGENT_TYPE_DISTRIBUTIONS:
        labels = AGENT_TYPE_DISTRIBUTIONS[num_agents]
    else:
        labels = tuple(("A", "B", "C")[index % 3] for index in range(num_agents))
    return [DEVICE_TYPES[label] for label in labels]


def sample_device_profile(spec: DeviceTypeSpec, rng: np.random.Generator) -> DeviceProfile:
    return DeviceProfile(
        type_name=spec.name,
        cpu_range_ghz=spec.cpu_range_ghz,
        cpu_ghz=float(rng.uniform(*spec.cpu_range_ghz)),
        max_tx_power_mw=float(rng.uniform(*spec.tx_power_range_mw)),
        init_distance_m=float(rng.uniform(*spec.init_distance_range_m)),
        speed_m_s=spec.speed_m_s,
        alpha=spec.alpha,
    )


def build_device_profiles(num_agents: int, rng: np.random.Generator) -> list[DeviceProfile]:
    return [sample_device_profile(spec, rng) for spec in get_type_sequence(num_agents)]
