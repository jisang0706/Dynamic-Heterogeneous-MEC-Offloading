from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MobilityState:
    position: np.ndarray
    velocity: np.ndarray
    mean_velocity: np.ndarray
    distance_m: float


class GaussMarkovMobility:
    def __init__(
        self,
        sigma_v: float = 0.5,
        dt: float = 0.5,
        min_distance_m: float = 50.0,
        max_distance_m: float = 250.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.sigma_v = sigma_v
        self.dt = dt
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.rng = rng if rng is not None else np.random.default_rng()

    def initialize(self, init_distance_m: float, mean_speed_m_s: float) -> MobilityState:
        position_angle = self.rng.uniform(0.0, 2.0 * np.pi)
        mean_angle = self.rng.uniform(0.0, 2.0 * np.pi)
        position = init_distance_m * np.array([np.cos(position_angle), np.sin(position_angle)], dtype=np.float32)
        mean_velocity = mean_speed_m_s * np.array([np.cos(mean_angle), np.sin(mean_angle)], dtype=np.float32)
        return MobilityState(
            position=position,
            velocity=mean_velocity.copy(),
            mean_velocity=mean_velocity,
            distance_m=float(init_distance_m),
        )

    def step(self, state: MobilityState, alpha: float) -> MobilityState:
        noise = self.rng.normal(size=2).astype(np.float32)
        scale = self.sigma_v * np.sqrt(max(1.0 - alpha ** 2, 0.0))
        velocity = alpha * state.velocity + (1.0 - alpha) * state.mean_velocity + scale * noise
        position = state.position + velocity * self.dt
        position, velocity = self._reflect(position, velocity)
        return MobilityState(
            position=position.astype(np.float32),
            velocity=velocity.astype(np.float32),
            mean_velocity=state.mean_velocity,
            distance_m=float(np.linalg.norm(position)),
        )

    def _reflect(self, position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radius = float(np.linalg.norm(position))
        if radius == 0.0:
            radial_direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            radial_direction = position / radius

        if radius > self.max_distance_m:
            position = radial_direction * self.max_distance_m
            velocity = velocity - 2.0 * np.dot(velocity, radial_direction) * radial_direction
        elif radius < self.min_distance_m:
            position = radial_direction * self.min_distance_m
            velocity = velocity - 2.0 * np.dot(velocity, radial_direction) * radial_direction

        return position, velocity
