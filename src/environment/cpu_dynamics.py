from __future__ import annotations

import numpy as np


class CPUDynamics:
    def __init__(self, step_scale: float = 0.15, rng: np.random.Generator | None = None) -> None:
        self.step_scale = step_scale
        self.rng = rng if rng is not None else np.random.default_rng()

    def step(self, current_freq_ghz: float, min_freq_ghz: float, max_freq_ghz: float) -> float:
        delta = self.step_scale * float(self.rng.uniform(-1.0, 1.0))
        return float(np.clip(current_freq_ghz + delta, min_freq_ghz, max_freq_ghz))
