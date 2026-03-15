from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IPPOBaselineSpec:
    name: str = "IPPO"
    description: str = "Placeholder spec for an independent PPO baseline on the dynamic environment."
