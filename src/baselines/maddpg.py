from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MADDPGBaselineSpec:
    name: str = "MADDPG"
    description: str = "Placeholder spec for an off-policy continuous-control baseline."
