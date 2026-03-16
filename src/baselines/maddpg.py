from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MADDPGBaselineSpec:
    name: str = "MADDPG"
    description: str = "Off-policy continuous-control baseline reserved for a separate implementation."


def run_maddpg_baseline(*args, **kwargs):
    raise NotImplementedError(
        "B8 MADDPG is intentionally left as a separate implementation path and is not wired into this trainer yet."
    )
