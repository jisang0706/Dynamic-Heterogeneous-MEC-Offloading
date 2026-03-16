from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LiOriginalBaselineSpec:
    name: str = "Li et al. Exact"
    description: str = "Original static Li et al. code path using untouched source files."


def li_original_required_files(root: Path | str = "li_code") -> list[Path]:
    base = Path(root)
    return [base / "main.py", base / "controller.py", base / "rollout.py"]


def li_original_missing_files(root: Path | str = "li_code") -> list[Path]:
    return [path for path in li_original_required_files(root) if not path.exists()]


def li_original_available(root: Path | str = "li_code") -> bool:
    return len(li_original_missing_files(root)) == 0


def build_li_original_command(root: Path | str = "li_code") -> list[str]:
    base = Path(root)
    return ["python", str(base / "main.py")]
