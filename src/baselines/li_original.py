from __future__ import annotations

from pathlib import Path


def li_original_available(root: Path | str = "li_code") -> bool:
    base = Path(root)
    required = [base / "main.py", base / "controller.py", base / "rollout.py"]
    return all(path.exists() for path in required)
