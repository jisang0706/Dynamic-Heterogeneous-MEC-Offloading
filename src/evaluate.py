from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation scaffold")
    parser.add_argument("--checkpoint", type=Path, default=Path("models"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"evaluation scaffold: add checkpoint loading and metrics here ({args.checkpoint})")


if __name__ == "__main__":
    main()
