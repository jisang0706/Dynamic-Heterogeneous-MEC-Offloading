from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualization scaffold")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"visualization scaffold: add plotting logic for {args.results_dir}")


if __name__ == "__main__":
    main()
