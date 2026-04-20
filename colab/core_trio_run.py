from __future__ import annotations

import argparse

from colab import paper_run

TRIO_STAGE = "core"
TRIO_VARIANTS = ("A1", "B1", "QAG")
TRIO_NUM_AGENTS = (5, 10)


def build_parser() -> argparse.ArgumentParser:
    parser = paper_run.build_parser()
    parser.description = (
        "Run the publication-stage core trio only: "
        "A1 (RC-P-GCN-MAPPO), B1 (MAPPO), and QAG for M in {5, 10}."
    )
    return parser


def lock_trio_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stage != TRIO_STAGE:
        raise SystemExit("core_trio_run fixes --stage core. Remove the override and rerun.")
    if args.variants is not None:
        raise SystemExit("core_trio_run fixes --variants to A1 B1 QAG. Remove the override and rerun.")
    if args.num_agents is not None:
        raise SystemExit("core_trio_run fixes --num-agents to 5 10. Remove the override and rerun.")

    args.stage = TRIO_STAGE
    args.variants = list(TRIO_VARIANTS)
    args.num_agents = list(TRIO_NUM_AGENTS)
    return args


def main(argv: list[str] | None = None) -> None:
    args = lock_trio_args(build_parser().parse_args(argv))
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    paper_run.run_stage(TRIO_STAGE, args, paper_run.REPO_ROOT)


if __name__ == "__main__":
    main()
