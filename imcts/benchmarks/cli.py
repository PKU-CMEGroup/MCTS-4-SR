"""Command-line parsing for benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run imcts benchmarks.")
    parser.add_argument("--group", default=None, help="Benchmark group name, e.g. Nguyen or BlackBox.")
    parser.add_argument("--cases", default="all", help="Case ids/names, e.g. all, 1,2,4, 1-4, Nguyen-5.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path. Defaults depend on the group.")
    parser.add_argument("--runs", type=int, default=None, help="Number of seeds per case.")
    parser.add_argument("--seed-start", type=int, default=None, help="Offset into the fixed benchmark seed list.")
    parser.add_argument("--samples", type=int, default=None, help="Override sample count for synthetic cases.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Directory containing black-box dataset subfolders.")
    parser.add_argument("--label", default=None, help="Target column name for black-box datasets.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Fraction reserved for the test split.")
    parser.add_argument("--output", type=Path, default=None, help="CSV output path.")
    parser.add_argument(
        "--split-by-case",
        action="store_true",
        help="Write one CSV per case. In this mode --output is treated as an output directory.",
    )
    parser.add_argument("--list", action="store_true", help="List available groups/cases and exit.")
    parser.add_argument("--ops", default=None, help="Comma-separated primitive set.")
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-unary", type=int, default=None)
    parser.add_argument("--max-constants", type=int, default=None)
    parser.add_argument("--lm-iterations", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gp-rate", type=float, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--exploration-rate", type=float, default=None)
    parser.add_argument("--succ-error-tol", type=float, default=None)
    parser.add_argument("--max-wall-time-hours", type=float, default=None)
    return parser.parse_args(argv)
