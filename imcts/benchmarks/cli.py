"""Command-line parsing for benchmark runs.

This module only defines the CLI surface. The effective benchmark settings are
resolved later in ``imcts.benchmarks.config.build_settings()``, where CLI
values override YAML, and YAML overrides code defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse raw benchmark CLI arguments.

    Notable behavior that is implemented elsewhere:

    - ``--group`` defaults to ``Nguyen`` at runner time, not here.
    - ``--cases`` accepts ``all``, comma-separated ids, ranges such as
      ``1-4``, and case names; selection is handled by the registry.
    - ``--output`` points at the exact per-group output directory, while
      ``--results-dir`` is treated as the root directory that later gets a
      ``/<group>`` suffix.
    """
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
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Root directory for benchmark outputs. Per-group CSV folders are created under this directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact output directory for this group's per-case CSV files. Overrides --results-dir for the current run.",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for independent seed runs. "
        "Defaults to physical core count. Use 1 to disable parallelism.",
    )
    return parser.parse_args(argv)
