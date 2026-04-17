"""Summarize benchmark CSV outputs under a configurable results directory."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import load_yaml_file, nested_get
from .defaults import DEFAULT_RESULTS_DIRNAME


@dataclass(frozen=True)
class SummaryRow:
    group: str
    case_id: int | None
    case_name: str
    runs: int
    mean_time_sec: float
    success_rate: float
    mean_evaluations: float
    median_test_r2: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark results under a configurable results directory.")
    parser.add_argument("groups", nargs="*", help="Optional benchmark groups to summarize, e.g. nguyen livermore.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path. If set, output.results_dir is used unless --results-dir overrides it.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Benchmark results directory. Defaults to output.results_dir from YAML or ./benchmark_results.",
    )
    parser.add_argument(
        "--level",
        choices=("group", "case", "both"),
        default="both",
        help="Whether to print per-group summaries, per-case summaries, or both.",
    )
    return parser.parse_args(argv)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def format_rate(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value * 100:.1f}%"


def format_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_eval_mean(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.1f}"


def finite_values(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean_or_nan(values: list[float]) -> float:
    filtered = finite_values(values)
    if not filtered:
        return float("nan")
    return statistics.fmean(filtered)


def median_or_nan(values: list[float]) -> float:
    filtered = finite_values(values)
    if not filtered:
        return float("nan")
    return float(statistics.median(filtered))


def iter_group_dirs(results_dir: Path, selected_groups: Sequence[str]) -> list[Path]:
    if selected_groups:
        return [results_dir / group.lower() for group in selected_groups]
    return sorted(path for path in results_dir.iterdir() if path.is_dir())


def load_rows(results_dir: Path, selected_groups: Sequence[str]) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    missing_groups: list[str] = []

    if not results_dir.exists():
        raise SystemExit(f"benchmark results directory not found: {results_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"benchmark results path is not a directory: {results_dir}")

    for group_dir in iter_group_dirs(results_dir, selected_groups):
        if not group_dir.exists():
            missing_groups.append(group_dir.name)
            continue
        csv_paths = sorted(group_dir.glob("*.csv"))
        for csv_path in csv_paths:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_rows.extend(reader)

    if missing_groups:
        missing_text = ", ".join(missing_groups)
        raise SystemExit(f"benchmark result groups not found under {results_dir}: {missing_text}")
    if not all_rows:
        raise SystemExit(f"no benchmark CSV files found under {results_dir}")
    return all_rows


def resolve_results_dir(args: argparse.Namespace, workspace_root: Path) -> Path:
    configured_dir = None
    if args.config is not None:
        config = load_yaml_file(args.config)
        configured_dir = nested_get(config, "output", "results_dir") or config.get("results_dir")

    raw_results_dir = args.results_dir or configured_dir or DEFAULT_RESULTS_DIRNAME
    results_dir = Path(raw_results_dir)
    if not results_dir.is_absolute():
        results_dir = workspace_root / results_dir
    return results_dir


def summarize_by_case(rows: Sequence[dict[str, str]]) -> list[SummaryRow]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["group"], int(row["case_id"]), row["case_name"])
        grouped[key].append(row)

    summaries: list[SummaryRow] = []
    for (group, case_id, case_name), case_rows in sorted(grouped.items()):
        times = [parse_float(row["time_sec"]) for row in case_rows]
        evals = [parse_float(row["evaluations"]) for row in case_rows]
        successes = [1.0 if parse_bool(row["success"]) else 0.0 for row in case_rows]
        test_r2_values = [parse_float(row["test_r2"]) for row in case_rows]
        summaries.append(
            SummaryRow(
                group=group,
                case_id=case_id,
                case_name=case_name,
                runs=len(case_rows),
                mean_time_sec=mean_or_nan(times),
                success_rate=mean_or_nan(successes),
                mean_evaluations=mean_or_nan(evals),
                median_test_r2=median_or_nan(test_r2_values),
            )
        )
    return summaries


def summarize_by_group(rows: Sequence[dict[str, str]]) -> list[SummaryRow]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    summaries: list[SummaryRow] = []
    for group, group_rows in sorted(grouped.items()):
        times = [parse_float(row["time_sec"]) for row in group_rows]
        evals = [parse_float(row["evaluations"]) for row in group_rows]
        successes = [1.0 if parse_bool(row["success"]) else 0.0 for row in group_rows]
        test_r2_values = [parse_float(row["test_r2"]) for row in group_rows]
        summaries.append(
            SummaryRow(
                group=group,
                case_id=None,
                case_name=f"{len({row['case_id'] for row in group_rows})} cases",
                runs=len(group_rows),
                mean_time_sec=mean_or_nan(times),
                success_rate=mean_or_nan(successes),
                mean_evaluations=mean_or_nan(evals),
                median_test_r2=median_or_nan(test_r2_values),
            )
        )
    return summaries


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def print_group_table(rows: Sequence[SummaryRow]) -> None:
    headers = ["group", "cases", "runs", "mean_time_s", "success_rate", "mean_evals", "median_test_r2"]
    body = [
        [
            row.group,
            row.case_name,
            str(row.runs),
            format_float(row.mean_time_sec),
            format_rate(row.success_rate),
            format_eval_mean(row.mean_evaluations),
            format_float(row.median_test_r2, digits=6),
        ]
        for row in rows
    ]
    print("Group Summary")
    print(make_table(headers, body))


def print_case_table(rows: Sequence[SummaryRow]) -> None:
    headers = ["group", "case_id", "case_name", "runs", "mean_time_s", "success_rate", "mean_evals", "median_test_r2"]
    body = [
        [
            row.group,
            str(row.case_id),
            row.case_name,
            str(row.runs),
            format_float(row.mean_time_sec),
            format_rate(row.success_rate),
            format_eval_mean(row.mean_evaluations),
            format_float(row.median_test_r2, digits=6),
        ]
        for row in rows
    ]
    print("Case Summary")
    print(make_table(headers, body))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = Path.cwd()
    results_dir = resolve_results_dir(args, workspace_root)
    rows = load_rows(results_dir, args.groups)

    if args.level in {"group", "both"}:
        print_group_table(summarize_by_group(rows))
    if args.level == "both":
        print()
    if args.level in {"case", "both"}:
        print_case_table(summarize_by_case(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
