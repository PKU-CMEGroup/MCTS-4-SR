"""CSV output helpers for benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

from .executor import BenchmarkResult


def slugify(value: str) -> str:
    chars = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "-")
    return "-".join(part for part in "".join(chars).split("-") if part)


def combined_output_path(group: str, explicit_output: Path | None, workspace_root: Path) -> Path:
    if explicit_output is not None:
        return explicit_output
    return workspace_root / "benchmark_results" / f"{group.lower()}_benchmark.csv"


def split_output_dir(group: str, explicit_output: Path | None, workspace_root: Path) -> Path:
    if explicit_output is not None:
        if explicit_output.suffix.lower() == ".csv":
            raise SystemExit("--output should be a directory when --split-by-case is used.")
        return explicit_output
    return workspace_root / "benchmark_results" / group.lower()


def case_output_path(output_dir: Path, group: str, case: dict) -> Path:
    filename = f"{slugify(group)}_{case['id']:03d}_{slugify(case['name'])}.csv"
    return output_dir / filename


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_row() for result in results]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
