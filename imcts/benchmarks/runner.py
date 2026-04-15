"""Unified benchmark runner entry point."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Sequence

from . import executor
from .cli import parse_args
from .config import build_settings, load_yaml_resource
from .registry import BenchmarkRegistry, load_bundled_registry, print_available_cases
from .sources import build_source, resolve_dataset_dir
from .writer import case_output_path, combined_output_path, split_output_dir, write_csv


def configure_environment(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)


def resolve_group_name(args_group: str | None, forced_group: str | None) -> str:
    if forced_group is not None and args_group is not None and args_group != forced_group:
        raise SystemExit(f"This entry point is fixed to group {forced_group!r}, but got --group={args_group!r}.")
    return forced_group or args_group or "Nguyen"


def list_requested_groups(args, registry: BenchmarkRegistry, forced_group: str | None) -> int:
    if forced_group is not None and args.group is not None and args.group != forced_group:
        raise SystemExit(f"This entry point is fixed to group {forced_group!r}, but got --group={args.group!r}.")
    print_available_cases(registry, selected_group=forced_group)
    return 0


def resolve_output_path(args, group_name: str, workspace_root: Path) -> Path:
    if args.split_by_case:
        return split_output_dir(group_name, args.output, workspace_root)
    return combined_output_path(group_name, args.output, workspace_root)


def main(
    argv: Sequence[str] | None = None,
    workspace_root: Path | None = None,
    forced_group: str | None = None,
) -> int:
    args = parse_args(argv)
    workspace_root = workspace_root or Path.cwd()
    registry = load_bundled_registry()

    if args.list:
        return list_requested_groups(args, registry, forced_group)

    group_name = resolve_group_name(args.group, forced_group)
    try:
        group = registry.get_group(group_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_yaml_resource(args.config, group.default_config_name)
    settings = build_settings(args, group, config)
    configure_environment(settings.threads)
    executor.require_imcts()

    try:
        selected_cases = registry.select_cases(group_name, args.cases)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source = build_source(group)
    output = resolve_output_path(args, group_name, workspace_root)
    if args.split_by_case:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    print(f"benchmark group : {group_name}")
    print(f"cases           : {', '.join(case['name'] for case in selected_cases)}")
    print(f"runs per case   : {settings.runs}")
    print(f"ops             : {','.join(settings.ops)}")
    if settings.source_type == "expression":
        print(f"sample_multiplier: {settings.sample_multiplier}")
    print(f"test_ratio      : {settings.test_ratio}")
    if settings.source_type == "dataset":
        print(f"dataset_dir     : {resolve_dataset_dir(settings, workspace_root)}")
    if settings.auto_added_constant:
        print(f"note            : auto-added constant op R for {group_name} because neither YAML nor CLI specified ops")
    if settings.max_wall_time_hours is not None:
        print(f"max_wall_time_h : {settings.max_wall_time_hours}")
    print(f"output          : {output}")

    wall_time_start = time.perf_counter()
    wall_time_limit = settings.max_wall_time_hours
    wall_time_limit_sec = None if wall_time_limit is None else max(0.0, float(wall_time_limit) * 3600.0)

    rows: list[executor.BenchmarkResult] = []
    stop_requested = False
    for case in selected_cases:
        case_rows: list[executor.BenchmarkResult] = []
        for run_index in range(settings.runs):
            if wall_time_limit_sec is not None and time.perf_counter() - wall_time_start > wall_time_limit_sec:
                print(f"stopping        : reached wall-clock limit before {case['name']} run={run_index}")
                stop_requested = True
                break

            seed = executor.seed_for_run(settings.seed_start, run_index)
            prepared = source.prepare(case, settings, seed, workspace_root)
            result = executor.run_case(group_name, case, run_index, seed, settings, prepared)
            rows.append(result)
            case_rows.append(result)

            test_r2_text = f"{result.test_r2:.6f}" if math.isfinite(result.test_r2) else "nan"
            complexity_text = f"{result.complexity:.0f}" if math.isfinite(result.complexity) else "nan"
            print(
                f"{case['name']:>18} run={run_index:<2} seed={seed:<5} "
                f"time={result.time_sec:.3f}s reward={result.reward:.6f} "
                f"test_r2={test_r2_text} complexity={complexity_text} evals={result.evaluations}"
            )

        if args.split_by_case and case_rows:
            case_path = case_output_path(output, group_name, case)
            write_csv(case_rows, case_path)
            print(f"wrote {len(case_rows)} rows to {case_path}")

        if stop_requested:
            break

    if not args.split_by_case:
        if rows:
            write_csv(rows, output)
            print(f"wrote {len(rows)} rows to {output}")
        else:
            print("no rows written : no runs completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
