"""Unified benchmark runner entry point.

The execution flow is:

1. Parse CLI arguments
2. Resolve the benchmark group and selected cases
3. Load YAML, then merge CLI/YAML/defaults into ``BenchmarkSettings``
4. Prepare case data through the source adapter
5. Run each ``(case, seed)`` pair, sequentially or in parallel
6. Write one CSV per case under the resolved output directory
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from . import executor
from .cli import parse_args
from .config import BenchmarkSettings, build_settings, load_yaml_resource
from .registry import BenchmarkRegistry, load_bundled_registry, print_available_cases
from .sources import build_source, resolve_dataset_dir
from .writer import case_output_path, split_output_dir, write_csv


def physical_core_count() -> int:
    """Return the number of physical CPU cores (not hyperthreads).

    Uses /proc/cpuinfo on Linux; falls back to os.cpu_count() elsewhere.
    """
    try:
        with open("/proc/cpuinfo") as f:
            cores = len(set(
                line.split(":")[1].strip()
                for line in f
                if "core id" in line
            ))
        if cores > 0:
            return cores
    except OSError:
        pass
    return os.cpu_count() or 1


def _init_worker() -> None:
    """Initializer for worker processes: disable OpenMP threading.

    Each worker is a separate process using one physical core for the MCTS
    search.  OpenMP parallelism within the C++ evaluator is counterproductive
    here (Nguyen benchmarks have <=40 samples = 1 batch, so the OMP region
    is a no-op anyway) and would over-subscribe the CPU.
    """
    os.environ["OMP_NUM_THREADS"] = "1"


def _run_one(
    group_name: str,
    case: dict[str, Any],
    run_index: int,
    seed: int,
    settings: BenchmarkSettings,
    source_type: str,
    workspace_root: Path,
) -> executor.BenchmarkResult:
    """Execute a single benchmark run inside a worker process."""
    source = build_source_for_type(source_type)
    prepared = source.prepare(case, settings, seed, workspace_root)
    return executor.run_case(group_name, case, run_index, seed, settings, prepared)


def build_source_for_type(source_type: str):
    """Reconstruct the benchmark source from its type string."""
    from .sources import DatasetSource, ExpressionSource

    if source_type == "expression":
        return ExpressionSource()
    return DatasetSource()


def _format_result(result: executor.BenchmarkResult) -> str:
    train_r2_text = f"{result.train_r2:.6f}" if math.isfinite(result.train_r2) else "nan"
    test_r2_text = f"{result.test_r2:.6f}" if math.isfinite(result.test_r2) else "nan"
    complexity_text = f"{result.complexity:.0f}" if math.isfinite(result.complexity) else "nan"
    return (
        f"{result.case_name:>18} run={result.run:<2} seed={result.seed:<5} "
        f"time={result.time_sec:.3f}s reward={result.reward:.6f} "
        f"train_r2={train_r2_text} test_r2={test_r2_text} "
        f"complexity={complexity_text} evals={result.evaluations}"
    )


def resolve_group_name(args_group: str | None, forced_group: str | None) -> str:
    if forced_group is not None and args_group is not None and args_group != forced_group:
        raise SystemExit(f"This entry point is fixed to group {forced_group!r}, but got --group={args_group!r}.")
    return forced_group or args_group or "Nguyen"


def list_requested_groups(args, registry: BenchmarkRegistry, forced_group: str | None) -> int:
    if forced_group is not None and args.group is not None and args.group != forced_group:
        raise SystemExit(f"This entry point is fixed to group {forced_group!r}, but got --group={args.group!r}.")
    print_available_cases(registry, selected_group=forced_group)
    return 0


def resolve_output_path(args, settings, group_name: str, workspace_root: Path) -> Path:
    return split_output_dir(group_name, args.output, settings.results_dir, workspace_root)


def _run_sequential(
    selected_cases: list[dict[str, Any]],
    settings: BenchmarkSettings,
    source,
    group_name: str,
    workspace_root: Path,
    output_dir: Path,
    wall_time_limit_sec: float | None,
    wall_time_start: float,
) -> list[executor.BenchmarkResult]:
    """Original sequential execution path.

    The optional wall-clock limit is enforced only in this path. Parallel
    execution currently runs every submitted task to completion.
    """
    rows: list[executor.BenchmarkResult] = []
    for case in selected_cases:
        case_rows: list[executor.BenchmarkResult] = []
        for run_index in range(settings.runs):
            if wall_time_limit_sec is not None and time.perf_counter() - wall_time_start > wall_time_limit_sec:
                print(f"stopping        : reached wall-clock limit before {case['name']} run={run_index}")
                return rows

            seed = executor.seed_for_run(settings.seed_start, run_index)
            prepared = source.prepare(case, settings, seed, workspace_root)
            result = executor.run_case(group_name, case, run_index, seed, settings, prepared)
            rows.append(result)
            case_rows.append(result)
            print(_format_result(result))

        if case_rows:
            case_path = case_output_path(output_dir, group_name, case)
            write_csv(case_rows, case_path)
            print(f"wrote {len(case_rows)} rows to {case_path}")
    return rows


def _run_parallel(
    selected_cases: list[dict[str, Any]],
    settings: BenchmarkSettings,
    group_name: str,
    workspace_root: Path,
    output_dir: Path,
    num_workers: int,
) -> list[executor.BenchmarkResult]:
    """Parallel execution: each ``(case, seed)`` pair runs in its own process."""
    tasks: list[tuple[str, dict, int, int, BenchmarkSettings, str, Path]] = []
    cases_by_name = {case["name"]: case for case in selected_cases}
    case_rows: dict[str, list[executor.BenchmarkResult]] = {case["name"]: [] for case in selected_cases}
    remaining_runs = {case["name"]: settings.runs for case in selected_cases}
    for case in selected_cases:
        for run_index in range(settings.runs):
            seed = executor.seed_for_run(settings.seed_start, run_index)
            tasks.append((group_name, case, run_index, seed, settings, settings.source_type, workspace_root))

    results: list[executor.BenchmarkResult] = []
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_run_one, *task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            case_name = result.case_name
            case_rows[case_name].append(result)
            remaining_runs[case_name] -= 1
            print(_format_result(result))

            if remaining_runs[case_name] == 0:
                finished_rows = sorted(case_rows[case_name], key=lambda row: row.run)
                case_path = case_output_path(output_dir, group_name, cases_by_name[case_name])
                write_csv(finished_rows, case_path)
                print(f"wrote {len(finished_rows)} rows to {case_path}")

    results.sort(key=lambda r: (r.case_name, r.run))
    return results


def main(
    argv: Sequence[str] | None = None,
    workspace_root: Path | None = None,
    forced_group: str | None = None,
) -> int:
    """Run the benchmark CLI entry point.

    ``forced_group`` is used by specialized entry points that pin the group and
    reject conflicting ``--group`` values from the CLI.
    """
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
    executor.require_imcts()

    try:
        selected_cases = registry.select_cases(group_name, args.cases)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source = build_source(group)
    output = resolve_output_path(args, settings, group_name, workspace_root)
    output.mkdir(parents=True, exist_ok=True)

    num_workers = args.workers if args.workers is not None else physical_core_count()
    num_workers = max(1, num_workers)

    print(f"benchmark group : {group_name}")
    print(f"cases           : {', '.join(case['name'] for case in selected_cases)}")
    print(f"runs per case   : {settings.runs}")
    print(f"ops             : {','.join(settings.ops)}")
    print(f"K               : {settings.K}")
    print(f"max_tree_nodes  : {settings.max_tree_nodes}")
    print(f"c               : {settings.c}")
    print(f"gamma           : {settings.gamma}")
    print(f"gp_rate         : {settings.gp_rate}")
    print(f"mutation_rate   : {settings.mutation_rate}")
    print(f"exploration_rate: {settings.exploration_rate}")
    if settings.source_type == "expression":
        print(f"sample_multiplier: {settings.sample_multiplier}")
    print(f"test_ratio      : {settings.test_ratio}")
    if settings.source_type == "dataset":
        print(f"dataset_dir     : {resolve_dataset_dir(settings, workspace_root)}")
    if settings.auto_added_constant:
        print(f"note            : auto-added constant op R for {group_name} because neither YAML nor CLI specified ops")
    if settings.max_wall_time_hours is not None:
        print(f"max_wall_time_h : {settings.max_wall_time_hours}")
    print(f"workers         : {num_workers}")
    if settings.results_dir is not None and args.output is None:
        print(f"results_root    : {settings.results_dir}")
    print(f"output          : {output}")

    wall_time_start = time.perf_counter()
    wall_time_limit = settings.max_wall_time_hours
    wall_time_limit_sec = None if wall_time_limit is None else max(0.0, float(wall_time_limit) * 3600.0)

    if num_workers <= 1:
        rows = _run_sequential(
            selected_cases, settings, source, group_name, workspace_root,
            output,
            wall_time_limit_sec, wall_time_start,
        )
    else:
        rows = _run_parallel(
            selected_cases, settings, group_name, workspace_root, output, num_workers,
        )

    if not rows:
        print("no rows written : no runs completed")

    elapsed = time.perf_counter() - wall_time_start
    print(f"total wall time : {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
