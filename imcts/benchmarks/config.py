"""Configuration loading and CLI/YAML merge rules for benchmarks."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .registry import BenchmarkGroup


DEFAULT_OPS = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
DEFAULT_THREADS = max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class DataSettings:
    samples: int | None
    sample_multiplier: float
    dataset_dir: Path | None
    label: str
    test_ratio: float


@dataclass(frozen=True)
class SearchSettings:
    ops: list[str]
    max_depth: int
    max_unary: int
    max_constants: int
    lm_iterations: int
    K: int
    c: float
    gamma: float
    gp_rate: float
    mutation_rate: float
    exploration_rate: float
    succ_error_tol: float
    max_evals: int


@dataclass(frozen=True)
class RuntimeSettings:
    threads: int
    max_wall_time_hours: float | None


@dataclass(frozen=True)
class BenchmarkSettings:
    group: str
    source_type: str
    runs: int
    seed_start: int
    data: DataSettings
    search: SearchSettings
    runtime: RuntimeSettings
    auto_added_constant: bool

    @property
    def samples(self) -> int | None:
        return self.data.samples

    @property
    def sample_multiplier(self) -> float:
        return self.data.sample_multiplier

    @property
    def dataset_dir(self) -> Path | None:
        return self.data.dataset_dir

    @property
    def label(self) -> str:
        return self.data.label

    @property
    def test_ratio(self) -> float:
        return self.data.test_ratio

    @property
    def threads(self) -> int:
        return self.runtime.threads

    @property
    def max_wall_time_hours(self) -> float | None:
        return self.runtime.max_wall_time_hours

    @property
    def ops(self) -> list[str]:
        return self.search.ops

    @property
    def max_depth(self) -> int:
        return self.search.max_depth

    @property
    def max_unary(self) -> int:
        return self.search.max_unary

    @property
    def max_constants(self) -> int:
        return self.search.max_constants

    @property
    def lm_iterations(self) -> int:
        return self.search.lm_iterations

    @property
    def K(self) -> int:
        return self.search.K

    @property
    def c(self) -> float:
        return self.search.c

    @property
    def gamma(self) -> float:
        return self.search.gamma

    @property
    def gp_rate(self) -> float:
        return self.search.gp_rate

    @property
    def mutation_rate(self) -> float:
        return self.search.mutation_rate

    @property
    def exploration_rate(self) -> float:
        return self.search.exploration_rate

    @property
    def succ_error_tol(self) -> float:
        return self.search.succ_error_tol

    @property
    def max_evals(self) -> int:
        return self.search.max_evals


def load_yaml_resource(path: Path | None, default_name: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise SystemExit("PyYAML is required. Reinstall with `python -m pip install -e .`.") from exc

    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    with resources.files("imcts.benchmarks").joinpath(default_name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def nested_get(mapping: Mapping[str, Any], *keys: str, default=None):
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def pick(*values):
    for value in values:
        if value is not None:
            return value
    return None


def normalize_ops(raw_ops) -> list[str] | None:
    if raw_ops is None:
        return None
    if isinstance(raw_ops, str):
        ops = [op.strip() for op in raw_ops.split(",") if op.strip()]
    else:
        ops = [str(op).strip() for op in raw_ops if str(op).strip()]
    unique_ops: list[str] = []
    for op in ops:
        if op not in unique_ops:
            unique_ops.append(op)
    return unique_ops


def optional_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def build_settings(
    args: argparse.Namespace,
    group: BenchmarkGroup,
    config: Mapping[str, Any],
) -> BenchmarkSettings:
    config_ops = normalize_ops(nested_get(config, "search", "ops"))
    cli_ops = normalize_ops(args.ops)
    ops = pick(cli_ops, config_ops)
    auto_added_constant = False
    if ops is None:
        ops = list(DEFAULT_OPS)
        if group.needs_constant_op:
            ops.append("R")
            auto_added_constant = True

    data = DataSettings(
        samples=pick(args.samples, nested_get(config, "data", "samples")),
        sample_multiplier=float(pick(nested_get(config, "data", "sample_multiplier"), 1.0)),
        dataset_dir=optional_path(pick(args.dataset_dir, nested_get(config, "data", "dataset_dir"))),
        label=str(pick(args.label, nested_get(config, "data", "label"), "target")),
        test_ratio=float(pick(args.test_ratio, nested_get(config, "data", "test_ratio"), 0.25)),
    )
    search = SearchSettings(
        ops=ops,
        max_depth=int(pick(args.max_depth, nested_get(config, "search", "max_depth"), 6)),
        max_unary=int(pick(args.max_unary, nested_get(config, "search", "max_unary"), 999)),
        max_constants=int(pick(args.max_constants, nested_get(config, "search", "max_constants"), 10)),
        lm_iterations=int(pick(args.lm_iterations, nested_get(config, "search", "lm_iterations"), 50)),
        K=int(pick(args.K, nested_get(config, "search", "K"), 500)),
        c=float(pick(args.c, nested_get(config, "search", "c"), 4.0)),
        gamma=float(pick(args.gamma, nested_get(config, "search", "gamma"), 0.5)),
        gp_rate=float(pick(args.gp_rate, nested_get(config, "search", "gp_rate"), 0.2)),
        mutation_rate=float(pick(args.mutation_rate, nested_get(config, "search", "mutation_rate"), 0.1)),
        exploration_rate=float(pick(args.exploration_rate, nested_get(config, "search", "exploration_rate"), 0.2)),
        succ_error_tol=float(pick(args.succ_error_tol, nested_get(config, "search", "succ_error_tol"), 1e-6)),
        max_evals=int(
            pick(
                args.max_evals,
                nested_get(config, "search", "max_evals"),
                500000 if group.source_type == "dataset" else 2000000,
            )
        ),
    )
    runtime = RuntimeSettings(
        threads=int(pick(args.threads, nested_get(config, "runtime", "threads"), DEFAULT_THREADS)),
        max_wall_time_hours=pick(args.max_wall_time_hours, nested_get(config, "runtime", "max_wall_time_hours")),
    )
    return BenchmarkSettings(
        group=group.name,
        source_type=group.source_type,
        runs=int(pick(args.runs, config.get("runs"), 10 if group.source_type == "dataset" else 100)),
        seed_start=int(pick(args.seed_start, config.get("seed_start"), 0)),
        data=data,
        search=search,
        runtime=runtime,
        auto_added_constant=auto_added_constant,
    )
