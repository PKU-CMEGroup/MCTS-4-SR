"""Configuration loading and CLI/YAML merge rules for benchmarks.

The benchmark runner builds one immutable ``BenchmarkSettings`` instance from
three sources, in descending priority:

1. CLI arguments
2. YAML config values
3. Built-in defaults from ``defaults.py``

That merged settings object is then passed through data preparation, execution,
and CSV reporting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .defaults import (
    DEFAULT_C,
    DEFAULT_DATASET_MAX_EVALS,
    DEFAULT_DATASET_RUNS,
    DEFAULT_EXPLORATION_RATE,
    DEFAULT_EXPRESSION_MAX_EVALS,
    DEFAULT_EXPRESSION_RUNS,
    DEFAULT_GAMMA,
    DEFAULT_GP_RATE,
    DEFAULT_K,
    DEFAULT_LABEL,
    DEFAULT_LM_ITERATIONS,
    DEFAULT_MAX_CONSTANTS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_UNARY,
    DEFAULT_MUTATION_RATE,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SAMPLE_MULTIPLIER,
    DEFAULT_SEED_START,
    DEFAULT_SUCC_ERROR_TOL,
    DEFAULT_TEST_RATIO,
)

if TYPE_CHECKING:
    from .registry import BenchmarkGroup


DEFAULT_OPS = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]


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
    max_wall_time_hours: float | None


@dataclass(frozen=True)
class OutputSettings:
    results_dir: Path | None


@dataclass(frozen=True)
class BenchmarkSettings:
    group: str
    source_type: str
    runs: int
    seed_start: int
    data: DataSettings
    search: SearchSettings
    runtime: RuntimeSettings
    output: OutputSettings
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
    def max_wall_time_hours(self) -> float | None:
        return self.runtime.max_wall_time_hours

    @property
    def results_dir(self) -> Path | None:
        return self.output.results_dir

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
    if path is not None:
        return load_yaml_file(path)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise SystemExit("PyYAML is required. Reinstall with `python -m pip install -e .`.") from exc

    with resources.files("imcts.benchmarks").joinpath(default_name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml_file(path: Path) -> dict[str, Any]:
    return _load_yaml_file(path.resolve(), seen=[])


def _load_yaml_file(path: Path, seen: list[Path]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise SystemExit("PyYAML is required. Reinstall with `python -m pip install -e .`.") from exc

    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise SystemExit(f"YAML config inheritance cycle detected: {chain}")

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, Mapping):
        raise SystemExit(f"YAML config must be a mapping at top level: {path}")

    extends = loaded.pop("extends", None)
    if extends is None:
        return dict(loaded)

    base_paths = [extends] if isinstance(extends, (str, Path)) else list(extends)
    merged: dict[str, Any] = {}
    for raw_base in base_paths:
        base_path = Path(raw_base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_config = _load_yaml_file(base_path.resolve(), seen=[*seen, path])
        merged = deep_merge_dicts(merged, base_config)
    return deep_merge_dicts(merged, dict(loaded))


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


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
    """Merge CLI, YAML, and built-in defaults into one settings object.

    The ``pick(...)`` helper encodes the priority order used throughout this
    function: CLI value first, then YAML value, then a fallback default.

    A few settings have extra benchmark-specific rules:

    - ``ops``: if neither CLI nor YAML specifies a primitive set, use
      ``DEFAULT_OPS`` and auto-append ``R`` for groups that require constants.
    - ``max_evals`` and ``runs``: expression and dataset benchmarks use
      different default budgets.
    - ``results_dir``: supports both ``output.results_dir`` and the legacy
      top-level ``results_dir`` YAML key.
    """
    config_ops = normalize_ops(nested_get(config, "search", "ops"))
    cli_ops = normalize_ops(args.ops)
    ops = pick(cli_ops, config_ops)
    auto_added_constant = False
    if ops is None:
        # Keep the default primitive set stable unless the caller overrides it.
        ops = list(DEFAULT_OPS)
        if group.needs_constant_op:
            ops.append("R")
            auto_added_constant = True

    data = DataSettings(
        samples=pick(args.samples, nested_get(config, "data", "samples")),
        sample_multiplier=float(pick(nested_get(config, "data", "sample_multiplier"), DEFAULT_SAMPLE_MULTIPLIER)),
        dataset_dir=optional_path(pick(args.dataset_dir, nested_get(config, "data", "dataset_dir"))),
        label=str(pick(args.label, nested_get(config, "data", "label"), DEFAULT_LABEL)),
        test_ratio=float(pick(args.test_ratio, nested_get(config, "data", "test_ratio"), DEFAULT_TEST_RATIO)),
    )
    search = SearchSettings(
        ops=ops,
        max_depth=int(pick(args.max_depth, nested_get(config, "search", "max_depth"), DEFAULT_MAX_DEPTH)),
        max_unary=int(pick(args.max_unary, nested_get(config, "search", "max_unary"), DEFAULT_MAX_UNARY)),
        max_constants=int(pick(args.max_constants, nested_get(config, "search", "max_constants"), DEFAULT_MAX_CONSTANTS)),
        lm_iterations=int(pick(args.lm_iterations, nested_get(config, "search", "lm_iterations"), DEFAULT_LM_ITERATIONS)),
        K=int(pick(args.K, nested_get(config, "search", "K"), DEFAULT_K)),
        c=float(pick(args.c, nested_get(config, "search", "c"), DEFAULT_C)),
        gamma=float(pick(args.gamma, nested_get(config, "search", "gamma"), DEFAULT_GAMMA)),
        gp_rate=float(pick(args.gp_rate, nested_get(config, "search", "gp_rate"), DEFAULT_GP_RATE)),
        mutation_rate=float(pick(args.mutation_rate, nested_get(config, "search", "mutation_rate"), DEFAULT_MUTATION_RATE)),
        exploration_rate=float(pick(args.exploration_rate, nested_get(config, "search", "exploration_rate"), DEFAULT_EXPLORATION_RATE)),
        succ_error_tol=float(pick(args.succ_error_tol, nested_get(config, "search", "succ_error_tol"), DEFAULT_SUCC_ERROR_TOL)),
        max_evals=int(
            pick(
                args.max_evals,
                nested_get(config, "search", "max_evals"),
                DEFAULT_DATASET_MAX_EVALS if group.source_type == "dataset" else DEFAULT_EXPRESSION_MAX_EVALS,
            )
        ),
    )
    runtime = RuntimeSettings(
        max_wall_time_hours=pick(args.max_wall_time_hours, nested_get(config, "runtime", "max_wall_time_hours")),
    )
    output = OutputSettings(
        results_dir=optional_path(
            pick(
                args.results_dir,
                nested_get(config, "output", "results_dir"),
                # Older YAMLs may still provide a top-level results_dir.
                config.get("results_dir"),
                DEFAULT_RESULTS_DIRNAME,
            )
        ),
    )
    return BenchmarkSettings(
        group=group.name,
        source_type=group.source_type,
        runs=int(pick(args.runs, config.get("runs"), DEFAULT_DATASET_RUNS if group.source_type == "dataset" else DEFAULT_EXPRESSION_RUNS)),
        seed_start=int(pick(args.seed_start, config.get("seed_start"), DEFAULT_SEED_START)),
        data=data,
        search=search,
        runtime=runtime,
        output=output,
        auto_added_constant=auto_added_constant,
    )
