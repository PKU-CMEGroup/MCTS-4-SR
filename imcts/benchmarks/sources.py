"""Input preparation for benchmark data sources."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import BenchmarkSettings
from .defaults import DEFAULT_DATASETS_DIRNAME
from .registry import BenchmarkGroup


@dataclass(frozen=True)
class PreparedCaseData:
    X_total: Any
    y_total: Any
    feature_names: list[str]
    target_expression: str
    source_type: str


class BenchmarkSource(Protocol):
    def prepare(
        self,
        case: dict[str, Any],
        settings: BenchmarkSettings,
        seed: int,
        workspace_root: Path,
    ) -> PreparedCaseData:
        ...


class ExpressionSource:
    def prepare(
        self,
        case: dict[str, Any],
        settings: BenchmarkSettings,
        seed: int,
        workspace_root: Path,
    ) -> PreparedCaseData:
        """Generate synthetic inputs/targets for symbolic benchmark cases.

        ``settings.samples`` is a hard override. When it is absent, we scale
        the case's bundled sample count by ``settings.sample_multiplier``.
        """
        del workspace_root
        samples = settings.samples
        if samples is None:
            samples = max(2, int(round(case["samples"] * settings.sample_multiplier)))
        X_total = make_inputs(case, samples, seed)
        y_total = evaluate_case_expression(case["expression"], X_total).astype("float64", copy=False)
        feature_names = [f"x{i}" for i in range(X_total.shape[1])]
        return PreparedCaseData(
            X_total=X_total,
            y_total=y_total,
            feature_names=feature_names,
            target_expression=case["expression"],
            source_type="expression",
        )


class DatasetSource:
    def prepare(
        self,
        case: dict[str, Any],
        settings: BenchmarkSettings,
        seed: int,
        workspace_root: Path,
    ) -> PreparedCaseData:
        """Load a black-box dataset case from disk.

        The dataset root comes from ``settings.dataset_dir`` when provided,
        otherwise we fall back to ``<workspace>/datasets``.
        """
        del seed
        dataset_dir = resolve_dataset_dir(settings, workspace_root)
        X_total, y_total, feature_names = load_case_dataset(dataset_dir, case["name"], settings.label)
        return PreparedCaseData(
            X_total=X_total,
            y_total=y_total,
            feature_names=list(feature_names),
            target_expression="",
            source_type="dataset",
        )


def build_source(group: BenchmarkGroup) -> BenchmarkSource:
    if group.source_type == "expression":
        return ExpressionSource()
    if group.source_type == "dataset":
        return DatasetSource()
    raise ValueError(f"Unsupported source type {group.source_type!r} for group {group.name!r}.")


def get_bounds(case: dict[str, Any]):
    import numpy as np

    bounds = np.asarray(case["data_range"], dtype=np.float64)
    n_vars = int(case["variables"])
    if bounds.ndim == 1:
        low = np.full(n_vars, bounds[0], dtype=np.float64)
        high = np.full(n_vars, bounds[1], dtype=np.float64)
    else:
        low = bounds[:, 0].astype(np.float64, copy=False)
        high = bounds[:, 1].astype(np.float64, copy=False)
    return low, high


def make_inputs(case: dict[str, Any], samples_override: int | None, seed: int):
    import numpy as np

    n_samples = int(samples_override or case["samples"])
    n_vars = int(case["variables"])
    low, high = get_bounds(case)
    sampling = case.get("sampling", "U").upper()

    if sampling == "E":
        if n_vars == 1:
            return np.linspace(low[0], high[0], n_samples, dtype=np.float64)[:, None]
        per_axis = max(2, int(round(n_samples ** (1.0 / n_vars))))
        axes = [np.linspace(low[i], high[i], per_axis, dtype=np.float64) for i in range(n_vars)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([axis.ravel() for axis in mesh])

    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_samples, n_vars)).astype(np.float64, copy=False)


def evaluate_case_expression(expression: str, X):
    import numpy as np

    context = {
        "__builtins__": {},
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "tanh": np.tanh,
        "abs": np.abs,
        "pow": np.power,
        "x": X.T,
    }
    for i in range(X.shape[1]):
        context[f"x{i}"] = X[:, i]

    with np.errstate(all="ignore"):
        y = eval(expression, context, {})
    if np.isscalar(y):
        return np.full(X.shape[0], float(y), dtype=np.float64)
    return np.asarray(y, dtype=np.float64)


def detect_delimiter(path: Path) -> str:
    if path.suffix == ".csv":
        return ","
    if path.suffixes[-2:] == [".csv", ".gz"]:
        return ","
    return "\t"


def open_text_file(path: Path):
    if path.suffix == ".gz":
        with path.open("rb") as probe:
            magic = probe.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def dataset_candidates(dataset_dir: Path, case_name: str) -> list[Path]:
    base = dataset_dir / case_name
    return [
        base / f"{case_name}.tsv.gz",
        base / f"{case_name}.tsv",
        base / f"{case_name}.csv.gz",
        base / f"{case_name}.csv",
    ]


def resolve_dataset_dir(settings: BenchmarkSettings, workspace_root: Path) -> Path:
    """Resolve the dataset root for BlackBox benchmarks."""
    if settings.dataset_dir is not None:
        return Path(settings.dataset_dir)

    candidate = workspace_root / DEFAULT_DATASETS_DIRNAME
    if candidate.exists():
        return candidate

    raise SystemExit(
        "Could not find a datasets directory. Pass --dataset-dir or set data.dataset_dir in YAML."
    )


def load_dataset(path: Path, label: str):
    import numpy as np

    delimiter = detect_delimiter(path)
    with open_text_file(path) as f:
        first_line = f.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{path} is a Git LFS pointer, not the real dataset file. "
                "Fetch the dataset contents first, then rerun the benchmark."
            )
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")

        cleaned_columns = [name.strip().replace(".", "_") for name in reader.fieldnames]
        rows = []
        for row in reader:
            cleaned_row = {}
            for original, cleaned in zip(reader.fieldnames, cleaned_columns):
                cleaned_row[cleaned] = row[original]
            rows.append(cleaned_row)

    normalized_label = label.strip().replace(".", "_")
    if normalized_label not in cleaned_columns:
        raise ValueError(f"Target column {normalized_label!r} not found in {path}.")

    feature_names = [name for name in cleaned_columns if name != normalized_label]
    X_rows: list[list[float]] = []
    y_values: list[float] = []
    for row in rows:
        X_rows.append([float(row[name]) for name in feature_names])
        y_values.append(float(row[normalized_label]))

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    return X, y, feature_names


def load_case_dataset(dataset_dir: Path, case_name: str, label: str):
    for candidate in dataset_candidates(dataset_dir, case_name):
        if candidate.exists():
            return load_dataset(candidate, label)
    searched = ", ".join(str(path) for path in dataset_candidates(dataset_dir, case_name))
    raise FileNotFoundError(f"Could not find dataset files for {case_name}. Looked for: {searched}")
