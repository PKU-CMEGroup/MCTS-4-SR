"""Unified benchmark runner for symbolic and black-box datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

from ..pretty import simplify_with_complexity

DEFAULT_OPS = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
DEFAULT_CONSTANT_GROUPS = {
    "requires_constants": {"NguyenC", "Jin", "BlackBox"},
    "no_constants": {"Nguyen", "Livermore"},
}
DEFAULT_THREADS = max(1, os.cpu_count() or 1)
MCTS_4_SR_SEEDS = [
    23654, 15795, 860, 5390, 16850, 29910, 4426, 21962, 14423, 28020,
    29802, 21575, 11964, 11284, 22118, 6265, 11363, 27495, 16023, 8322,
    1685, 32052, 769, 26967, 30187, 32157, 23333, 2433, 5311, 5051,
    6420, 17568, 20939, 19769, 28693, 6396, 29419, 27480, 32304, 8666,
    25658, 18942, 24233, 18431, 32219, 2747, 25551, 26382, 189, 31677,
    19118, 3005, 21042, 1899, 24118, 1267, 31551, 17912, 11394, 3556,
    3890, 8838, 30740, 27464, 14502, 21777, 10627, 8792, 10555, 10253,
    8433, 10233, 11016, 23897, 2612, 23425, 25939, 22619, 21870, 23483,
    26054, 15787, 27132, 17159, 12206, 8226, 14541, 3152, 26531, 1585,
    3943, 23939, 19457, 1021, 11653, 10805, 13417, 20227, 7989, 9692,
]


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
    parser.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS value.")
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


def require_imcts():
    try:
        import imcts
    except ModuleNotFoundError as exc:  # pragma: no cover - user environment issue
        raise SystemExit(
            "imcts is not installed. From the repo root, run `python3 -m pip install -e .`."
        ) from exc
    return imcts


def load_json_resource(name: str) -> dict:
    with resources.files("imcts.benchmarks").joinpath(name).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml_resource(path: Path | None, default_name: str) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise SystemExit("PyYAML is required. Reinstall with `python -m pip install -e .`.") from exc

    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    with resources.files("imcts.benchmarks").joinpath(default_name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_synthetic_cases() -> dict:
    return load_json_resource("basic.json")


def load_blackbox_cases() -> list[dict]:
    config = load_json_resource("blackbox.json")
    return [{"id": idx, "name": name} for idx, name in enumerate(config["BlackBox"], start=1)]


def available_groups() -> dict[str, list[dict]]:
    synthetic = load_synthetic_cases()["groups"]
    groups = {name: list(cases) for name, cases in synthetic.items()}
    groups["BlackBox"] = load_blackbox_cases()
    return groups


def source_type_for_group(group: str) -> str:
    return "dataset" if group == "BlackBox" else "expression"


def constant_groups() -> dict[str, set[str]]:
    configured = load_synthetic_cases().get("constant_groups", {})
    requires = set(configured.get("requires_constants", DEFAULT_CONSTANT_GROUPS["requires_constants"]))
    no_constants = set(configured.get("no_constants", DEFAULT_CONSTANT_GROUPS["no_constants"]))
    requires.add("BlackBox")
    return {"requires_constants": requires, "no_constants": no_constants}


def print_available_cases(groups: dict[str, list[dict]], selected_group: str | None = None) -> None:
    for group, cases in groups.items():
        if selected_group is not None and group != selected_group:
            continue
        print(f"{group}:")
        for case in cases:
            expr = case.get("expression")
            if expr:
                print(f"  {case['id']:>3}: {case['name']}  y = {expr}")
            else:
                print(f"  {case['id']:>3}: {case['name']}")


def resolve_group(args: argparse.Namespace, forced_group: str | None) -> str:
    if forced_group is not None and args.group is not None and args.group != forced_group:
        raise SystemExit(f"This entry point is fixed to group {forced_group!r}, but got --group={args.group!r}.")
    return forced_group or args.group or "Nguyen"


def select_cases(groups: dict[str, list[dict]], group: str, selection: str) -> list[dict]:
    if group not in groups:
        available = ", ".join(sorted(groups))
        raise SystemExit(f"Unknown group '{group}'. Available groups: {available}")

    cases = groups[group]
    if selection.strip().lower() == "all":
        return cases

    wanted_ids: set[int] = set()
    wanted_names: set[str] = set()
    for token in (part.strip() for part in selection.split(",") if part.strip()):
        if "-" in token and token.replace("-", "").isdigit():
            start, end = (int(x) for x in token.split("-", 1))
            wanted_ids.update(range(min(start, end), max(start, end) + 1))
        elif token.isdigit():
            wanted_ids.add(int(token))
        else:
            wanted_names.add(token.lower())

    selected = [
        case
        for case in cases
        if case["id"] in wanted_ids or case["name"].lower() in wanted_names
    ]
    if not selected:
        raise SystemExit(f"No cases matched --cases={selection!r} in group {group!r}.")
    return selected


def nested_get(mapping: dict, *keys: str, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
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


def resolve_settings(args: argparse.Namespace, group: str, config: dict) -> dict[str, Any]:
    source_type = source_type_for_group(group)
    required_constant_groups = constant_groups()["requires_constants"]
    config_ops = normalize_ops(nested_get(config, "search", "ops"))
    cli_ops = normalize_ops(args.ops)
    ops = pick(cli_ops, config_ops)
    auto_added_constant = False
    if ops is None:
        ops = list(DEFAULT_OPS)
        if group in required_constant_groups:
            ops.append("R")
            auto_added_constant = True

    return {
        "group": group,
        "source_type": source_type,
        "runs": int(pick(args.runs, config.get("runs"), 10 if source_type == "dataset" else 100)),
        "seed_start": int(pick(args.seed_start, config.get("seed_start"), 0)),
        "samples": pick(args.samples, nested_get(config, "data", "samples")),
        "sample_multiplier": float(pick(nested_get(config, "data", "sample_multiplier"), 1.0)),
        "dataset_dir": pick(args.dataset_dir, nested_get(config, "data", "dataset_dir")),
        "label": pick(args.label, nested_get(config, "data", "label"), "target"),
        "test_ratio": float(pick(args.test_ratio, nested_get(config, "data", "test_ratio"), 0.25)),
        "threads": int(pick(args.threads, nested_get(config, "runtime", "threads"), DEFAULT_THREADS)),
        "max_wall_time_hours": pick(args.max_wall_time_hours, nested_get(config, "runtime", "max_wall_time_hours")),
        "ops": ops,
        "auto_added_constant": auto_added_constant,
        "max_depth": int(pick(args.max_depth, nested_get(config, "search", "max_depth"), 6)),
        "max_unary": int(pick(args.max_unary, nested_get(config, "search", "max_unary"), 999)),
        "max_constants": int(pick(args.max_constants, nested_get(config, "search", "max_constants"), 6)),
        "lm_iterations": int(pick(args.lm_iterations, nested_get(config, "search", "lm_iterations"), 50)),
        "K": int(pick(args.K, nested_get(config, "search", "K"), 500)),
        "c": float(pick(args.c, nested_get(config, "search", "c"), 4.0)),
        "gamma": float(pick(args.gamma, nested_get(config, "search", "gamma"), 0.5)),
        "gp_rate": float(pick(args.gp_rate, nested_get(config, "search", "gp_rate"), 0.2)),
        "mutation_rate": float(pick(args.mutation_rate, nested_get(config, "search", "mutation_rate"), 0.1)),
        "exploration_rate": float(pick(args.exploration_rate, nested_get(config, "search", "exploration_rate"), 0.2)),
        "succ_error_tol": float(pick(args.succ_error_tol, nested_get(config, "search", "succ_error_tol"), 1e-6)),
        "max_evals": int(pick(args.max_evals, nested_get(config, "search", "max_evals"), 500000 if source_type == "dataset" else 2000000)),
    }


def configure_environment(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)


def get_bounds(case: dict):
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


def make_inputs(case: dict, samples_override: int | None, seed: int):
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


def resolve_dataset_dir(settings: dict[str, Any], workspace_root: Path) -> Path:
    if settings["dataset_dir"] is not None:
        return Path(settings["dataset_dir"])

    candidates = [
        workspace_root / "datasets",
        # Legacy fallback path removed.
    ]
    for candidate in candidates:
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


def build_case_dataset(case: dict, settings: dict[str, Any], seed: int, workspace_root: Path):
    if settings["source_type"] == "expression":
        samples = settings["samples"]
        if samples is None:
            samples = max(2, int(round(case["samples"] * settings["sample_multiplier"])))
        X_total = make_inputs(case, samples, seed)
        y_total = eval_expression(case["expression"], X_total).astype("float64", copy=False)
        feature_names = [f"x{i}" for i in range(X_total.shape[1])]
        return X_total, y_total, feature_names, case["expression"]

    dataset_dir = resolve_dataset_dir(settings, workspace_root)
    X_total, y_total, feature_names = load_case_dataset(dataset_dir, case["name"], settings["label"])
    return X_total, y_total, list(feature_names), ""


def split_train_test(X_total, y_total, test_ratio: float, seed: int):
    import numpy as np

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    n_samples = X_total.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least two samples to create a train/test split.")

    n_test = max(1, int(round(n_samples * test_ratio)))
    n_test = min(n_test, n_samples - 1)
    permutation = np.random.default_rng(seed).permutation(n_samples)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    return X_total[train_idx], X_total[test_idx], y_total[train_idx], y_total[test_idx]


def eval_expression(expression: str, X, coefficients: list[float] | None = None):
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
    for i, value in enumerate(coefficients or []):
        context[f"c{i}"] = float(value)
        context[f"C{i}"] = float(value)

    with np.errstate(all="ignore"):
        y = eval(expression, context, {})
    if np.isscalar(y):
        return np.full(X.shape[0], float(y), dtype=np.float64)
    return np.asarray(y, dtype=np.float64)


def regression_r2(y_true, y_pred) -> float:
    import numpy as np

    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        return float("nan")
    mse = float(np.mean((y_true - y_pred) ** 2))
    variance = float(np.var(y_true))
    if variance == 0.0:
        variance = 1e-9
    return 1.0 - mse / variance


def materialize_expression(expression: str, coefficients: Sequence[float]) -> str:
    materialized = expression
    indexed_coefficients = list(enumerate(coefficients))
    indexed_coefficients.sort(key=lambda item: item[0], reverse=True)
    for idx, value in indexed_coefficients:
        numeric = repr(float(value))
        materialized = materialized.replace(f"C{idx}", numeric)
        materialized = materialized.replace(f"c{idx}", numeric)
    return materialized


def make_regressor_config(settings: dict[str, Any]):
    imcts = require_imcts()

    cfg = imcts.RegressorConfig()
    cfg.ops = settings["ops"]
    cfg.max_depth = settings["max_depth"]
    cfg.max_unary = settings["max_unary"]
    cfg.max_constants = settings["max_constants"]
    cfg.max_evals = settings["max_evals"]
    cfg.lm_iterations = settings["lm_iterations"]
    cfg.K = settings["K"]
    cfg.c = settings["c"]
    cfg.gamma = settings["gamma"]
    cfg.gp_rate = settings["gp_rate"]
    cfg.mutation_rate = settings["mutation_rate"]
    cfg.exploration_rate = settings["exploration_rate"]
    cfg.succ_error_tol = settings["succ_error_tol"]
    return cfg


def run_case(group: str, case: dict, run_index: int, seed: int, settings: dict[str, Any], workspace_root: Path) -> dict:
    import numpy as np

    imcts = require_imcts()
    X_total, y_total, feature_names, target_expression = build_case_dataset(case, settings, seed, workspace_root)
    X_train, X_test, y_train, y_test = split_train_test(X_total, y_total, settings["test_ratio"], seed)
    cfg = make_regressor_config(settings)
    model = imcts.Regressor(
        X_train.T.astype(np.float32, copy=False),
        y_train.astype(np.float32, copy=False),
        cfg,
    )

    t0 = time.perf_counter()
    result = model.fit(seed=seed)
    elapsed = time.perf_counter() - t0

    coefficients = list(result.best_coefficients)
    materialized_expression = materialize_expression(result.expression, coefficients)
    y_pred_train = eval_expression(materialized_expression, X_train)
    y_pred_test = eval_expression(materialized_expression, X_test)
    train_r2 = regression_r2(y_train, y_pred_train)
    test_r2 = regression_r2(y_test, y_pred_test)

    # Apply precision truncation logic to numeric properties globally via SymPy
    simplified_expression, complexity = simplify_with_complexity(materialized_expression, precision=4, threshold=1e-4)

    return {
        "group": group,
        "case_id": case["id"],
        "case_name": case["name"],
        "source_type": settings["source_type"],
        "run": run_index,
        "seed": seed,
        "samples_total": int(X_total.shape[0]),
        "samples_train": int(X_train.shape[0]),
        "samples_test": int(X_test.shape[0]),
        "variables": int(X_total.shape[1]),
        "feature_names": json.dumps(list(feature_names)),
        "target_expression": target_expression,
        "reward": float(result.best_reward),
        "success": bool(result.best_reward >= 1.0 - settings["succ_error_tol"]),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "complexity": complexity,
        "evaluations": int(result.n_evals),
        "time_sec": float(elapsed),
        "expression": result.expression,
        "materialized_expression": materialized_expression,
        "simplified_expression": simplified_expression,
        "coefficients": json.dumps(coefficients),
        "ops": ",".join(cfg.ops),
        "max_depth": settings["max_depth"],
        "max_unary": settings["max_unary"],
        "max_constants": settings["max_constants"],
        "max_evals": settings["max_evals"],
        "lm_iterations": settings["lm_iterations"],
        "test_ratio": float(settings["test_ratio"]),
    }


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


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_config_name(group: str) -> str:
    return "blackbox.yaml" if source_type_for_group(group) == "dataset" else "basic.yaml"


def main(
    argv: Sequence[str] | None = None,
    workspace_root: Path | None = None,
    forced_group: str | None = None,
) -> int:
    args = parse_args(argv)
    workspace_root = workspace_root or Path.cwd()
    groups = available_groups()
    group = resolve_group(args, forced_group)
    config = load_yaml_resource(args.config, default_config_name(group))
    settings = resolve_settings(args, group, config)
    configure_environment(settings["threads"])

    if args.list:
        print_available_cases(groups, selected_group=group if forced_group is not None else None)
        return 0

    require_imcts()

    selected = select_cases(groups, group, args.cases)
    output = split_output_dir(group, args.output, workspace_root) if args.split_by_case else combined_output_path(group, args.output, workspace_root)
    if args.split_by_case:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    wall_time_start = time.perf_counter()
    wall_time_limit = settings["max_wall_time_hours"]
    wall_time_limit_sec = None if wall_time_limit is None else max(0.0, float(wall_time_limit) * 3600.0)

    rows: list[dict] = []
    print(f"benchmark group : {group}")
    print(f"cases           : {', '.join(case['name'] for case in selected)}")
    print(f"runs per case   : {settings['runs']}")
    print(f"ops             : {','.join(settings['ops'])}")
    if settings["source_type"] == "expression":
        print(f"sample_multiplier: {settings['sample_multiplier']}")
    print(f"test_ratio      : {settings['test_ratio']}")
    if settings["source_type"] == "dataset":
        print(f"dataset_dir     : {resolve_dataset_dir(settings, workspace_root)}")
    if settings["auto_added_constant"]:
        print(f"note            : auto-added constant op R for {group} because neither YAML nor CLI specified ops")
    if wall_time_limit_sec is not None:
        print(f"max_wall_time_h : {wall_time_limit}")
    print(f"output          : {output}")

    stop_requested = False
    for case in selected:
        case_rows: list[dict] = []
        for run_index in range(settings["runs"]):
            if wall_time_limit_sec is not None and time.perf_counter() - wall_time_start > wall_time_limit_sec:
                print(f"stopping        : reached wall-clock limit before {case['name']} run={run_index}")
                stop_requested = True
                break

            seed = MCTS_4_SR_SEEDS[(settings["seed_start"] + run_index) % len(MCTS_4_SR_SEEDS)]
            row = run_case(group, case, run_index, seed, settings, workspace_root)
            rows.append(row)
            case_rows.append(row)
            test_r2_text = f"{row['test_r2']:.6f}" if math.isfinite(row["test_r2"]) else "nan"
            complexity_text = f"{row['complexity']:.0f}" if math.isfinite(row["complexity"]) else "nan"
            print(
                f"{case['name']:>18} run={run_index:<2} seed={seed:<5} "
                f"time={row['time_sec']:.3f}s reward={row['reward']:.6f} "
                f"test_r2={test_r2_text} complexity={complexity_text} evals={row['evaluations']}"
            )

        if args.split_by_case and case_rows:
            case_path = case_output_path(output, group, case)
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
