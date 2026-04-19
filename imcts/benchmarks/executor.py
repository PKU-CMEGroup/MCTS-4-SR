"""Execution and result shaping for benchmark runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..pretty import simplify_with_complexity
from .config import BenchmarkSettings
from .sources import PreparedCaseData


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


@dataclass(frozen=True)
class BenchmarkResult:
    group: str
    case_id: int
    case_name: str
    source_type: str
    run: int
    seed: int
    samples_total: int
    samples_train: int
    samples_test: int
    variables: int
    feature_names: list[str]
    target_expression: str
    reward: float
    success: bool
    train_r2: float
    test_r2: float
    complexity: float
    evaluations: int
    time_sec: float
    expression: str
    materialized_expression: str
    simplified_expression: str
    coefficients: list[float]
    ops: list[str]
    max_depth: int
    max_unary: int
    max_constants: int
    max_evals: int
    lm_iterations: int
    max_tree_nodes: int
    test_ratio: float

    def to_row(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "source_type": self.source_type,
            "run": self.run,
            "seed": self.seed,
            "samples_total": self.samples_total,
            "samples_train": self.samples_train,
            "samples_test": self.samples_test,
            "variables": self.variables,
            "feature_names": json.dumps(self.feature_names),
            "target_expression": self.target_expression,
            "reward": self.reward,
            "success": self.success,
            "train_r2": self.train_r2,
            "test_r2": self.test_r2,
            "complexity": self.complexity,
            "evaluations": self.evaluations,
            "time_sec": self.time_sec,
            "expression": self.expression,
            "materialized_expression": self.materialized_expression,
            "simplified_expression": self.simplified_expression,
            "coefficients": json.dumps(self.coefficients),
            "ops": ",".join(self.ops),
            "max_depth": self.max_depth,
            "max_unary": self.max_unary,
            "max_constants": self.max_constants,
            "max_evals": self.max_evals,
            "lm_iterations": self.lm_iterations,
            "max_tree_nodes": self.max_tree_nodes,
            "test_ratio": self.test_ratio,
        }


def require_imcts():
    try:
        import imcts
    except ModuleNotFoundError as exc:  # pragma: no cover - user environment issue
        raise SystemExit(
            "imcts is not installed. From the repo root, run `python3 -m pip install -e .`."
        ) from exc
    return imcts


def seed_for_run(seed_start: int, run_index: int) -> int:
    return MCTS_4_SR_SEEDS[(seed_start + run_index) % len(MCTS_4_SR_SEEDS)]


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


def evaluate_expression(expression: str, X, coefficients: list[float] | None = None):
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


def make_regressor_config(settings: BenchmarkSettings):
    """Translate benchmark search settings into ``imcts.RegressorConfig``.

    This is the handoff point where benchmark CLI/YAML parameters become the
    actual search budget and exploration hyperparameters used by the core
    regressor implementation.
    """
    imcts = require_imcts()

    cfg = imcts.RegressorConfig()
    cfg.ops = settings.ops
    cfg.max_depth = settings.max_depth
    cfg.max_unary = settings.max_unary
    cfg.max_constants = settings.max_constants
    cfg.max_evals = settings.max_evals
    cfg.lm_iterations = settings.lm_iterations
    cfg.K = settings.K
    cfg.max_tree_nodes = settings.max_tree_nodes
    cfg.c = settings.c
    cfg.gamma = settings.gamma
    cfg.gp_rate = settings.gp_rate
    cfg.mutation_rate = settings.mutation_rate
    cfg.exploration_rate = settings.exploration_rate
    cfg.succ_error_tol = settings.succ_error_tol
    return cfg


def run_case(
    group_name: str,
    case: dict[str, Any],
    run_index: int,
    seed: int,
    settings: BenchmarkSettings,
    prepared: PreparedCaseData,
) -> BenchmarkResult:
    """Run one benchmark seed for one case and collect reporting fields.

    The benchmark runner prepares the full dataset first, then we apply the
    seeded train/test split here so repeated runs on the same case can differ
    only by their benchmark seed.
    """
    import numpy as np

    imcts = require_imcts()
    X_train, X_test, y_train, y_test = split_train_test(prepared.X_total, prepared.y_total, settings.test_ratio, seed)
    cfg = make_regressor_config(settings)
    model = imcts.Regressor(
        X_train.T.astype(np.float32, copy=False),
        y_train.astype(np.float32, copy=False),
        cfg,
    )

    t0 = time.perf_counter()
    result = model.fit(seed=seed)
    elapsed = time.perf_counter() - t0

    coefficients = [float(value) for value in result.best_coefficients]
    materialized = materialize_expression(result.expression, coefficients)
    y_pred_train = evaluate_expression(materialized, X_train)
    y_pred_test = evaluate_expression(materialized, X_test)
    train_r2 = regression_r2(y_train, y_pred_train)
    test_r2 = regression_r2(y_test, y_pred_test)
    simplified_expression, complexity = simplify_with_complexity(materialized, precision=4, threshold=1e-4)

    return BenchmarkResult(
        group=group_name,
        case_id=int(case["id"]),
        case_name=case["name"],
        source_type=prepared.source_type,
        run=run_index,
        seed=seed,
        samples_total=int(prepared.X_total.shape[0]),
        samples_train=int(X_train.shape[0]),
        samples_test=int(X_test.shape[0]),
        variables=int(prepared.X_total.shape[1]),
        feature_names=list(prepared.feature_names),
        target_expression=prepared.target_expression,
        reward=float(result.best_reward),
        success=bool(result.best_reward >= 1.0 - settings.succ_error_tol),
        train_r2=float(train_r2),
        test_r2=float(test_r2),
        complexity=complexity,
        evaluations=int(result.n_evals),
        time_sec=float(elapsed),
        expression=result.expression,
        materialized_expression=materialized,
        simplified_expression=simplified_expression,
        coefficients=coefficients,
        ops=list(cfg.ops),
        max_depth=settings.max_depth,
        max_unary=settings.max_unary,
        max_constants=settings.max_constants,
        max_evals=settings.max_evals,
        lm_iterations=settings.lm_iterations,
        max_tree_nodes=settings.max_tree_nodes,
        test_ratio=float(settings.test_ratio),
    )
