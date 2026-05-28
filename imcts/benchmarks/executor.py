"""Execution and result shaping for benchmark runs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
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

MAX_TRAINING_SAMPLES = 10_000


@dataclass(frozen=True)
class BenchmarkResult:
    algorithm: str
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
    tuning_time_sec: float
    tuning_evaluations: int
    expression: str
    materialized_expression: str
    simplified_expression: str
    coefficients: list[float]
    algorithm_params: dict[str, Any]
    tuned_params: dict[str, Any]
    test_ratio: float

    def to_row(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
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
            "tuning_time_sec": self.tuning_time_sec,
            "tuning_evaluations": self.tuning_evaluations,
            "expression": self.expression,
            "materialized_expression": self.materialized_expression,
            "simplified_expression": self.simplified_expression,
            "coefficients": json.dumps(self.coefficients),
            "algorithm_params": json.dumps(self.algorithm_params, sort_keys=True),
            "tuned_params": json.dumps(self.tuned_params, sort_keys=True),
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

    _, n_test = training_split_counts(n_samples, test_ratio, max_samples=None)
    permutation = np.random.default_rng(seed).permutation(n_samples)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    return X_total[train_idx], X_total[test_idx], y_total[train_idx], y_total[test_idx]


def training_split_counts(n_samples: int, test_ratio: float, max_samples: int | None = MAX_TRAINING_SAMPLES) -> tuple[int, int]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    if n_samples < 2:
        raise ValueError("Need at least two samples to create a train/test split.")

    n_test = max(1, int(round(n_samples * test_ratio)))
    n_test = min(n_test, n_samples - 1)
    n_train = n_samples - n_test
    if max_samples is not None:
        n_train = min(n_train, max_samples)
    return n_train, n_test


def subsample_training_data(X_train, y_train, max_samples: int = MAX_TRAINING_SAMPLES, seed: int = 0):
    import numpy as np

    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    n_samples = X_train.shape[0]
    if n_samples <= max_samples:
        return X_train, y_train

    sample_idx = np.random.default_rng(seed).choice(n_samples, size=max_samples, replace=False)
    return X_train[sample_idx], y_train[sample_idx]


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


def search_params(settings: BenchmarkSettings) -> dict[str, Any]:
    params = asdict(settings.search)
    params["ops"] = list(settings.ops)
    return params


def _with_search_params(settings: BenchmarkSettings, params: dict[str, Any]) -> BenchmarkSettings:
    search = replace(settings.search, **params)
    return replace(settings, search=search)


def _with_max_wall_time(settings: BenchmarkSettings, max_wall_time_hours: float | None) -> BenchmarkSettings:
    runtime = replace(settings.runtime, max_wall_time_hours=max_wall_time_hours)
    return replace(settings, runtime=runtime)


def iter_tuning_candidates(settings: BenchmarkSettings) -> list[BenchmarkSettings]:
    grid = settings.tuning.parameters
    if not grid:
        return [settings]

    base_params = search_params(settings)
    keys = list(grid)
    candidates: list[BenchmarkSettings] = []
    for values in product(*(grid[key] for key in keys)):
        params = dict(base_params)
        params.update(dict(zip(keys, values)))
        candidates.append(_with_search_params(settings, params))
    return candidates


@dataclass(frozen=True)
class FitSummary:
    raw_result: Any
    elapsed: float
    evaluations: int
    expression: str
    coefficients: list[float]
    reward: float


@dataclass(frozen=True)
class TunedFit:
    settings: BenchmarkSettings
    summary: FitSummary
    tuning_elapsed: float
    tuning_evaluations: int

    @property
    def elapsed(self) -> float:
        return self.tuning_elapsed + self.summary.elapsed

    @property
    def evaluations(self) -> int:
        return self.tuning_evaluations + self.summary.evaluations


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
    cfg.max_time_sec = 0.0 if settings.max_wall_time_hours is None else max(0.0, float(settings.max_wall_time_hours) * 3600.0)
    cfg.lm_iterations = settings.lm_iterations
    cfg.K = settings.K
    cfg.c = settings.c
    cfg.gamma = settings.gamma
    cfg.gp_rate = settings.gp_rate
    cfg.mutation_rate = settings.mutation_rate
    cfg.exploration_rate = settings.exploration_rate
    cfg.succ_error_tol = settings.succ_error_tol
    return cfg


def _fit_imcts(X_train, y_train, settings: BenchmarkSettings, seed: int) -> FitSummary:
    import numpy as np

    imcts = require_imcts()
    cfg = make_regressor_config(settings)
    model = imcts.Regressor(
        X_train.T.astype(np.float32, copy=False),
        y_train.astype(np.float32, copy=False),
        cfg,
    )

    t0 = time.perf_counter()
    result = model.fit(seed=seed)
    elapsed = time.perf_counter() - t0
    coefficients = [float(value) for value in getattr(result, "best_coefficients", [])]
    return FitSummary(
        raw_result=result,
        elapsed=float(elapsed),
        evaluations=int(getattr(result, "n_evals", 0)),
        expression=str(getattr(result, "expression", "")),
        coefficients=coefficients,
        reward=float(getattr(result, "best_reward", float("nan"))),
    )


def _score_fit_on_validation(summary: FitSummary, X_valid, y_valid) -> float:
    materialized = materialize_expression(summary.expression, summary.coefficients)
    y_pred = evaluate_expression(materialized, X_valid)
    score = regression_r2(y_valid, y_pred)
    if not math.isfinite(score):
        return float("-inf")
    return float(score)


def _kfold_indices(n_samples: int, n_splits: int, seed: int):
    import numpy as np

    n_splits = min(n_splits, n_samples)
    if n_splits < 2:
        raise ValueError("Need at least two samples for cross-validation.")

    indices = np.random.default_rng(seed).permutation(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        valid_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        current = stop
        yield train_idx, valid_idx


def _halving_resources(n_samples: int, n_candidates: int, cv_folds: int, factor: int) -> list[int]:
    if n_samples < 2:
        raise ValueError("Need at least two training samples for tuning.")
    if n_candidates <= 1:
        return [n_samples]

    n_iterations = 1 + int(math.floor(math.log(n_candidates, factor)))
    min_resources = max(cv_folds * 2, n_samples // (factor ** (n_iterations - 1)))
    while n_iterations > 1 and min_resources * (factor ** (n_iterations - 1)) > n_samples:
        n_iterations -= 1

    resources = [min(n_samples, min_resources * (factor ** iteration)) for iteration in range(n_iterations)]
    unique_resources: list[int] = []
    for resource in resources:
        resource = max(2, int(resource))
        if not unique_resources or resource > unique_resources[-1]:
            unique_resources.append(resource)
    return unique_resources


def _select_resource_subset(X_train, y_train, resource: int, seed: int):
    import numpy as np

    if resource >= X_train.shape[0]:
        return X_train, y_train
    indices = np.random.default_rng(seed).permutation(X_train.shape[0])[:resource]
    return X_train[indices], y_train[indices]


def _run_halving_tuning(
    X_train,
    y_train,
    settings: BenchmarkSettings,
    seed: int,
) -> tuple[BenchmarkSettings, float, int]:
    candidates = iter_tuning_candidates(settings)
    resources = _halving_resources(
        n_samples=int(X_train.shape[0]),
        n_candidates=len(candidates),
        cv_folds=settings.tuning.cv_folds,
        factor=settings.tuning.factor,
    )
    tuning_elapsed = 0.0
    tuning_evaluations = 0
    tuning_budget_sec = None
    if settings.tuning.max_wall_time_hours is not None:
        tuning_budget_sec = max(0.0, float(settings.tuning.max_wall_time_hours) * 3600.0)

    for iteration, resource in enumerate(resources):
        X_resource, y_resource = _select_resource_subset(X_train, y_train, resource, seed + iteration)
        scored_candidates: list[tuple[float, int, BenchmarkSettings]] = []
        for candidate_index, candidate_settings in enumerate(candidates):
            fold_scores: list[float] = []
            for fold_index, (train_idx, valid_idx) in enumerate(
                _kfold_indices(
                    int(X_resource.shape[0]),
                    settings.tuning.cv_folds,
                    seed + iteration * 10_000 + candidate_index * 100,
                )
            ):
                fit_settings = candidate_settings
                if tuning_budget_sec is not None:
                    remaining_budget_sec = tuning_budget_sec - tuning_elapsed
                    if remaining_budget_sec <= 0.0:
                        break
                    current_limit = settings.max_wall_time_hours
                    current_limit_sec = None if current_limit is None else max(0.0, float(current_limit) * 3600.0)
                    fit_limit_sec = remaining_budget_sec if current_limit_sec is None else min(remaining_budget_sec, current_limit_sec)
                    fit_settings = _with_max_wall_time(candidate_settings, fit_limit_sec / 3600.0)

                summary = _fit_imcts(
                    X_resource[train_idx],
                    y_resource[train_idx],
                    fit_settings,
                    seed + iteration * 10_000 + candidate_index * 100 + fold_index,
                )
                tuning_elapsed += summary.elapsed
                tuning_evaluations += summary.evaluations
                fold_scores.append(_score_fit_on_validation(summary, X_resource[valid_idx], y_resource[valid_idx]))

            mean_score = float("-inf")
            finite_scores = [score for score in fold_scores if math.isfinite(score)]
            if finite_scores:
                mean_score = float(sum(finite_scores) / len(finite_scores))
            scored_candidates.append((mean_score, candidate_index, candidate_settings))
            if tuning_budget_sec is not None and tuning_elapsed >= tuning_budget_sec:
                break

        if not scored_candidates:
            break

        scored_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        keep = max(1, len(scored_candidates) // settings.tuning.factor)
        candidates = [candidate for _, _, candidate in scored_candidates[:keep]]
        if tuning_budget_sec is not None and tuning_elapsed >= tuning_budget_sec:
            break
        if len(candidates) == 1:
            break

    return candidates[0], tuning_elapsed, tuning_evaluations


def fit_with_optional_tuning(X_train, y_train, settings: BenchmarkSettings, seed: int) -> TunedFit:
    if settings.source_type != "dataset" or not settings.tuning.enabled or not settings.tuning.parameters:
        summary = _fit_imcts(X_train, y_train, settings, seed)
        return TunedFit(settings=settings, summary=summary, tuning_elapsed=0.0, tuning_evaluations=0)

    tuned_settings, tuning_elapsed, tuning_evaluations = _run_halving_tuning(X_train, y_train, settings, seed)
    final_summary = _fit_imcts(X_train, y_train, tuned_settings, seed)
    return TunedFit(
        settings=tuned_settings,
        summary=final_summary,
        tuning_elapsed=tuning_elapsed,
        tuning_evaluations=tuning_evaluations,
    )


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
    X_train, X_test, y_train, y_test = split_train_test(prepared.X_total, prepared.y_total, settings.test_ratio, seed)
    X_train, y_train = subsample_training_data(X_train, y_train, seed=seed)
    fitted = fit_with_optional_tuning(X_train, y_train, settings, seed)
    fit_result = fitted.summary

    coefficients = list(fit_result.coefficients)
    materialized = materialize_expression(fit_result.expression, coefficients)
    y_pred_train = evaluate_expression(materialized, X_train)
    y_pred_test = evaluate_expression(materialized, X_test)
    train_r2 = regression_r2(y_train, y_pred_train)
    test_r2 = regression_r2(y_test, y_pred_test)
    simplified_expression, complexity = simplify_with_complexity(materialized)
    reward = fit_result.reward if math.isfinite(fit_result.reward) else train_r2

    return BenchmarkResult(
        algorithm="imcts",
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
        reward=float(reward),
        success=bool(reward >= 1.0 - fitted.settings.succ_error_tol),
        train_r2=float(train_r2),
        test_r2=float(test_r2),
        complexity=complexity,
        evaluations=int(fitted.evaluations),
        time_sec=float(fitted.elapsed),
        tuning_time_sec=float(fitted.tuning_elapsed),
        tuning_evaluations=int(fitted.tuning_evaluations),
        expression=fit_result.expression,
        materialized_expression=materialized,
        simplified_expression=simplified_expression,
        coefficients=coefficients,
        algorithm_params=search_params(settings),
        tuned_params=search_params(fitted.settings),
        test_ratio=float(settings.test_ratio),
    )
