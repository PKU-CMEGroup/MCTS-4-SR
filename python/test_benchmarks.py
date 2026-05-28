from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import imcts
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMCTS = REPO_ROOT / "imcts"
LOCAL_BENCHMARKS = LOCAL_IMCTS / "benchmarks"
if str(LOCAL_IMCTS) not in imcts.__path__:
    imcts.__path__.insert(0, str(LOCAL_IMCTS))

import imcts.benchmarks as benchmarks_pkg

if str(LOCAL_BENCHMARKS) not in benchmarks_pkg.__path__:
    benchmarks_pkg.__path__.insert(0, str(LOCAL_BENCHMARKS))

from imcts.benchmarks import executor, runner
from imcts import pretty
from imcts.benchmarks.config import build_settings, load_yaml_resource
from imcts.benchmarks.registry import load_bundled_registry
from imcts.benchmarks.sources import DatasetSource, ExpressionSource, PreparedCaseData, inspect_case_dataset
from imcts.benchmarks.writer import case_output_path


def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "group": None,
        "cases": "all",
        "config": None,
        "runs": None,
        "seed_start": None,
        "samples": None,
        "dataset_dir": None,
        "label": None,
        "test_ratio": None,
        "results_dir": None,
        "output": None,
        "split_by_case": False,
        "list": False,
        "tune": None,
        "ops": None,
        "max_evals": None,
        "max_depth": None,
        "max_unary": None,
        "max_constants": None,
        "lm_iterations": None,
        "K": None,
        "c": None,
        "gamma": None,
        "gp_rate": None,
        "mutation_rate": None,
        "exploration_rate": None,
        "succ_error_tol": None,
        "max_wall_time_hours": None,
        "workers": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_simplify_with_complexity_uses_srbench_node_counting():
    simplified, complexity = pretty.simplify_with_complexity("x0 + 0.00001*x1 + 1.23456")

    assert simplified == "x0 + 1.24"
    assert complexity == 3.0


def test_simplify_with_complexity_runs_in_process(monkeypatch: pytest.MonkeyPatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("simplification should not spawn subprocesses")

    monkeypatch.setattr(subprocess, "run", fake_run)

    simplified, complexity = pretty.simplify_with_complexity("x0 + x1", timeout_sec=1.0)

    assert not called
    assert simplified == "x0 + x1"
    assert complexity == 3.0


def test_registry_group_metadata_and_cases():
    registry = load_bundled_registry()

    nguyen = registry.get_group("Nguyen")
    blackbox = registry.get_group("BlackBox")

    assert nguyen.source_type == "expression"
    assert nguyen.default_config_name == "basic.yaml"
    assert not nguyen.needs_constant_op
    assert registry.get_cases("Nguyen")[0]["name"] == "Nguyen-1"

    assert blackbox.source_type == "dataset"
    assert blackbox.default_config_name == "blackbox.yaml"
    assert blackbox.needs_constant_op
    assert registry.get_cases("BlackBox")[0]["name"] == "1027_ESL"


def test_build_settings_prefers_cli_over_yaml_and_defaults():
    registry = load_bundled_registry()
    group = registry.get_group("Nguyen")
    args = make_args(runs=3, ops="sin,cos", max_evals=1234, test_ratio=0.4)
    raw_config = {
        "runs": 8,
        "data": {"test_ratio": 0.5},
        "search": {"ops": ["+", "-"], "max_evals": 999},
    }

    settings = build_settings(args, group, raw_config)

    assert settings.runs == 3
    assert settings.ops == ["sin", "cos"]
    assert settings.max_evals == 1234
    assert settings.test_ratio == 0.4


def test_load_yaml_resource_deep_merges_user_config_with_bundled_default(tmp_path: Path):
    registry = load_bundled_registry()
    group = registry.get_group("Nguyen")
    config_path = tmp_path / "nguyen_override.yaml"
    config_path.write_text(
        "search:\n"
        "  max_depth: 3\n",
        encoding="utf-8",
    )

    raw_config = load_yaml_resource(config_path, group.default_config_name)
    settings = build_settings(make_args(), group, raw_config)

    assert settings.runs == 100
    assert settings.max_depth == 3
    assert settings.K == 500


def test_build_settings_parses_tuning_section_and_cli_override():
    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    raw_config = {
        "tuning": {
            "enabled": True,
            "cv_folds": 3,
            "factor": 2,
            "max_wall_time_hours": 1.5,
            "parameters": {
                "max_depth": [4, 6],
                "K": [250, 500],
            },
        },
    }

    enabled = build_settings(make_args(), group, raw_config)
    disabled = build_settings(make_args(tune=False), group, raw_config)

    assert enabled.tuning.enabled
    assert enabled.tuning.cv_folds == 3
    assert enabled.tuning.factor == 2
    assert enabled.tuning.max_wall_time_hours == 1.5
    assert enabled.tuning.parameters == {"max_depth": [4, 6], "K": [250, 500]}
    assert not disabled.tuning.enabled


def test_make_regressor_config_sets_per_run_wall_time_seconds(monkeypatch: pytest.MonkeyPatch):
    class FakeConfig:
        pass

    fake_imcts = SimpleNamespace(RegressorConfig=FakeConfig)
    monkeypatch.setattr("imcts.benchmarks.executor.require_imcts", lambda: fake_imcts)

    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    settings = build_settings(
        make_args(max_wall_time_hours=2.5),
        group,
        load_yaml_resource(None, group.default_config_name),
    )

    cfg = executor.make_regressor_config(settings)

    assert cfg.max_time_sec == 9000.0


def test_iter_tuning_candidates_expands_search_parameter_grid():
    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    settings = build_settings(
        make_args(),
        group,
        {
            "search": {"max_depth": 2, "K": 10, "max_evals": 11},
            "tuning": {"parameters": {"max_depth": [2, 3], "K": [10, 20]}},
        },
    )

    candidates = executor.iter_tuning_candidates(settings)

    assert [candidate.max_depth for candidate in candidates] == [2, 2, 3, 3]
    assert [candidate.K for candidate in candidates] == [10, 20, 10, 20]
    assert all(candidate.max_evals == 11 for candidate in candidates)


def test_expression_source_prepares_symbolic_case():
    registry = load_bundled_registry()
    group = registry.get_group("Nguyen")
    case = registry.get_cases("Nguyen")[0]
    settings = build_settings(make_args(), group, load_yaml_resource(None, group.default_config_name))

    prepared = ExpressionSource().prepare(case, settings, seed=7, workspace_root=Path.cwd())

    assert prepared.source_type == "expression"
    assert prepared.target_expression == case["expression"]
    assert prepared.X_total.shape == (40, 1)
    assert prepared.feature_names == ["x0"]


def test_dataset_source_loads_csv_and_rejects_lfs_pointer(tmp_path: Path):
    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    dataset_dir = tmp_path / "datasets"
    valid_case = {"id": 1, "name": "toy"}
    invalid_case = {"id": 2, "name": "pointer"}

    valid_dir = dataset_dir / "toy"
    valid_dir.mkdir(parents=True)
    (valid_dir / "toy.csv").write_text("x0,target\n1,2\n3,4\n", encoding="utf-8")

    invalid_dir = dataset_dir / "pointer"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "pointer.tsv").write_text(
        "version https://git-lfs.github.com/spec/v1\n",
        encoding="utf-8",
    )

    settings = build_settings(make_args(dataset_dir=dataset_dir), group, load_yaml_resource(None, group.default_config_name))
    prepared = DatasetSource().prepare(valid_case, settings, seed=0, workspace_root=tmp_path)

    assert prepared.source_type == "dataset"
    assert prepared.target_expression == ""
    assert prepared.feature_names == ["x0"]
    assert prepared.X_total.shape == (2, 1)

    with pytest.raises(ValueError, match="Git LFS pointer"):
        DatasetSource().prepare(invalid_case, settings, seed=0, workspace_root=tmp_path)


def test_dataset_source_loads_deprecated_prefixed_dataset_directory(tmp_path: Path):
    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    dataset_dir = tmp_path / "datasets"
    case = {"id": 15, "name": "legacy"}

    legacy_dir = dataset_dir / "_deprecated_legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "_deprecated_legacy.tsv").write_text(
        "x0\ttarget\n"
        "1\t2\n"
        "3\t4\n",
        encoding="utf-8",
    )

    settings = build_settings(make_args(dataset_dir=dataset_dir), group, load_yaml_resource(None, group.default_config_name))
    prepared = DatasetSource().prepare(case, settings, seed=0, workspace_root=tmp_path)

    assert prepared.source_type == "dataset"
    assert prepared.feature_names == ["x0"]
    assert prepared.X_total.shape == (2, 1)
    assert prepared.y_total.tolist() == [2.0, 4.0]


def test_dataset_source_skips_rows_with_empty_values(tmp_path: Path):
    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    dataset_dir = tmp_path / "datasets"
    case = {"id": 1, "name": "toy"}

    case_dir = dataset_dir / "toy"
    case_dir.mkdir(parents=True)
    (case_dir / "toy.csv").write_text(
        "x0,x1,target\n"
        "1,10,2\n"
        "3,,4\n"
        "5,50,6\n",
        encoding="utf-8",
    )

    settings = build_settings(make_args(dataset_dir=dataset_dir), group, load_yaml_resource(None, group.default_config_name))
    prepared = DatasetSource().prepare(case, settings, seed=0, workspace_root=tmp_path)

    assert prepared.X_total.tolist() == [[1.0, 10.0], [5.0, 50.0]]
    assert prepared.y_total.tolist() == [2.0, 6.0]


def test_inspect_case_dataset_prefers_summary_stats_over_data_file(tmp_path: Path):
    case_dir = tmp_path / "datasets" / "toy"
    case_dir.mkdir(parents=True)
    (case_dir / "summary_stats.tsv").write_text(
        "dataset\tn_instances\tn_features\ttask\n"
        "toy\t159\t15\tregression\n",
        encoding="utf-8",
    )
    (case_dir / "toy.csv").write_text("x0,target\n1,2\n3,4\n", encoding="utf-8")

    metadata = inspect_case_dataset(tmp_path / "datasets", "toy", "target")

    assert metadata is not None
    assert metadata.samples == 159
    assert metadata.features == 15
    assert metadata.path == case_dir / "summary_stats.tsv"


def test_inspect_case_dataset_supports_deprecated_summary_stats_directory(tmp_path: Path):
    case_dir = tmp_path / "datasets" / "_deprecated_legacy"
    case_dir.mkdir(parents=True)
    (case_dir / "summary_stats.tsv").write_text(
        "dataset\tn_instances\tn_features\ttask\n"
        "_deprecated_legacy\t47\t7\tregression\n",
        encoding="utf-8",
    )

    metadata = inspect_case_dataset(tmp_path / "datasets", "legacy", "target")

    assert metadata is not None
    assert metadata.samples == 47
    assert metadata.features == 7
    assert metadata.path == case_dir / "summary_stats.tsv"


def test_run_case_subsamples_large_training_split(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeConfig:
        pass

    class FakeResult:
        def __init__(self):
            self.best_coefficients = []
            self.expression = "x0"
            self.best_reward = 0.75
            self.n_evals = 7

    class FakeRegressor:
        def __init__(self, x, y, cfg):
            captured["x_shape"] = x.shape
            captured["y_shape"] = y.shape
            self.cfg = cfg

        def fit(self, seed):
            assert seed is not None
            return FakeResult()

    fake_imcts = SimpleNamespace(RegressorConfig=FakeConfig, Regressor=FakeRegressor)
    monkeypatch.setattr("imcts.benchmarks.executor.require_imcts", lambda: fake_imcts)

    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    settings = build_settings(make_args(), group, load_yaml_resource(None, group.default_config_name))
    n_samples = 13_336
    X_total = np.arange(n_samples, dtype=np.float64).reshape(-1, 1)
    prepared = PreparedCaseData(
        X_total=X_total,
        y_total=X_total[:, 0],
        feature_names=["x0"],
        target_expression="",
        source_type="dataset",
    )

    result = executor.run_case(
        "BlackBox",
        {"id": 1, "name": "large"},
        run_index=0,
        seed=42,
        settings=settings,
        prepared=prepared,
    )

    assert captured["x_shape"] == (1, 10_000)
    assert captured["y_shape"] == (10_000,)
    assert result.samples_train == 10_000
    assert result.samples_test == 3_334
    assert result.samples_total == n_samples
    assert result.algorithm == "imcts"
    assert result.algorithm_params["max_depth"] == settings.max_depth
    assert result.tuned_params == result.algorithm_params


def test_run_case_tunes_blackbox_candidates_before_final_fit(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    class FakeConfig:
        pass

    class FakeResult:
        def __init__(self, cfg):
            self.best_coefficients = []
            self.expression = "x0" if cfg.max_depth == 3 else "0.0"
            self.best_reward = 0.75
            self.n_evals = cfg.max_evals

    class FakeRegressor:
        def __init__(self, x, y, cfg):
            calls.append(
                {
                    "x_shape": x.shape,
                    "cfg": cfg,
                }
            )
            self.cfg = cfg

        def fit(self, seed):
            return FakeResult(self.cfg)

    fake_imcts = SimpleNamespace(RegressorConfig=FakeConfig, Regressor=FakeRegressor)
    monkeypatch.setattr("imcts.benchmarks.executor.require_imcts", lambda: fake_imcts)

    registry = load_bundled_registry()
    group = registry.get_group("BlackBox")
    settings = build_settings(
        make_args(),
        group,
        {
            "data": {"test_ratio": 0.25},
            "runtime": {"max_wall_time_hours": 42.0},
            "search": {"max_depth": 2, "K": 10, "max_evals": 11},
            "tuning": {
                "enabled": True,
                "cv_folds": 2,
                "factor": 2,
                "max_wall_time_hours": 6.0,
                "parameters": {"max_depth": [2, 3], "K": [10, 20]},
            },
        },
    )
    X_total = np.arange(16, dtype=np.float64).reshape(-1, 1)
    prepared = PreparedCaseData(
        X_total=X_total,
        y_total=X_total[:, 0],
        feature_names=["x0"],
        target_expression="",
        source_type="dataset",
    )

    result = executor.run_case(
        "BlackBox",
        {"id": 1, "name": "toy"},
        run_index=0,
        seed=42,
        settings=settings,
        prepared=prepared,
    )

    assert len(calls) == 13
    assert {call["x_shape"][1] for call in calls[:8]} == {2}
    assert {call["x_shape"][1] for call in calls[8:12]} == {4}
    assert calls[-1]["x_shape"] == (1, 12)
    assert all(call["cfg"].max_time_sec <= 21600.0 for call in calls[:-1])
    assert calls[-1]["cfg"].max_time_sec == 151200.0
    assert result.tuned_params["max_depth"] == 3
    assert result.tuned_params["K"] in {10, 20}
    assert result.test_r2 == 1.0
    assert result.evaluations == 13 * 11


def test_run_case_does_not_tune_expression_benchmarks(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    class FakeConfig:
        pass

    class FakeResult:
        best_coefficients = []
        expression = "x0"
        best_reward = 0.75
        n_evals = 7

    class FakeRegressor:
        def __init__(self, x, y, cfg):
            calls.append(cfg.max_depth)

        def fit(self, seed):
            return FakeResult()

    fake_imcts = SimpleNamespace(RegressorConfig=FakeConfig, Regressor=FakeRegressor)
    monkeypatch.setattr("imcts.benchmarks.executor.require_imcts", lambda: fake_imcts)

    registry = load_bundled_registry()
    group = registry.get_group("Nguyen")
    settings = build_settings(
        make_args(),
        group,
        {
            "data": {"test_ratio": 0.25},
            "search": {"max_depth": 2, "max_evals": 7},
            "tuning": {
                "enabled": True,
                "cv_folds": 2,
                "factor": 2,
                "parameters": {"max_depth": [2, 3]},
            },
        },
    )
    X_total = np.arange(8, dtype=np.float64).reshape(-1, 1)
    prepared = PreparedCaseData(
        X_total=X_total,
        y_total=X_total[:, 0],
        feature_names=["x0"],
        target_expression="x0",
        source_type="expression",
    )

    result = executor.run_case(
        "Nguyen",
        {"id": 1, "name": "Nguyen-1"},
        run_index=0,
        seed=42,
        settings=settings,
        prepared=prepared,
    )

    assert calls == [2]
    assert result.tuned_params["max_depth"] == 2


def test_format_result_omits_training_sample_count():
    result = executor.BenchmarkResult(
        algorithm="imcts",
        group="BlackBox",
        case_id=1,
        case_name="1027_ESL",
        source_type="dataset",
        run=0,
        seed=23654,
        samples_total=13_336,
        samples_train=10_000,
        samples_test=3_334,
        variables=4,
        feature_names=["x0"],
        target_expression="",
        reward=0.750642,
        success=False,
        train_r2=0.889648,
        test_r2=0.850435,
        complexity=44.0,
        evaluations=500_002,
        time_sec=78.088,
        expression="x0",
        materialized_expression="x0",
        simplified_expression="x0",
        coefficients=[],
        tuning_time_sec=0.0,
        tuning_evaluations=0,
        algorithm_params={"max_depth": 4},
        tuned_params={"max_depth": 4},
        test_ratio=0.25,
    )

    assert "train_n=" not in runner._format_result(result)


def make_benchmark_result(case_name: str, run: int) -> executor.BenchmarkResult:
    return executor.BenchmarkResult(
        algorithm="imcts",
        group="BlackBox",
        case_id=1,
        case_name=case_name,
        source_type="dataset",
        run=run,
        seed=executor.seed_for_run(0, run),
        samples_total=4,
        samples_train=3,
        samples_test=1,
        variables=1,
        feature_names=["x0"],
        target_expression="",
        reward=0.5,
        success=False,
        train_r2=0.5,
        test_r2=0.5,
        complexity=1.0,
        evaluations=7,
        time_sec=1.0,
        expression="x0",
        materialized_expression="x0",
        simplified_expression="x0",
        coefficients=[],
        tuning_time_sec=0.0,
        tuning_evaluations=0,
        algorithm_params={"max_depth": 2},
        tuned_params={"max_depth": 2},
        test_ratio=0.25,
    )


def test_parallel_runner_checkpoints_case_csv_after_each_completed_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    case = {"id": 1, "name": "slow_case"}
    first_result = make_benchmark_result("slow_case", run=0)
    second_result = make_benchmark_result("slow_case", run=1)
    future_results = [first_result, second_result]

    class FakeFuture:
        def __init__(self, result: executor.BenchmarkResult):
            self._result = result

        def result(self):
            return self._result

    class FakePool:
        def __init__(self, max_workers, initializer):
            self.futures: list[FakeFuture] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            future = FakeFuture(future_results[len(self.futures)])
            self.futures.append(future)
            return future

    def fake_as_completed(futures):
        futures = list(futures)
        output_path = case_output_path(tmp_path, "BlackBox", case)
        yield futures[0]
        assert output_path.exists()
        with output_path.open("r", encoding="utf-8", newline="") as f:
            checkpoint_rows = list(csv.DictReader(f))
        assert [row["run"] for row in checkpoint_rows] == ["0"]
        yield futures[1]

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(runner, "as_completed", fake_as_completed)

    settings = SimpleNamespace(runs=2, seed_start=0, source_type="dataset")
    runner._run_parallel([case], settings, "BlackBox", tmp_path, tmp_path, num_workers=2)


def test_parallel_runner_interleaves_cases_by_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cases = [{"id": 1, "name": "case_a"}, {"id": 2, "name": "case_b"}]
    submitted: list[tuple[str, int]] = []

    class FakeFuture:
        def __init__(self, case_name: str, run_index: int):
            self._result = make_benchmark_result(case_name, run_index)

        def result(self):
            return self._result

    class FakePool:
        def __init__(self, max_workers, initializer):
            self.futures: list[FakeFuture] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            case = args[1]
            run_index = args[2]
            submitted.append((case["name"], run_index))
            future = FakeFuture(case["name"], run_index)
            self.futures.append(future)
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(runner, "as_completed", lambda futures: list(futures))

    settings = SimpleNamespace(runs=2, seed_start=0, source_type="dataset")
    runner._run_parallel(cases, settings, "BlackBox", tmp_path, tmp_path, num_workers=2)

    assert submitted == [
        ("case_a", 0),
        ("case_b", 0),
        ("case_a", 1),
        ("case_b", 1),
    ]


def test_split_output_dir_nests_imcts_before_group(tmp_path: Path):
    from imcts.benchmarks.writer import split_output_dir

    output = split_output_dir(
        group="Nguyen",
        explicit_output=None,
        results_dir=Path("benchmark_results"),
        workspace_root=tmp_path,
    )

    assert output == tmp_path / "benchmark_results" / "imcts" / "nguyen"


def test_sequential_runner_does_not_apply_group_wall_time_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    case = {"id": 1, "name": "case_a"}

    class FakeSource:
        def prepare(self, case, settings, seed, workspace_root):
            return object()

    def fake_run_case(group_name, case, run_index, seed, settings, prepared):
        return make_benchmark_result(case["name"], run_index)

    monkeypatch.setattr(runner.executor, "run_case", fake_run_case)

    settings = SimpleNamespace(runs=2, seed_start=0, source_type="dataset")
    rows = runner._run_sequential(
        [case],
        settings,
        FakeSource(),
        "BlackBox",
        tmp_path,
        tmp_path,
    )

    assert [row.run for row in rows] == [0, 1]


def test_list_blackbox_without_datasets_does_not_require_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert runner.main(["--list", "--group", "BlackBox"], workspace_root=tmp_path) == 0

    output = capsys.readouterr().out
    assert "BlackBox:" in output
    assert "1027_ESL" in output
    assert "train_n=" not in output


def test_list_blackbox_includes_dataset_shape_when_available(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dataset_name = load_bundled_registry().get_cases("BlackBox")[0]["name"]
    summary_file = tmp_path / "datasets" / dataset_name / "summary_stats.tsv"
    summary_file.parent.mkdir(parents=True)
    summary_file.write_text(
        "dataset\tn_instances\tn_features\ttask\n"
        f"{dataset_name}\t4\t2\tregression\n",
        encoding="utf-8",
    )

    assert runner.main(["--list", "--group", "BlackBox"], workspace_root=tmp_path) == 0

    output = capsys.readouterr().out
    assert "  1: 1027_ESL" in output
    assert "samples=4" in output
    assert "features=2" in output
    assert "train_n=3" in output
    assert "test_n=1" in output


def test_list_symbolic_includes_shape_before_expression(capsys: pytest.CaptureFixture[str]):
    assert runner.main(["--list", "--group", "Nguyen"]) == 0

    output = capsys.readouterr().out
    first_case_line = next(line for line in output.splitlines() if "Nguyen-1" in line)
    assert "samples=40" in first_case_line
    assert "features=1" in first_case_line
    assert "train_n=20" in first_case_line
    assert "test_n=20" in first_case_line
    assert "expr=x[0]**3 + x[0]**2 + x[0]" in first_case_line
    assert first_case_line.index("test_n=20") < first_case_line.index("expr=")
    assert " y = " not in first_case_line


def test_runner_smoke_for_nguyen_and_blackbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeConfig:
        pass

    class FakeResult:
        def __init__(self):
            self.best_coefficients = []
            self.expression = "x0"
            self.best_reward = 0.75
            self.n_evals = 7

    class FakeRegressor:
        def __init__(self, x, y, cfg):
            self.x = x
            self.y = y
            self.cfg = cfg

        def fit(self, seed):
            assert seed is not None
            return FakeResult()

    fake_imcts = SimpleNamespace(RegressorConfig=FakeConfig, Regressor=FakeRegressor)
    monkeypatch.setattr("imcts.benchmarks.executor.require_imcts", lambda: fake_imcts)

    nguyen_results_dir = tmp_path / "nguyen-results"
    assert runner.main(["--group", "Nguyen", "--cases", "1", "--runs", "1", "--workers", "1", "--results-dir", str(nguyen_results_dir)], workspace_root=tmp_path) == 0

    dataset_name = load_bundled_registry().get_cases("BlackBox")[0]["name"]
    dataset_file = tmp_path / "datasets" / dataset_name / f"{dataset_name}.csv"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text("x0,target\n1,1\n2,2\n3,3\n4,4\n", encoding="utf-8")

    blackbox_results_dir = tmp_path / "blackbox-results"
    assert runner.main(["--group", "BlackBox", "--cases", "1", "--runs", "1", "--workers", "1", "--results-dir", str(blackbox_results_dir)], workspace_root=tmp_path) == 0

    nguyen_case = load_bundled_registry().get_cases("Nguyen")[0]
    blackbox_case = load_bundled_registry().get_cases("BlackBox")[0]
    nguyen_output = case_output_path(nguyen_results_dir / "imcts" / "nguyen", "Nguyen", nguyen_case)
    blackbox_output = case_output_path(blackbox_results_dir / "imcts" / "blackbox", "BlackBox", blackbox_case)
    assert nguyen_output.exists()
    assert blackbox_output.exists()

    with nguyen_output.open("r", encoding="utf-8", newline="") as f:
        nguyen_rows = list(csv.DictReader(f))
    with blackbox_output.open("r", encoding="utf-8", newline="") as f:
        blackbox_rows = list(csv.DictReader(f))

    assert len(nguyen_rows) == 1
    assert len(blackbox_rows) == 1
    assert nguyen_rows[0]["case_name"] == "Nguyen-1"
    assert blackbox_rows[0]["case_name"] == dataset_name
    assert nguyen_rows[0]["algorithm"] == "imcts"
    assert "algorithm_params" in nguyen_rows[0]
    assert "tuned_params" in nguyen_rows[0]
    assert "tuning_time_sec" in nguyen_rows[0]
    assert "materialized_expression" in nguyen_rows[0]
    assert "materialized_expression" in blackbox_rows[0]


def test_report_loads_algorithm_group_result_layout(tmp_path: Path):
    from imcts.benchmarks.report import load_rows, summarize_by_group

    csv_dir = tmp_path / "results" / "imcts" / "nguyen"
    csv_dir.mkdir(parents=True)
    (csv_dir / "nguyen_001_nguyen-1.csv").write_text(
        "algorithm,group,case_id,case_name,time_sec,success,evaluations,test_r2\n"
        "imcts,Nguyen,1,Nguyen-1,1.5,true,7,1.0\n",
        encoding="utf-8",
    )

    rows = load_rows(tmp_path / "results", ["nguyen"])
    summaries = summarize_by_group(rows)

    assert rows[0]["algorithm"] == "imcts"
    assert summaries[0].algorithm == "imcts"
    assert summaries[0].group == "Nguyen"
