from __future__ import annotations

import argparse
import csv
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
from imcts.benchmarks.config import build_settings, load_yaml_resource
from imcts.benchmarks.registry import load_bundled_registry
from imcts.benchmarks.sources import DatasetSource, ExpressionSource, PreparedCaseData
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


def test_format_result_omits_training_sample_count():
    result = executor.BenchmarkResult(
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
        ops=["+", "-"],
        max_depth=4,
        max_unary=2,
        max_constants=1,
        max_evals=500_000,
        lm_iterations=10,
        test_ratio=0.25,
    )

    assert "train_n=" not in runner._format_result(result)


def make_benchmark_result(case_name: str, run: int) -> executor.BenchmarkResult:
    return executor.BenchmarkResult(
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
        ops=["+"],
        max_depth=2,
        max_unary=1,
        max_constants=1,
        max_evals=10,
        lm_iterations=1,
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
    dataset_file = tmp_path / "datasets" / dataset_name / f"{dataset_name}.csv"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text("x0,x1,target\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n", encoding="utf-8")

    assert runner.main(["--list", "--group", "BlackBox"], workspace_root=tmp_path) == 0

    output = capsys.readouterr().out
    assert "1: 1027_ESL  samples=4 features=2 train_n=3 test_n=1" in output


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

    nguyen_output_dir = tmp_path / "nguyen"
    assert runner.main(["--group", "Nguyen", "--cases", "1", "--runs", "1", "--workers", "1", "--output", str(nguyen_output_dir)], workspace_root=tmp_path) == 0

    dataset_name = load_bundled_registry().get_cases("BlackBox")[0]["name"]
    dataset_file = tmp_path / "datasets" / dataset_name / f"{dataset_name}.csv"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text("x0,target\n1,1\n2,2\n3,3\n4,4\n", encoding="utf-8")

    blackbox_output_dir = tmp_path / "blackbox"
    assert runner.main(["--group", "BlackBox", "--cases", "1", "--runs", "1", "--workers", "1", "--output", str(blackbox_output_dir)], workspace_root=tmp_path) == 0

    nguyen_case = load_bundled_registry().get_cases("Nguyen")[0]
    blackbox_case = load_bundled_registry().get_cases("BlackBox")[0]
    nguyen_output = case_output_path(nguyen_output_dir, "Nguyen", nguyen_case)
    blackbox_output = case_output_path(blackbox_output_dir, "BlackBox", blackbox_case)
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
    assert "materialized_expression" in nguyen_rows[0]
    assert "materialized_expression" in blackbox_rows[0]
