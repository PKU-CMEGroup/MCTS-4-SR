from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import imcts

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMCTS = REPO_ROOT / "imcts"
LOCAL_BENCHMARKS = LOCAL_IMCTS / "benchmarks"
if str(LOCAL_IMCTS) not in imcts.__path__:
    imcts.__path__.insert(0, str(LOCAL_IMCTS))

import imcts.benchmarks as benchmarks_pkg

if str(LOCAL_BENCHMARKS) not in benchmarks_pkg.__path__:
    benchmarks_pkg.__path__.insert(0, str(LOCAL_BENCHMARKS))

from imcts.benchmarks import runner
from imcts.benchmarks.config import build_settings, load_yaml_resource
from imcts.benchmarks.registry import load_bundled_registry
from imcts.benchmarks.sources import DatasetSource, ExpressionSource


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
        "output": None,
        "split_by_case": False,
        "list": False,
        "threads": None,
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

    nguyen_output = tmp_path / "nguyen.csv"
    assert runner.main(["--group", "Nguyen", "--cases", "1", "--runs", "1", "--output", str(nguyen_output)], workspace_root=tmp_path) == 0
    assert nguyen_output.exists()

    dataset_name = load_bundled_registry().get_cases("BlackBox")[0]["name"]
    dataset_file = tmp_path / "datasets" / dataset_name / f"{dataset_name}.csv"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text("x0,target\n1,1\n2,2\n3,3\n4,4\n", encoding="utf-8")

    blackbox_output = tmp_path / "blackbox.csv"
    assert runner.main(["--group", "BlackBox", "--cases", "1", "--runs", "1", "--output", str(blackbox_output)], workspace_root=tmp_path) == 0
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
