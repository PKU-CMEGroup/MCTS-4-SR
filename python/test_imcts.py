import numpy as np

try:
    import imcts
except ModuleNotFoundError as exc:  # pragma: no cover - user environment issue
    raise SystemExit(
        "imcts is not importable. Install the package with `python -m pip install -e .`, "
        "or run the CMake-built `python_smoke` test target."
    ) from exc


def test_basic():
    n = 20
    x = np.linspace(0, 2, n, dtype=np.float32).reshape(1, n)
    y = (x[0] + 1).astype(np.float32)

    cfg = imcts.RegressorConfig()
    cfg.ops = ["+", "-", "*", "/", "sin"]
    cfg.max_depth = 4
    cfg.K = 50
    cfg.max_evals = 1000

    model = imcts.Regressor(x, y, cfg)
    result = model.fit(seed=42)

    print(f"Best reward: {result.best_reward:.4f}, evals: {result.n_evals}")
    assert result.best_reward > 0.50, f"Expected >0.50, got {result.best_reward}"
    print("test_basic PASSED")


def test_pretty_expression_fallback_or_simplify():
    expr = "(x0 + 0) * 1"
    simplified = imcts.simplify_expression(expr)
    assert isinstance(simplified, str)
    assert simplified
    print(f"pretty expression: {simplified}")


def test_openmp_info():
    info = imcts.openmp_info()

    assert isinstance(info, dict)
    assert isinstance(info["enabled"], bool)
    if info["enabled"]:
        assert info["max_threads"] >= 1
        assert info["num_procs"] >= 1


def test_timing_stats():
    imcts.reset_timing_stats()
    stats = imcts.timing_stats()

    expected_sections = {
        "coefficient_optimize",
        "bridge_to_tree",
        "lm_residual",
        "lm_jacobian",
        "mcts_backpropagate",
        "mcts_crossover",
        "mcts_mutation",
        "mcts_rollout",
        "mcts_search",
        "normal_equation_accumulate",
        "optimizer_lm_minimize",
        "interpreter_evaluate",
        "interpreter_evaluate_residual",
        "interpreter_evaluate_with_jacobian",
    }
    assert expected_sections.issubset(stats.keys())
    for section in expected_sections:
        assert stats[section]["calls"] == 0
        assert stats[section]["total_seconds"] == 0.0
        assert stats[section]["average_seconds"] == 0.0


def test_timing_stats_record_fit_work():
    n = 128
    x = np.linspace(-1, 1, n, dtype=np.float32).reshape(1, n)
    y = (2.0 * x[0] + 1.0).astype(np.float32)

    cfg = imcts.RegressorConfig()
    cfg.ops = ["+", "*", "R"]
    cfg.max_depth = 3
    cfg.K = 10
    cfg.max_evals = 50
    cfg.lm_iterations = 2
    cfg.succ_error_tol = 0.0

    imcts.reset_timing_stats()
    imcts.Regressor(x, y, cfg).fit(seed=1)
    stats = imcts.timing_stats()

    assert stats["coefficient_optimize"]["calls"] > 0
    assert stats["bridge_to_tree"]["calls"] > 0
    assert "lm_residual" in stats
    assert "lm_jacobian" in stats
    assert stats["mcts_backpropagate"]["calls"] > 0
    assert stats["mcts_rollout"]["calls"] > 0
    assert stats["mcts_search"]["calls"] > 0
    assert "mcts_mutation" in stats
    assert "mcts_crossover" in stats
    assert stats["normal_equation_accumulate"]["calls"] > 0
    assert stats["optimizer_lm_minimize"]["calls"] > 0
    assert stats["interpreter_evaluate"]["calls"] > 0
    assert "interpreter_evaluate_residual" in stats
    assert "interpreter_evaluate_with_jacobian" in stats


def main():
    test_openmp_info()
    test_timing_stats()
    test_timing_stats_record_fit_work()
    test_basic()
    test_pretty_expression_fallback_or_simplify()


if __name__ == "__main__":
    main()
