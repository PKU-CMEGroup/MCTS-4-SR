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


def main():
    test_basic()
    test_pretty_expression_fallback_or_simplify()


if __name__ == "__main__":
    main()
