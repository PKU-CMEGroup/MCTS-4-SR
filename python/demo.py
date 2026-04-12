"""iMCTS symbolic regression demo."""

import time

import numpy as np

try:
    import imcts
except ModuleNotFoundError as exc:  # pragma: no cover - user environment issue
    raise SystemExit(
        "imcts is not installed. From the repo root, run `python3 -m pip install -e .`."
    ) from exc


def main() -> None:
    n = 20
    x1 = np.random.uniform(-1, 1, n).astype(np.float32).reshape(1, n)
    y1 = (x1[0]**6 + x1[0]**5 + x1[0]**4 + x1[0]**3 + x1[0]**2 + x1[0]).astype(np.float32)
    name = "y = x^6 + x^5 + x^4 + x^3 + x^2 + x"

    cfg = imcts.RegressorConfig()
    # cfg.ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
    # cfg.max_depth = 6
    # cfg.K = 500
    # cfg.max_evals = 2_000_000
    # cfg.c = 4.0
    # cfg.gamma = 0.5
    # cfg.gp_rate = 0.2
    # cfg.mutation_rate = 0.1
    # cfg.exploration_rate = 0.2
    # cfg.max_unary = 999
    # cfg.max_constants = 5
    # cfg.lm_iterations = 100
    # cfg.succ_error_tol = 1e-6

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  samples={x1.shape[1]}, vars={x1.shape[0]}, max_evals={cfg.max_evals}")
    print(f"{'=' * 60}")

    # model = imcts.Regressor(x1, y1, cfg)
    model = imcts.Regressor(x1, y1)
    t0 = time.time()
    result = model.fit(seed=42)
    elapsed = time.time() - t0

    print(f"  best_reward : {result.best_reward:.6f}")
    print(f"  evaluations : {result.n_evals}")
    print(f"  time        : {elapsed:.2f}s")
    print(f"  best_path   : {list(result.best_path)}")
    print(f"  coefficients: {list(result.best_coefficients)}")
    simplified_expr = imcts.simplify_expression(result.expression)
    print(f"  expression  : {result.expression}")
    if simplified_expr != result.expression:
        print(f"  simplified  : {simplified_expr}")


if __name__ == "__main__":
    main()
