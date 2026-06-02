import numpy as np
import imcts

x = np.random.uniform(-2, 2, size=(5, 200))
y = ( 2.026 * np.cos(x[4]) + 0.530 * x[1] ** 2 - 0.1757 * x[0]).astype(np.float32)

cfg = imcts.RegressorConfig()
cfg.ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "R"]
cfg.max_depth = 6
cfg.K = 500
cfg.c = 6.0
cfg.gamma = 0.5
cfg.gp_rate = 0.5
cfg.mutation_rate = 0.1
cfg.exploration_rate = 0.2
cfg.max_unary = 999
cfg.max_constants = 999
cfg.lm_iterations = 10
cfg.max_evals = 100000
cfg.succ_error_tol = 1e-6

model = imcts.Regressor(x, y, cfg)
result = model.fit(seed=42)

print(result.best_reward)
print(result.expression)
print(imcts.simplify_expression(result.expression, digits=4))