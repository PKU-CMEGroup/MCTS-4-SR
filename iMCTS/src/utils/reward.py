import numpy as np
from typing import Tuple, Callable, Any
import numba
import nlopt

@numba.njit(fastmath=True, cache=True)
def cal_res_numba(y_pred: np.ndarray, y_train: np.ndarray, sigma: float) -> float:
    """
    Fast reward component: compute a normalized (1 / (1+RMSE/σ)).
    Numba accelerates this hot path.
    """
    diff = y_pred - y_train
    # mean squared error (avoid np.mean for slightly less overhead inside nopython)
    mse = np.dot(diff, diff) / diff.size
    # If sigma is zero (constant target), fall back to raw RMSE (avoid div-by-zero)
    if sigma == 0.0:
        sigma = 1.0
    res1 = np.sqrt(mse) / sigma

    # Protect against nan / inf propagating upward
    if np.isnan(res1) or np.isinf(res1):
        return 0.0
    return 1.0 / (1.0 + res1)

class Optimizer:
    """Optimize symbolic expression constants and compute reward.

    Performance oriented adjustments:
    - Avoid repeated regex compilation inside loops (use str.replace for exact tokens).
    - Use Numba accelerated RMSE-based reward.
    - Early exits for simple / non-optimizable states.
    - Reduce attribute / global lookups in hot sections via local bindings.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, context: dict[str, Any],
                 cal_reward: Callable | None = None, optimization_method: str = 'LN_NELDERMEAD'):
        self.x_train = x_train
        self.y_train = y_train
        # Precompute target std (σ). If y_train is None, default to 1.0 to avoid div-by-zero.
        self.sigma = float(np.std(y_train)) if y_train is not None else 1.0
        self.context = context
        # Default reward function is _cal_res, which uses numba accelerated metric.
        self.cal_reward = cal_reward if cal_reward is not None else self._cal_res
        # Optimization method for constant optimization
        self.optimization_method = getattr(nlopt, optimization_method)

    def optimize_constants(self, state) -> Tuple[str, float]:
        """Optimize constants in the state's expression and compute reward.

        Returns
        -------
        (expression, reward)
            The (possibly) constant-substituted expression and its computed reward.
        """
        expression: str = state.get_expression()

        # Quick rejection: if expression already contains invalid tokens, skip optimization entirely.
        if any(tok in expression for tok in ("zoo", "nan", "inf")):
            if state.constant_count > 0:
                return expression, 0.0
            return expression, 0.0

        constant_count = state.constant_count
        if constant_count > 0:
            # Number of purely real constants (R) and complex constants (C)
            r_len = state.real_constant_count
            c_len = constant_count - r_len

            # Build once (slightly faster than eval of string each param iteration inside minimize)
            # NOTE: context is trusted upstream. If untrusted, this is a code injection risk.
            f_pred_const = eval(compile('lambda x, C, R: ' + expression, '<expr>', 'eval'), self.context)

            # Initial guess: complex constants represented by separate real / imag components.
            # Use normal distribution; could be improved by heuristic seeding.
            if c_len:
                guess_c = np.random.randn(c_len * 2)
            else:
                guess_c = np.empty(0)
            if r_len:
                guess_r = np.random.randn(r_len)
            else:
                guess_r = np.empty(0)
            initial_guess = np.concatenate((guess_c, guess_r)) if constant_count else np.empty(0)
            n_params = len(initial_guess)

            # Create an NLopt optimizer object with the specified algorithm.
            opt = nlopt.opt(self.optimization_method, n_params)
            
            # Set the objective function to be maximized. We minimize the negative reward.
            opt.set_min_objective(lambda p, grad: -self._cal_reward_wrapper(p, f_pred_const, c_len))

            # Set a relative tolerance for the optimization.
            opt.set_xtol_rel(1e-6)
            # Set a maximum number of evaluations to prevent infinite loops.
            opt.set_maxeval(250)

            # --- Start: Set bounds for the optimization parameters ---
            # Set a uniform lower and upper bound for all parameters.
            # This is equivalent to the `bounds` parameter in `differential_evolution`.
            bounds = np.array([-10.0] * n_params)
            opt.set_lower_bounds(bounds)
            opt.set_upper_bounds(-bounds) # Using -bounds to get [10.0, 10.0, ...]
            # --- End: Set bounds ---

            try:
                # Run the optimization.
                optimized_params = opt.optimize(initial_guess)
                min_neg_reward = opt.last_optimum_value()
            # except nlopt.Failure as e:
            #     print(f"NLopt optimization failed: {e}")
            #     return expression, 0.0
            except Exception as e:
                print(f"An unexpected error occurred during NLopt optimization: {e}")
                return expression, 0.0

            # Split optimized params back into complex + real sets.
            if c_len:
                complex_constants = optimized_params[:c_len] + 1j * optimized_params[c_len:2 * c_len]
            else:
                complex_constants = []
            real_constants = optimized_params[2 * c_len:]

            # Fast token substitution (exact token replacement) without regex overhead.
            # Use repr to preserve precision and include complex literal syntax for Python.
            for idx, c in enumerate(complex_constants):
                expression = expression.replace(f'C[{idx}]', repr(c))
            for idx, r in enumerate(real_constants):
                expression = expression.replace(f'R[{idx}]', repr(float(r)))

        elif constant_count == 0:
            # No constants: skip all optimization ceremony.
            pass

        # Compile final expression to a single lambda for evaluation.
        try:
            f_pred = eval(compile(f"lambda x: {expression}", "<expr-final>", "eval"), self.context)
        except Exception:
            return expression, 0.0

        # Compute reward safely.
        try:
            reward = self.cal_reward(self.x_train, self.y_train, f_pred)
        except ZeroDivisionError:
            reward = 0.0
        except Exception:
            # Any other runtime failure yields zero reward instead of aborting search.
            reward = 0.0
        return expression, float(reward)

    def _cal_res(self, x_train: np.ndarray, y_train: np.ndarray, f_pred: Callable[[np.ndarray], np.ndarray]) -> float:
        """Compute reward given prediction function.

        Fast path + guards around numerical issues. Assumes f_pred returns np.ndarray.
        """
        try:
            y_pred = f_pred(x_train)
            # Ensure ndarray (some lambdas might return list)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.asarray(y_pred)
            return float(cal_res_numba(y_pred, y_train, self.sigma))
        except ZeroDivisionError:
            return 0.0
        except Exception:
            # Any unexpected failure yields zero reward instead of aborting search.
            return 0.0

    def _cal_reward_wrapper(self, params: np.ndarray, f_pred_const: Callable, c_len: int) -> float:
        """Internal helper used during optimization.

        Converts flattened parameter vector into complex / real constant arrays
        and evaluates reward for current parameter set.
        """
        if c_len:
            complex_params = params[:c_len] + 1j * params[c_len:2 * c_len]
            real_params = params[2 * c_len:]
        else:
            complex_params = []
            real_params = params
        # Inline lambda to reduce nested scope overhead.
        reward = self.cal_reward(self.x_train, self.y_train,
                                 lambda x: f_pred_const(x, C=complex_params, R=real_params))
        return float(reward)