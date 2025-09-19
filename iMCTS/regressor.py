import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sympy import sympify, expand, expand_log
import time
import random
from iMCTS.mcts import MCTS
from iMCTS.src import ExpTree, Optimizer
from iMCTS.gp import GPManager

def simplify_expression(exp_str: str, verbose: bool = False) -> str:
    """Simplify mathematical expression string without relying on class methods."""
    try:
        cleaned = exp_str.replace("[", "").replace("]", "")
        expr = sympify(cleaned)
        expanded = expand(expand_log(expr, force=True))
        return str(expanded)
    except Exception as e:
        if verbose:
            print(f"Simplification failed: {e}")
        return exp_str

class Regressor:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        ops: List[str] = None,
        arity_dict: Dict[str, int] = None,
        context: Dict = None,
        max_depth: int = 6,
        K: int = 500,
        c: float = 4.0,
        gamma: float = 0.5,
        gp_rate: float = 0.2,
        mutation_rate: float = 0.1,
        exploration_rate: float = 0.2,
        max_single_arity_ops: int = 999,
        max_constants: int = 10,
        max_expressions: float = 2e6,
        verbose: bool = False,
        reward_func: Optional[Callable] = None,
        optimization_method: str = 'LN_NELDERMEAD'
    ):
        """
        Symbolic Regression Regressor with MCTS optimization

        Parameters:
        x_train (np.ndarray): Training data features of shape (n_features, n_samples)
        y_train (np.ndarray): Training data labels of shape (n_samples,)
        seed (int, optional): Random seed for reproducibility
        """
        # Input validation
        self._validate_inputs(x_train, y_train, max_depth)

        # Initialize core components
        self.x_train = x_train
        self.y_train = y_train
        self.verbose = verbose
        self.reward_func = reward_func
        
        # Initialize operations and context
        self.ops = self._init_operations(ops, x_train.shape[0])
        self.arity_dict = self._init_arity_dict(arity_dict, x_train.shape[0])
        self.global_context = self._init_context(context)
        
        # Initialize optimization parameters
        self._init_optimization_params(
            max_depth,
            K,
            c,
            gamma,
            gp_rate,
            mutation_rate,
            exploration_rate,
            max_single_arity_ops,
            max_constants,
            max_expressions
        )

        # Initialize core components
        self.optimizer = Optimizer(
            x_train,
            y_train,
            self.global_context,
            reward_func,
            optimization_method
        )
        
        self.exp_tree = self._create_exp_tree()

    def fit(self, seed: int = None) -> Tuple[str, float, int, int, List]:
        """Perform symbolic regression search"""
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        with np.errstate(all='ignore'):
            mcts = self._create_mcts()
            self.start_time = time.time()
            self.find_best(mcts)
            exp_str, reward = mcts.exp_queue.best()
            
            return (
                simplify_expression(exp_str, self.verbose),
                exp_str,
                mcts.count_num,
                mcts.root.path_queue.best()[0]
            )
    
    def find_best(self, mcts: MCTS):
        """Find best expression within given time limit"""
        import time
        search_num = 0
        # Track last report time (store on instance so future extensions can reuse)
        last_report_time = getattr(self, '_last_report_time', self.start_time)
        REPORT_INTERVAL = 10.0  # seconds
        while mcts.count_num < self.max_expressions:
            search_num += 1
            best_reward = mcts.search(self.exp_tree)

            now = time.time()
            # Time-based periodic status report (every ~10s)
            if self.verbose and (now - last_report_time >= REPORT_INTERVAL):
                self.print_status(mcts)
                last_report_time = now
                self._last_report_time = last_report_time

            # Exit if exceeding 48 hours total runtime
            if now - self.start_time > 172800:  # 48 hours = 172800 seconds
                if self.verbose:
                    self.print_status(mcts)
                break

            # Success criterion reached (reward close enough to 1)
            if 1 - best_reward < mcts.succ_error_tol:
                if self.verbose:
                    self.print_status(mcts)
                break

    def predict(self, x: np.ndarray, vec_exp_str) -> np.ndarray:
        """Make predictions using the given expression"""
        try:
            f_pred = eval(f'lambda x: {vec_exp_str}', self.optimizer.context)
            return f_pred(x)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def print_status(self, mcts) -> None:
        """Format and print search status report with key metrics and exploration statistics"""
        try:
            best_expr, best_reward = mcts.exp_queue.best()
            f_pred = eval(f'lambda x: {best_expr}', self.optimizer.context)
            y_pred = f_pred(self.x_train)
            y_true = self.y_train
            elapsed_time = time.time() - self.start_time

            # Calculate evaluation metrics
            with np.errstate(divide='ignore', invalid='ignore'):
                ss_res = np.sum((y_pred - y_true) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
                nmse = np.mean((y_pred - y_true) ** 2) / np.var(y_true) if np.var(y_true) != 0 else float('nan')

            # Build status report
            report = [
                "\n\033[1;36m=== Symbolic Regression Progress Report ===\033[0m",
                f"\033[1mEvaluated Expressions:\033[0m {mcts.count_num}",
                f"\033[1mElapsed Time:\033[0m {elapsed_time:.1f}s",
                "\n\033[1mTop Performance Metrics:\033[0m",
                f"  \033[32mBest Expression:\033[0m {simplify_expression(best_expr, self.verbose)}",
                f"  \033[33mReward (↑):\033[0m {best_reward:.3e}",
                f" R² (↑): {r_squared:.4f} | NMSE (↓): {nmse:.3e}",
                "\n\033[1mExploration Profile:\033[0m",
                f"  Total Nodes: {mcts.total_nodes} | Active Branches: {len(mcts.root.children)}",
                f"  Exploration Rate: {self.exploration_rate:.2f} | Mutation Rate: {self.mutation_rate:.2f}"
            ]

            # Add branch statistics table
            if mcts.root.children:
                report.append("\n\033[1mBranch Statistics:\033[0m")
                header = f"{'Operator':<10} {'Visits':<8} {'Best Reward':<12} {'Dominance':<10}"
                report.append("\033[34m" + header + "\033[0m")
            
                for child in sorted(mcts.root.children, key=lambda x: x.visits, reverse=True):
                    reward = child.path_queue.best()[1]
                    dominance = child.visits / mcts.root.visits
                    report.append(
                        f"{child.move:<10} {child.visits:<8} {reward:.3e} {dominance:>7.1%}"
                    )

            print("\n".join(report))
            print("\033[1;35m" + "═" * 60 + "\033[0m")

        except Exception as e:
            print(f"\033[31mStatus Update Error: {str(e)}\033[0m")

    # Initialization helper methods
    def _validate_inputs(self, x_train, y_train, max_depth):
        """Validate input parameters"""
        if x_train.shape[1] != y_train.shape[0]:
            raise ValueError("Mismatched dimensions between x_train and y_train")
            
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

    def _init_operations(self, ops, var_count) -> List[str]:
        """Initialize operations list with variables"""
        default_ops = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
        ops = ops or default_ops
        return ops + [f'x{i}' for i in range(var_count)]

    def _init_arity_dict(self, arity_dict, var_count) -> Dict[str, int]:
        """Initialize arity dictionary with variables"""
        default_arity = {
            '+': 2, '-': 2, '*': 2, '/': 2,
            'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'tanh': 1,
            'C': 0, 'R': 0
        }
        arity_dict = arity_dict or default_arity.copy()
        arity_dict.update({f'x{i}': 0 for i in range(var_count)})
        return arity_dict

    def _init_context(self, context) -> Dict:
        """Initialize evaluation context"""
        default_context = {
            'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'log': np.log, 'tanh': np.tanh,
        }
        return {**default_context, **(context or {})}

    def _init_optimization_params(self, max_depth, K, c, gamma, gp_rate,
                                 mutation_rate, exploration_rate,
                                 max_single_arity_ops, max_constants, max_expressions):
        """Initialize optimization parameters with validation"""
        self.max_depth = max_depth
        self.K = K
        self.c = c
        self.gamma = gamma
        self.gp_rate = np.clip(gp_rate, 0.0, 1.0)
        self.mutation_rate = np.clip(mutation_rate, 0.0, 1.0)
        self.exploration_rate = np.clip(exploration_rate, 0.0, 1.0)
        self.max_single_arity_ops = max_single_arity_ops
        self.max_constants = max_constants
        self.max_expressions = max_expressions

    def _create_exp_tree(self):
        """Create expression tree instance"""
        T = ExpTree(
            max_depth=self.max_depth,
            max_single_arity_ops=self.max_single_arity_ops,
            max_constants=self.max_constants,
            arity_dict=self.arity_dict,
            ops=self.ops
        )
        return T

    def _create_mcts(self):
        """Dynamically create new MCTS instance"""
        return MCTS(
            optimizer=self.optimizer,
            gp_manager=GPManager(self.ops, self.arity_dict),
            gp_rate=self.gp_rate,
            mutation_rate=self.mutation_rate,
            exploration_rate=self.exploration_rate,
            K=self.K,
            c=self.c,
            gamma=self.gamma,
            verbose=self.verbose
        )