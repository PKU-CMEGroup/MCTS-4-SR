import numpy as np
from sklearn.metrics import mean_squared_error
import sympy as sp
from typing import Dict, Optional

"""
For all of these, higher is better
"""
def accuracy(est, X, y, vec_exp_str):
    pred = est.predict(X, vec_exp_str)
    pred = np.array(pred).flatten()

    mse = mean_squared_error(y, pred)
    y_var = np.var(y)
    if y_var == 0.0:
        y_var = 1e-9

    r2 = 1 - mse / y_var
    # r2 = np.round(r2, 3)
    return r2

"""
Utilities for symbolic expression handling
"""

def round_floats(ex1):
    ex2 = ex1
    for a in sp.preorder_traversal(ex1):
        if isinstance(a, sp.Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, sp.Integer(0))
            else:
                ex2 = ex2.subs(a, round(a, 3))
    return ex2


def get_symbolic_model(pred_model, local_dict):
    """Parse string to Sympy expression, round floats, and simplify"""
    sp_model = sp.parse_expr(pred_model, local_dict=local_dict)
    sp_model = round_floats(sp_model)

    try:
        sp_model = sp.simplify(sp_model)
    except Exception as e:
        print('Warning: simplify failed. Msg:', e)
    return sp_model


def simplicity(pred_model, feature_names):
    """Compute model simplicity by counting expression components"""
    local_dict = {f: sp.Symbol(f) for f in feature_names}
    sp_model = get_symbolic_model(pred_model, local_dict)
    # count number of nodes in the expression tree
    num_components = 0
    for _ in sp.preorder_traversal(sp_model):
        num_components += 1
    return num_components, sp_model


def metrics(
    model,
    exp_str: str,
    vec_exp_str: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_train: Optional[np.ndarray],
    y_train: Optional[np.ndarray]
) -> Dict[str, float]:
    """Evaluate model performance on test and train sets"""
    metrics_dict = {}

    # test set evaluation
    metrics_dict["test_accuracy"] = accuracy(model, x_test, y_test, vec_exp_str)
    # train set evaluation
    metrics_dict["train_accuracy"] = accuracy(model, x_train, y_train, vec_exp_str)
    # model simplicity (data-independent)
    feature_names = [f"x{i}" for i in range(x_test.shape[1])]
    metrics_dict["simplicity"], sp_model = simplicity(exp_str, feature_names)

    return metrics_dict, sp_model