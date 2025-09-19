from iMCTS import Regressor
from sympy import sympify, expand, expand_log
import numpy as np

X = np.random.uniform(0, 2, (1, 20))

def f(x):
    return np.log(x[0] + 1) + np.log(x[0]**2 + 1)

Y = f(X)
model = Regressor(
    x_train=X,
    y_train=Y,
    ops=['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log'], # if you need constants, add 'R' to the list
    verbose=True,
)

sym_exp, vec_exp, evaluations, path = model.fit()
print("Sympy Expression: ", sym_exp)