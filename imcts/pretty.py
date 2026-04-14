"""Shared SymPy helpers for expression simplification and complexity."""

from __future__ import annotations

import re


def _simplify_sympy(expr: str, *, rationalize_constants: bool = False, precision: int = 4, threshold: float = 1e-4):
    if not expr:
        return None

    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    local_dict = {
        "sin": sp.sin,
        "cos": sp.cos,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "log": sp.log,
        "tanh": sp.tanh,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
    }
    for name in sorted(set(re.findall(r"\bx\d+\b", expr))):
        local_dict[name] = sp.Symbol(name)

    try:
        parsed = parse_expr(expr, evaluate=True, local_dict=local_dict)
        simplified = sp.expand(sp.simplify(parsed))
        if precision is not None and precision > 0:
            simplified = simplified.evalf(precision, chop=threshold)
        if rationalize_constants:
            simplified = sp.nsimplify(simplified)
        return simplified
    except Exception:
        return None


def simplify_expression(expr: str, *, rationalize_constants: bool = False, precision: int = 4, threshold: float = 1e-4) -> str:
    """Return a prettier symbolic expression if SymPy is installed."""
    try:
        import sympy as sp
    except ImportError:
        return expr

    simplified = _simplify_sympy(expr, rationalize_constants=rationalize_constants, precision=precision, threshold=threshold)
    if simplified is None:
        return expr
    return sp.sstr(simplified)


def expression_complexity(expr: str, *, rationalize_constants: bool = False, precision: int = 4, threshold: float = 1e-4) -> float:
    """Return the SymPy preorder-traversal node count for an expression."""
    try:
        import sympy as sp
    except ImportError:
        return float("nan")

    simplified = _simplify_sympy(expr, rationalize_constants=rationalize_constants, precision=precision, threshold=threshold)
    if simplified is None:
        return float("nan")
    return float(sum(1 for _ in sp.preorder_traversal(simplified)))


def simplify_with_complexity(expr: str, *, rationalize_constants: bool = False, precision: int = 4, threshold: float = 1e-4) -> tuple[str, float]:
    """Return `(simplified_expression, complexity)` using a shared SymPy pipeline."""
    try:
        import sympy as sp
    except ImportError:
        return expr, float("nan")

    simplified = _simplify_sympy(expr, rationalize_constants=rationalize_constants, precision=precision, threshold=threshold)
    if simplified is None:
        return expr, float("nan")
    return sp.sstr(simplified), float(sum(1 for _ in sp.preorder_traversal(simplified)))
