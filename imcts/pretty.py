"""Shared SymPy helpers for expression simplification and complexity."""

from __future__ import annotations

import re


DEFAULT_THRESHOLD = 1e-4
DEFAULT_DIGITS = 3
DEFAULT_TIMEOUT_SEC = 30.0


def _sympy_locals(expr_str: str):
    import sympy as sp

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
    for name in sorted(set(re.findall(r"\bx\d+\b", expr_str))):
        local_dict[name] = sp.Symbol(name)
    return local_dict


def _round_floats(expr, threshold: float, digits: int):
    import sympy as sp

    rounded = expr
    for atom in sp.preorder_traversal(expr):
        if isinstance(atom, sp.Float):
            if abs(atom) < threshold:
                rounded = rounded.subs(atom, sp.Integer(0))
            else:
                rounded = rounded.subs(atom, sp.Float(round(atom, digits), digits))
    return rounded


def _fallback_complexity(expr_str: str, threshold: float = DEFAULT_THRESHOLD, digits: int = DEFAULT_DIGITS) -> float:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        parsed = parse_expr(expr_str, evaluate=True, local_dict=_sympy_locals(expr_str))
        rounded = _round_floats(parsed, threshold=threshold, digits=digits)
        return float(sum(1 for _ in sp.preorder_traversal(rounded)))
    except Exception:
        return float("nan")


def simplify_with_complexity(
    expr: str,
    timeout_sec: float | None = DEFAULT_TIMEOUT_SEC,
    threshold: float = DEFAULT_THRESHOLD,
    digits: int = DEFAULT_DIGITS,
    **_ignored,
) -> tuple[str, float]:
    """Return `(simplified_expression, complexity)` using SRBench-style counting."""
    del timeout_sec
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        parsed = parse_expr(expr, evaluate=True, local_dict=_sympy_locals(expr))
        rounded = _round_floats(parsed, threshold=threshold, digits=digits)
        simplified = sp.simplify(rounded, ratio=1)
        complexity = float(sum(1 for _ in sp.preorder_traversal(simplified)))
        return sp.sstr(simplified), complexity
    except Exception:
        return expr, _fallback_complexity(expr, threshold=threshold, digits=digits)


def simplify_expression(expr: str, **kwargs) -> str:
    return simplify_with_complexity(
        expr,
        timeout_sec=kwargs.get("timeout_sec", DEFAULT_TIMEOUT_SEC),
        threshold=kwargs.get("threshold", DEFAULT_THRESHOLD),
        digits=kwargs.get("digits", DEFAULT_DIGITS),
    )[0]


def expression_complexity(expr: str, **kwargs) -> float:
    return simplify_with_complexity(
        expr,
        timeout_sec=kwargs.get("timeout_sec", DEFAULT_TIMEOUT_SEC),
        threshold=kwargs.get("threshold", DEFAULT_THRESHOLD),
        digits=kwargs.get("digits", DEFAULT_DIGITS),
    )[1]
