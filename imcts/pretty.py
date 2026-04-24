"""Shared SymPy helpers for expression simplification and complexity."""

from __future__ import annotations

import re


def _is_small_numeric_value(value, *, threshold: float) -> bool:
    if threshold <= 0 or getattr(value, "free_symbols", None):
        return False

    try:
        return abs(complex(value.evalf())) < threshold
    except (TypeError, ValueError):
        return False


def _zero_small_coefficients(expr, *, threshold: float):
    import sympy as sp

    if threshold <= 0:
        return expr

    def prune(node):
        if node.is_Atom:
            return node

        updated_args = tuple(prune(arg) for arg in node.args)
        if updated_args != node.args:
            node = node.func(*updated_args, evaluate=True)

        if isinstance(node, sp.Add):
            kept_terms = []
            for term in node.args:
                coeff, _ = term.as_coeff_Mul()
                if _is_small_numeric_value(coeff, threshold=threshold):
                    continue
                kept_terms.append(term)
            if not kept_terms:
                return sp.Integer(0)
            return sp.Add(*kept_terms, evaluate=True)

        if isinstance(node, sp.Mul):
            coeff, _ = node.as_coeff_Mul()
            if _is_small_numeric_value(coeff, threshold=threshold):
                return sp.Integer(0)

        return node

    return prune(expr)


def _simplify_sympy(
    expr: str,
    *,
    rationalize_constants: bool = False,
    precision: int = 4,
    threshold: float = 1e-4,
    coefficient_threshold: float | None = None,
):
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
        if coefficient_threshold is not None:
            parsed = _zero_small_coefficients(parsed, threshold=coefficient_threshold)
        simplified = sp.factor_terms(sp.simplify(parsed))
        if coefficient_threshold is not None:
            simplified = _zero_small_coefficients(simplified, threshold=coefficient_threshold)
        if precision is not None and precision > 0:
            simplified = simplified.evalf(precision, chop=threshold)
        if rationalize_constants:
            simplified = sp.nsimplify(simplified)
        return simplified
    except Exception:
        return None


def simplify_expression(
    expr: str,
    *,
    rationalize_constants: bool = False,
    precision: int = 4,
    threshold: float = 1e-4,
    coefficient_threshold: float | None = None,
) -> str:
    """Return a prettier symbolic expression if SymPy is installed."""
    try:
        import sympy as sp
    except ImportError:
        return expr

    simplified = _simplify_sympy(
        expr,
        rationalize_constants=rationalize_constants,
        precision=precision,
        threshold=threshold,
        coefficient_threshold=coefficient_threshold,
    )
    if simplified is None:
        return expr
    return sp.sstr(simplified)


def expression_complexity(
    expr: str,
    *,
    rationalize_constants: bool = False,
    precision: int = 4,
    threshold: float = 1e-4,
    coefficient_threshold: float | None = None,
) -> float:
    """Return the SymPy preorder-traversal node count for an expression."""
    try:
        import sympy as sp
    except ImportError:
        return float("nan")

    simplified = _simplify_sympy(
        expr,
        rationalize_constants=rationalize_constants,
        precision=precision,
        threshold=threshold,
        coefficient_threshold=coefficient_threshold,
    )
    if simplified is None:
        return float("nan")
    return float(sum(1 for _ in sp.preorder_traversal(simplified)))


def simplify_with_complexity(
    expr: str,
    *,
    rationalize_constants: bool = False,
    precision: int = 4,
    threshold: float = 1e-4,
    coefficient_threshold: float | None = None,
) -> tuple[str, float]:
    """Return `(simplified_expression, complexity)` using a shared SymPy pipeline."""
    try:
        import sympy as sp
    except ImportError:
        return expr, float("nan")

    simplified = _simplify_sympy(
        expr,
        rationalize_constants=rationalize_constants,
        precision=precision,
        threshold=threshold,
        coefficient_threshold=coefficient_threshold,
    )
    if simplified is None:
        return expr, float("nan")
    return sp.sstr(simplified), float(sum(1 for _ in sp.preorder_traversal(simplified)))
