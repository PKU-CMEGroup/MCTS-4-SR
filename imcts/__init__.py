"""Python package interface for the iMCTS C++ extension."""

from __future__ import annotations

try:
    from .imcts_py import FitResult, Regressor, RegressorConfig
except ImportError:  # pragma: no cover - developer-tree fallback
    from imcts_py import FitResult, Regressor, RegressorConfig

from .pretty import expression_complexity, simplify_expression, simplify_with_complexity

__all__ = [
    "FitResult",
    "Regressor",
    "RegressorConfig",
    "expression_complexity",
    "simplify_expression",
    "simplify_with_complexity",
]
