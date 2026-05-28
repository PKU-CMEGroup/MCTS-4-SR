"""Python package interface for the iMCTS C++ extension."""

from __future__ import annotations

try:
    from imcts_py import (
        FitResult,
        Regressor,
        RegressorConfig,
        openmp_info,
        reset_timing_stats,
        timing_stats,
    )
except ImportError:  # pragma: no cover - installed-wheel fallback
    from .imcts_py import (
        FitResult,
        Regressor,
        RegressorConfig,
        openmp_info,
        reset_timing_stats,
        timing_stats,
    )

from .pretty import expression_complexity, simplify_expression, simplify_with_complexity

__all__ = [
    "FitResult",
    "Regressor",
    "RegressorConfig",
    "openmp_info",
    "reset_timing_stats",
    "timing_stats",
    "expression_complexity",
    "simplify_expression",
    "simplify_with_complexity",
]
