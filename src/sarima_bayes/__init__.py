"""
sarima_bayes — deprecated compatibility shim for boa_forecaster.

This package has been renamed to ``boa_forecaster``.
All symbols are re-exported from there.

.. deprecated::
    ``sarima_bayes`` will be removed in v3.0.
    Update your imports::

        # old
        from sarima_bayes import optimize_arima
        # new
        from boa_forecaster import optimize_arima
"""

import warnings

warnings.warn(
    "The 'sarima_bayes' package has been renamed to 'boa_forecaster'. "
    "Please update your imports. 'sarima_bayes' will be removed in v3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from boa_forecaster import *  # noqa: E402, F401, F403
from boa_forecaster import (  # noqa: E402, F401 — explicit re-exports for tooling
    METRIC_REGISTRY,
    build_combined_metric,
    combined_metric,
    forecast_arima,
    mae,
    mape,
    optimize_arima,
    pred_arima,
    rmse,
    rmsle,
    run_benchmark_comparison,
    smape,
    summary_table,
    validate_by_group,
    walk_forward_validation,
)

__all__ = [
    "optimize_arima",
    "forecast_arima",
    "pred_arima",
    "combined_metric",
    "build_combined_metric",
    "METRIC_REGISTRY",
    "smape",
    "rmsle",
    "mae",
    "rmse",
    "mape",
    "walk_forward_validation",
    "validate_by_group",
    "run_benchmark_comparison",
    "summary_table",
]
