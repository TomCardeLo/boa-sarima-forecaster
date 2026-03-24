"""
sarima_bayes: SARIMA + Bayesian Optimisation for monthly demand forecasting.

Public API
----------
The most commonly used symbols are re-exported here for convenience:

.. code-block:: python

    from sarima_bayes import optimize_arima, forecast_arima, combined_metric

Modules
-------
config
    Global constants (date formats, search-space bounds, optimisation budget).
data_loader
    Excel ingestion and minimal data cleaning.
preprocessor
    Missing-date fill and zero-series removal.
standardization
    Weighted moving-average smoother and outlier clipping.
metrics
    sMAPE, RMSLE, MAE, RMSE, MAPE, metric registry, and the combined cost
    function factory used by the optimiser.
optimizer
    Bayesian Optimisation (Optuna TPE) for ARIMA order search.
model
    SARIMA model fitting and forecast generation.
"""

from sarima_bayes.benchmarks import run_benchmark_comparison, summary_table
from sarima_bayes.metrics import (
    METRIC_REGISTRY,
    build_combined_metric,
    combined_metric,
    mae,
    mape,
    rmse,
    rmsle,
    smape,
)
from sarima_bayes.model import forecast_arima, pred_arima
from sarima_bayes.optimizer import optimize_arima
from sarima_bayes.validation import validate_by_group, walk_forward_validation

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
