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
    Missing-date fill, zero-series removal, and representative-SKU consolidation.
standardization
    Weighted moving-average smoother and outlier clipping.
metrics
    sMAPE, RMSLE, and the combined cost function used by the optimiser.
optimizer
    Bayesian Optimisation (Optuna TPE) for ARIMA order search.
model
    SARIMA model fitting and forecast generation.
"""

from sarima_bayes.metrics import combined_metric, rmsle, smape
from sarima_bayes.model import forecast_arima, pred_arima
from sarima_bayes.optimizer import optimize_arima

__all__ = [
    "optimize_arima",
    "forecast_arima",
    "pred_arima",
    "combined_metric",
    "smape",
    "rmsle",
]
