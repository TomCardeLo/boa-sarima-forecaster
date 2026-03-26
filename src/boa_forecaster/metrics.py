"""Evaluation metrics for time-series forecasting models.

This module provides pure-NumPy implementations of:

- ``smape``                – Symmetric Mean Absolute Percentage Error
- ``rmsle``                – Root Mean Squared Logarithmic Error
- ``mae``                  – Mean Absolute Error
- ``rmse``                 – Root Mean Squared Error
- ``mape``                 – Mean Absolute Percentage Error
- ``combined_metric``      – Weighted hybrid cost function (0.7 sMAPE + 0.3 RMSLE, default)
- ``METRIC_REGISTRY``      – Dict mapping metric names to callables
- ``build_combined_metric`` – Factory that builds a weighted objective from any metric mix

``combined_metric`` (and the default ``METRIC_REGISTRY`` profile) is the
primary objective minimised by the Bayesian optimiser.  The metric
composition is fully configurable via ``build_combined_metric``, making the
library applicable beyond demand forecasting — e.g. revenue or price series
can use ``0.6 × MAE + 0.4 × RMSE``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE is symmetric around zero and bounded in ``[0, 200]``, avoiding
    the unbounded behaviour of classical MAPE for small true values.  A
    small ``epsilon`` prevents division-by-zero when both ``y_true`` and
    ``y_pred`` are simultaneously zero (common in intermittent demand).

    Formula::

        sMAPE = 100 * mean( |y_true - y_pred| / ((|y_true| + |y_pred|)/2 + ε) )

    Args:
        y_true: Array of observed values. Shape ``(n,)``.
        y_pred: Array of predicted values. Shape ``(n,)``.

    Returns:
        sMAPE score as a percentage (lower is better).

    Example:
        >>> import numpy as np
        >>> smape(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        5.238...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Epsilon guards against simultaneous zeros in intermittent demand
    epsilon = 1e-10
    diff = np.abs(y_true - y_pred) / (denominator + epsilon)
    return float(100.0 * np.mean(diff))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Root Mean Squared Logarithmic Error (RMSLE).

    RMSLE penalises under-predictions more than over-predictions and is
    less sensitive to very large absolute errors than RMSE, making it
    suitable for demand series spanning several orders of magnitude.
    ``log1p`` is used for numerical safety when values are near zero.

    Formula::

        RMSLE = sqrt( mean( (log(1 + y_true) - log(1 + y_pred))^2 ) )

    Args:
        y_true: Array of observed values. Clipped to ``[0, ∞)`` internally.
        y_pred: Array of predicted values. Clipped to ``[0, ∞)`` internally.

    Returns:
        RMSLE score (non-negative float, lower is better).

    Example:
        >>> import numpy as np
        >>> rmsle(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        0.052...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Clip to non-negative so log1p is always real-valued
    log_true = np.log1p(np.maximum(y_true, 0.0))
    log_pred = np.log1p(np.maximum(y_pred, 0.0))
    return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))


def combined_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w_smape: float = 0.7,
    w_rmsle: float = 0.3,
) -> float:
    """Compute the weighted hybrid cost function used during optimisation.

    Combines percentage accuracy (sMAPE) with log-scale stability (RMSLE)::

        combined = w_smape * sMAPE(y_true, y_pred) + w_rmsle * RMSLE(y_true, y_pred)

    Default weights (0.7 / 0.3) place more emphasis on relative percentage
    accuracy while giving RMSLE enough influence to penalise large absolute
    errors in high-volume SKUs.  This function is the objective minimised
    by the Optuna study in ``optimize_arima()``.

    Args:
        y_true: Array of observed values. Shape ``(n,)``.
        y_pred: Array of predicted values. Shape ``(n,)``.
        w_smape: Weight assigned to sMAPE. Defaults to ``0.7``.
        w_rmsle: Weight assigned to RMSLE. Defaults to ``0.3``.

    Returns:
        Scalar cost value (lower is better).

    Example:
        >>> import numpy as np
        >>> combined_metric(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        3.683...
    """
    val_smape = smape(y_true, y_pred)
    val_rmsle = rmsle(y_true, y_pred)
    return (w_smape * val_smape) + (w_rmsle * val_rmsle)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Mean Absolute Error (MAE).

    Formula::

        MAE = mean( |y_true - y_pred| )

    Args:
        y_true: Array of observed values. Shape ``(n,)``.
        y_pred: Array of predicted values. Shape ``(n,)``.

    Returns:
        MAE score (non-negative float, lower is better).

    Example:
        >>> import numpy as np
        >>> mae(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        10.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Root Mean Squared Error (RMSE).

    Formula::

        RMSE = sqrt( mean( (y_true - y_pred)^2 ) )

    Args:
        y_true: Array of observed values. Shape ``(n,)``.
        y_pred: Array of predicted values. Shape ``(n,)``.

    Returns:
        RMSE score (non-negative float, lower is better).

    Example:
        >>> import numpy as np
        >>> rmse(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        10.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Mean Absolute Percentage Error (MAPE).

    A small ``epsilon`` prevents division-by-zero when ``y_true`` is zero.
    Note that MAPE is asymmetric and unbounded for small true values — prefer
    sMAPE for series with intermittent zeros.

    Formula::

        MAPE = 100 * mean( |y_true - y_pred| / (|y_true| + ε) )

    Args:
        y_true: Array of observed values. Shape ``(n,)``.
        y_pred: Array of predicted values. Shape ``(n,)``.

    Returns:
        MAPE score as a percentage (lower is better).

    Example:
        >>> import numpy as np
        >>> mape(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        7.5
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    epsilon = 1e-10
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)))


# Registry mapping metric names (as used in config.yaml) to their callables.
# Extend this dict to expose additional metrics to the configuration layer.
METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "smape": smape,
    "rmsle": rmsle,
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
}


def build_combined_metric(
    components: list[dict],
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build a weighted metric callable from a list of ``{metric, weight}`` dicts.

    This factory is the extension point for customising the optimisation
    objective.  Any combination of registered metrics and weights is
    supported, making the library applicable to contexts beyond demand
    forecasting (e.g. revenue, price, or count series).

    Args:
        components: List of dicts, each with:
            - ``"metric"`` (str): name of a metric in ``METRIC_REGISTRY``.
            - ``"weight"`` (float): scalar weight for that metric.
            Weights do not need to sum to 1, but it is recommended for
            interpretability.

    Returns:
        A callable ``fn(y_true, y_pred) -> float`` that returns the
        weighted sum of the specified metrics.

    Raises:
        ValueError: If any metric name is not present in ``METRIC_REGISTRY``.

    Example:
        >>> import numpy as np
        >>> fn = build_combined_metric([
        ...     {"metric": "mae",  "weight": 0.6},
        ...     {"metric": "rmse", "weight": 0.4},
        ... ])
        >>> fn(np.array([100.0]), np.array([110.0]))
        10.0
    """
    for item in components:
        name = item["metric"]
        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Available metrics: {sorted(METRIC_REGISTRY)}"
            )

    pairs = [
        (METRIC_REGISTRY[item["metric"]], float(item["weight"])) for item in components
    ]

    def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return sum(w * fn(y_true, y_pred) for fn, w in pairs)

    return _metric
