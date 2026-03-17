"""Evaluation metrics for time-series forecasting models.

This module provides pure-NumPy implementations of:

- :func:`smape`  – Symmetric Mean Absolute Percentage Error
- :func:`rmsle`  – Root Mean Squared Logarithmic Error
- :func:`combined_metric` – Weighted hybrid cost function (0.7 sMAPE + 0.3 RMSLE)

The :func:`combined_metric` is the primary objective minimised by the Bayesian
optimiser in :mod:`sarima_bayes.optimizer`.  The combination balances
percentage-level accuracy (sMAPE) with log-scale stability (RMSLE), which
works well for demand series that contain intermittent zeros and occasional
large spikes.
"""

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
    errors in high-volume SKUs.  This function is the *objective minimised*
    by the Optuna study in :func:`~sarima_bayes.optimizer.optimize_arima`.

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
