"""Probabilistic forecasting metrics: pinball loss and interval coverage.

This module is deliberately kept import-free from ``boa_forecaster.metrics``
to avoid circular imports.  The ``metrics.py`` module imports from here
(one-way dependency).
"""

from __future__ import annotations

import numpy as np


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Pinball (quantile) loss for a single quantile.

    Formula::

        mean( max(q * (y_true - y_pred), (q - 1) * (y_true - y_pred)) )

    At q=0.5 this equals half of MAE.  At q<0.5, under-prediction
    (y_true > y_pred) is penalised less than over-prediction and vice versa.

    Args:
        y_true: Array-like of observed values.
        y_pred: Array-like of predicted values.
        quantile: Target quantile in the open interval ``(0, 1)``.

    Returns:
        Pinball loss as a plain Python ``float`` (lower is better).

    Raises:
        ValueError: If ``quantile`` is not in ``(0, 1)``.

    Example:
        >>> import numpy as np
        >>> pinball_loss(np.array([10.0, 20.0]), np.array([10.0, 20.0]), quantile=0.5)
        0.0
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(
            f"quantile must be in the open interval (0, 1); got {quantile!r}"
        )
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1.0) * errors)
    return float(np.mean(loss))


def interval_coverage(y_true, lower, upper) -> float:
    """Empirical coverage: fraction of y_true values in [lower, upper].

    Both boundaries are inclusive.

    Args:
        y_true: Array-like of observed values.
        lower: Array-like of lower-bound forecasts (same shape as y_true).
        upper: Array-like of upper-bound forecasts (same shape as y_true).

    Returns:
        Coverage fraction in ``[0.0, 1.0]`` (higher is better for a target
        nominal coverage).

    Raises:
        ValueError: If ``y_true``, ``lower``, and ``upper`` do not share
            the same shape.

    Example:
        >>> interval_coverage([5.0], [4.0], [6.0])
        1.0
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if y_true.shape != lower.shape or y_true.shape != upper.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, lower={lower.shape}, "
            f"upper={upper.shape}. All arrays must have the same shape."
        )
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
