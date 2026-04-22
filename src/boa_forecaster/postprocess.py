"""Post-training seasonal bias correction for boa-forecaster.

The module provides a two-step multiplicative correction that can be
applied after any model forecasts:

1. **Compute** per-period factors from held-out CV residuals using a median
   of ``y_true / y_pred`` per calendar period.  Median is used (not mean)
   for robustness against outliers, matching the ``sesgo_mensual_para_ajuste.csv``
   pattern validated in the CAR PM2.5 production pipeline (see feedback_aire §5).

2. **Apply** the resulting factor array multiplicatively to a forecast,
   aligned by calendar month (when a DatetimeIndex is available) or by
   position modulo ``len(bias)``.

The correction is generalizable beyond monthly data — the ``periods``
parameter defaults to 12 but accepts any positive integer (e.g. 7 for
weekly seasonality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_seasonal_bias(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    periods: int = 12,
    start_period: int = 1,
    clip_range: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """Compute per-calendar-period median bias factors from residuals.

    For each period bucket, the factor is the median of ``y_true / y_pred``
    across all observations that fall in that bucket.  Pairs are dropped when
    either value is NaN or when ``y_pred == 0`` (divide-by-zero guard).
    Empty buckets default to 1.0.  All factors are clipped to ``clip_range``
    to avoid blow-ups.

    When ``y_true`` is a ``pd.Series`` with a ``DatetimeIndex`` and
    ``periods == 12``, the bucket for each observation is determined by
    ``(y_true.index.month - 1) % 12`` — i.e. ``bias[0]`` is always January,
    ``bias[6]`` is always July, regardless of ``start_period``.
    ``start_period`` is honored only for position-based (ndarray) alignment.
    Otherwise observations are assumed to be ordered sequentially starting at
    ``start_period`` and the bucket is ``(i % periods)``.

    Args:
        y_true: Actual observed values.
        y_pred: Model-predicted values aligned with ``y_true``.
        periods: Number of seasonal periods (default 12 for monthly data).
        start_period: Period index of the first observation; used for
            position-based alignment only.  For monthly data ``start_period=1``
            means the first observation is January.  Ignored when ``y_true``
            has a ``DatetimeIndex`` and ``periods == 12`` (calendar-month
            alignment takes precedence).
        clip_range: ``(lower, upper)`` hard bounds applied to every factor
            after computing the per-bucket median.

    Notes:
        Intended for non-negative target series.  Negative values in
        ``y_true`` or ``y_pred`` produce arithmetically valid but semantically
        misleading ratios; the clipping bounds prevent blow-ups but do not
        recover the correct sign.

    Returns:
        ``np.ndarray`` of shape ``(periods,)`` where element ``k`` is the
        multiplicative correction factor for period bucket ``k``.  When
        inputs have a ``DatetimeIndex`` and ``periods=12``, ``bias[k]`` is
        the factor for calendar month ``k+1`` (``bias[0]`` = January,
        ``bias[6]`` = July).  ``start_period`` is honored only for
        position-based (ndarray) alignment.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        >>> perfect = pd.Series(np.ones(24) * 100.0, index=idx)
        >>> factors = compute_seasonal_bias(perfect, perfect)
        >>> np.allclose(factors, 1.0)
        True
    """
    # Normalise inputs to arrays, preserving the original Series for index check.
    true_series = y_true if isinstance(y_true, pd.Series) else None
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    n = len(yt)
    use_datetime = (
        true_series is not None
        and isinstance(true_series.index, pd.DatetimeIndex)
        and periods == 12
    )

    if use_datetime:
        # Bucket by calendar month: month ∈ [1..12] → index ∈ [0..11]
        # bias[0]=January, bias[6]=July regardless of start_period.
        # start_period is used ONLY for ndarray (position-based) alignment.
        buckets = (true_series.index.month - 1) % periods
    else:
        # Position-based: observation i starts at bucket (start_period - 1)
        buckets = (np.arange(n) + (start_period - 1)) % periods

    factors = np.ones(periods, dtype=float)
    for k in range(periods):
        mask = buckets == k
        yt_k = yt[mask]
        yp_k = yp[mask]

        # Drop NaN pairs
        valid = ~(np.isnan(yt_k) | np.isnan(yp_k))
        yt_k = yt_k[valid]
        yp_k = yp_k[valid]

        # Drop zero-prediction pairs (divide-by-zero guard)
        nonzero = yp_k != 0.0
        yt_k = yt_k[nonzero]
        yp_k = yp_k[nonzero]

        if len(yt_k) == 0:
            factors[k] = 1.0
        else:
            factors[k] = float(np.median(yt_k / yp_k))

    return np.clip(factors, clip_range[0], clip_range[1])


def apply_seasonal_bias(
    forecast: np.ndarray | pd.Series,
    bias: np.ndarray,
    start_period: int = 1,
) -> np.ndarray | pd.Series:
    """Multiplicatively apply per-period bias factors to a forecast.

    Alignment strategy:

    - If ``forecast`` is a ``pd.Series`` with a ``DatetimeIndex`` and
      ``len(bias) == 12``, use ``forecast.index.month`` to look up the factor
      for each observation.
    - Otherwise align by position modulo ``len(bias)``, starting at
      ``start_period``.

    Args:
        forecast: Point-forecast values to correct.
        bias: Per-period multiplicative factors from ``compute_seasonal_bias``.
        start_period: Period index of the first forecast observation; used for
            position-based alignment only.  Ignored when DatetimeIndex alignment
            is active.  When ``bias`` has 12 elements and ``forecast`` carries a
            ``DatetimeIndex``, ``bias[k]`` is looked up by calendar month
            (``bias[0]`` = January, ``bias[6]`` = July) — matching the
            convention established by ``compute_seasonal_bias``.

    Returns:
        Corrected forecast in the same container type as ``forecast`` (i.e.
        ``pd.Series`` in → ``pd.Series`` out, preserving the original index).

    Examples:
        >>> import numpy as np
        >>> forecast = np.array([100.0] * 12)
        >>> bias = np.ones(12)
        >>> bias[0] = 1.5  # January factor
        >>> result = apply_seasonal_bias(forecast, bias, start_period=1)
        >>> result[0]
        150.0
    """
    is_series = isinstance(forecast, pd.Series)
    periods = len(bias)

    use_datetime = (
        is_series and isinstance(forecast.index, pd.DatetimeIndex) and periods == 12
    )

    if use_datetime:
        month_buckets = (forecast.index.month - 1) % 12
        factors = bias[month_buckets]
    else:
        n = len(forecast)
        positions = (np.arange(n) + (start_period - 1)) % periods
        factors = bias[positions]

    if is_series:
        return pd.Series(
            np.asarray(forecast, dtype=float) * factors,
            index=forecast.index,
            name=forecast.name,
        )

    return np.asarray(forecast, dtype=float) * factors
