"""SARIMA model fitting and forecast generation.

This module wraps statsmodels' ``SARIMAX`` to provide a clean interface for:

- Fitting ``SARIMA(p, d, q)(P, D, Q, m)`` models on monthly sales series.
- Generating multi-step ahead point forecasts with confidence intervals.
- Formatting output DataFrames for downstream aggregation and export.

Seasonal components ``(P, D, Q, m)`` are optional — pass ``m=None`` to fit a
plain ``ARIMA(p, d, q)`` model, or provide ``m`` to enable full SARIMA.
"""

from __future__ import annotations

import logging

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


def pred_arima(
    bd: pd.DataFrame,
    col_x: str,
    col_y: str,
    order: tuple[int, int, int],
    s_order: tuple[int, int, int, int] | None = None,
    n_per: int = 12,
    alpha: float = 0.05,
    freq: str = "MS",
) -> tuple[pd.DataFrame, pd.DataFrame | None, tuple, tuple | None, pd.Series | None]:
    """Fit a SARIMA model and produce a multi-step forecast.

    This is the core forecasting engine.  The series index is set to the
    specified ``freq`` before fitting so that statsmodels generates properly
    date-indexed forecasts.

    ``enforce_stationarity=False`` and ``enforce_invertibility=False`` are
    deliberately set so that the Bayesian optimiser can probe the full
    ``(p, d, q)`` search space without raising hard exceptions on
    numerically challenging parameter combinations — the cost function
    naturally penalises poor fits.

    Args:
        bd: DataFrame containing at least the date column (``col_x``) and
            the target demand column (``col_y``).
        col_x: Name of the date column (e.g. ``"Date"``).
        col_y: Name of the target / demand column (e.g. ``"CS"``).
        order: ARIMA non-seasonal order ``(p, d, q)``.
        s_order: SARIMA seasonal order ``(P, D, Q, s)``.  Pass ``None``
            to fit a plain ARIMA model.  Defaults to ``None``.
        n_per: Number of forecast periods to generate ahead.
            Defaults to ``12``.
        alpha: Significance level for confidence intervals.
            Defaults to ``0.05`` (95 % CI).
        freq: Pandas DateOffset alias used to set the series frequency before
            fitting (e.g. ``"MS"``, ``"W"``, ``"D"``, ``"h"``).
            Must match the actual sampling rate of the input data.
            Defaults to ``"MS"`` (month start).

    Returns:
        5-tuple ``(forecast_df, conf_int, order, s_order, forecast_series)``
        where:

        - ``forecast_df``      – DataFrame indexed by forecast date with
          columns ``["Predictions", "Std Error", "Confidence Interval"]``.
        - ``conf_int``         – DataFrame with lower / upper CI columns.
        - ``order``            – The ``(p, d, q)`` tuple that was fitted.
        - ``s_order``          – The seasonal order that was passed in.
        - ``forecast_series``  – Raw ``pd.Series`` of point forecasts.

        On failure, returns ``(empty_df, None, order, s_order, None)``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> dates = pd.date_range("2022-01", periods=36, freq="MS")
        >>> df = pd.DataFrame({"Date": dates, "CS": np.random.rand(36) * 100})
        >>> fcast_df, _, _, _, _ = pred_arima(df, "Date", "CS", order=(1, 1, 1))
        >>> len(fcast_df)
        12
    """
    try:
        # Set temporal index so statsmodels produces date-indexed forecasts
        series = bd.set_index(col_x)[col_y].asfreq(freq)

        # NOTE: enforce_stationarity / enforce_invertibility = False allows the
        # Bayesian optimiser to test edge-case parameter combinations without
        # crashing.  The cost function penalises numerically unstable fits.
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model.fit(disp=False)  # disp=False suppresses convergence output

        forecast_result = model_fit.get_forecast(steps=n_per)
        forecast = forecast_result.predicted_mean
        stderr = forecast_result.se_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        forecast_df = pd.DataFrame(
            {
                "Predictions": forecast.values,
                "Std Error": stderr.values,
                "Confidence Interval": conf_int.values.tolist(),
            },
            index=forecast.index,
        )

        return forecast_df, conf_int, order, s_order, forecast

    except Exception as exc:
        logger.debug("pred_arima failed for order=%s: %s", order, exc)
        return pd.DataFrame(), None, order, s_order, None


def forecast_arima(
    bd: pd.DataFrame,
    col_x: str,
    col_y: str,
    p: int,
    d: int,
    q: int,
    n_per: int,
    country: str,
    sku: int,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    m: int | None = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """Fit SARIMA and return a tidy forecast DataFrame (no Forecast group).

    Convenience wrapper around ``pred_arima()`` that clips negative
    predictions to zero and formats the output for concatenation.

    Args:
        bd: Historical sales DataFrame for the given ``country`` / ``sku``.
        col_x: Date column name.
        col_y: Target demand column name.
        p: Autoregressive order.
        d: Integration order.
        q: Moving-average order.
        n_per: Number of forecast periods.
        country: Country label (for output tagging).
        sku: SKU identifier (for output tagging).
        P: Seasonal AR order.  Defaults to ``0``.
        D: Seasonal differencing order.  Defaults to ``0``.
        Q: Seasonal MA order.  Defaults to ``0``.
        m: Seasonal period.  Pass ``None`` to fit a non-seasonal ARIMA
            (equivalent to ``P=D=Q=0``).  Defaults to ``None``.
        freq: Pandas DateOffset alias for the series sampling frequency.
            Defaults to ``"MS"`` (month start).

    Returns:
        DataFrame with columns ``["Date", "Pred", "Country", "Sku"]``.
        Returns an empty DataFrame on failure.

    Example:
        >>> result = forecast_arima(df, "Date", "CS", 1, 1, 1, 12, "US", 1001)
        >>> result.columns.tolist()
        ['Date', 'Pred', 'Country', 'Sku']
    """
    try:
        s_order = (P, D, Q, m) if m is not None else None
        forecast_df, _, _, _, _ = pred_arima(
            bd=bd,
            col_x=col_x,
            col_y=col_y,
            order=(p, d, q),
            s_order=s_order,
            n_per=n_per,
            freq=freq,
        )

        if forecast_df.empty:
            return pd.DataFrame()

        # Demand cannot be negative — clip lower bound to zero
        preds = forecast_df["Predictions"].clip(lower=0)

        return pd.DataFrame(
            {
                "Date": preds.index,
                "Pred": preds.values,
                "Country": country,
                "Sku": sku,
            }
        )

    except Exception as exc:
        logger.error("forecast_arima error for %s-%s: %s", country, sku, exc)
        return pd.DataFrame()


def forecast_arima_with_group(
    bd: pd.DataFrame,
    col_x: str,
    col_y: str,
    p: int,
    d: int,
    q: int,
    n_per: int,
    country: str,
    sku: int,
    forecast_group: str,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    m: int | None = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """Fit SARIMA and return a tidy forecast DataFrame (with Forecast group).

    Identical to ``forecast_arima()`` but includes a ``"Forecast group"``
    column in the output.  Used when the dataset is segmented by product
    channel or distribution channel.

    Args:
        bd: Historical sales DataFrame for the given combination of
            ``country``, ``sku``, and ``forecast_group``.
        col_x: Date column name.
        col_y: Target demand column name.
        p: Autoregressive order.
        d: Integration order.
        q: Moving-average order.
        n_per: Number of forecast periods.
        country: Country label.
        sku: SKU identifier.
        forecast_group: Forecast group / channel identifier.
        P: Seasonal AR order.  Defaults to ``0``.
        D: Seasonal differencing order.  Defaults to ``0``.
        Q: Seasonal MA order.  Defaults to ``0``.
        m: Seasonal period.  Pass ``None`` to fit a non-seasonal ARIMA.
            Defaults to ``None``.
        freq: Pandas DateOffset alias for the series sampling frequency.
            Defaults to ``"MS"`` (month start).

    Returns:
        DataFrame with columns
        ``["Date", "Pred", "Country", "Sku", "Forecast group"]``.
        Returns an empty DataFrame on failure.

    Example:
        >>> result = forecast_arima_with_group(
        ...     df, "Date", "CS", 1, 1, 1, 12, "US", 1001, "Traditional"
        ... )
        >>> result.columns.tolist()
        ['Date', 'Pred', 'Country', 'Sku', 'Forecast group']
    """
    try:
        s_order = (P, D, Q, m) if m is not None else None
        forecast_df, _, _, _, _ = pred_arima(
            bd=bd,
            col_x=col_x,
            col_y=col_y,
            order=(p, d, q),
            s_order=s_order,
            n_per=n_per,
            freq=freq,
        )

        if forecast_df.empty:
            return pd.DataFrame()

        preds = forecast_df["Predictions"].clip(lower=0)

        return pd.DataFrame(
            {
                "Date": preds.index,
                "Pred": preds.values,
                "Country": country,
                "Sku": sku,
                "Forecast group": forecast_group,
            }
        )

    except Exception as exc:
        logger.error(
            "forecast_arima_with_group error for %s-%s-%s: %s",
            country,
            sku,
            forecast_group,
            exc,
        )
        return pd.DataFrame()
