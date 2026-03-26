"""SARIMA model plugin for the boa-forecaster TPE engine.

Implements ``ModelSpec`` for ``SARIMA(p,d,q)(P,D,Q,m)`` via statsmodels
``SARIMAX``.  Also provides the ``pred_arima`` and ``forecast_arima`` free
functions ported from ``sarima_bayes.model`` so that v2 users can import them
from ``boa_forecaster.models.sarima``.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from boa_forecaster.config import (
    DEFAULT_D_RANGE,
    DEFAULT_D_SEASONAL_RANGE,
    DEFAULT_P_RANGE,
    DEFAULT_P_SEASONAL_RANGE,
    DEFAULT_Q_RANGE,
    DEFAULT_Q_SEASONAL_RANGE,
    DEFAULT_SEASONAL_PERIOD,
    OPTIMIZER_PENALTY,
)
from boa_forecaster.models.base import IntParam, suggest_from_space

logger = logging.getLogger(__name__)


class SARIMASpec:
    """``ModelSpec`` implementation for SARIMA(p,d,q)(P,D,Q,m).

    Args:
        p_range: Inclusive ``(min, max)`` for the AR order *p*.
        d_range: Inclusive ``(min, max)`` for the differencing order *d*.
        q_range: Inclusive ``(min, max)`` for the MA order *q*.
        P_range: Inclusive ``(min, max)`` for the seasonal AR order *P*.
        D_range: Inclusive ``(min, max)`` for the seasonal differencing *D*.
        Q_range: Inclusive ``(min, max)`` for the seasonal MA order *Q*.
        m: Fixed seasonal period (not optimised).  Default is 12 (monthly).
    """

    name = "sarima"
    needs_features = False

    def __init__(
        self,
        p_range: tuple[int, int] = DEFAULT_P_RANGE,
        d_range: tuple[int, int] = DEFAULT_D_RANGE,
        q_range: tuple[int, int] = DEFAULT_Q_RANGE,
        P_range: tuple[int, int] = DEFAULT_P_SEASONAL_RANGE,
        D_range: tuple[int, int] = DEFAULT_D_SEASONAL_RANGE,
        Q_range: tuple[int, int] = DEFAULT_Q_SEASONAL_RANGE,
        m: int = DEFAULT_SEASONAL_PERIOD,
    ) -> None:
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.P_range = P_range
        self.D_range = D_range
        self.Q_range = Q_range
        self.m = m

    # ── ModelSpec properties ──────────────────────────────────────────────────

    @property
    def search_space(self) -> dict:
        """One ``IntParam`` per SARIMA order (p, d, q, P, D, Q)."""
        return {
            "p": IntParam(self.p_range[0], self.p_range[1]),
            "d": IntParam(self.d_range[0], self.d_range[1]),
            "q": IntParam(self.q_range[0], self.q_range[1]),
            "P": IntParam(self.P_range[0], self.P_range[1]),
            "D": IntParam(self.D_range[0], self.D_range[1]),
            "Q": IntParam(self.Q_range[0], self.Q_range[1]),
        }

    @property
    def warm_starts(self) -> list[dict]:
        """Two sensible starting configurations before TPE takes over.

        - ``SARIMA(1,1,1)(1,1,1,m)``: trending seasonal baseline.
        - ``ARIMA(1,0,0)``: simple AR(1) for stationary series.
        """
        return [
            {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1},
            {"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0},
        ]

    # ── ModelSpec methods ─────────────────────────────────────────────────────

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Sample (p, d, q, P, D, Q) from the search space via Optuna."""
        return suggest_from_space(trial, self.search_space)

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config=None,
    ) -> float:
        """Fit SARIMAX in-sample and return the metric score.

        Applies complexity constraints before fitting:

        - ``(p + q) > 4`` → return ``OPTIMIZER_PENALTY`` immediately.
        - ``(P + Q) > 3`` → return ``OPTIMIZER_PENALTY`` immediately.

        Any SARIMAX exception also returns ``OPTIMIZER_PENALTY`` so the
        optimiser can continue without crashing.

        Args:
            series: Training series (``pd.Series`` or array-like).
            params: Dict with keys ``p, d, q, P, D, Q``.
            metric_fn: Callable ``(y_true, y_pred) -> float``.
            feature_config: Ignored (SARIMA does not use features).

        Returns:
            Scalar metric score, or ``OPTIMIZER_PENALTY`` on failure.
        """
        p, d, q = params["p"], params["d"], params["q"]
        P, D, Q = params["P"], params["D"], params["Q"]

        if (p + q) > 4 or (P + Q) > 3:
            return OPTIMIZER_PENALTY

        data = series.values if isinstance(series, pd.Series) else np.asarray(series)
        try:
            model = SARIMAX(
                data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, self.m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            pred = fit.predict(start=0, end=len(data) - 1)
            return float(metric_fn(data, pred))
        except Exception:
            return OPTIMIZER_PENALTY

    def build_forecaster(self, params: dict, feature_config=None):
        """Return a closure ``forecaster(train: pd.Series) -> pd.Series``.

        The returned callable fits SARIMAX on *train* with the given *params*
        and produces a 12-step-ahead forecast as a ``pd.Series`` with a
        ``DatetimeIndex`` (when *train* has one).

        Args:
            params: Dict with keys ``p, d, q`` and optionally ``P, D, Q, m``.
            feature_config: Ignored.

        Returns:
            Callable ``(train: pd.Series) -> pd.Series``.
        """
        p, d, q = params["p"], params["d"], params["q"]
        P = params.get("P", 0)
        D = params.get("D", 0)
        Q = params.get("Q", 0)
        m = params.get("m", self.m)

        def forecaster(train: pd.Series) -> pd.Series:
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            return fit.get_forecast(steps=12).predicted_mean

        return forecaster


# ── Free functions (backwards compatibility) ──────────────────────────────────
# Ported from sarima_bayes.model so that
#   from boa_forecaster.models.sarima import pred_arima, forecast_arima
# works in v2.0.


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

    Port of ``sarima_bayes.model.pred_arima``.  See that module for the full
    docstring.

    Returns:
        5-tuple ``(forecast_df, conf_int, order, s_order, forecast_series)``.
        On failure returns ``(empty_df, None, order, s_order, None)``.
    """
    try:
        series = bd.set_index(col_x)[col_y].asfreq(freq)
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        result = fit.get_forecast(steps=n_per)
        forecast = result.predicted_mean
        stderr = result.se_mean
        conf_int = result.conf_int(alpha=alpha)
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
    country: str = "_",
    sku: int = 1,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    m: int | None = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """Fit SARIMA and return a tidy forecast DataFrame.

    Port of ``sarima_bayes.model.forecast_arima``.  See that module for the
    full docstring.

    Returns:
        DataFrame with columns ``["Date", "Pred", "Country", "Sku"]``.
        Returns an empty DataFrame on failure.
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
            }
        )
    except Exception as exc:
        logger.error("forecast_arima error for %s-%s: %s", country, sku, exc)
        return pd.DataFrame()
