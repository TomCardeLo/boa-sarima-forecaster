"""Feature engineering for tabular ML models (Random Forest, XGBoost, LightGBM).

``FeatureEngineer`` transforms a ``pd.Series`` into a ``pd.DataFrame`` of
lagged, rolling, calendar, and trend features that ML models can consume.

Temporal integrity guarantee
-----------------------------
Every feature at position ``t`` depends **only** on observations at ``t-1``
and earlier.  This is enforced by:

- Lag features: ``series.shift(n)``  →  ``lag_n[t] = y[t-n]``
- Rolling features: ``series.shift(1).rolling(w)``  →  uses ``y[t-1..t-w]``
- Expanding features: ``series.shift(1).expanding()``  →  uses ``y[0..t-1]``
- Calendar and trend features: deterministic from the DatetimeIndex, never
  derived from future values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for ``FeatureEngineer``.

    Attributes
    ----------
    lag_periods
        List of lag offsets to include (e.g. ``[1, 2, 3, 6, 12]``).
    rolling_windows
        List of window sizes for rolling mean/std (e.g. ``[3, 6, 12]``).
    include_calendar
        If ``True`` and the series has a ``DatetimeIndex``, add cyclic month
        and quarter encodings (sin/cos).
    include_trend
        If ``True`` and the series has a ``DatetimeIndex``, add a normalised
        trend index ``[0, 1]`` over the training window and a year normaliser.
    include_expanding
        If ``True``, add expanding mean and std (causal, using shift(1)).
    target_col
        Name assigned to the target ``pd.Series`` returned by ``fit_transform``.
    """

    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 6, 12])
    include_calendar: bool = True
    include_trend: bool = True
    include_expanding: bool = False
    target_col: str = "y"


class FeatureEngineer:
    """Transforms ``pd.Series`` → ``(X: pd.DataFrame, y: pd.Series)``.

    Usage in cross-validation
    -------------------------
    ::

        fe = FeatureEngineer(config)
        X_train, y_train = fe.fit_transform(train_series)
        X_test = fe.transform(test_series)  # uses train normalisation stats

    ``fit_transform`` stores the training-window size and year bounds so that
    ``transform`` can extrapolate trend/year features consistently.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._n_train: int | None = None
        self._min_year: int | None = None
        self._year_range: int | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on *series* and return ``(X, y)`` with NaN rows removed.

        Args:
            series: Time series with at least ``max(lag_periods) + 1`` points.

        Returns:
            Tuple ``(X, y)`` where ``X`` is the feature DataFrame and ``y``
            is the aligned target series (both start at index ``max_lag``).

        Raises:
            ValueError: If the series is too short to produce any non-NaN rows.
        """
        max_lag = max(self.config.lag_periods)
        if len(series) <= max_lag:
            raise ValueError(
                f"Series too short: {len(series)} points but "
                f"max lag = {max_lag}. Need at least {max_lag + 1} points."
            )

        # Store training stats for transform()
        self._n_train = len(series)
        if isinstance(series.index, pd.DatetimeIndex):
            self._min_year = int(series.index.year.min())
            self._year_range = max(1, int(series.index.year.max()) - self._min_year)

        features = self._build_features(series, fit=True)
        y = series.iloc[max_lag:].copy()
        y.name = self.config.target_col
        return features, y

    def transform(self, series: pd.Series) -> pd.DataFrame:
        """Apply the same feature pipeline using stats stored by ``fit_transform``.

        Used for recursive forecasting: the extended series (train + predictions
        appended one step at a time) is transformed to produce the feature row
        for the next prediction step.

        Args:
            series: Series to transform (may be longer than the training set).

        Returns:
            Feature DataFrame (NaN rows for the first ``max_lag`` positions
            are dropped, matching the behaviour of ``fit_transform``).

        Raises:
            RuntimeError: If ``fit_transform`` has not been called yet.
        """
        if self._n_train is None:
            raise RuntimeError("Call fit_transform before transform.")
        return self._build_features(series, fit=False)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of column names produced by this engineer.

        Note: Calendar and trend names are included when the corresponding
        config flags are ``True``.  Assumes a ``DatetimeIndex`` input —
        consistent with ``fit_transform`` / ``transform``.
        """
        names: list[str] = [f"lag_{n}" for n in self.config.lag_periods]
        for w in self.config.rolling_windows:
            names += [f"rolling_mean_{w}", f"rolling_std_{w}"]
        if self.config.include_calendar:
            names += ["month_sin", "month_cos", "quarter_sin", "quarter_cos"]
        if self.config.include_trend:
            names += ["year_norm", "trend_idx"]
        if self.config.include_expanding:
            names += ["expanding_mean", "expanding_std"]
        return names

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_features(self, series: pd.Series, fit: bool) -> pd.DataFrame:
        """Build the full feature matrix for *series*.

        Args:
            series: Input time series.
            fit: If ``True``, derive normalisation constants from *series*
                 (used by ``fit_transform``).  If ``False``, use stored
                 constants (used by ``transform``).

        Returns:
            Feature DataFrame with the first ``max_lag`` rows removed.
        """
        max_lag = max(self.config.lag_periods)
        df = pd.DataFrame(index=series.index)

        # ── Lag features ─────────────────────────────────────────────────────
        # lag_n[t] = y[t - n]  →  fully causal
        for n in self.config.lag_periods:
            df[f"lag_{n}"] = series.shift(n)

        # ── Rolling features ─────────────────────────────────────────────────
        # shift(1) ensures rolling window at t uses y[t-1..t-w], not y[t]
        shifted = series.shift(1)
        for w in self.config.rolling_windows:
            df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
            df[f"rolling_std_{w}"] = shifted.rolling(w).std()

        # ── Calendar features ─────────────────────────────────────────────────
        if self.config.include_calendar and isinstance(series.index, pd.DatetimeIndex):
            months = series.index.month
            quarters = series.index.quarter
            df["month_sin"] = np.sin(2 * np.pi * months / 12)
            df["month_cos"] = np.cos(2 * np.pi * months / 12)
            df["quarter_sin"] = np.sin(2 * np.pi * quarters / 4)
            df["quarter_cos"] = np.cos(2 * np.pi * quarters / 4)

        # ── Trend features ────────────────────────────────────────────────────
        if self.config.include_trend and isinstance(series.index, pd.DatetimeIndex):
            n_total = len(series)
            n_train = self._n_train  # set in fit_transform before _build_features

            if fit:
                # year_norm: normalised year relative to training year range
                min_yr = int(series.index.year.min())
                yr_range = max(1, int(series.index.year.max()) - min_yr)
                df["year_norm"] = (series.index.year - min_yr) / yr_range
                # trend_idx: [0.0, ..., 1.0] over training window
                df["trend_idx"] = np.arange(n_total) / max(1, n_train - 1)
            else:
                # Use stored training stats; extrapolates beyond training window
                df["year_norm"] = (
                    series.index.year - self._min_year
                ) / self._year_range
                df["trend_idx"] = np.arange(n_total) / max(1, n_train - 1)

        # ── Expanding features ────────────────────────────────────────────────
        # shift(1) ensures causal: expanding stats at t use y[0..t-1]
        if self.config.include_expanding:
            df["expanding_mean"] = series.shift(1).expanding().mean()
            df["expanding_std"] = series.shift(1).expanding().std()

        # Drop the first max_lag rows (all NaN from lag shifts)
        return df.iloc[max_lag:]
