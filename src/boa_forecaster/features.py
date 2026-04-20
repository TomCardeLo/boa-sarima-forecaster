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

Deterministic-feature caching (v2.2 P1)
---------------------------------------
Calendar (sin/cos) and trend (``year_norm``, ``trend_idx``) features are
pure functions of the ``DatetimeIndex``.  In the optimiser's per-trial CV
loop they are recomputed every fold even though the index is the same —
that is wasted work.  ``_compute_deterministic_features`` exposes them as
a standalone ``pd.DataFrame`` that the optimiser can build **once** for
the full series index and thread through every fold / recursive-forecast
step via ``fit_transform(..., feature_cache=...)``.

The cached path is **bit-identical** to the fresh-compute path when the
cache was built from the same index (covered by
``tests/unit/test_features.py::test_cache_equivalence_fit_transform``).
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

    @classmethod
    def for_frequency(cls, freq: str, **overrides: object) -> FeatureConfig:
        """Return a ``FeatureConfig`` with sensible defaults for *freq*.

        The default ``FeatureConfig()`` is calibrated for **monthly** series
        (``lag_periods=[1, 2, 3, 6, 12]``, ``rolling_windows=[3, 6, 12]``).
        Those values are meaningless for hourly PM2.5 or daily inventory
        data, where the relevant seasonalities live at different offsets.
        This factory picks defaults per pandas frequency alias so callers
        don't have to re-derive them from first principles.

        Chosen defaults (and the reasoning behind each):

        - ``"MS"`` / ``"M"`` (monthly) — lags ``[1, 2, 3, 6, 12]``, windows
          ``[3, 6, 12]``.  Covers the last quarter, half-year and full year;
          matches the library's historical default.
        - ``"W"`` (weekly) — lags ``[1, 2, 4, 13, 26, 52]``, windows
          ``[4, 13, 52]``.  Captures same-week, two-weeks-back, ~month
          (4w), ~quarter (13w), ~half-year (26w), and full-year (52w)
          seasonality common in retail and utility demand.
        - ``"D"`` (daily) — lags ``[1, 2, 3, 7, 14, 30, 365]``, windows
          ``[7, 14, 30]``.  Day-of-week effect (lag 7), bi-weekly pay-cycle
          (lag 14), monthly (lag 30), and annual (lag 365).
        - ``"h"`` / ``"H"`` (hourly) — lags ``[1, 2, 3, 24, 48, 168]``,
          windows ``[24, 168]``.  Last three hours for short-horizon
          momentum; 24 h / 48 h for daily seasonality; 168 h for weekly.
          Matches the air-quality (PM2.5) feedback in
          ``tasks/feedback_aire.md`` §1.

        These are **starting points**, not dogma.  Pass ``**overrides`` to
        patch any field without rebuilding the whole config, e.g.
        ``FeatureConfig.for_frequency("D", include_expanding=True)`` or
        ``FeatureConfig.for_frequency("h", lag_periods=[1, 24])``.

        Alias handling mirrors ``_freq_to_period_alias`` in
        ``preprocessor.py``: both pandas 1.x (``"H"``) and pandas 2.x
        (``"h"``) spellings are accepted for hourly.  Unrecognised aliases
        raise ``ValueError`` listing the supported set.

        Args:
            freq: Pandas DateOffset alias (``"MS"``, ``"M"``, ``"W"``,
                ``"D"``, ``"h"``, ``"H"``).
            **overrides: Any ``FeatureConfig`` field to override — e.g.
                ``lag_periods=[1, 2]`` or ``include_expanding=True``.
                Invalid field names propagate as ``TypeError`` from the
                dataclass constructor.

        Returns:
            A new ``FeatureConfig`` instance.

        Raises:
            ValueError: If *freq* is not a supported alias.
            TypeError: If *overrides* contains a key that is not a
                ``FeatureConfig`` field.
        """
        # Frequency → (lag_periods, rolling_windows) map.
        # Kept inside the method so each call constructs fresh lists —
        # prevents aliasing between returned configs (mirrors the
        # default_factory guard on the dataclass fields).
        presets: dict[str, tuple[list[int], list[int]]] = {
            "MS": ([1, 2, 3, 6, 12], [3, 6, 12]),
            "M": ([1, 2, 3, 6, 12], [3, 6, 12]),
            "W": ([1, 2, 4, 13, 26, 52], [4, 13, 52]),
            "D": ([1, 2, 3, 7, 14, 30, 365], [7, 14, 30]),
            "h": ([1, 2, 3, 24, 48, 168], [24, 168]),
            "H": ([1, 2, 3, 24, 48, 168], [24, 168]),
        }
        if freq not in presets:
            raise ValueError(
                f"Unsupported frequency alias {freq!r}. "
                f"Supported aliases: {sorted(presets)} "
                "(monthly: 'MS'/'M', weekly: 'W', daily: 'D', hourly: 'h'/'H')."
            )
        lag_periods, rolling_windows = presets[freq]
        defaults: dict[str, object] = {
            "lag_periods": list(lag_periods),
            "rolling_windows": list(rolling_windows),
        }
        defaults.update(overrides)
        return cls(**defaults)  # type: ignore[arg-type]


def _compute_deterministic_features(
    index: pd.DatetimeIndex,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Compute the subset of features that are pure functions of *index*.

    Calendar (month/quarter sin/cos) and trend (``year_norm``, ``trend_idx``)
    depend only on the ``DatetimeIndex``, never on the observed values.  They
    can be pre-computed once for the full optimiser series index and sliced
    per CV fold, avoiding ~3× the work of the current per-fold implementation.

    Args:
        index: Target ``DatetimeIndex`` (anything else short-circuits to a
            column-less frame).
        config: Feature configuration; ``include_calendar`` / ``include_trend``
            gate which columns are produced.

    Returns:
        ``pd.DataFrame`` indexed by *index*.  Columns appear in the same order
        as ``FeatureEngineer.get_feature_names`` produces them so
        ``_build_features`` can splice the slice in directly.
    """
    df = pd.DataFrame(index=index)
    if not isinstance(index, pd.DatetimeIndex):
        return df

    if config.include_calendar:
        months = index.month
        quarters = index.quarter
        df["month_sin"] = np.sin(2 * np.pi * months / 12)
        df["month_cos"] = np.cos(2 * np.pi * months / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * quarters / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * quarters / 4)

    if config.include_trend:
        n = len(index)
        min_yr = int(index.year.min())
        yr_range = max(1, int(index.year.max()) - min_yr)
        df["year_norm"] = (index.year - min_yr) / yr_range
        df["trend_idx"] = np.arange(n) / max(1, n - 1)

    return df


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

    Deterministic-feature cache (optional)
    --------------------------------------
    ``fit_transform`` and ``transform`` accept an optional ``feature_cache``
    kwarg — a ``pd.DataFrame`` produced by
    ``_compute_deterministic_features`` whose index must be a **superset** of
    the series index passed to the method.  When provided:

    - The calendar / trend columns are sliced out of the cache via
      ``cache.loc[series.index]`` instead of being recomputed.
    - The cache is retained on the instance (``self._cache``) so that
      subsequent ``transform`` calls inside ``recursive_forecast`` pick it up
      without the caller having to thread it through explicitly.

    Numerical equivalence vs the no-cache path is guaranteed whenever the
    cache was computed from the same (or a superset of) the series index.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._n_train: int | None = None
        self._min_year: int | None = None
        self._year_range: int | None = None
        self._cache: pd.DataFrame | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        series: pd.Series,
        feature_cache: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on *series* and return ``(X, y)`` with NaN rows removed.

        Args:
            series: Time series with at least ``max(lag_periods) + 1`` points.
            feature_cache: Optional pre-computed deterministic-feature frame
                whose index must cover ``series.index``.  When supplied the
                calendar / trend columns are taken from this cache instead of
                being recomputed and the cache is retained on the instance so
                subsequent ``transform`` calls reuse it.

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

        self._n_train = len(series)
        if isinstance(series.index, pd.DatetimeIndex):
            self._min_year = int(series.index.year.min())
            self._year_range = max(1, int(series.index.year.max()) - self._min_year)

        # Retain the cache so recursive_forecast (which calls fe.transform
        # without knowing about the cache) can still benefit from it.
        self._cache = feature_cache

        features = self._build_features(series, fit=True, cache=feature_cache)
        y = series.iloc[max_lag:].copy()
        y.name = self.config.target_col
        return features, y

    def transform(
        self,
        series: pd.Series,
        feature_cache: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Apply the same feature pipeline using stats stored by ``fit_transform``.

        Used for recursive forecasting: the extended series (train + predictions
        appended one step at a time) is transformed to produce the feature row
        for the next prediction step.

        Args:
            series: Series to transform (may be longer than the training set).
            feature_cache: Optional pre-computed deterministic-feature frame;
                when ``None`` falls back to the cache stored on the instance
                by ``fit_transform``.

        Returns:
            Feature DataFrame (NaN rows for the first ``max_lag`` positions
            are dropped, matching the behaviour of ``fit_transform``).

        Raises:
            RuntimeError: If ``fit_transform`` has not been called yet.
        """
        if self._n_train is None:
            raise RuntimeError("Call fit_transform before transform.")
        cache = feature_cache if feature_cache is not None else self._cache
        return self._build_features(series, fit=False, cache=cache)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of column names produced by this engineer."""
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

    def _compute_window_features(self, series: pd.Series) -> pd.DataFrame:
        """Compute lag / rolling features (value-dependent).

        Builds columns as a dict of numpy arrays and wraps them into a single
        ``DataFrame`` at the end to avoid the per-column pandas block-manager
        overhead that the v2.1 loop incurred.
        """
        cols: dict[str, np.ndarray] = {}

        values = series.to_numpy()
        for n in self.config.lag_periods:
            shifted: np.ndarray = np.empty(len(values), dtype=np.float64)
            shifted[:n] = np.nan
            shifted[n:] = values[:-n]
            cols[f"lag_{n}"] = shifted

        if self.config.rolling_windows:
            shifted_series = series.shift(1)
            for w in self.config.rolling_windows:
                cols[f"rolling_mean_{w}"] = shifted_series.rolling(w).mean().to_numpy()
                cols[f"rolling_std_{w}"] = shifted_series.rolling(w).std().to_numpy()
        return pd.DataFrame(cols, index=series.index)

    def _build_features(
        self,
        series: pd.Series,
        fit: bool,
        cache: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the full feature matrix for *series*.

        Args:
            series: Input time series.
            fit: If ``True``, derive normalisation constants from *series*
                 (used by ``fit_transform``).  If ``False``, use stored
                 constants (used by ``transform``).
            cache: Optional pre-computed deterministic-feature frame covering
                ``series.index``.  When supplied, calendar / trend columns are
                sliced from it rather than being recomputed.

        Returns:
            Feature DataFrame with the first ``max_lag`` rows removed.
        """
        max_lag = max(self.config.lag_periods)
        df = self._compute_window_features(series)

        # ── Deterministic block (calendar + trend) ───────────────────────────
        if cache is not None:
            # ``cache`` is allowed to be a superset — align to series.index.
            det_slice = cache.loc[series.index]
            for col in det_slice.columns:
                df[col] = det_slice[col].to_numpy()
        else:
            if self.config.include_calendar and isinstance(
                series.index, pd.DatetimeIndex
            ):
                months = series.index.month
                quarters = series.index.quarter
                df["month_sin"] = np.sin(2 * np.pi * months / 12)
                df["month_cos"] = np.cos(2 * np.pi * months / 12)
                df["quarter_sin"] = np.sin(2 * np.pi * quarters / 4)
                df["quarter_cos"] = np.cos(2 * np.pi * quarters / 4)

            if self.config.include_trend and isinstance(series.index, pd.DatetimeIndex):
                n_total = len(series)
                n_train = self._n_train
                assert n_train is not None  # noqa: S101 — invariant documented above

                if fit:
                    min_yr = int(series.index.year.min())
                    yr_range = max(1, int(series.index.year.max()) - min_yr)
                    df["year_norm"] = (series.index.year - min_yr) / yr_range
                    df["trend_idx"] = np.arange(n_total) / max(1, n_train - 1)
                else:
                    df["year_norm"] = (
                        series.index.year - self._min_year
                    ) / self._year_range
                    df["trend_idx"] = np.arange(n_total) / max(1, n_train - 1)

        # ── Expanding block (always value-dependent, so never cached) ────────
        if self.config.include_expanding:
            df["expanding_mean"] = series.shift(1).expanding().mean()
            df["expanding_std"] = series.shift(1).expanding().std()

        # Drop the first max_lag rows (all NaN from lag shifts)
        return df.iloc[max_lag:]
