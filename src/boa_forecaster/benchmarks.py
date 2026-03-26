"""Baseline forecasting models and multi-model benchmark comparison.

Public functions
----------------
seasonal_naive
    Repeat the value from the same period one year ago.
ets_model
    Holt-Winters additive trend + seasonal (statsmodels).
auto_arima_nixtla
    AutoARIMA via the statsforecast library (Nixtla).
run_model_comparison
    Bayesian-optimise each ModelSpec, then compare all against baselines.
run_benchmark_comparison
    [DEPRECATED v1 API] Compare a pre-built SARIMA+BO callable against baselines.
summary_table
    Aggregate fold-level results into a mean/std table with a beats_naive flag.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable

import pandas as pd

from boa_forecaster.optimizer import optimize_model
from boa_forecaster.validation import walk_forward_validation

if TYPE_CHECKING:
    from boa_forecaster.models.base import ModelSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual baseline models
# ---------------------------------------------------------------------------


def seasonal_naive(
    train: pd.Series,
    forecast_horizon: int,
    m: int = 12,
) -> pd.Series:
    """Repeat the value from the same period one seasonal cycle ago.

    Args:
        train: Historical series with a ``pd.DatetimeIndex``.
        forecast_horizon: Number of steps to forecast.
        m: Seasonal period. Defaults to ``12`` for monthly data.

    Returns:
        Forecast Series of length ``forecast_horizon`` with a future
        ``pd.DatetimeIndex``.

    Example:
        >>> forecast = seasonal_naive(train, forecast_horizon=6)
        >>> len(forecast)
        6
    """
    freq = train.index.freq or "MS"
    future_index = pd.date_range(
        start=train.index[-1],
        periods=forecast_horizon + 1,
        freq=freq,
    )[1:]

    if len(train) < m:
        logger.warning(
            "seasonal_naive: train length %d < m=%d; falling back to last value.",
            len(train),
            m,
        )
        values = [train.iloc[-1]] * forecast_horizon
    else:
        values = [train.iloc[-(m - (i % m))] for i in range(forecast_horizon)]

    return pd.Series(values, index=future_index, dtype=float)


def ets_model(train: pd.Series, forecast_horizon: int, m: int = 12) -> pd.Series:
    """Holt-Winters additive ETS model via statsmodels.

    Falls back to ``seasonal_naive()`` on any exception (e.g. series too
    short for seasonal decomposition).

    Args:
        train: Historical series with a ``pd.DatetimeIndex``.
        forecast_horizon: Number of steps to forecast.
        m: Seasonal period for Holt-Winters decomposition.
            Defaults to ``12`` (monthly data, annual cycle).

    Returns:
        Forecast Series of length ``forecast_horizon``.

    Example:
        >>> forecast = ets_model(train, forecast_horizon=6)
        >>> len(forecast)
        6
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=m,
        ).fit(optimized=True)
        forecast = model.forecast(forecast_horizon)

        _freq = train.index.freq or "MS"
        future_index = pd.date_range(
            start=train.index[-1],
            periods=forecast_horizon + 1,
            freq=_freq,
        )[1:]
        return pd.Series(forecast.values, index=future_index, dtype=float)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ets_model failed (%s); falling back to seasonal_naive.", exc)
        return seasonal_naive(train, forecast_horizon, m=m)


def auto_arima_nixtla(
    train: pd.Series,
    forecast_horizon: int,
    m: int = 12,
    freq: str = "MS",
) -> pd.Series:
    """AutoARIMA via the statsforecast (Nixtla) library.

    Falls back to ``seasonal_naive()`` on any exception (e.g. if
    ``statsforecast`` is not installed).

    Args:
        train: Historical series with a ``pd.DatetimeIndex``.
        forecast_horizon: Number of steps to forecast.
        m: Seasonal period passed to ``AutoARIMA``.
            Defaults to ``12`` (monthly data, annual cycle).
        freq: Pandas DateOffset alias passed to ``StatsForecast``.
            Defaults to ``"MS"`` (month start).

    Returns:
        Forecast Series of length ``forecast_horizon``.

    Example:
        >>> forecast = auto_arima_nixtla(train, forecast_horizon=6)
        >>> len(forecast)
        6
    """
    _freq = train.index.freq or freq
    future_index = pd.date_range(
        start=train.index[-1],
        periods=forecast_horizon + 1,
        freq=_freq,
    )[1:]

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        sf_df = pd.DataFrame(
            {
                "unique_id": "item",
                "ds": train.index,
                "y": train.values,
            }
        )

        sf = StatsForecast(models=[AutoARIMA(season_length=m)], freq=freq)
        predictions = sf.forecast(df=sf_df, h=forecast_horizon)

        return pd.Series(
            predictions["AutoARIMA"].values,
            index=future_index,
            dtype=float,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "auto_arima_nixtla failed (%s); falling back to seasonal_naive.", exc
        )
        return seasonal_naive(train, forecast_horizon, m=m)


# ---------------------------------------------------------------------------
# Multi-model comparison runner (v2 primary API)
# ---------------------------------------------------------------------------


def run_model_comparison(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    date_col: str,
    model_specs: list[ModelSpec],
    n_calls_per_model: int = 50,
    n_folds: int = 3,
    test_size: int = 12,
    min_train_size: int = 24,
    m: int = 12,
    freq: str = "MS",
    seed: int = 42,
) -> pd.DataFrame:
    """Bayesian-optimise each ModelSpec and compare all models against baselines.

    For every combination of group and model spec, runs Bayesian TPE optimisation
    via :func:`~boa_forecaster.optimizer.optimize_model`, builds the forecaster,
    then evaluates it with :func:`~boa_forecaster.validation.walk_forward_validation`.
    Three baselines (Seasonal Naïve, ETS, AutoARIMA) are always included.

    Args:
        df: Multi-series DataFrame.
        group_cols: Columns that identify each individual series.
        target_col: Target variable column name.
        date_col: Date column name.
        model_specs: List of :class:`~boa_forecaster.models.base.ModelSpec`
            instances to optimise and evaluate.
        n_calls_per_model: Optuna trials per model per group. Defaults to ``50``.
        n_folds: Walk-forward CV folds. Defaults to ``3``.
        test_size: Length of each test window. Defaults to ``12``.
        min_train_size: Minimum initial training window. Defaults to ``24``.
        m: Seasonal period for baselines. Defaults to ``12``.
        freq: Pandas DateOffset alias for the series index. Defaults to ``"MS"``.
        seed: Optuna sampler seed for reproducibility. Defaults to ``42``.

    Returns:
        DataFrame with columns:
        ``[group_cols..., fold, train_start, train_end, test_start, test_end,
        sMAPE, RMSLE, model, optimized]``.
        ``optimized=True`` for Bayesian-optimised models; ``False`` for baselines.

    Example:
        >>> results = run_model_comparison(
        ...     df, group_cols=["SKU"],
        ...     target_col="demand", date_col="date",
        ...     model_specs=[SARIMASpec(), RandomForestSpec()],
        ...     n_calls_per_model=10,
        ... )
        >>> "optimized" in results.columns
        True
    """
    baselines: dict[str, Callable[[pd.Series], pd.Series]] = {
        "seasonal_naive": lambda t: seasonal_naive(t, test_size, m=m),
        "ETS": lambda t: ets_model(t, test_size, m=m),
        "AutoARIMA": lambda t: auto_arima_nixtla(t, test_size, m=m, freq=freq),
    }

    all_rows: list[pd.DataFrame] = []

    for group_key, group_df in df.groupby(group_cols):
        group_df_sorted = group_df.sort_values(date_col)
        index = pd.DatetimeIndex(group_df_sorted[date_col])
        series = pd.Series(group_df_sorted[target_col].values, index=index)
        series.index.freq = pd.tseries.frequencies.to_offset(freq)

        def _annotate(
            fold_df: pd.DataFrame, model_name: str, is_optimized: bool
        ) -> pd.DataFrame:
            fold_df = fold_df.copy()
            fold_df["model"] = model_name
            fold_df["optimized"] = is_optimized
            if isinstance(group_key, tuple):
                for col, val in zip(group_cols, group_key):
                    fold_df.insert(0, col, val)
            else:
                fold_df.insert(0, group_cols[0], group_key)
            return fold_df

        # ── Bayesian-optimised models ─────────────────────────────────────
        for spec in model_specs:
            try:
                result = optimize_model(series, spec, n_calls=n_calls_per_model, seed=seed)
                forecaster = spec.build_forecaster(result.best_params)
                fold_df = walk_forward_validation(
                    series,
                    forecaster,
                    n_folds=n_folds,
                    test_size=test_size,
                    min_train_size=min_train_size,
                )
                all_rows.append(_annotate(fold_df, spec.name, True))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run_model_comparison: group=%s model=%s failed: %s",
                    group_key,
                    spec.name,
                    exc,
                )

        # ── Baselines ─────────────────────────────────────────────────────
        for model_name, fn in baselines.items():
            try:
                fold_df = walk_forward_validation(
                    series,
                    fn,
                    n_folds=n_folds,
                    test_size=test_size,
                    min_train_size=min_train_size,
                )
                all_rows.append(_annotate(fold_df, model_name, False))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run_model_comparison: group=%s baseline=%s failed: %s",
                    group_key,
                    model_name,
                    exc,
                )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Deprecated v1 wrapper
# ---------------------------------------------------------------------------


def run_benchmark_comparison(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    date_col: str,
    sarima_model_fn: Callable[[pd.Series], pd.Series],
    n_folds: int = 3,
    test_size: int = 12,
    min_train_size: int = 24,
    m: int = 12,
    freq: str = "MS",
) -> pd.DataFrame:
    """Compare a pre-built SARIMA+BO callable against baselines.

    .. deprecated::
        Use :func:`run_model_comparison` instead, which accepts
        :class:`~boa_forecaster.models.base.ModelSpec` objects and handles
        Bayesian optimisation internally.

    Args:
        df: Multi-series DataFrame.
        group_cols: Columns identifying each series.
        target_col: Target variable column name.
        date_col: Date column name.
        sarima_model_fn: Pre-built SARIMA+BO callable ``(train) -> forecast``.
        n_folds: Number of walk-forward folds.
        test_size: Length of each test window.
        min_train_size: Minimum initial training size.
        m: Seasonal period forwarded to baselines. Defaults to ``12``.
        freq: Pandas DateOffset alias. Defaults to ``"MS"``.

    Returns:
        DataFrame with columns:
        ``[group_cols..., fold, train_start, train_end, test_start, test_end,
        sMAPE, RMSLE, model, optimized]``.

    Example:
        >>> results = run_benchmark_comparison(
        ...     df, group_cols=["Country", "SKU"],
        ...     target_col="CS", date_col="Date",
        ...     sarima_model_fn=make_sarima_fn(),
        ...     n_folds=3, test_size=6,
        ... )
        >>> "model" in results.columns
        True
    """
    warnings.warn(
        "run_benchmark_comparison() is deprecated in boa-forecaster v2.0. "
        "Use run_model_comparison(model_specs=[...]) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    models: dict[str, tuple[Callable[[pd.Series], pd.Series], bool]] = {
        "SARIMA+BO": (sarima_model_fn, True),
        "seasonal_naive": (lambda t: seasonal_naive(t, test_size, m=m), False),
        "ETS": (lambda t: ets_model(t, test_size, m=m), False),
        "AutoARIMA": (lambda t: auto_arima_nixtla(t, test_size, m=m, freq=freq), False),
    }

    all_rows: list[pd.DataFrame] = []

    for group_key, group_df in df.groupby(group_cols):
        group_df_sorted = group_df.sort_values(date_col)
        index = pd.DatetimeIndex(group_df_sorted[date_col])
        series = pd.Series(group_df_sorted[target_col].values, index=index)
        series.index.freq = pd.tseries.frequencies.to_offset(freq)

        for model_name, (fn, is_optimized) in models.items():
            try:
                fold_df = walk_forward_validation(
                    series,
                    fn,
                    n_folds=n_folds,
                    test_size=test_size,
                    min_train_size=min_train_size,
                )
                fold_df["model"] = model_name
                fold_df["optimized"] = is_optimized

                if isinstance(group_key, tuple):
                    for col, val in zip(group_cols, group_key):
                        fold_df.insert(0, col, val)
                else:
                    fold_df.insert(0, group_cols[0], group_key)

                all_rows.append(fold_df)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run_benchmark_comparison: group=%s model=%s failed: %s",
                    group_key,
                    model_name,
                    exc,
                )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def summary_table(results_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate fold-level benchmark results into a summary table.

    Works with both :func:`run_model_comparison` and the deprecated
    :func:`run_benchmark_comparison` output.

    Args:
        results_df: Output of :func:`run_model_comparison` or
            :func:`run_benchmark_comparison`.
        group_cols: Same group columns used during comparison.

    Returns:
        DataFrame with columns:
        ``[group_cols..., model, sMAPE_mean, sMAPE_std, RMSLE_mean,
        RMSLE_std, optimized (if present), beats_naive]``,
        sorted by ``[group_cols..., sMAPE_mean]``.

    Example:
        >>> tbl = summary_table(results, group_cols=["Country", "SKU"])
        >>> "beats_naive" in tbl.columns
        True
    """
    agg = (
        results_df.groupby([*group_cols, "model"])[["sMAPE", "RMSLE"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        *group_cols,
        "model",
        "sMAPE_mean",
        "sMAPE_std",
        "RMSLE_mean",
        "RMSLE_std",
    ]

    # Carry over the `optimized` flag (first value per model is deterministic)
    if "optimized" in results_df.columns:
        opt_map = (
            results_df.groupby("model")["optimized"].first().reset_index()
        )
        agg = agg.merge(opt_map, on="model", how="left")

    # beats_naive: compare sMAPE_mean against seasonal_naive per group
    naive_smape = agg[agg["model"] == "seasonal_naive"][
        [*group_cols, "sMAPE_mean"]
    ].rename(columns={"sMAPE_mean": "_naive_smape"})

    agg = agg.merge(naive_smape, on=group_cols, how="left")
    agg["beats_naive"] = agg["sMAPE_mean"] < agg["_naive_smape"]
    agg.loc[agg["model"] == "seasonal_naive", "beats_naive"] = False
    agg["beats_naive"] = agg["beats_naive"].astype(bool)
    agg = agg.drop(columns=["_naive_smape"])

    return agg.sort_values([*group_cols, "sMAPE_mean"]).reset_index(drop=True)
