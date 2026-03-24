"""Walk-forward (expanding window) cross-validation for time-series models.

Public functions
----------------
walk_forward_validation
    Evaluate a single model over multiple out-of-sample folds.
validate_by_group
    Apply walk_forward_validation to every group in a multi-series DataFrame.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from sarima_bayes.metrics import rmsle, smape

logger = logging.getLogger(__name__)


def walk_forward_validation(
    series: pd.Series,
    model_fn: Callable[[pd.Series], pd.Series],
    n_folds: int = 3,
    test_size: int = 12,
    min_train_size: int = 24,
    metrics_fn: dict[str, Callable] | None = None,
) -> pd.DataFrame:
    """Evaluate *model_fn* using an expanding-window walk-forward scheme.

    Each fold extends the training window by ``test_size`` observations while
    keeping the test window at a fixed length of ``test_size``, preventing any
    look-ahead bias.

    Args:
        series: Univariate time series with a ``pd.DatetimeIndex``.
        model_fn: Callable ``(train: pd.Series) -> pd.Series``.  The returned
            Series must contain exactly ``test_size`` predictions aligned with
            the test window.
        n_folds: Number of walk-forward folds.  Must be >= 3.
        test_size: Number of observations in each test window.
        min_train_size: Minimum number of observations in the first training
            window (fold 0).
        metrics_fn: Mapping of metric-name → callable ``(y_true, y_pred) ->
            float``.  Defaults to ``{"sMAPE": smape, "RMSLE": rmsle}``.

    Returns:
        DataFrame with one row per fold and columns:
        ``fold``, ``train_start``, ``train_end``, ``test_start``,
        ``test_end``, plus one column per metric.

    Raises:
        ValueError: If ``n_folds < 3`` or the series is too short.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from statsmodels.tsa.statespace.sarimax import SARIMAX
        >>> dates = pd.date_range("2020-01", periods=48, freq="MS")
        >>> series = pd.Series(np.ones(48) * 100.0, index=dates)
        >>> def naive_fn(train):
        ...     idx = pd.date_range(train.index[-1], periods=5, freq="MS")[1:]
        ...     return pd.Series([train.iloc[-1]] * 4, index=idx)
        >>> result = walk_forward_validation(series, naive_fn, n_folds=3, test_size=4, min_train_size=24)
        >>> len(result)
        3
    """
    if n_folds < 3:
        raise ValueError(f"n_folds must be >= 3, got {n_folds}")

    n = len(series)
    required = min_train_size + n_folds * test_size
    if n < required:
        raise ValueError(
            f"Series length {n} is too short for {n_folds} folds with "
            f"min_train_size={min_train_size} and test_size={test_size}. "
            f"Need at least {required}."
        )

    if metrics_fn is None:
        metrics_fn = {"sMAPE": smape, "RMSLE": rmsle}

    rows = []
    for i in range(n_folds):
        train_end_idx = min_train_size + i * test_size
        test_end_idx = min_train_size + (i + 1) * test_size

        train = series.iloc[:train_end_idx]
        test = series.iloc[train_end_idx:test_end_idx]

        row: dict = {
            "fold": i + 1,
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
        }

        try:
            predictions = model_fn(train)
            y_true = test.values
            y_pred = predictions.values
            for name, fn in metrics_fn.items():
                row[name] = fn(y_true, y_pred)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fold %d failed: %s", i + 1, exc)
            for name in metrics_fn:
                row[name] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def validate_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    date_col: str,
    model_fn: Callable[[pd.Series], pd.Series],
    freq: str = "MS",
    **kwargs,
) -> pd.DataFrame:
    """Run walk-forward validation for every group in *df*.

    Args:
        df: Multi-series DataFrame containing ``group_cols``, ``date_col``,
            and ``target_col``.
        group_cols: Columns used to identify each individual series
            (e.g. ``["Country", "SKU"]``).
        target_col: Name of the column holding the target variable.
        date_col: Name of the date column (must be parseable as datetime).
        model_fn: Passed unchanged to ``walk_forward_validation()``.
        freq: Pandas DateOffset alias used to assign the index frequency
            before passing each group's series to ``walk_forward_validation``.
            Defaults to ``"MS"`` (month start).
        **kwargs: Extra keyword arguments forwarded to
            ``walk_forward_validation()`` — ``n_folds``, ``test_size``,
            ``min_train_size``, ``metrics_fn``.

    Returns:
        Concatenated DataFrame of all fold results, with ``group_cols``
        prepended as additional columns.  Returns an empty DataFrame if
        every group fails.

    Raises:
        No exceptions are propagated — per-group failures are logged as
        warnings and the group is skipped.

    Example:
        >>> validate_by_group(
        ...     df, group_cols=["Country", "SKU"],
        ...     target_col="CS", date_col="Date",
        ...     model_fn=naive_fn, n_folds=3, test_size=6,
        ... )
    """
    all_results: list[pd.DataFrame] = []

    for group_key, group_df in df.groupby(group_cols):
        try:
            group_df_sorted = group_df.sort_values(date_col)
            index = pd.DatetimeIndex(group_df_sorted[date_col])
            series = pd.Series(
                group_df_sorted[target_col].values,
                index=index,
            )
            series.index.freq = pd.tseries.frequencies.to_offset(freq)

            fold_df = walk_forward_validation(series, model_fn, **kwargs)

            if isinstance(group_key, tuple):
                for col, val in zip(group_cols, group_key):
                    fold_df.insert(0, col, val)
            else:
                fold_df.insert(0, group_cols[0], group_key)

            all_results.append(fold_df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Group %s failed: %s", group_key, exc)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
