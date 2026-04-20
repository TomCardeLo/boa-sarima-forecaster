"""Walk-forward (expanding window) cross-validation for time-series models.

Public functions
----------------
walk_forward_validation
    Evaluate a single model over multiple out-of-sample folds.  Supports
    optional process-parallelism over folds via ``n_jobs``.
validate_by_group
    Apply walk_forward_validation to every group in a multi-series DataFrame.
"""

from __future__ import annotations

import logging
import pickle
from typing import Callable

import pandas as pd

from boa_forecaster.metrics import rmsle, smape

logger = logging.getLogger(__name__)


def _run_fold(
    i: int,
    series: pd.Series,
    model_fn: Callable[[pd.Series], pd.Series],
    min_train_size: int,
    test_size: int,
    metrics_fn: dict[str, Callable[..., float]],
) -> dict:
    """Evaluate a single walk-forward fold.

    Extracted from the inline loop so ``joblib.Parallel`` can dispatch folds
    to worker processes.  Returns the result row as a plain ``dict`` — pandas
    concatenation is deferred to the caller to keep this function
    pickle-friendly.
    """
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
        if len(y_pred) > len(y_true):
            y_pred = y_pred[: len(y_true)]
        elif len(y_pred) < len(y_true):
            raise ValueError(
                f"model_fn returned {len(y_pred)} predictions but test window "
                f"has {len(y_true)} observations."
            )
        for name, fn in metrics_fn.items():
            row[name] = fn(y_true, y_pred)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fold %d failed: %s", i + 1, exc)
        for name in metrics_fn:
            row[name] = float("nan")

    return row


def walk_forward_validation(
    series: pd.Series,
    model_fn: Callable[[pd.Series], pd.Series],
    n_folds: int = 3,
    test_size: int = 12,
    min_train_size: int = 24,
    metrics_fn: dict[str, Callable[..., float]] | None = None,
    n_jobs: int = 1,
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
        n_folds: Number of walk-forward folds.  Must be >= 1.  Values below 3
            are accepted but carry high variance in the resulting metric
            estimate — prefer ``n_folds >= 3`` in production use.  The default
            of 3 is kept for backwards compatibility.
        test_size: Number of observations in each test window.
        min_train_size: Minimum number of observations in the first training
            window (fold 0).
        metrics_fn: Mapping of metric-name → callable ``(y_true, y_pred) ->
            float``.  Defaults to ``{"sMAPE": smape, "RMSLE": rmsle}``.
        n_jobs: Number of parallel worker processes for fold evaluation.
            ``1`` (default) keeps the historical sequential behaviour; values
            ``>= 2`` dispatch via ``joblib.Parallel(backend="loky")``.  Every
            worker is wrapped in ``threadpoolctl.threadpool_limits(1)`` to
            stop BLAS/OpenMP oversubscription from erasing the parallel win.
            If ``model_fn`` (or an object it closes over) is not picklable,
            the call automatically falls back to sequential execution and
            logs a warning.

    Returns:
        DataFrame with one row per fold and columns:
        ``fold``, ``train_start``, ``train_end``, ``test_start``,
        ``test_end``, plus one column per metric.

    Raises:
        ValueError: If ``n_folds < 1`` or the series is too short.

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
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

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

    rows: list[dict]
    if n_jobs is not None and n_jobs >= 2:
        rows = _run_folds_parallel(
            n_folds, series, model_fn, min_train_size, test_size, metrics_fn, n_jobs
        )
    else:
        rows = [
            _run_fold(i, series, model_fn, min_train_size, test_size, metrics_fn)
            for i in range(n_folds)
        ]

    return pd.DataFrame(rows)


def _run_fold_pinned(
    i: int,
    series: pd.Series,
    model_fn: Callable[[pd.Series], pd.Series],
    min_train_size: int,
    test_size: int,
    metrics_fn: dict[str, Callable[..., float]],
) -> dict:
    """Module-level worker: run a fold with BLAS/OpenMP pinned to 1 thread.

    Defined at module scope (not as a nested closure) so loky can pickle it
    by import path rather than via cloudpickle's closure machinery — the
    latter fails to un-serialise on the worker side in some environments.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _run_fold(i, series, model_fn, min_train_size, test_size, metrics_fn)


def _run_folds_parallel(
    n_folds: int,
    series: pd.Series,
    model_fn: Callable[[pd.Series], pd.Series],
    min_train_size: int,
    test_size: int,
    metrics_fn: dict[str, Callable[..., float]],
    n_jobs: int,
) -> list[dict]:
    """Dispatch ``_run_fold`` across ``n_jobs`` worker processes.

    Each worker pins BLAS / OpenMP to a single thread to prevent
    oversubscription.  If pickling fails (e.g. ``model_fn`` closes over a
    local lambda that cloudpickle cannot serialise), we log a warning and
    fall back to a sequential loop so the caller's contract
    (``pd.DataFrame`` with the right shape) is never broken by a
    performance optimisation.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError as exc:  # pragma: no cover — joblib is a core dep
        logger.warning(
            "Parallel fold execution unavailable (%s); running sequentially.",
            exc,
        )
        return [
            _run_fold(i, series, model_fn, min_train_size, test_size, metrics_fn)
            for i in range(n_folds)
        ]

    try:
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_fold_pinned)(
                i, series, model_fn, min_train_size, test_size, metrics_fn
            )
            for i in range(n_folds)
        )
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        # ``loky`` wraps pickling errors in several flavours (PicklingError,
        # AttributeError for un-importable closures, TypeError for
        # non-picklable objects).  Fall back to sequential so the run still
        # completes.
        logger.warning(
            "Parallel fold dispatch failed (%s); falling back to sequential.",
            exc,
        )
        return [
            _run_fold(i, series, model_fn, min_train_size, test_size, metrics_fn)
            for i in range(n_folds)
        ]


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
