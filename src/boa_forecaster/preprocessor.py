"""Data preprocessing utilities for monthly sales time series.

This module provides three operations applied before model fitting:

1. ``clean_zeros`` – removes SKU series with zero cumulative demand.
2. ``fill_blanks`` – fills missing calendar months with zero demand.
3. ``flag_intermittent`` – returns a boolean mask identifying groups whose
   zero-ratio meets a threshold (does not mutate or remove rows).

All functions operate on ``pd.DataFrame`` inputs and return copies so that
the caller's original data is never modified in-place.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _freq_to_period_alias(freq: str) -> str:
    """Map a pandas DateOffset alias to a pandas Period alias.

    ``pd.Series.dt.to_period()`` requires a Period alias (e.g. ``"M"``)
    rather than a DateOffset alias (e.g. ``"MS"``).  This helper performs
    the translation for the frequency values the library supports.

    Args:
        freq: Pandas DateOffset alias string (e.g. ``"MS"``, ``"W"``, ``"D"``).

    Returns:
        Corresponding pandas Period alias string (e.g. ``"M"``, ``"W"``, ``"D"``).

    Raises:
        ValueError: If ``freq`` has no known Period alias mapping.

    Example:
        >>> _freq_to_period_alias("MS")
        'M'
        >>> _freq_to_period_alias("W")
        'W'
    """
    _MAP: dict[str, str] = {
        # Monthly
        "MS": "M",
        "ME": "M",
        "M": "M",
        # Quarterly
        "QS": "Q",
        "QE": "Q",
        "Q": "Q",
        # Annual
        "YS": "Y",
        "YE": "Y",
        "Y": "Y",
        "AS": "Y",
        "A": "Y",
        # Weekly
        "W": "W",
        # Daily
        "D": "D",
        # Hourly (pandas 2.x uses lowercase "h")
        "H": "h",
        "h": "h",
        # Minutely (pandas 2.x uses "min")
        "T": "min",
        "min": "min",
    }
    try:
        return _MAP[freq]
    except KeyError:
        raise ValueError(
            f"Cannot map DateOffset alias {freq!r} to a Period alias. "
            f"Supported aliases: {sorted(_MAP)}"
        )


def clean_zeros(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    value_col: str = "CS",
) -> pd.DataFrame:
    """Remove groups whose cumulative demand is zero.

    Flat-zero series carry no signal for the ARIMA model and waste
    optimisation budget.  This function removes them early in the pipeline
    using a vectorised ``transform`` mask that avoids an explicit merge.

    Args:
        df: Input DataFrame with at least ``group_cols`` and ``value_col``.
        group_cols: Columns that define a unique time series
            (e.g. ``["SKU"]`` or ``["Country", "SKU"]``).
            Defaults to ``["SKU"]``.
        value_col: Column containing demand values to aggregate.
            Defaults to ``"CS"``.

    Returns:
        Filtered DataFrame (copy) with zero-sum groups removed.

    Example:
        >>> clean_zeros(df, group_cols=["Country", "SKU"], value_col="CS")
    """
    if group_cols is None:
        group_cols = ["SKU"]

    # transform keeps the same index alignment as the original DataFrame,
    # enabling a vectorised boolean mask without an explicit merge step.
    mask = df.groupby(group_cols)[value_col].transform("sum") != 0
    n_removed = (~mask).sum()
    if n_removed:
        logger.info("clean_zeros: removed %d zero-demand rows.", n_removed)
    return df[mask].copy()


def flag_intermittent(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    value_col: str = "CS",
    threshold: float = 0.7,
) -> pd.Series:
    """Flag groups whose zero-ratio meets or exceeds a threshold.

    Intermittent demand (many zeros interleaved with sparse non-zero spikes)
    is poorly handled by ARIMA and gradient-boosting models; specialised
    methods such as Croston or SBA are typically more accurate on such
    series.  This function identifies them without removing or modifying the
    data, leaving the choice of downstream treatment to the caller.

    NaN values in ``value_col`` are treated as zero demand.  This mirrors the
    convention used by :func:`fill_blanks` (missing periods are zero-filled)
    and yields a conservative flag — a group with many NaNs is a candidate
    for specialised treatment regardless of whether those gaps originated as
    recorded zeros or as missing data.

    Args:
        df: Input DataFrame with at least ``group_cols`` and ``value_col``.
        group_cols: Columns that define a unique time series
            (e.g. ``["SKU"]`` or ``["Country", "SKU"]``).  Defaults to
            ``["SKU"]``.
        value_col: Column containing demand values.  Defaults to ``"CS"``.
        threshold: Minimum zero-ratio required for a group to be flagged.
            Defaults to ``0.7`` (70% zero observations).  Must be in
            ``[0.0, 1.0]``.

    Returns:
        Boolean ``pd.Series`` aligned with ``df.index``: ``True`` where the
        row belongs to a flagged group, ``False`` otherwise.  The caller
        decides whether to filter, route, or annotate the flagged rows.

    Example:
        >>> mask = flag_intermittent(df, group_cols=["SKU"])
        >>> intermittent_df = df[mask]
        >>> continuous_df   = df[~mask]
    """
    if group_cols is None:
        group_cols = ["SKU"]

    # ``fillna(0.0)`` before comparison so NaN counts as a zero observation.
    zero_ratio = df.groupby(group_cols)[value_col].transform(
        lambda x: (x.fillna(0.0) == 0).mean()
    )
    return zero_ratio >= threshold


def fill_blanks(
    df: pd.DataFrame,
    date_col: str = "Date",
    group_cols: list[str] | None = None,
    value_col: str = "CS",
    end_date: str | None = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """Fill missing calendar months with zero demand.

    The SARIMAX model requires a complete, gap-free time series (``asfreq``
    raises on series with missing periods).  This function builds the target
    ``(date × group)`` MultiIndex directly from ``np.repeat`` / ``np.tile``
    and reindexes a grouped source Series, avoiding the intermediate
    cross-join DataFrame a merge-based approach would allocate.

    Duplicate ``(date, group)`` rows in the input are summed before reindexing
    (a no-op when the pipeline runs ``clean_zeros`` first).

    Args:
        df: Input DataFrame with at least ``date_col``, ``group_cols``,
            and ``value_col``.
        date_col: Name of the date column. Defaults to ``"Date"``.
        group_cols: Grouping columns (e.g. ``["SKU"]`` or
            ``["Country", "SKU"]``).  Defaults to ``["SKU"]``.
        value_col: Demand column name. Defaults to ``"CS"``.
        end_date: Extend the time axis up to this date (inclusive).
            If ``None``, uses the maximum date already in the data.
        freq: Pandas offset alias for the time frequency.
            Defaults to ``"MS"`` (month start).

    Returns:
        Fully filled DataFrame with no missing month/group combinations.

    Example:
        >>> fill_blanks(
        ...     df,
        ...     group_cols=["Country", "SKU"],
        ...     end_date="2026-01-01",
        ... )
    """
    if group_cols is None:
        group_cols = ["SKU"]

    df = df.copy()
    period_alias = _freq_to_period_alias(freq)
    # "W" (weekly) periods use end-of-period convention (Sunday) to match the
    # anchoring of pd.date_range(freq="W").  All other periods use start ("S").
    period_how = "E" if period_alias == "W" else "S"
    # Normalise all dates to the canonical period boundary so mid-period
    # timestamps (e.g. "2022-01-15") land in the correct bucket before reindexing.
    df[date_col] = (
        pd.to_datetime(df[date_col])
        .dt.to_period(period_alias)
        .dt.to_timestamp(how=period_how)
    )

    target_end = (
        pd.to_datetime(end_date).to_period(period_alias).to_timestamp(how=period_how)
        if end_date
        else df[date_col].max()
    )

    full_range = pd.date_range(
        start=df[date_col].min(),
        end=target_end,
        freq=freq,
    )

    # Build the target (date × group) MultiIndex directly from numpy arrays.
    # Cartesian product via np.repeat / np.tile avoids materialising the
    # intermediate (n_dates × n_groups)-row DataFrame that a cross-join merge
    # would allocate. Row order matches the legacy cross-join: iterate groups
    # within each date, i.e. (d1,g1), (d1,g2), …, (d2,g1), ….
    unique_groups = df[group_cols].drop_duplicates()
    n_dates = len(full_range)
    n_groups = len(unique_groups)
    date_values: np.ndarray = np.repeat(full_range.to_numpy(), n_groups)
    group_arrays = [np.tile(unique_groups[c].to_numpy(), n_dates) for c in group_cols]
    full_idx = pd.MultiIndex.from_arrays(
        [date_values] + group_arrays,
        names=[date_col] + group_cols,
    )

    # Fast path: set_index → reindex. If duplicate (date, group) rows are
    # present, collapse them by summing — matches the intent of "total demand
    # per period" and yields a unique index that reindex requires. Pipelines
    # that run clean_zeros first never produce duplicates, so the groupby
    # branch is skipped in practice.
    source = df.set_index([date_col] + group_cols)[value_col]
    if source.index.has_duplicates:
        source = source.groupby(
            level=list(range(len(group_cols) + 1)), sort=False
        ).sum()

    df_filled = source.reindex(full_idx, fill_value=0.0).reset_index()

    return df_filled
