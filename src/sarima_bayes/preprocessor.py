"""Data preprocessing utilities for monthly sales time series.

This module provides two operations applied before model fitting:

1. ``clean_zeros`` – removes SKU series with zero cumulative demand.
2. ``fill_blanks`` – fills missing calendar months with zero demand.

All functions operate on ``pd.DataFrame`` inputs and return copies so that
the caller's original data is never modified in-place.
"""

from __future__ import annotations

import logging

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
    raises on series with missing periods).  This function uses a MultiIndex
    ``unstack / reindex / stack`` pattern to efficiently introduce zeros for
    any ``(date, group)`` combination absent from the data.

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

    # Build full index: all dates × all group combinations that exist in data.
    # Using a cross join (via a temporary key) is robust for any number of
    # group columns and avoids the unstack/reindex MultiIndex issue.
    full_dates = pd.DataFrame({date_col: full_range, "_key": 1})
    unique_groups = df[group_cols].drop_duplicates().copy()
    unique_groups["_key"] = 1
    full_index = full_dates.merge(unique_groups, on="_key").drop(columns="_key")

    # Left join to the original data; missing combinations get NaN → 0.
    df_filled = full_index.merge(
        df[[date_col] + group_cols + [value_col]],
        on=[date_col] + group_cols,
        how="left",
    )
    df_filled[value_col] = df_filled[value_col].fillna(0)

    return df_filled
