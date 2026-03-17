"""Data preprocessing utilities for monthly sales time series.

This module provides three operations applied before model fitting:

1. :func:`clean_zeros`           – removes SKU series with zero cumulative demand.
2. :func:`fill_blanks`           – fills missing calendar months with zero demand.
3. :func:`merge_representatives` – consolidates sales under representative SKUs.

All functions operate on ``pd.DataFrame`` inputs and return copies so that
the caller's original data is never modified in-place.
"""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def clean_zeros(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
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
    group_cols: Optional[List[str]] = None,
    value_col: str = "CS",
    end_date: Optional[str] = None,
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
    # Normalise all dates to period start so "2022-01-15" and "2022-01-01"
    # land in the same monthly bucket before reindexing.
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()

    target_end = (
        pd.to_datetime(end_date).to_period("M").to_timestamp()
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


def merge_representatives(
    bd: pd.DataFrame,
    bd_rep: pd.DataFrame,
    tradicional: bool = True,
) -> pd.DataFrame:
    """Consolidate sales of representative SKUs.

    In some product hierarchies, multiple detailed SKUs are grouped under a
    single "representative" SKU for forecasting purposes.  This function
    performs a left-join on the mapping table and sums demand by the
    representative SKU identifier.

    Args:
        bd: Historical sales DataFrame with columns
            ``["Country", "Date", "SKU", "CS"]`` (and optionally
            ``"Forecast group"``).
        bd_rep: Mapping DataFrame with columns
            ``["Country", "SKU", "To SKU"]``.  Rows where ``"To SKU"`` is
            ``NaN`` are treated as self-mapping (the SKU is its own
            representative).
        tradicional: If ``True``, groups by ``["Country", "Date", "To SKU"]``.
            If ``False``, also groups by ``"Forecast group"``.
            Defaults to ``True``.

    Returns:
        Consolidated DataFrame with ``"SKU"`` replaced by the representative
        value and ``"CS"`` summed within each group.

    Example:
        >>> merge_representatives(sales_df, mapping_df, tradicional=True)
    """
    bd = bd.merge(bd_rep, how="left", on=["Country", "SKU"])

    # SKUs not present in the mapping table self-map (NaN → original SKU)
    bd["To SKU"] = bd["To SKU"].fillna(bd["SKU"]).astype(int)

    group_cols = ["Country", "Date", "To SKU"]
    if not tradicional:
        group_cols.append("Forecast group")

    bd = bd.groupby(group_cols, as_index=False)[["CS"]].sum()

    return bd.rename(columns={"To SKU": "SKU"})
