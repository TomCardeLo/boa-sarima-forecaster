"""Performance benchmark: preprocessor.fill_blanks reindex-based path.

The v2.x ``fill_blanks`` uses ``np.repeat``/``np.tile`` MultiIndex
reindexing in place of a merge-based cross-join.  This benchmark guards
against a regression to the older ``pd.merge`` pattern, which scales
O(n_dates * n_groups) in allocation rather than O(n_rows_output).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.preprocessor import fill_blanks

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def sparse_multi_group_df() -> pd.DataFrame:
    """50 SKUs × 120 months (10 years), ~40% of cells present."""
    rng = np.random.default_rng(7)
    all_dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    rows = []
    for sku in range(50):
        keep = rng.random(len(all_dates)) < 0.4
        dates = all_dates[keep]
        values = rng.normal(100, 5, keep.sum())
        for d, v in zip(dates, values):
            rows.append({"Date": d, "SKU": sku, "CS": float(v)})
    return pd.DataFrame(rows)


def test_bench_fill_blanks_single_group(benchmark, sparse_multi_group_df):
    benchmark(
        fill_blanks,
        sparse_multi_group_df,
        date_col="Date",
        group_cols=["SKU"],
        value_col="CS",
    )


def test_bench_fill_blanks_with_end_date(benchmark, sparse_multi_group_df):
    """Extending the time axis via ``end_date`` grows the output reindex."""
    benchmark(
        fill_blanks,
        sparse_multi_group_df,
        date_col="Date",
        group_cols=["SKU"],
        value_col="CS",
        end_date="2026-12-01",
    )
