"""Unit tests for sarima_bayes.preprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from boa_forecaster.preprocessor import _freq_to_period_alias, clean_zeros, fill_blanks


class TestFreqToPeriodAlias:
    """_freq_to_period_alias must map every supported DateOffset alias."""

    def test_ms_maps_to_m(self):
        assert _freq_to_period_alias("MS") == "M"

    def test_me_maps_to_m(self):
        assert _freq_to_period_alias("ME") == "M"

    def test_m_maps_to_m(self):
        assert _freq_to_period_alias("M") == "M"

    def test_weekly_maps_to_w(self):
        assert _freq_to_period_alias("W") == "W"

    def test_daily_maps_to_d(self):
        assert _freq_to_period_alias("D") == "D"

    def test_hourly_upper_maps_to_lower_h(self):
        assert _freq_to_period_alias("H") == "h"

    def test_hourly_lower_maps_to_lower_h(self):
        assert _freq_to_period_alias("h") == "h"

    def test_minutely_t_maps_to_min(self):
        assert _freq_to_period_alias("T") == "min"

    def test_minutely_min_maps_to_min(self):
        assert _freq_to_period_alias("min") == "min"

    def test_quarterly_qs_maps_to_q(self):
        assert _freq_to_period_alias("QS") == "Q"

    def test_annual_ys_maps_to_y(self):
        assert _freq_to_period_alias("YS") == "Y"

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot map"):
            _freq_to_period_alias("UNKNOWN")


class TestFillBlanksMonthlyDefaultUnchanged:
    """Regression: existing monthly behaviour must be unaffected."""

    def test_monthly_fills_gap(self):
        # Jan and Mar present; Feb is missing
        dates = pd.to_datetime(["2023-01-01", "2023-03-01"])
        df = pd.DataFrame({"Date": dates, "SKU": [1, 1], "CS": [10.0, 20.0]})
        result = fill_blanks(df)
        assert len(result) == 3

    def test_monthly_filled_row_has_zero(self):
        dates = pd.to_datetime(["2023-01-01", "2023-03-01"])
        df = pd.DataFrame({"Date": dates, "SKU": [1, 1], "CS": [10.0, 20.0]})
        result = fill_blanks(df)
        assert (result["CS"] == 0.0).any()

    def test_monthly_no_gaps_unchanged_count(self):
        dates = pd.date_range("2023-01-01", periods=4, freq="MS")
        df = pd.DataFrame({"Date": dates, "SKU": [1] * 4, "CS": [5.0] * 4})
        result = fill_blanks(df)
        assert len(result) == 4


class TestFillBlanksWeekly:
    """fill_blanks must work correctly with freq="W"."""

    def test_weekly_fills_gap(self):
        # Two dates 2 weeks apart — middle week is missing
        dates = pd.to_datetime(["2024-01-01", "2024-01-15"])
        df = pd.DataFrame({"Date": dates, "SKU": [1, 1], "CS": [10.0, 20.0]})
        result = fill_blanks(df, freq="W")
        # Should have 3 rows: weeks of Jan 1, Jan 8, Jan 15
        assert len(result) == 3

    def test_weekly_filled_row_has_zero(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-15"])
        df = pd.DataFrame({"Date": dates, "SKU": [1, 1], "CS": [10.0, 20.0]})
        result = fill_blanks(df, freq="W")
        assert (result["CS"] == 0.0).any()

    def test_weekly_no_gaps_unchanged_count(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="W")
        df = pd.DataFrame({"Date": dates, "SKU": [1] * 4, "CS": [5.0] * 4})
        result = fill_blanks(df, freq="W")
        assert len(result) == 4

    def test_weekly_multiple_groups(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="W")
        rows = []
        for sku in [1, 2]:
            for d in dates:
                rows.append({"Date": d, "SKU": sku, "CS": 1.0})
        df = pd.DataFrame(rows)
        result = fill_blanks(df, group_cols=["SKU"], freq="W")
        assert len(result) == 8  # 4 weeks × 2 groups


class TestFillBlanksMultiGroupCols:
    """fill_blanks must honour multi-column grouping (e.g. Country × SKU)."""

    def test_preserves_existing_combinations_only(self):
        # Input has only (US,1) and (MX,2); expect those combos × full date range,
        # NOT a full Cartesian over Country × SKU.
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "Country": ["US", "MX"],
                "SKU": [1, 2],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df, group_cols=["Country", "SKU"])
        # 2 combos × 2 dates = 4 rows
        assert len(result) == 4
        combos = set(zip(result["Country"], result["SKU"]))
        assert combos == {("US", 1), ("MX", 2)}

    def test_fills_zero_for_missing_date(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "Country": ["US", "MX"],
                "SKU": [1, 2],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df, group_cols=["Country", "SKU"])
        # (US,1) on Feb, (MX,2) on Jan should be filled with 0
        mask_us1_feb = (
            (result["Country"] == "US")
            & (result["SKU"] == 1)
            & (result["Date"] == pd.Timestamp("2023-02-01"))
        )
        mask_mx2_jan = (
            (result["Country"] == "MX")
            & (result["SKU"] == 2)
            & (result["Date"] == pd.Timestamp("2023-01-01"))
        )
        assert result.loc[mask_us1_feb, "CS"].iloc[0] == 0.0
        assert result.loc[mask_mx2_jan, "CS"].iloc[0] == 0.0

    def test_three_group_cols(self):
        # Region × Country × SKU with a single row per combo.
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "Region": ["LATAM", "NA"],
                "Country": ["MX", "US"],
                "SKU": [1, 2],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df, group_cols=["Region", "Country", "SKU"])
        assert len(result) == 4  # 2 combos × 2 dates
        assert list(result.columns) == ["Date", "Region", "Country", "SKU", "CS"]


class TestFillBlanksInvariants:
    """Output shape, dtype, ordering and idempotence guarantees."""

    def test_column_order_single_group(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01"]),
                "SKU": [1],
                "CS": [10.0],
            }
        )
        result = fill_blanks(df)
        assert list(result.columns) == ["Date", "SKU", "CS"]

    def test_value_col_dtype_is_float(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-03-01"]),
                "SKU": [1, 1],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df)
        assert result["CS"].dtype == float

    def test_date_col_is_datetime(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-03-01"]),
                "SKU": [1, 1],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_caller_df_not_mutated(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-03-01"]),
                "SKU": [1, 1],
                "CS": [10.0, 20.0],
            }
        )
        df_before = df.copy()
        _ = fill_blanks(df)
        pd.testing.assert_frame_equal(df, df_before)

    def test_idempotent_on_gap_free_input(self):
        dates = pd.date_range("2023-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"Date": dates, "SKU": [1] * 6, "CS": [1.0, 2, 3, 4, 5, 6]})
        once = fill_blanks(df)
        twice = fill_blanks(once)
        pd.testing.assert_frame_equal(
            once.reset_index(drop=True), twice.reset_index(drop=True)
        )

    def test_end_date_extends_range(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "SKU": [1, 1],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df, end_date="2023-04-01")
        assert len(result) == 4  # Jan, Feb, Mar, Apr
        assert (result["CS"] == 0.0).sum() == 2  # Mar + Apr


class TestFillBlanksDuplicateAggregation:
    """Duplicate (date, group) rows are summed rather than duplicated."""

    def test_duplicates_summed(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "SKU": [1, 1],
                "CS": [10.0, 20.0],
            }
        )
        result = fill_blanks(df)
        assert len(result) == 1
        assert result["CS"].iloc[0] == 30.0

    def test_duplicates_summed_with_gap(self):
        # Two rows on Jan (10 + 20), zero rows on Feb, one row on Mar (5)
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-03-01"]),
                "SKU": [1, 1, 1],
                "CS": [10.0, 20.0, 5.0],
            }
        )
        result = fill_blanks(df)
        assert len(result) == 3
        assert result.sort_values("Date")["CS"].tolist() == [30.0, 0.0, 5.0]


class TestCleanZeros:
    """clean_zeros was previously untested."""

    def test_all_zero_group_removed(self):
        df = pd.DataFrame({"SKU": [1, 1, 2, 2], "CS": [0.0, 0.0, 100.0, 200.0]})
        result = clean_zeros(df)
        assert set(result["SKU"].unique()) == {2}

    def test_nonzero_group_kept(self):
        df = pd.DataFrame({"SKU": [1, 1], "CS": [10.0, 20.0]})
        result = clean_zeros(df)
        assert len(result) == 2

    def test_mixed_group_kept(self):
        # Group has some zeros but sum != 0 — must be kept
        df = pd.DataFrame({"SKU": [1, 1, 1], "CS": [0.0, 0.0, 5.0]})
        result = clean_zeros(df)
        assert len(result) == 3

    def test_all_groups_zero_returns_empty(self):
        df = pd.DataFrame({"SKU": [1, 1, 2, 2], "CS": [0.0, 0.0, 0.0, 0.0]})
        result = clean_zeros(df)
        assert len(result) == 0

    def test_custom_group_cols(self):
        df = pd.DataFrame(
            {
                "Country": ["US", "US", "MX", "MX"],
                "SKU": [1, 1, 1, 1],
                "CS": [0.0, 0.0, 50.0, 50.0],
            }
        )
        result = clean_zeros(df, group_cols=["Country", "SKU"])
        assert list(result["Country"].unique()) == ["MX"]

    def test_returns_copy_not_inplace(self):
        df = pd.DataFrame({"SKU": [1, 1, 2, 2], "CS": [0.0, 0.0, 10.0, 20.0]})
        result = clean_zeros(df)
        assert len(df) == 4  # original unchanged
        assert len(result) == 2
