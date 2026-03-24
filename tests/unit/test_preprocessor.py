"""Unit tests for sarima_bayes.preprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from sarima_bayes.preprocessor import _freq_to_period_alias, fill_blanks


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
