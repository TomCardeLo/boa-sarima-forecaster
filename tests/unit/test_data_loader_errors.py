"""Error-path coverage for :mod:`boa_forecaster.data_loader`.

Targets the previously uncovered branches:

- ``data_loader.py:116-118`` — ``except Exception`` around ``pd.read_excel``
  (e.g. non-existent sheet name).
- ``data_loader.py:156-158`` — SKU cast-to-int failure.
- ``data_loader.py:163-165`` — CS cast-to-float failure.
- The flat-workbook branch (both ``SKU`` and ``Country`` missing) that
  auto-injects sentinel values.
- Malformed-date handling surfaces a WARNING rather than silently returning
  NaN rows.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from boa_forecaster.data_loader import load_data

# ---------------------------------------------------------------------------
# Missing / non-existent sheet (covers 116-118)
# ---------------------------------------------------------------------------


class TestLoadDataMissingSheet:
    """A non-existent sheet name must raise a clear ``ValueError``."""

    @pytest.fixture
    def wrong_sheet_xlsx(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202"],
                "SKU": [1, 1],
                "CS": [100.0, 200.0],
                "Country": ["US", "US"],
            }
        )
        path = tmp_path / "only_data_sheet.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        return path

    def test_raises_value_error_on_unknown_sheet(self, wrong_sheet_xlsx):
        with pytest.raises(ValueError):
            load_data(str(wrong_sheet_xlsx), sheet_name="DoesNotExist", skip_rows=0)

    def test_error_log_emitted_on_unknown_sheet(self, wrong_sheet_xlsx, caplog):
        caplog.set_level(logging.ERROR, logger="boa_forecaster.data_loader")
        with pytest.raises(ValueError):
            load_data(str(wrong_sheet_xlsx), sheet_name="DoesNotExist", skip_rows=0)
        assert any(
            "DoesNotExist" in rec.message or "Error reading" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# SKU cast failure (covers 156-158)
# ---------------------------------------------------------------------------


class TestLoadDataSkuCastFailure:
    """Non-int-convertible SKU values must surface as a ``ValueError``."""

    @pytest.fixture
    def bad_sku_xlsx(self, tmp_path):
        # "abc"/"xyz" pass the '##' marker filter but fail .astype(int).
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202"],
                "SKU": ["abc", "xyz"],
                "CS": [100.0, 200.0],
                "Country": ["US", "US"],
            }
        )
        path = tmp_path / "bad_sku.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        return path

    def test_raises_on_uncastable_sku(self, bad_sku_xlsx):
        with pytest.raises((ValueError, TypeError)):
            load_data(str(bad_sku_xlsx), skip_rows=0)

    def test_error_logged_on_sku_cast_failure(self, bad_sku_xlsx, caplog):
        caplog.set_level(logging.ERROR, logger="boa_forecaster.data_loader")
        with pytest.raises((ValueError, TypeError)):
            load_data(str(bad_sku_xlsx), skip_rows=0)
        assert any("SKU" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# CS cast failure (covers 163-165)
# ---------------------------------------------------------------------------


class TestLoadDataCsCastFailure:
    """Non-float-convertible CS values must surface as a ``ValueError``."""

    @pytest.fixture
    def bad_cs_xlsx(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202"],
                "SKU": [1, 2],
                "CS": ["not_a_number", "also_bad"],
                "Country": ["US", "US"],
            }
        )
        path = tmp_path / "bad_cs.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        return path

    def test_raises_on_uncastable_cs(self, bad_cs_xlsx):
        with pytest.raises((ValueError, TypeError)):
            load_data(str(bad_cs_xlsx), skip_rows=0)

    def test_error_logged_on_cs_cast_failure(self, bad_cs_xlsx, caplog):
        caplog.set_level(logging.ERROR, logger="boa_forecaster.data_loader")
        with pytest.raises((ValueError, TypeError)):
            load_data(str(bad_cs_xlsx), skip_rows=0)
        assert any("CS" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Flat-workbook branch: both SKU and Country absent
# ---------------------------------------------------------------------------


class TestLoadDataFlatWorkbook:
    """Workbooks without SKU or Country columns (single-series / flat) must
    auto-inject sentinel values without raising."""

    @pytest.fixture
    def flat_xlsx(self, tmp_path):
        df = pd.DataFrame({"Date": ["202201", "202202"], "CS": [100.0, 200.0]})
        path = tmp_path / "flat.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        return path

    def test_returns_non_empty(self, flat_xlsx):
        result = load_data(str(flat_xlsx), skip_rows=0)
        assert len(result) == 2

    def test_injects_default_sku(self, flat_xlsx):
        result = load_data(str(flat_xlsx), skip_rows=0)
        assert "SKU" in result.columns
        assert (result["SKU"] == 1).all()

    def test_injects_default_country(self, flat_xlsx):
        result = load_data(str(flat_xlsx), skip_rows=0)
        assert "Country" in result.columns
        assert (result["Country"] == "_").all()

    def test_both_defaults_logged(self, flat_xlsx, caplog):
        caplog.set_level(logging.INFO, logger="boa_forecaster.data_loader")
        load_data(str(flat_xlsx), skip_rows=0)
        messages = [rec.message for rec in caplog.records]
        assert any("SKU" in m and "defaulting" in m for m in messages)
        assert any("Country" in m and "defaulting" in m for m in messages)


# ---------------------------------------------------------------------------
# Malformed dates surface via WARNING (not silent NaN)
# ---------------------------------------------------------------------------


class TestLoadDataMalformedDatesObservable:
    """Malformed dates must be dropped AND surfaced via a WARNING log."""

    @pytest.fixture
    def malformed_dates_xlsx(self, tmp_path):
        # "2020-13" is not parseable by the default "%Y%m" format.
        df = pd.DataFrame(
            {
                "Date": ["202201", "2020-13", "202203"],
                "SKU": [1, 1, 1],
                "CS": [100.0, 200.0, 150.0],
                "Country": ["US", "US", "US"],
            }
        )
        path = tmp_path / "malformed_dates.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        return path

    def test_warning_emitted_for_unparseable_dates(self, malformed_dates_xlsx, caplog):
        caplog.set_level(logging.WARNING, logger="boa_forecaster.data_loader")
        result = load_data(str(malformed_dates_xlsx), skip_rows=0)
        # Bad row is dropped — not left as silent NaT.
        assert len(result) == 2
        assert result["Date"].isnull().sum() == 0
        # And the drop was observable via a WARNING.
        assert any(
            "unparseable" in rec.message.lower() for rec in caplog.records
        ), "Expected a WARNING mentioning unparseable dates."
