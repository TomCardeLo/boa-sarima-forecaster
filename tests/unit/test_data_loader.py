"""Unit tests for boa_forecaster.data_loader."""

import pandas as pd
import pytest

from boa_forecaster.data_loader import load_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_xlsx(tmp_path):
    """Minimal valid Excel file with all required columns."""
    df = pd.DataFrame(
        {
            "Date": ["202201", "202202", "202203"],
            "SKU": [1, 1, 1],
            "CS": [100.0, 200.0, 150.0],
            "Country": ["US", "US", "US"],
        }
    )
    path = tmp_path / "sales.xlsx"
    df.to_excel(path, index=False, sheet_name="Data")
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadDataHappyPath:
    def test_returns_dataframe(self, simple_xlsx):
        result = load_data(str(simple_xlsx), skip_rows=0)
        assert isinstance(result, pd.DataFrame)

    def test_date_parsed_as_datetime(self, simple_xlsx):
        result = load_data(str(simple_xlsx), skip_rows=0)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_sku_cast_to_int(self, simple_xlsx):
        result = load_data(str(simple_xlsx), skip_rows=0)
        assert result["SKU"].dtype == int

    def test_cs_cast_to_float(self, simple_xlsx):
        result = load_data(str(simple_xlsx), skip_rows=0)
        assert result["CS"].dtype == float

    def test_row_count(self, simple_xlsx):
        result = load_data(str(simple_xlsx), skip_rows=0)
        assert len(result) == 3

    def test_required_columns_present(self, simple_xlsx):
        result = load_data(str(simple_xlsx), skip_rows=0)
        assert {"Date", "SKU", "CS", "Country"}.issubset(result.columns)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


class TestLoadDataMissingFile:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nonexistent.xlsx"), skip_rows=0)


# ---------------------------------------------------------------------------
# Malformed dates
# ---------------------------------------------------------------------------


class TestLoadDataMalformedDates:
    def test_bad_date_rows_dropped(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "NOTADATE", "202203"],
                "SKU": [1, 1, 1],
                "CS": [100.0, 200.0, 150.0],
                "Country": ["US", "US", "US"],
            }
        )
        path = tmp_path / "bad_dates.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        result = load_data(str(path), skip_rows=0)
        assert len(result) == 2

    def test_all_bad_dates_returns_empty(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["NOTADATE", "ALSOBAD"],
                "SKU": [1, 1],
                "CS": [100.0, 200.0],
                "Country": ["US", "US"],
            }
        )
        path = tmp_path / "all_bad.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        result = load_data(str(path), skip_rows=0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Auto-inject optional columns
# ---------------------------------------------------------------------------


class TestLoadDataAutoInjectColumns:
    def test_missing_sku_defaults_to_1(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202"],
                "CS": [100.0, 200.0],
                "Country": ["US", "US"],
            }
        )
        path = tmp_path / "no_sku.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        result = load_data(str(path), skip_rows=0)
        assert "SKU" in result.columns
        assert (result["SKU"] == 1).all()

    def test_missing_country_defaults_to_underscore(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202"],
                "SKU": [1, 1],
                "CS": [100.0, 200.0],
            }
        )
        path = tmp_path / "no_country.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        result = load_data(str(path), skip_rows=0)
        assert "Country" in result.columns
        assert (result["Country"] == "_").all()


# ---------------------------------------------------------------------------
# Invalid SKU marker
# ---------------------------------------------------------------------------


class TestLoadDataInvalidSkuMarker:
    def test_double_hash_rows_dropped(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202", "202203"],
                "SKU": ["##", 1, 1],
                "CS": [0.0, 200.0, 150.0],
                "Country": ["US", "US", "US"],
            }
        )
        path = tmp_path / "with_marker.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        result = load_data(str(path), skip_rows=0)
        assert len(result) == 2

    def test_custom_marker_dropped(self, tmp_path):
        df = pd.DataFrame(
            {
                "Date": ["202201", "202202"],
                "SKU": ["TOTAL", 1],
                "CS": [0.0, 100.0],
                "Country": ["US", "US"],
            }
        )
        path = tmp_path / "custom_marker.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        result = load_data(str(path), skip_rows=0, invalid_sku_marker="TOTAL")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Missing required columns
# ---------------------------------------------------------------------------


class TestLoadDataMissingRequiredColumns:
    def test_missing_date_raises_key_error(self, tmp_path):
        df = pd.DataFrame({"SKU": [1, 2], "CS": [100.0, 200.0]})
        path = tmp_path / "no_date.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        with pytest.raises(KeyError):
            load_data(str(path), skip_rows=0)


# ---------------------------------------------------------------------------
# skip_rows behaviour
# ---------------------------------------------------------------------------


class TestLoadDataSkipRows:
    def test_skip_rows_removes_metadata(self, tmp_path):
        # First 2 rows are metadata, actual data starts at row 3
        df = pd.DataFrame(
            {
                "Date": ["meta", "meta", "202201", "202202"],
                "SKU": ["meta", "meta", 1, 1],
                "CS": [0, 0, 100.0, 200.0],
                "Country": ["meta", "meta", "US", "US"],
            }
        )
        path = tmp_path / "with_meta.xlsx"
        df.to_excel(path, index=False, sheet_name="Data")
        # skip_rows=2 drops the first 2 data rows (the meta rows)
        result = load_data(str(path), skip_rows=2)
        assert len(result) == 2
        assert result["CS"].tolist() == [100.0, 200.0]
