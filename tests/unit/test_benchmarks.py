"""Unit tests for sarima_bayes.benchmarks."""

import numpy as np
import pandas as pd
import pytest

from sarima_bayes.benchmarks import (
    auto_arima_nixtla,
    ets_model,
    run_benchmark_comparison,
    seasonal_naive,
    summary_table,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_series():
    """36-point monthly series with no zeros or NaNs."""
    rng = np.random.default_rng(99)
    values = 100 + np.arange(36) * 0.3 + rng.normal(0, 2, 36)
    values = np.maximum(values, 1.0)
    index = pd.date_range("2021-01-01", periods=36, freq="MS")
    return pd.Series(values, index=index)


@pytest.fixture
def two_group_df():
    """Synthetic DataFrame with 2 Country/SKU groups, 36 months each."""
    rng = np.random.default_rng(11)
    rows = []
    for country, sku in [("US", 1), ("MX", 2)]:
        values = 50 + np.arange(36) * 0.5 + rng.normal(0, 3, 36)
        values = np.maximum(values, 1.0)
        dates = pd.date_range("2021-01-01", periods=36, freq="MS")
        for d, v in zip(dates, values):
            rows.append({"Country": country, "SKU": sku, "CS": round(v, 1), "Date": d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# seasonal_naive
# ---------------------------------------------------------------------------


class TestSeasonalNaive:
    def test_seasonal_naive_length(self, clean_series):
        result = seasonal_naive(clean_series, forecast_horizon=6)
        assert len(result) == 6

    def test_seasonal_naive_index(self, clean_series):
        result = seasonal_naive(clean_series, forecast_horizon=6)
        assert isinstance(result.index, pd.DatetimeIndex)
        # All future dates must be after the last training date
        assert (result.index > clean_series.index[-1]).all()

    def test_seasonal_naive_no_nan(self, clean_series):
        result = seasonal_naive(clean_series, forecast_horizon=6)
        assert not result.isna().any()

    def test_seasonal_naive_fallback(self):
        """Series shorter than m should not raise and should return valid output."""
        short = pd.Series(
            [10.0, 20.0, 15.0, 18.0, 12.0, 14.0],
            index=pd.date_range("2023-01-01", periods=6, freq="MS"),
        )
        result = seasonal_naive(short, forecast_horizon=4, m=12)
        assert len(result) == 4
        assert not result.isna().any()


# ---------------------------------------------------------------------------
# ets_model
# ---------------------------------------------------------------------------


class TestEtsModel:
    def test_ets_length(self, clean_series):
        result = ets_model(clean_series, forecast_horizon=6)
        assert len(result) == 6

    def test_ets_no_nan(self, clean_series):
        result = ets_model(clean_series, forecast_horizon=6)
        assert not result.isna().any()


# ---------------------------------------------------------------------------
# auto_arima_nixtla
# ---------------------------------------------------------------------------


class TestAutoArimaNixtla:
    @pytest.mark.slow
    def test_auto_arima_length(self, clean_series):
        pytest.importorskip("statsforecast", reason="statsforecast not installed")
        result = auto_arima_nixtla(clean_series, forecast_horizon=6)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# run_benchmark_comparison
# ---------------------------------------------------------------------------


class TestRunBenchmarkComparison:
    def test_run_benchmark_comparison_shape(self, two_group_df):
        # Use a trivial SARIMA stand-in to keep the test fast
        def _dummy_sarima(train: pd.Series) -> pd.Series:
            freq = train.index.freq or "MS"
            idx = pd.date_range(start=train.index[-1], periods=4, freq=freq)[1:]
            return pd.Series([train.iloc[-1]] * 3, index=idx)

        result = run_benchmark_comparison(
            two_group_df,
            group_cols=["Country", "SKU"],
            target_col="CS",
            date_col="Date",
            sarima_model_fn=_dummy_sarima,
            n_folds=3,
            test_size=3,
            min_train_size=12,
        )

        # 4 models × 3 folds × 2 groups = 24 rows
        assert len(result) == 24

        required_cols = {"Country", "SKU", "fold", "model", "sMAPE", "RMSLE"}
        assert required_cols.issubset(result.columns)

    def test_run_benchmark_comparison_model_names(self, two_group_df):
        def _dummy_sarima(train: pd.Series) -> pd.Series:
            freq = train.index.freq or "MS"
            idx = pd.date_range(start=train.index[-1], periods=4, freq=freq)[1:]
            return pd.Series([train.iloc[-1]] * 3, index=idx)

        result = run_benchmark_comparison(
            two_group_df,
            group_cols=["Country", "SKU"],
            target_col="CS",
            date_col="Date",
            sarima_model_fn=_dummy_sarima,
            n_folds=3,
            test_size=3,
            min_train_size=12,
        )
        expected_models = {"SARIMA+BO", "seasonal_naive", "ETS", "AutoARIMA"}
        assert set(result["model"].unique()) == expected_models


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_summary_table_beats_naive(self, two_group_df):
        def _dummy_sarima(train: pd.Series) -> pd.Series:
            freq = train.index.freq or "MS"
            idx = pd.date_range(start=train.index[-1], periods=4, freq=freq)[1:]
            return pd.Series([train.iloc[-1]] * 3, index=idx)

        results = run_benchmark_comparison(
            two_group_df,
            group_cols=["Country", "SKU"],
            target_col="CS",
            date_col="Date",
            sarima_model_fn=_dummy_sarima,
            n_folds=3,
            test_size=3,
            min_train_size=12,
        )
        tbl = summary_table(results, group_cols=["Country", "SKU"])
        assert "beats_naive" in tbl.columns
        assert tbl["beats_naive"].dtype == bool

    def test_summary_table_columns(self, two_group_df):
        def _dummy_sarima(train: pd.Series) -> pd.Series:
            freq = train.index.freq or "MS"
            idx = pd.date_range(start=train.index[-1], periods=4, freq=freq)[1:]
            return pd.Series([train.iloc[-1]] * 3, index=idx)

        results = run_benchmark_comparison(
            two_group_df,
            group_cols=["Country", "SKU"],
            target_col="CS",
            date_col="Date",
            sarima_model_fn=_dummy_sarima,
            n_folds=3,
            test_size=3,
            min_train_size=12,
        )
        tbl = summary_table(results, group_cols=["Country", "SKU"])
        required = {"sMAPE_mean", "sMAPE_std", "RMSLE_mean", "RMSLE_std", "beats_naive"}
        assert required.issubset(tbl.columns)


# ---------------------------------------------------------------------------
# Tests for new freq / m parameters
# ---------------------------------------------------------------------------


class TestEtsModelWithM:
    """ets_model must respect the m parameter."""

    def test_ets_custom_m_length(self, clean_series):
        # m=4 fits easily within a 36-point series
        result = ets_model(clean_series, forecast_horizon=4, m=4)
        assert len(result) == 4

    def test_ets_custom_m_no_nan(self, clean_series):
        result = ets_model(clean_series, forecast_horizon=4, m=4)
        assert not result.isna().any()

    def test_ets_fallback_uses_m(self):
        # A very short series forces the ETS fallback to seasonal_naive;
        # the fallback must honour m so the returned length is still correct.
        short = pd.Series(
            [10.0, 12.0, 11.0, 13.0],
            index=pd.date_range("2023-01-01", periods=4, freq="MS"),
        )
        result = ets_model(short, forecast_horizon=3, m=4)
        assert len(result) == 3
        assert not result.isna().any()


class TestAutoArimaNixtlaWithM:
    """auto_arima_nixtla must accept m and freq parameters."""

    @pytest.mark.slow
    def test_auto_arima_custom_m_length(self, clean_series):
        pytest.importorskip("statsforecast")
        result = auto_arima_nixtla(clean_series, forecast_horizon=4, m=4, freq="MS")
        assert len(result) == 4


class TestRunBenchmarkComparisonFreq:
    """run_benchmark_comparison must forward freq and m to all baselines."""

    def test_explicit_defaults_match_implicit(self, two_group_df):
        # Passing m=12, freq="MS" explicitly must produce the same row count
        # as calling without those args (the old implicit defaults).
        def _dummy(train: pd.Series) -> pd.Series:
            _f = train.index.freq or "MS"
            idx = pd.date_range(start=train.index[-1], periods=4, freq=_f)[1:]
            return pd.Series([train.iloc[-1]] * 3, index=idx)

        result = run_benchmark_comparison(
            two_group_df,
            group_cols=["Country", "SKU"],
            target_col="CS",
            date_col="Date",
            sarima_model_fn=_dummy,
            n_folds=3,
            test_size=3,
            min_train_size=12,
            m=12,
            freq="MS",
        )
        # 4 models × 3 folds × 2 groups = 24 rows
        assert len(result) == 24

    def test_accepts_freq_param_without_error(self, two_group_df):
        def _dummy(train: pd.Series) -> pd.Series:
            _f = train.index.freq or "MS"
            idx = pd.date_range(start=train.index[-1], periods=4, freq=_f)[1:]
            return pd.Series([train.iloc[-1]] * 3, index=idx)

        # Should not raise even when freq is explicitly supplied
        result = run_benchmark_comparison(
            two_group_df,
            group_cols=["Country", "SKU"],
            target_col="CS",
            date_col="Date",
            sarima_model_fn=_dummy,
            n_folds=3,
            test_size=3,
            min_train_size=12,
            freq="MS",
        )
        assert "model" in result.columns
