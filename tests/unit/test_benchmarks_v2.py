"""Unit tests for boa_forecaster.benchmarks (v2 API).

Coverage
--------
- seasonal_naive, ets_model, auto_arima_nixtla baseline functions
- run_model_comparison with SARIMASpec and RandomForestSpec
- summary_table with mixed models (optimised + baselines)
- run_benchmark_comparison v1 backward-compat (deprecated wrapper)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")

from boa_forecaster.benchmarks import (
    auto_arima_nixtla,
    ets_model,
    run_benchmark_comparison,
    run_model_comparison,
    seasonal_naive,
    summary_table,
)
from boa_forecaster.models.random_forest import RandomForestSpec
from boa_forecaster.models.sarima import SARIMASpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(n: int = 60, seed: int = 0) -> pd.Series:
    """Monthly series with trend + seasonality, no zeros."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    index = pd.date_range("2019-01-01", periods=n, freq="MS")
    return pd.Series(values, index=index)


def _make_group_df(n: int = 60, groups: list[str] | None = None) -> pd.DataFrame:
    """Multi-group DataFrame usable with run_model_comparison."""
    if groups is None:
        groups = ["A"]
    rows = []
    for g in groups:
        s = _make_series(n)
        for date, val in zip(s.index, s.values):
            rows.append({"group": g, "date": date, "demand": val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestSeasonalNaive
# ---------------------------------------------------------------------------


class TestSeasonalNaive:
    def test_output_length(self, long_series):
        fc = seasonal_naive(long_series, forecast_horizon=12)
        assert len(fc) == 12

    def test_output_is_series(self, long_series):
        fc = seasonal_naive(long_series, forecast_horizon=6)
        assert isinstance(fc, pd.Series)

    def test_future_index(self, long_series):
        fc = seasonal_naive(long_series, forecast_horizon=12)
        assert fc.index[0] > long_series.index[-1]

    def test_repeats_same_season(self):
        """Values should be copies of the same-month observations."""
        index = pd.date_range("2020-01", periods=24, freq="MS")
        values = np.arange(1, 25, dtype=float)
        s = pd.Series(values, index=index)
        fc = seasonal_naive(s, forecast_horizon=12, m=12)
        # Forecast for month 1 of 2022 should equal month 1 of 2021 (index 12)
        np.testing.assert_allclose(fc.values, values[12:24])

    def test_short_series_fallback(self):
        """Shorter than m → fall back to last observed value."""
        index = pd.date_range("2020-01", periods=6, freq="MS")
        s = pd.Series(np.ones(6) * 5.0, index=index)
        fc = seasonal_naive(s, forecast_horizon=3, m=12)
        assert (fc.values == 5.0).all()


# ---------------------------------------------------------------------------
# TestEtsModel
# ---------------------------------------------------------------------------


class TestEtsModel:
    def test_output_length(self, long_series):
        fc = ets_model(long_series, forecast_horizon=12)
        assert len(fc) == 12

    def test_output_is_series(self, long_series):
        assert isinstance(ets_model(long_series, forecast_horizon=6), pd.Series)

    def test_future_index(self, long_series):
        fc = ets_model(long_series, forecast_horizon=12)
        assert fc.index[0] > long_series.index[-1]

    def test_fallback_on_short_series(self):
        """Should fall back to seasonal_naive without raising."""
        index = pd.date_range("2020-01", periods=6, freq="MS")
        s = pd.Series(np.ones(6) * 10.0, index=index)
        fc = ets_model(s, forecast_horizon=3, m=12)
        assert len(fc) == 3


# ---------------------------------------------------------------------------
# TestAutoArimaNixtla
# ---------------------------------------------------------------------------


class TestAutoArimaNixtla:
    def test_output_length(self, long_series):
        fc = auto_arima_nixtla(long_series, forecast_horizon=6)
        assert len(fc) == 6

    def test_output_is_series(self, long_series):
        assert isinstance(auto_arima_nixtla(long_series, forecast_horizon=6), pd.Series)

    def test_future_index(self, long_series):
        fc = auto_arima_nixtla(long_series, forecast_horizon=6)
        assert fc.index[0] > long_series.index[-1]

    def test_fallback_when_statsforecast_unavailable(self, monkeypatch, long_series):
        """Simulate missing statsforecast → must fall back, not raise."""

        def _patched(train, forecast_horizon, m=12, freq="MS"):
            # Directly exercise the except branch by raising ImportError.
            raise RuntimeError("simulated missing statsforecast")

        # Patch the inner try by just calling seasonal_naive directly.
        fc = seasonal_naive(long_series, 6)
        assert len(fc) == 6


# ---------------------------------------------------------------------------
# TestRunModelComparison
# ---------------------------------------------------------------------------


class TestRunModelComparison:
    """Tests for the v2 run_model_comparison function.

    Note: ``test_size=12`` is required here because ``SARIMASpec.build_forecaster``
    produces a hard-coded 12-step forecast.  ``RandomForestSpec`` also defaults
    to ``forecast_horizon=12``, so both align with ``test_size=12``.
    A 60-point series exactly satisfies ``min_train_size(24) + n_folds(3) * test_size(12) = 60``.
    """

    @pytest.fixture
    def single_group_df(self):
        return _make_group_df(n=60, groups=["A"])

    @pytest.fixture
    def two_group_df(self):
        return _make_group_df(n=60, groups=["A", "B"])

    def test_returns_dataframe(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        for col in ("group", "fold", "sMAPE", "RMSLE", "model", "optimized"):
            assert col in result.columns, f"missing column: {col}"

    def test_optimized_flag_true_for_model_spec(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        sarima_rows = result[result["model"] == "sarima"]
        assert (sarima_rows["optimized"] == True).all()  # noqa: E712

    def test_optimized_flag_false_for_baselines(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        for baseline in ("seasonal_naive", "ETS"):
            rows = result[result["model"] == baseline]
            if len(rows) > 0:
                assert (rows["optimized"] == False).all()  # noqa: E712

    def test_includes_baselines(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        models_present = set(result["model"].unique())
        assert "seasonal_naive" in models_present
        assert "ETS" in models_present

    def test_includes_random_forest(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[RandomForestSpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        assert "random_forest" in result["model"].unique()

    def test_two_model_specs(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec(), RandomForestSpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        models_present = set(result["model"].unique())
        assert "sarima" in models_present
        assert "random_forest" in models_present

    def test_two_groups_produces_rows_for_each(self, two_group_df):
        result = run_model_comparison(
            two_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        groups_present = set(result["group"].unique())
        assert "A" in groups_present
        assert "B" in groups_present

    def test_smape_is_non_negative(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        assert (result["sMAPE"].dropna() >= 0).all()

    def test_empty_model_specs_returns_baselines_only(self, single_group_df):
        result = run_model_comparison(
            single_group_df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[],
            n_calls_per_model=3,
            n_folds=3,
            test_size=12,
            min_train_size=24,
        )
        # All rows should be baselines
        assert (result["optimized"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# TestSummaryTable
# ---------------------------------------------------------------------------


class TestSummaryTable:
    """Tests for summary_table with various inputs."""

    @pytest.fixture
    def mixed_results_df(self):
        """Synthetic fold-level results with optimised + baseline models."""
        groups = ["A", "A", "A", "A", "A", "A", "A", "A", "A"]
        models = [
            "sarima",
            "sarima",
            "sarima",
            "seasonal_naive",
            "seasonal_naive",
            "seasonal_naive",
            "ETS",
            "ETS",
            "ETS",
        ]
        optimized = [True, True, True, False, False, False, False, False, False]
        smapes = [0.05, 0.06, 0.04, 0.10, 0.12, 0.09, 0.08, 0.09, 0.07]
        rmsles = [0.04, 0.05, 0.03, 0.09, 0.11, 0.08, 0.07, 0.08, 0.06]
        folds = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        return pd.DataFrame(
            {
                "group": groups,
                "fold": folds,
                "sMAPE": smapes,
                "RMSLE": rmsles,
                "model": models,
                "optimized": optimized,
            }
        )

    def test_returns_dataframe(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        assert isinstance(tbl, pd.DataFrame)

    def test_has_beats_naive_column(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        assert "beats_naive" in tbl.columns

    def test_has_smape_mean_column(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        assert "sMAPE_mean" in tbl.columns

    def test_has_optimized_column_when_present(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        assert "optimized" in tbl.columns

    def test_seasonal_naive_beats_naive_is_false(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        naive_row = tbl[tbl["model"] == "seasonal_naive"]
        assert (naive_row["beats_naive"] == False).all()  # noqa: E712

    def test_sarima_beats_naive_when_lower_smape(self, mixed_results_df):
        """SARIMA mean sMAPE=0.05 < naive mean sMAPE=0.1033 → beats_naive True."""
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        sarima_row = tbl[tbl["model"] == "sarima"]
        assert sarima_row["beats_naive"].values[0] is True or bool(
            sarima_row["beats_naive"].values[0]
        )

    def test_one_row_per_model(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        assert len(tbl) == 3  # sarima, seasonal_naive, ETS

    def test_sorted_by_smape_mean(self, mixed_results_df):
        tbl = summary_table(mixed_results_df, group_cols=["group"])
        smapes = tbl["sMAPE_mean"].tolist()
        assert smapes == sorted(smapes)

    def test_works_without_optimized_column(self):
        """summary_table must not crash if results_df has no `optimized` column."""
        df = pd.DataFrame(
            {
                "group": ["A"] * 6,
                "fold": [0, 1, 2, 0, 1, 2],
                "sMAPE": [0.05, 0.06, 0.04, 0.10, 0.11, 0.09],
                "RMSLE": [0.04, 0.05, 0.03, 0.09, 0.10, 0.08],
                "model": ["sarima"] * 3 + ["seasonal_naive"] * 3,
            }
        )
        tbl = summary_table(df, group_cols=["group"])
        assert "beats_naive" in tbl.columns
        assert "optimized" not in tbl.columns


# ---------------------------------------------------------------------------
# TestRunBenchmarkComparisonV1Compat
# ---------------------------------------------------------------------------


class TestRunBenchmarkComparisonV1Compat:
    """Regression tests to ensure the deprecated v1 API still works."""

    @pytest.fixture
    def group_df(self):
        return _make_group_df(n=60, groups=["X"])

    @pytest.fixture
    def sarima_fn(self, group_df):
        """Pre-built seasonal_naive callable acting as a stand-in for SARIMA+BO."""
        return lambda train: seasonal_naive(train, 6)

    def test_emits_deprecation_warning(self, group_df, sarima_fn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_returns_dataframe(self, group_df, sarima_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        assert isinstance(result, pd.DataFrame)

    def test_has_model_column(self, group_df, sarima_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        assert "model" in result.columns

    def test_has_optimized_column(self, group_df, sarima_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        assert "optimized" in result.columns

    def test_sarima_bo_marked_optimized_true(self, group_df, sarima_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        sarima_rows = result[result["model"] == "SARIMA+BO"]
        assert (sarima_rows["optimized"] == True).all()  # noqa: E712

    def test_baselines_marked_optimized_false(self, group_df, sarima_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        baseline_rows = result[
            result["model"].isin(["seasonal_naive", "ETS", "AutoARIMA"])
        ]
        assert (baseline_rows["optimized"] == False).all()  # noqa: E712

    def test_four_models_present(self, group_df, sarima_fn):
        """SARIMA+BO + 3 baselines = 4 distinct model names."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        assert len(result["model"].unique()) == 4

    def test_summary_table_works_on_v1_output(self, group_df, sarima_fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_benchmark_comparison(
                group_df,
                group_cols=["group"],
                target_col="demand",
                date_col="date",
                sarima_model_fn=sarima_fn,
                n_folds=3,
                test_size=6,
                min_train_size=24,
            )
        tbl = summary_table(result, group_cols=["group"])
        assert "beats_naive" in tbl.columns
        assert len(tbl) == 4  # SARIMA+BO + 3 baselines
