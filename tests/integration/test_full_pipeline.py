"""End-to-end integration tests for the full boa-forecaster pipeline.

Exercises the complete chain from a raw Excel workbook to per-fold metrics::

    data_loader.load_data
        └─▶ preprocessor.clean_zeros
            └─▶ preprocessor.fill_blanks
                └─▶ standardization.weighted_moving_stats_series
                    └─▶ optimizer.optimize_model
                        └─▶ ModelSpec.build_forecaster
                            └─▶ validation.walk_forward_validation

Two model branches are covered:

* **SARIMA** — statistical path (no feature engineering).
* **RandomForest** — ML path that exercises the ``FeatureEngineer`` →
  tree-model → ``recursive_forecast`` trio, which SARIMA does not touch.

A zero-demand SKU is included in the raw fixture to verify ``clean_zeros`` and
an injected spike is included to verify that standardisation dampens outliers
in the series the optimiser eventually sees.

Marked ``integration``; deselect with ``pytest -m "not integration"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")

from boa_forecaster.data_loader import load_data
from boa_forecaster.models.random_forest import RandomForestSpec
from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.optimizer import optimize_model
from boa_forecaster.preprocessor import clean_zeros, fill_blanks
from boa_forecaster.standardization import weighted_moving_stats_series
from boa_forecaster.validation import walk_forward_validation

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

_N_MONTHS: int = 60
_START: pd.Timestamp = pd.Timestamp("2019-01-01")
_COUNTRY: str = "US"
_SKU_CLEAN: int = 1  # well-behaved series (used by RandomForest test)
_SKU_SPIKED: int = 2  # series with an injected outlier (used by SARIMA test)
_SKU_ZERO: int = 9999  # cumulative-zero series — must be removed by clean_zeros
_OUTLIER_POS: int = 30  # row index in _SKU_SPIKED where we inject a 800-unit spike
_OUTLIER_VALUE: float = 800.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monthly_rows(n_months: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic trend + yearly seasonality + noise."""
    t = np.arange(n_months, dtype=float)
    trend = 0.4 * t
    season = 12.0 * np.sin(2 * np.pi * t / 12.0)
    noise = rng.normal(0.0, 1.5, size=n_months)
    return 80.0 + trend + season + noise


def _series_for_sku(df: pd.DataFrame, country: str, sku: int) -> pd.Series:
    """Extract one SKU/country series from the filled DataFrame, freq='MS'."""
    sub = df[(df["Country"] == country) & (df["SKU"] == sku)].sort_values("Date")
    idx = pd.DatetimeIndex(sub["Date"].to_numpy(), freq="MS")
    return pd.Series(sub["CS"].to_numpy(dtype=float), index=idx, name="CS")


def _load_and_preprocess(xlsx_path: Path) -> pd.DataFrame:
    raw = load_data(str(xlsx_path), skip_rows=0)
    cleaned = clean_zeros(raw, group_cols=["Country", "SKU"], value_col="CS")
    return fill_blanks(
        cleaned,
        date_col="Date",
        group_cols=["Country", "SKU"],
        value_col="CS",
        freq="MS",
    )


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped — expensive writes / optimiser runs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xlsx_path(tmp_path_factory) -> Path:
    """Write a realistic monthly sales workbook to a temp ``.xlsx`` file.

    Layout (before any preprocessing):

    * SKU=1, Country=US: 60 months, trend+seasonality+noise (clean).
    * SKU=2, Country=US: 60 months, trend+seasonality+noise, one 800-unit spike.
    * SKU=9999, Country=US: 60 rows of zeros — must be removed by clean_zeros.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range(_START, periods=_N_MONTHS, freq="MS")
    yyyymm = [f"{d.year:04d}{d.month:02d}" for d in dates]

    sku_clean_vals = _make_monthly_rows(_N_MONTHS, rng)
    sku_spiked_vals = _make_monthly_rows(_N_MONTHS, rng)
    sku_spiked_vals[_OUTLIER_POS] = _OUTLIER_VALUE
    sku_zero_vals = np.zeros(_N_MONTHS)

    frames = []
    for sku, values in (
        (_SKU_CLEAN, sku_clean_vals),
        (_SKU_SPIKED, sku_spiked_vals),
        (_SKU_ZERO, sku_zero_vals),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "Date": yyyymm,
                    "SKU": sku,
                    "CS": values,
                    "Country": _COUNTRY,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    path = tmp_path_factory.mktemp("e2e") / "sales.xlsx"
    df.to_excel(path, index=False, sheet_name="Data")
    return path


@pytest.fixture(scope="module")
def prepared(xlsx_path) -> pd.DataFrame:
    """Post-loader + preprocessor DataFrame, shared across test classes."""
    return _load_and_preprocess(xlsx_path)


# ---------------------------------------------------------------------------
# Stage-level assertions: loader + preprocessor + standardisation
# ---------------------------------------------------------------------------


class TestLoaderAndPreprocessor:
    """Non-model stages must produce a clean, gap-free frame per group."""

    def test_zero_sku_removed_by_clean_zeros(self, prepared):
        assert _SKU_ZERO not in prepared["SKU"].unique()

    def test_real_skus_preserved(self, prepared):
        assert {_SKU_CLEAN, _SKU_SPIKED}.issubset(set(prepared["SKU"].unique()))

    def test_each_group_has_full_calendar(self, prepared):
        for sku in (_SKU_CLEAN, _SKU_SPIKED):
            series = _series_for_sku(prepared, _COUNTRY, sku)
            assert len(series) == _N_MONTHS
            assert series.index.is_monotonic_increasing
            # Index equals the canonical month-start range — no gaps, no dupes.
            expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
            assert series.index.equals(expected)

    def test_no_nan_after_fill_blanks(self, prepared):
        assert not prepared["CS"].isna().any()

    def test_row_count_matches_groups_times_dates(self, prepared):
        n_groups = prepared[["Country", "SKU"]].drop_duplicates().shape[0]
        n_dates = prepared["Date"].drop_duplicates().shape[0]
        assert len(prepared) == n_groups * n_dates

    def test_standardization_dampens_outlier(self, prepared):
        series = _series_for_sku(prepared, _COUNTRY, _SKU_SPIKED)
        raw_max = float(series.max())
        _, _, clipped = weighted_moving_stats_series(
            series.to_numpy(), window_size=3, threshold=2.5
        )
        assert clipped.shape == series.shape
        assert (clipped >= 0).all()
        assert float(clipped.max()) < raw_max
        # The injected 800 spike must be clipped below its raw value.
        assert clipped[_OUTLIER_POS] < _OUTLIER_VALUE


# ---------------------------------------------------------------------------
# Full pipeline: SARIMA branch (standardised series, no feature engineering)
# ---------------------------------------------------------------------------


class TestFullPipelineSARIMA:
    """data_loader → preprocessor → standardisation → SARIMA → walk-forward CV."""

    @pytest.fixture(scope="class")
    def pipeline(self, prepared):
        series = _series_for_sku(prepared, _COUNTRY, _SKU_SPIKED)
        _, _, clipped = weighted_moving_stats_series(
            series.to_numpy(), window_size=3, threshold=2.5
        )
        clipped_series = pd.Series(clipped, index=series.index, name="CS")

        spec = SARIMASpec(forecast_horizon=6)
        result = optimize_model(clipped_series, spec, n_calls=3, seed=0)
        forecaster = spec.build_forecaster(result.best_params)
        folds = walk_forward_validation(
            clipped_series,
            forecaster,
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        return {"series": clipped_series, "result": result, "folds": folds}

    def test_optimizer_not_fallback(self, pipeline):
        assert pipeline["result"].is_fallback is False

    def test_optimizer_reports_sarima(self, pipeline):
        assert pipeline["result"].model_name == "sarima"

    def test_optimizer_score_is_finite_nonneg(self, pipeline):
        score = pipeline["result"].best_score
        assert np.isfinite(score)
        assert score >= 0.0

    def test_optimizer_trial_count_matches_budget(self, pipeline):
        assert pipeline["result"].n_trials == 3

    def test_validation_returns_three_folds(self, pipeline):
        folds = pipeline["folds"]
        assert isinstance(folds, pd.DataFrame)
        assert len(folds) == 3
        assert folds["fold"].tolist() == [1, 2, 3]

    def test_validation_metrics_finite_nonneg(self, pipeline):
        folds = pipeline["folds"]
        for col in ("sMAPE", "RMSLE"):
            assert folds[col].notna().all(), f"Fold metric {col} has NaN"
            assert (folds[col] >= 0).all()
            assert np.isfinite(folds[col]).all()

    def test_validation_test_windows_within_series(self, pipeline):
        folds = pipeline["folds"]
        series_end = pipeline["series"].index[-1]
        series_start = pipeline["series"].index[0]
        assert (folds["test_end"] <= series_end).all()
        assert (folds["train_start"] >= series_start).all()


# ---------------------------------------------------------------------------
# Full pipeline: RandomForest branch (with feature engineering)
# ---------------------------------------------------------------------------


class TestFullPipelineRandomForest:
    """End-to-end through FeatureEngineer → tree model → recursive_forecast."""

    @pytest.fixture(scope="class")
    def pipeline(self, prepared):
        # Use the clean SKU (no outlier) for the RF branch so CV metrics reflect
        # feature-engineering quality rather than clipping behaviour.
        series = _series_for_sku(prepared, _COUNTRY, _SKU_CLEAN)

        spec = RandomForestSpec(forecast_horizon=6)
        result = optimize_model(series, spec, n_calls=3, seed=0)
        forecaster = spec.build_forecaster(result.best_params)
        folds = walk_forward_validation(
            series,
            forecaster,
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        return {"series": series, "result": result, "folds": folds}

    def test_optimizer_not_fallback(self, pipeline):
        assert pipeline["result"].is_fallback is False

    def test_optimizer_reports_random_forest(self, pipeline):
        assert pipeline["result"].model_name == "random_forest"

    def test_optimizer_score_is_finite_nonneg(self, pipeline):
        score = pipeline["result"].best_score
        assert np.isfinite(score)
        assert score >= 0.0

    def test_best_params_cover_search_space_keys(self, pipeline):
        expected = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        }
        assert expected.issubset(pipeline["result"].best_params.keys())

    def test_validation_returns_three_folds(self, pipeline):
        folds = pipeline["folds"]
        assert isinstance(folds, pd.DataFrame)
        assert len(folds) == 3

    def test_validation_metrics_finite_nonneg(self, pipeline):
        folds = pipeline["folds"]
        for col in ("sMAPE", "RMSLE"):
            assert folds[col].notna().all(), f"Fold metric {col} has NaN"
            assert (folds[col] >= 0).all()
            assert np.isfinite(folds[col]).all()

    def test_fold_windows_span_six_months(self, pipeline):
        # Each fold's test window spans test_size=6 consecutive months.
        folds = pipeline["folds"]
        for _, row in folds.iterrows():
            span = pd.date_range(row["test_start"], row["test_end"], freq="MS")
            assert len(span) == 6
