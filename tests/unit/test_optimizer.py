"""Tests for sarima_bayes.optimizer: optimize_arima, optimize_model validation."""

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.config import DEFAULT_SEASONAL_PERIOD
from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.optimizer import OPTIMIZER_PENALTY, optimize_arima, optimize_model

# Expected full parameter key set after BREAKING CHANGE (seasonal params added)
_EXPECTED_KEYS = {"p", "d", "q", "P", "D", "Q", "m"}


def test_optimize_arima_returns_tuple(synthetic_series):
    result = optimize_arima(synthetic_series.values, n_calls=5)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_optimize_arima_param_keys(synthetic_series):
    best_params, _ = optimize_arima(synthetic_series.values, n_calls=5)
    assert set(best_params.keys()) == _EXPECTED_KEYS


def test_optimize_arima_param_types_and_bounds(synthetic_series):
    best_params, _ = optimize_arima(synthetic_series.values, n_calls=5)
    for key in ("p", "d", "q", "P", "D", "Q"):
        assert isinstance(best_params[key], int), f"{key} should be int"
        assert best_params[key] >= 0, f"{key} should be non-negative"
    # Non-seasonal bounds (narrowed from (0,6) to (0,3))
    assert best_params["p"] <= 3
    assert best_params["d"] <= 2
    assert best_params["q"] <= 3
    # Seasonal bounds
    assert best_params["P"] <= 2
    assert best_params["D"] <= 1
    assert best_params["Q"] <= 2


def test_optimize_arima_m_equals_config_default(synthetic_series):
    best_params, _ = optimize_arima(synthetic_series.values, n_calls=5)
    assert best_params["m"] == DEFAULT_SEASONAL_PERIOD


def test_optimize_arima_m_respects_custom_value(synthetic_series):
    best_params, _ = optimize_arima(synthetic_series.values, n_calls=5, m=6)
    assert best_params["m"] == 6


def test_optimize_arima_best_value_finite(synthetic_series):
    _, best_value = optimize_arima(synthetic_series.values, n_calls=5)
    assert best_value < OPTIMIZER_PENALTY


def test_optimize_arima_no_exception(synthetic_series):
    # Should not raise regardless of what Optuna does internally
    best_params, best_value = optimize_arima(synthetic_series.values, n_calls=5)
    assert best_params is not None
    assert best_value is not None


def test_optimizer_penalty_constant():
    assert OPTIMIZER_PENALTY == 1e6


# ---------------------------------------------------------------------------
# _validate_series — exercised via optimize_model()
# ---------------------------------------------------------------------------


class TestValidateSeries:
    def test_rejects_non_series(self):
        with pytest.raises(TypeError, match="pd.Series"):
            optimize_model([1.0] * 30, SARIMASpec(), n_calls=1)

    def test_rejects_ndarray_not_series(self):
        with pytest.raises(TypeError):
            optimize_model(np.ones(30), SARIMASpec(), n_calls=1)

    def test_rejects_non_datetime_index(self):
        s = pd.Series(np.ones(30))  # default RangeIndex
        with pytest.raises(ValueError, match="DatetimeIndex"):
            optimize_model(s, SARIMASpec(), n_calls=1)

    def test_rejects_too_short(self):
        dates = pd.date_range("2020-01", periods=10, freq="MS")
        s = pd.Series(np.ones(10), index=dates)
        with pytest.raises(ValueError, match="at least"):
            optimize_model(s, SARIMASpec(), n_calls=1)

    def test_rejects_nan(self):
        dates = pd.date_range("2020-01", periods=30, freq="MS")
        s = pd.Series(np.ones(30), index=dates)
        s.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            optimize_model(s, SARIMASpec(), n_calls=1)

    def test_rejects_inf(self):
        dates = pd.date_range("2020-01", periods=30, freq="MS")
        s = pd.Series(np.ones(30), index=dates)
        s.iloc[5] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            optimize_model(s, SARIMASpec(), n_calls=1)

    def test_accepts_valid_series(self, synthetic_series):
        result = optimize_model(synthetic_series, SARIMASpec(), n_calls=2)
        assert result is not None
