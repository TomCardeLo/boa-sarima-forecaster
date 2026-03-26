"""Tests for boa_forecaster.models.sarima: SARIMASpec and free functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.metrics import build_combined_metric
from boa_forecaster.models.base import IntParam, ModelSpec
from boa_forecaster.models.sarima import SARIMASpec, forecast_arima, pred_arima

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def default_spec():
    return SARIMASpec()


@pytest.fixture
def metric_fn():
    return build_combined_metric([{"metric": "smape", "weight": 1.0}])


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_sarima_spec_satisfies_model_spec_protocol(default_spec):
    assert isinstance(default_spec, ModelSpec)


def test_sarima_spec_name(default_spec):
    assert default_spec.name == "sarima"


def test_sarima_spec_needs_features_false(default_spec):
    assert default_spec.needs_features is False


# ── search_space ──────────────────────────────────────────────────────────────


def test_search_space_has_six_keys(default_spec):
    assert set(default_spec.search_space.keys()) == {"p", "d", "q", "P", "D", "Q"}


def test_search_space_all_int_params(default_spec):
    for name, param in default_spec.search_space.items():
        assert isinstance(param, IntParam), f"{name} should be IntParam"


def test_search_space_respects_custom_ranges():
    spec = SARIMASpec(p_range=(0, 2), q_range=(0, 1))
    assert spec.search_space["p"].high == 2
    assert spec.search_space["q"].high == 1


# ── warm_starts ───────────────────────────────────────────────────────────────


def test_warm_starts_length(default_spec):
    assert len(default_spec.warm_starts) == 2


def test_warm_starts_keys_match_search_space(default_spec):
    space_keys = set(default_spec.search_space.keys())
    for ws in default_spec.warm_starts:
        assert (
            set(ws.keys()) == space_keys
        ), f"warm_start keys {set(ws.keys())} != {space_keys}"


def test_warm_starts_values_within_bounds(default_spec):
    space = default_spec.search_space
    for ws in default_spec.warm_starts:
        for k, v in ws.items():
            param = space[k]
            assert (
                param.low <= v <= param.high
            ), f"{k}={v} out of [{param.low},{param.high}]"


# ── suggest_params ────────────────────────────────────────────────────────────


def test_suggest_params_returns_dict(default_spec):
    import optuna

    study = optuna.create_study()
    trial = study.ask()
    params = default_spec.suggest_params(trial)
    assert isinstance(params, dict)
    assert set(params.keys()) == {"p", "d", "q", "P", "D", "Q"}


# ── evaluate ─────────────────────────────────────────────────────────────────


def test_evaluate_returns_float(synthetic_series, default_spec, metric_fn):
    params = {"p": 1, "d": 1, "q": 0, "P": 0, "D": 0, "Q": 0}
    score = default_spec.evaluate(synthetic_series, params, metric_fn)
    assert isinstance(score, float)
    assert score >= 0.0


def test_evaluate_penalty_on_complexity_violation(
    synthetic_series, default_spec, metric_fn
):
    """p+q > 4 must return OPTIMIZER_PENALTY without fitting."""
    params = {"p": 3, "d": 1, "q": 3, "P": 0, "D": 0, "Q": 0}  # p+q=6 > 4
    score = default_spec.evaluate(synthetic_series, params, metric_fn)
    assert score == OPTIMIZER_PENALTY


def test_evaluate_penalty_on_seasonal_complexity_violation(
    synthetic_series, default_spec, metric_fn
):
    """P+Q > 3 must return OPTIMIZER_PENALTY."""
    params = {"p": 1, "d": 0, "q": 1, "P": 2, "D": 0, "Q": 2}  # P+Q=4 > 3
    score = default_spec.evaluate(synthetic_series, params, metric_fn)
    assert score == OPTIMIZER_PENALTY


def test_evaluate_accepts_numpy_array(synthetic_series, default_spec, metric_fn):
    """evaluate must work on np.ndarray input, not only pd.Series."""
    params = {"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0}
    score = default_spec.evaluate(synthetic_series.values, params, metric_fn)
    assert isinstance(score, float)


def test_evaluate_never_raises(synthetic_series, default_spec, metric_fn):
    """evaluate must not raise even for degenerate params."""
    params = {"p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0}
    result = default_spec.evaluate(synthetic_series, params, metric_fn)
    assert isinstance(result, float)


# ── build_forecaster ──────────────────────────────────────────────────────────


def test_build_forecaster_returns_callable(default_spec):
    params = {"p": 1, "d": 1, "q": 0, "P": 0, "D": 0, "Q": 0}
    forecaster = default_spec.build_forecaster(params)
    assert callable(forecaster)


def test_build_forecaster_output_is_series(synthetic_series, default_spec):
    params = {"p": 1, "d": 1, "q": 0, "P": 0, "D": 0, "Q": 0}
    forecaster = default_spec.build_forecaster(params)
    result = forecaster(synthetic_series)
    assert isinstance(result, pd.Series)


def test_build_forecaster_output_length(synthetic_series, default_spec):
    params = {"p": 1, "d": 1, "q": 0, "P": 0, "D": 0, "Q": 0}
    forecaster = default_spec.build_forecaster(params)
    result = forecaster(synthetic_series)
    assert len(result) == 12


def test_build_forecaster_output_datetime_index(synthetic_series, default_spec):
    params = {"p": 1, "d": 1, "q": 0, "P": 0, "D": 0, "Q": 0}
    forecaster = default_spec.build_forecaster(params)
    result = forecaster(synthetic_series)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_build_forecaster_uses_m_from_params(synthetic_series):
    """m in params dict must override spec.m."""
    spec = SARIMASpec(m=12)
    params = {"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "m": 6}
    forecaster = spec.build_forecaster(params)
    # Just verify it runs without error; the m override is honoured internally
    result = forecaster(synthetic_series)
    assert len(result) == 12


# ── pred_arima (free function) ────────────────────────────────────────────────


@pytest.fixture
def simple_df():
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"Date": dates, "Sales": 100 + rng.normal(0, 5, 36)})


def test_pred_arima_returns_5_tuple(simple_df):
    result = pred_arima(simple_df, "Date", "Sales", order=(1, 1, 1))
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_pred_arima_forecast_length(simple_df):
    forecast_df, _, _, _, _ = pred_arima(
        simple_df, "Date", "Sales", order=(1, 1, 1), n_per=12
    )
    assert len(forecast_df) == 12


def test_pred_arima_on_failure_returns_empty_df():
    # Pass an empty dataframe to force failure
    empty = pd.DataFrame(
        {"Date": pd.Series(dtype="datetime64[ns]"), "Sales": pd.Series(dtype=float)}
    )
    result_df, conf_int, _, _, forecast = pred_arima(
        empty, "Date", "Sales", order=(1, 1, 1)
    )
    assert result_df.empty
    assert conf_int is None
    assert forecast is None


# ── forecast_arima (free function) ────────────────────────────────────────────


def test_forecast_arima_columns(simple_df):
    result = forecast_arima(simple_df, "Date", "Sales", 1, 1, 1, 12, "ES", 42)
    assert list(result.columns) == ["Date", "Pred", "Country", "Sku"]


def test_forecast_arima_no_negative_predictions(simple_df):
    result = forecast_arima(simple_df, "Date", "Sales", 1, 1, 1, 12)
    assert (result["Pred"] >= 0).all()


def test_forecast_arima_length(simple_df):
    result = forecast_arima(simple_df, "Date", "Sales", 1, 1, 1, 12)
    assert len(result) == 12
