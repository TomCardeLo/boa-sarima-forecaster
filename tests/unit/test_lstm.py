"""Tests for LSTMSpec — Track H3.

Requires torch: collected/run only when torch is importable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytestmark = [pytest.mark.slow, pytest.mark.requires_torch]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_series() -> pd.Series:
    """36-point monthly series with sinusoidal pattern + small noise."""
    rng = np.random.default_rng(0)
    t = np.arange(36)
    values = 100 + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 36)
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    return pd.Series(values, index=idx, name="test_sku")


@pytest.fixture
def tiny_params() -> dict:
    """Tiny hyperparameters to keep tests fast on CPU."""
    return {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "n_epochs": 5,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixed_trial(params: dict):
    """Return an Optuna FixedTrial for the given params dict."""
    import optuna

    return optuna.trial.FixedTrial(params)


# ---------------------------------------------------------------------------
# Import & protocol checks
# ---------------------------------------------------------------------------


def test_import() -> None:
    """LSTMSpec is importable from top-level package."""
    from boa_forecaster import LSTMSpec  # noqa: F401

    assert LSTMSpec is not None


def test_isinstance_model_spec() -> None:
    """LSTMSpec instance satisfies the ModelSpec Protocol."""
    from boa_forecaster import LSTMSpec
    from boa_forecaster.models.base import ModelSpec

    spec = LSTMSpec()
    assert isinstance(spec, ModelSpec)


# ---------------------------------------------------------------------------
# Class-attribute checks
# ---------------------------------------------------------------------------


def test_name_attribute() -> None:
    from boa_forecaster import LSTMSpec

    assert LSTMSpec.name == "lstm"


def test_needs_features_false() -> None:
    from boa_forecaster import LSTMSpec

    assert LSTMSpec.needs_features is False


def test_uses_early_stopping_true() -> None:
    from boa_forecaster import LSTMSpec

    assert LSTMSpec.uses_early_stopping is True


# ---------------------------------------------------------------------------
# search_space
# ---------------------------------------------------------------------------


def test_search_space_keys() -> None:
    from boa_forecaster import LSTMSpec

    space = LSTMSpec().search_space
    expected_keys = {
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "n_epochs",
    }
    assert set(space.keys()) == expected_keys


def test_search_space_types() -> None:
    from boa_forecaster import LSTMSpec
    from boa_forecaster.models.base import FloatParam, IntParam

    space = LSTMSpec().search_space
    assert isinstance(space["hidden_size"], IntParam)
    assert isinstance(space["num_layers"], IntParam)
    assert isinstance(space["dropout"], FloatParam)
    assert isinstance(space["learning_rate"], FloatParam)
    assert isinstance(space["n_epochs"], IntParam)


def test_search_space_bounds() -> None:
    from boa_forecaster import LSTMSpec

    space = LSTMSpec().search_space
    assert space["hidden_size"].low == 16
    assert space["hidden_size"].high == 128
    assert space["num_layers"].low == 1
    assert space["num_layers"].high == 3
    assert space["dropout"].low == 0.0
    assert space["dropout"].high == 0.4
    assert space["learning_rate"].low == pytest.approx(1e-4)
    assert space["learning_rate"].high == pytest.approx(1e-2)
    assert space["learning_rate"].log is True
    assert space["n_epochs"].low == 10
    assert space["n_epochs"].high == 100


# ---------------------------------------------------------------------------
# warm_starts
# ---------------------------------------------------------------------------


def test_warm_starts_length() -> None:
    from boa_forecaster import LSTMSpec

    ws = LSTMSpec().warm_starts
    assert isinstance(ws, list)
    assert len(ws) >= 2


def test_warm_starts_cover_search_space() -> None:
    """Every warm-start dict must cover every search-space key."""
    from boa_forecaster import LSTMSpec

    spec = LSTMSpec()
    space_keys = set(spec.search_space.keys())
    for ws in spec.warm_starts:
        assert space_keys.issubset(
            set(ws.keys())
        ), f"warm_start missing keys: {space_keys - set(ws.keys())}"


# ---------------------------------------------------------------------------
# suggest_params
# ---------------------------------------------------------------------------


def test_suggest_params_returns_all_keys(tiny_params) -> None:
    from boa_forecaster import LSTMSpec

    spec = LSTMSpec()
    trial = _make_fixed_trial(tiny_params)
    result = spec.suggest_params(trial)
    assert set(result.keys()) == set(spec.search_space.keys())


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_finite(train_series, tiny_params) -> None:
    from boa_forecaster import LSTMSpec
    from boa_forecaster.metrics import combined_metric

    spec = LSTMSpec(window_size=6, batch_size=8)
    score = spec.evaluate(train_series, tiny_params, combined_metric)
    assert np.isfinite(score)


def test_evaluate_degenerate_series_returns_penalty() -> None:
    """Series shorter than window_size should return OPTIMIZER_PENALTY, not crash."""
    from boa_forecaster import LSTMSpec
    from boa_forecaster.config import OPTIMIZER_PENALTY
    from boa_forecaster.metrics import combined_metric

    # window_size defaults to 12; 5-point series is degenerate
    idx = pd.date_range("2021-01-01", periods=5, freq="MS")
    short = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=idx)
    spec = LSTMSpec()
    score = spec.evaluate(
        short,
        {
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 1e-3,
            "n_epochs": 3,
        },
        combined_metric,
    )
    assert score == OPTIMIZER_PENALTY


# ---------------------------------------------------------------------------
# build_forecaster
# ---------------------------------------------------------------------------


def test_forecast_length(train_series, tiny_params) -> None:
    from boa_forecaster import LSTMSpec

    horizon = 6
    spec = LSTMSpec(forecast_horizon=horizon, window_size=6, batch_size=8)
    forecaster = spec.build_forecaster(tiny_params)
    forecast = forecaster(train_series)
    assert len(forecast) == horizon


def test_forecast_is_finite(train_series, tiny_params) -> None:
    from boa_forecaster import LSTMSpec

    spec = LSTMSpec(forecast_horizon=6, window_size=6, batch_size=8)
    forecaster = spec.build_forecaster(tiny_params)
    forecast = forecaster(train_series)
    assert np.isfinite(forecast.values).all()


def test_forecast_within_3sigma(train_series) -> None:
    """Forecast with more training epochs should land within ±3σ of the mean."""
    from boa_forecaster import LSTMSpec

    # Use slightly more training than tiny_params so the model learns the scale.
    params = {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "n_epochs": 30,
    }
    spec = LSTMSpec(forecast_horizon=6, window_size=6, batch_size=8)
    forecaster = spec.build_forecaster(params)
    forecast = forecaster(train_series)
    mu = float(train_series.mean())
    sigma = float(train_series.std())
    assert (
        (forecast.values >= mu - 3 * sigma) & (forecast.values <= mu + 3 * sigma)
    ).all()


def test_forecast_index_follows_train(train_series, tiny_params) -> None:
    """Forecast index must be strictly after the last training observation."""
    from boa_forecaster import LSTMSpec

    spec = LSTMSpec(forecast_horizon=6, window_size=6, batch_size=8)
    forecaster = spec.build_forecaster(tiny_params)
    forecast = forecaster(train_series)
    assert forecast.index[0] > train_series.index[-1]


def test_forecast_index_freq(train_series, tiny_params) -> None:
    """Forecast index frequency must match training index frequency."""
    from boa_forecaster import LSTMSpec

    spec = LSTMSpec(forecast_horizon=6, window_size=6, batch_size=8)
    forecaster = spec.build_forecaster(tiny_params)
    forecast = forecaster(train_series)
    inferred = pd.infer_freq(forecast.index)
    train_freq = pd.infer_freq(train_series.index)
    assert inferred == train_freq


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism(train_series, tiny_params) -> None:
    """Two LSTMSpec(seed=42) forecasts on same data must be element-wise equal."""
    from boa_forecaster import LSTMSpec

    spec_a = LSTMSpec(forecast_horizon=6, window_size=6, batch_size=8, seed=42)
    spec_b = LSTMSpec(forecast_horizon=6, window_size=6, batch_size=8, seed=42)
    fc_a = spec_a.build_forecaster(tiny_params)(train_series)
    fc_b = spec_b.build_forecaster(tiny_params)(train_series)
    np.testing.assert_allclose(fc_a.values, fc_b.values, atol=1e-6)


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def test_device_auto_accepted() -> None:
    from boa_forecaster import LSTMSpec

    spec = LSTMSpec(device="auto")
    assert spec is not None


def test_device_invalid_raises() -> None:
    from boa_forecaster import LSTMSpec

    with pytest.raises(ValueError, match="device"):
        LSTMSpec(device="gpu")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_lstm() -> None:
    from boa_forecaster.models import MODEL_REGISTRY

    assert "lstm" in MODEL_REGISTRY
    from boa_forecaster import LSTMSpec

    assert MODEL_REGISTRY["lstm"] is LSTMSpec


def test_get_model_spec_returns_lstm_instance() -> None:
    from boa_forecaster.models import get_model_spec

    spec = get_model_spec("lstm")
    from boa_forecaster import LSTMSpec

    assert isinstance(spec, LSTMSpec)
