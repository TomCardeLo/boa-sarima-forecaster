"""Tests for boa_forecaster.optimizer: optimize_model and optimize_arima."""

from __future__ import annotations

import warnings

import optuna
import pytest

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.models.base import IntParam, OptimizationResult
from boa_forecaster.optimizer import optimize_arima, optimize_model

# ── MockModelSpec ──────────────────────────────────────────────────────────────


class MockModelSpec:
    """Trivial model: single IntParam ``k``.  Objective = |k - 5|."""

    name = "mock"
    needs_features = False

    @property
    def search_space(self) -> dict:
        return {"k": IntParam(1, 10)}

    @property
    def warm_starts(self) -> list[dict]:
        return [{"k": 5}]

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {"k": trial.suggest_int("k", 1, 10)}

    def evaluate(self, series, params: dict, metric_fn, feature_config=None) -> float:
        return float(abs(params["k"] - 5))

    def build_forecaster(self, params: dict, feature_config=None):
        return lambda train: train


class AlwaysPenaltySpec:
    """Mock that always returns OPTIMIZER_PENALTY — tests soft failure."""

    name = "penalty_model"
    needs_features = False

    @property
    def search_space(self) -> dict:
        return {"k": IntParam(1, 3)}

    @property
    def warm_starts(self) -> list[dict]:
        return [{"k": 1}]

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {"k": trial.suggest_int("k", 1, 3)}

    def evaluate(self, series, params: dict, metric_fn, feature_config=None) -> float:
        return OPTIMIZER_PENALTY

    def build_forecaster(self, params: dict, feature_config=None):
        return lambda train: train


# ── optimize_model ─────────────────────────────────────────────────────────────


def test_returns_optimization_result(synthetic_series):
    result = optimize_model(synthetic_series, MockModelSpec(), n_calls=3)
    assert isinstance(result, OptimizationResult)


def test_result_model_name(synthetic_series):
    result = optimize_model(synthetic_series, MockModelSpec(), n_calls=3)
    assert result.model_name == "mock"


def test_result_n_trials(synthetic_series):
    result = optimize_model(synthetic_series, MockModelSpec(), n_calls=3)
    assert result.n_trials == 3


def test_result_best_score_non_negative(synthetic_series):
    result = optimize_model(synthetic_series, MockModelSpec(), n_calls=3)
    assert result.best_score >= 0.0


def test_result_best_params_has_k(synthetic_series):
    result = optimize_model(synthetic_series, MockModelSpec(), n_calls=3)
    assert "k" in result.best_params


def test_warm_start_enqueued_as_first_trial(synthetic_series):
    """With n_calls=1 the only trial must use the warm_start params."""
    spec = MockModelSpec()
    result = optimize_model(synthetic_series, spec, n_calls=1)
    # warm_start is {"k": 5}; with 1 trial it must be the best
    assert result.best_params["k"] == 5


def test_soft_failure_no_exception(synthetic_series):
    """AlwaysPenaltySpec must not raise even though all scores = 1e6."""
    result = optimize_model(synthetic_series, AlwaysPenaltySpec(), n_calls=3)
    assert isinstance(result, OptimizationResult)
    assert result.best_score == OPTIMIZER_PENALTY


def test_seed_produces_reproducible_results(synthetic_series):
    spec = MockModelSpec()
    r1 = optimize_model(synthetic_series, spec, n_calls=5, seed=0)
    r2 = optimize_model(synthetic_series, spec, n_calls=5, seed=0)
    assert r1.best_params == r2.best_params
    assert r1.best_score == r2.best_score


def test_different_seeds_may_differ(synthetic_series):
    spec = MockModelSpec()
    r1 = optimize_model(synthetic_series, spec, n_calls=5, seed=0)
    r2 = optimize_model(synthetic_series, spec, n_calls=5, seed=999)
    # With only 5 trials the results might coincide by chance, but the API
    # must work — just check it returns valid objects
    assert isinstance(r1, OptimizationResult)
    assert isinstance(r2, OptimizationResult)


# ── optimize_arima backwards compatibility ─────────────────────────────────────

_EXPECTED_KEYS = {"p", "d", "q", "P", "D", "Q", "m"}


def test_optimize_arima_returns_tuple(synthetic_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = optimize_arima(synthetic_series.values, n_calls=3)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_optimize_arima_param_keys(synthetic_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        best_params, _ = optimize_arima(synthetic_series.values, n_calls=3)
    assert set(best_params.keys()) == _EXPECTED_KEYS


def test_optimize_arima_m_in_params(synthetic_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        best_params, _ = optimize_arima(synthetic_series.values, n_calls=3, m=6)
    assert best_params["m"] == 6


def test_optimize_arima_best_value_is_float(synthetic_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _, best_value = optimize_arima(synthetic_series.values, n_calls=3)
    assert isinstance(best_value, float)


def test_optimize_arima_deprecation_warning(synthetic_series):
    with pytest.warns(DeprecationWarning):
        optimize_arima(synthetic_series.values, n_calls=3)


def test_optimize_arima_accepts_pd_series(synthetic_series):
    """optimize_arima must also accept pd.Series (not just np.ndarray)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        best_params, _ = optimize_arima(synthetic_series, n_calls=3)
    assert set(best_params.keys()) == _EXPECTED_KEYS


def test_optimize_arima_param_types(synthetic_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        best_params, _ = optimize_arima(synthetic_series.values, n_calls=3)
    for key in ("p", "d", "q", "P", "D", "Q"):
        assert isinstance(best_params[key], int), f"{key} must be int"
    assert isinstance(best_params["m"], int)


# ── MODEL_REGISTRY integration ────────────────────────────────────────────────


def test_get_model_spec_sarima():
    from boa_forecaster.models import get_model_spec

    spec = get_model_spec("sarima")
    assert spec.name == "sarima"


def test_get_model_spec_unknown_raises():
    from boa_forecaster.models import get_model_spec

    with pytest.raises(KeyError, match="not registered"):
        get_model_spec("nonexistent_model_xyz")
