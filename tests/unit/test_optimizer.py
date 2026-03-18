"""Tests for sarima_bayes.optimizer: optimize_arima."""

from sarima_bayes.optimizer import optimize_arima


def test_optimize_arima_returns_tuple(synthetic_series):
    result = optimize_arima(synthetic_series.values, n_calls=5)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_optimize_arima_param_keys(synthetic_series):
    best_params, _ = optimize_arima(synthetic_series.values, n_calls=5)
    assert set(best_params.keys()) == {"p", "d", "q"}


def test_optimize_arima_param_types_and_bounds(synthetic_series):
    best_params, _ = optimize_arima(synthetic_series.values, n_calls=5)
    for key in ("p", "d", "q"):
        assert isinstance(best_params[key], int)
        assert best_params[key] >= 0
    assert best_params["p"] <= 6
    assert best_params["d"] <= 2
    assert best_params["q"] <= 6


def test_optimize_arima_best_value_finite(synthetic_series):
    _, best_value = optimize_arima(synthetic_series.values, n_calls=5)
    assert best_value < 1e6


def test_optimize_arima_no_exception(synthetic_series):
    # Should not raise regardless of what Optuna does internally
    best_params, best_value = optimize_arima(synthetic_series.values, n_calls=5)
    assert best_params is not None
    assert best_value is not None
