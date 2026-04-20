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

    def test_rejects_inf_at_position_zero(self):
        # Regression test for the np.isinf(to_numpy()).any() refactor —
        # ensures the vectorised check catches an Inf at the very first
        # element (the old isin([inf, -inf]).any() path already did; this
        # guards against future short-circuit mistakes).
        dates = pd.date_range("2020-01", periods=30, freq="MS")
        s = pd.Series(np.ones(30), index=dates)
        s.iloc[0] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            optimize_model(s, SARIMASpec(), n_calls=1)

    def test_rejects_negative_inf(self):
        dates = pd.date_range("2020-01", periods=30, freq="MS")
        s = pd.Series(np.ones(30), index=dates)
        s.iloc[3] = -np.inf
        with pytest.raises(ValueError, match="Inf"):
            optimize_model(s, SARIMASpec(), n_calls=1)

    def test_accepts_valid_series(self, synthetic_series):
        result = optimize_model(synthetic_series, SARIMASpec(), n_calls=2)
        assert result is not None


# ---------------------------------------------------------------------------
# E4 — MedianPruner activation
# ---------------------------------------------------------------------------


class TestMedianPruner:
    """Optuna ``MedianPruner`` should discard obviously-bad trials early.

    We synthesise a noisy series where many SARIMA parameterisations are
    very bad, so the pruner has ample material to prune against.  The key
    assertions:

    1. At least one trial reaches state ``PRUNED`` after ~30 trials.
    2. A study that successfully returns (any mix of COMPLETE + PRUNED) is
       **not** a fallback — pruning is normal, not a crash.
    """

    def _study(self, synthetic_series):
        # 40 trials beats the n_startup_trials=5 + n_warmup_steps=1 defaults
        # by enough margin that pruning is near-deterministic on noisy data.
        spec = SARIMASpec()
        result = optimize_model(synthetic_series, spec, n_calls=40, seed=7)
        return result

    def test_median_pruner_discards_bad_trials(self, long_series):
        """With ~30 trials on a noisy series, ≥1 trial should be PRUNED.

        Uses ``RandomForestSpec`` because it reports per-fold (multiple
        ``trial.report`` calls) whereas SARIMA has only one evaluation
        step and can never reach the ``n_warmup_steps=1`` threshold.
        """
        pytest.importorskip("sklearn")

        import optuna
        from optuna.samplers import TPESampler

        from boa_forecaster.features import FeatureConfig
        from boa_forecaster.metrics import build_combined_metric
        from boa_forecaster.models.random_forest import RandomForestSpec

        # Keep the feature matrix small so 30 trials stay fast.
        cfg = FeatureConfig(
            lag_periods=[1, 2, 3],
            rolling_windows=[3],
            include_calendar=False,
            include_trend=False,
        )
        spec = RandomForestSpec(feature_config=cfg, forecast_horizon=6)
        metric_fn = build_combined_metric(
            [{"metric": "smape", "weight": 0.7}, {"metric": "rmsle", "weight": 0.3}]
        )

        def objective(trial):
            params = spec.suggest_params(trial)
            return spec.evaluate(long_series, params, metric_fn, trial=trial)

        sampler = TPESampler(seed=7, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )
        for w in spec.warm_starts:
            study.enqueue_trial(w)
        study.optimize(objective, n_trials=30)

        pruned = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)
        )
        assert len(pruned) >= 1, (
            f"expected at least one pruned trial, got states "
            f"{[t.state.name for t in study.trials]}"
        )

    def test_pruned_study_is_not_a_fallback(self, long_series):
        """A study that completes normally (with or without prunes) is NOT a fallback.

        Pruning is expected flow-control; ``is_fallback`` must stay ``False``
        so downstream callers don't treat pruned studies as crashed studies.
        """
        pytest.importorskip("sklearn")

        from boa_forecaster.features import FeatureConfig
        from boa_forecaster.models.random_forest import RandomForestSpec

        cfg = FeatureConfig(
            lag_periods=[1, 2, 3],
            rolling_windows=[3],
            include_calendar=False,
            include_trend=False,
        )
        spec = RandomForestSpec(feature_config=cfg, forecast_horizon=6)
        result = optimize_model(long_series, spec, n_calls=20, seed=7)
        assert result.is_fallback is False
