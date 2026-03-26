"""Unit tests for boa_forecaster.models.random_forest (RandomForestSpec).

All tests are skipped when scikit-learn is not installed.
The ``long_series`` fixture (60 monthly points) from conftest.py provides
enough data for all 3 cross-validation folds (min_train=24, horizon=12).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.features import FeatureConfig
from boa_forecaster.metrics import combined_metric
from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ModelSpec,
)
from boa_forecaster.models.random_forest import RandomForestSpec

# ── Protocol & metadata ───────────────────────────────────────────────────────


class TestRandomForestSpecProtocol:
    def test_implements_model_spec_protocol(self):
        assert isinstance(RandomForestSpec(), ModelSpec)

    def test_name(self):
        assert RandomForestSpec().name == "random_forest"

    def test_needs_features(self):
        assert RandomForestSpec().needs_features is True

    def test_default_forecast_horizon(self):
        assert RandomForestSpec().forecast_horizon == 12

    def test_custom_forecast_horizon(self):
        spec = RandomForestSpec(forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_default_feature_config(self):
        spec = RandomForestSpec()
        assert isinstance(spec.feature_config, FeatureConfig)


# ── Search space ──────────────────────────────────────────────────────────────


class TestRandomForestSearchSpace:
    def test_search_space_has_five_keys(self):
        space = RandomForestSpec().search_space
        assert len(space) == 5

    def test_search_space_keys(self):
        space = RandomForestSpec().search_space
        assert set(space.keys()) == {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        }

    def test_n_estimators_is_int_param(self):
        assert isinstance(RandomForestSpec().search_space["n_estimators"], IntParam)

    def test_max_depth_is_int_param(self):
        assert isinstance(RandomForestSpec().search_space["max_depth"], IntParam)

    def test_min_samples_split_is_float_param(self):
        assert isinstance(
            RandomForestSpec().search_space["min_samples_split"], FloatParam
        )

    def test_min_samples_leaf_is_int_param(self):
        assert isinstance(RandomForestSpec().search_space["min_samples_leaf"], IntParam)

    def test_max_features_is_categorical_param(self):
        assert isinstance(
            RandomForestSpec().search_space["max_features"], CategoricalParam
        )


# ── Warm starts ───────────────────────────────────────────────────────────────


class TestRandomForestWarmStarts:
    def test_warm_starts_has_two_entries(self):
        assert len(RandomForestSpec().warm_starts) == 2

    def test_warm_starts_keys_match_search_space(self):
        spec = RandomForestSpec()
        expected_keys = set(spec.search_space.keys())
        for ws in spec.warm_starts:
            assert set(ws.keys()) == expected_keys

    def test_warm_starts_n_estimators_positive(self):
        for ws in RandomForestSpec().warm_starts:
            assert ws["n_estimators"] > 0

    def test_warm_starts_max_depth_within_range(self):
        space = RandomForestSpec().search_space
        low = space["max_depth"].low
        high = space["max_depth"].high
        for ws in RandomForestSpec().warm_starts:
            assert low <= ws["max_depth"] <= high


# ── suggest_params ────────────────────────────────────────────────────────────


class TestRandomForestSuggestParams:
    def test_suggest_params_returns_all_keys(self):
        import optuna

        spec = RandomForestSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert set(params.keys()) == set(spec.search_space.keys())

    def test_suggest_params_types(self):
        import optuna

        spec = RandomForestSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert isinstance(params["n_estimators"], int)
        assert isinstance(params["max_depth"], int)
        assert isinstance(params["min_samples_split"], float)
        assert isinstance(params["min_samples_leaf"], int)
        assert params["max_features"] in ["sqrt", "log2", 0.5, 0.8, 1.0]


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestRandomForestEvaluate:
    def test_evaluate_returns_finite_float(self, long_series):
        spec = RandomForestSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_evaluate_score_below_penalty(self, long_series):
        spec = RandomForestSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert score < OPTIMIZER_PENALTY

    def test_evaluate_soft_failure_bad_params(self, long_series):
        spec = RandomForestSpec()
        bad_params = {
            "n_estimators": -1,
            "max_depth": 5,
            "min_samples_split": 0.1,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        score = spec.evaluate(long_series, bad_params, combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_returns_penalty_for_too_short_series(self):
        spec = RandomForestSpec()
        short = pd.Series(
            range(10), index=pd.date_range("2020-01-01", periods=10, freq="MS")
        )
        score = spec.evaluate(short, spec.warm_starts[0], combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_does_not_raise(self, long_series):
        spec = RandomForestSpec()
        # Should never raise — only return OPTIMIZER_PENALTY on failure
        try:
            spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        except Exception as exc:
            pytest.fail(f"evaluate() raised unexpectedly: {exc}")

    def test_evaluate_second_warm_start(self, long_series):
        spec = RandomForestSpec()
        score = spec.evaluate(long_series, spec.warm_starts[1], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)


# ── build_forecaster ──────────────────────────────────────────────────────────


class TestRandomForestBuildForecaster:
    def test_build_forecaster_returns_callable(self):
        spec = RandomForestSpec()
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        assert callable(forecaster)

    def test_forecaster_output_length_12(self, long_series):
        spec = RandomForestSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert len(preds) == 12

    def test_forecaster_output_length_custom_horizon(self, long_series):
        spec = RandomForestSpec(forecast_horizon=6)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert len(preds) == 6

    def test_forecaster_returns_series(self, long_series):
        spec = RandomForestSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert isinstance(preds, pd.Series)

    def test_forecaster_datetime_index(self, long_series):
        spec = RandomForestSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert isinstance(preds.index, pd.DatetimeIndex)

    def test_forecaster_future_dates_after_train(self, long_series):
        spec = RandomForestSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index[0] > long_series.index[-1]

    def test_forecaster_monthly_frequency(self, long_series):
        spec = RandomForestSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index.freqstr == "MS"

    def test_forecaster_no_nans_in_output(self, long_series):
        spec = RandomForestSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert not preds.isna().any()


# ── Model registry ────────────────────────────────────────────────────────────


class TestRandomForestRegistry:
    def test_registered_in_model_registry(self):
        from boa_forecaster.models import MODEL_REGISTRY

        assert "random_forest" in MODEL_REGISTRY

    def test_get_model_spec_returns_instance(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("random_forest")
        assert isinstance(spec, RandomForestSpec)

    def test_get_model_spec_passes_kwargs(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("random_forest", forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_exported_from_top_level(self):
        from boa_forecaster import RandomForestSpec as RFS

        assert RFS is RandomForestSpec
