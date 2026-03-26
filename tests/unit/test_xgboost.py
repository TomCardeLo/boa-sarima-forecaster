"""Unit tests for boa_forecaster.models.xgboost (XGBoostSpec).

All tests are skipped when xgboost is not installed.
The ``long_series`` fixture (60 monthly points) from conftest.py provides
enough data for all 3 cross-validation folds (min_train=24, horizon=12).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost", reason="xgboost not installed")

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.features import FeatureConfig
from boa_forecaster.metrics import combined_metric
from boa_forecaster.models.base import FloatParam, IntParam, ModelSpec
from boa_forecaster.models.xgboost import HAS_XGBOOST, XGBoostSpec

# ── Protocol & metadata ───────────────────────────────────────────────────────


class TestXGBoostSpecProtocol:
    def test_implements_model_spec_protocol(self):
        assert isinstance(XGBoostSpec(), ModelSpec)

    def test_name(self):
        assert XGBoostSpec().name == "xgboost"

    def test_needs_features(self):
        assert XGBoostSpec().needs_features is True

    def test_default_forecast_horizon(self):
        assert XGBoostSpec().forecast_horizon == 12

    def test_custom_forecast_horizon(self):
        spec = XGBoostSpec(forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_default_early_stopping_rounds(self):
        assert XGBoostSpec().early_stopping_rounds == 20

    def test_custom_early_stopping_rounds(self):
        spec = XGBoostSpec(early_stopping_rounds=10)
        assert spec.early_stopping_rounds == 10

    def test_default_feature_config(self):
        spec = XGBoostSpec()
        assert isinstance(spec.feature_config, FeatureConfig)


# ── Search space ──────────────────────────────────────────────────────────────


class TestXGBoostSpecSearchSpace:
    def test_search_space_has_nine_keys(self):
        space = XGBoostSpec().search_space
        assert len(space) == 9

    def test_search_space_keys(self):
        space = XGBoostSpec().search_space
        assert set(space.keys()) == {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "gamma",
        }

    def test_n_estimators_is_int_param(self):
        assert isinstance(XGBoostSpec().search_space["n_estimators"], IntParam)

    def test_n_estimators_log_scale(self):
        assert XGBoostSpec().search_space["n_estimators"].log is True

    def test_max_depth_is_int_param(self):
        assert isinstance(XGBoostSpec().search_space["max_depth"], IntParam)

    def test_learning_rate_is_float_param(self):
        assert isinstance(XGBoostSpec().search_space["learning_rate"], FloatParam)

    def test_learning_rate_log_scale(self):
        assert XGBoostSpec().search_space["learning_rate"].log is True

    def test_subsample_is_float_param(self):
        assert isinstance(XGBoostSpec().search_space["subsample"], FloatParam)

    def test_colsample_bytree_is_float_param(self):
        assert isinstance(XGBoostSpec().search_space["colsample_bytree"], FloatParam)

    def test_min_child_weight_is_int_param(self):
        assert isinstance(XGBoostSpec().search_space["min_child_weight"], IntParam)

    def test_reg_alpha_is_float_param_log_scale(self):
        param = XGBoostSpec().search_space["reg_alpha"]
        assert isinstance(param, FloatParam)
        assert param.log is True

    def test_reg_lambda_is_float_param_log_scale(self):
        param = XGBoostSpec().search_space["reg_lambda"]
        assert isinstance(param, FloatParam)
        assert param.log is True

    def test_gamma_is_float_param_not_log(self):
        param = XGBoostSpec().search_space["gamma"]
        assert isinstance(param, FloatParam)
        assert param.log is False

    def test_no_categorical_params(self):
        from boa_forecaster.models.base import CategoricalParam

        space = XGBoostSpec().search_space
        for v in space.values():
            assert not isinstance(v, CategoricalParam)


# ── Warm starts ───────────────────────────────────────────────────────────────


class TestXGBoostSpecWarmStarts:
    def test_warm_starts_has_two_entries(self):
        assert len(XGBoostSpec().warm_starts) == 2

    def test_warm_starts_keys_match_search_space(self):
        spec = XGBoostSpec()
        expected_keys = set(spec.search_space.keys())
        for ws in spec.warm_starts:
            assert set(ws.keys()) == expected_keys

    def test_warm_starts_n_estimators_positive(self):
        for ws in XGBoostSpec().warm_starts:
            assert ws["n_estimators"] > 0

    def test_warm_starts_max_depth_within_range(self):
        space = XGBoostSpec().search_space
        low = space["max_depth"].low
        high = space["max_depth"].high
        for ws in XGBoostSpec().warm_starts:
            assert low <= ws["max_depth"] <= high

    def test_warm_starts_learning_rate_within_range(self):
        space = XGBoostSpec().search_space
        for ws in XGBoostSpec().warm_starts:
            assert space["learning_rate"].low <= ws["learning_rate"] <= space["learning_rate"].high

    def test_warm_starts_reg_alpha_within_log_bounds(self):
        """reg_alpha must be >= 1e-8 (log-scale floor), not 0.0."""
        space = XGBoostSpec().search_space
        for ws in XGBoostSpec().warm_starts:
            assert ws["reg_alpha"] >= space["reg_alpha"].low

    def test_warm_starts_subsample_in_unit_interval(self):
        for ws in XGBoostSpec().warm_starts:
            assert 0.0 < ws["subsample"] <= 1.0

    def test_warm_starts_gamma_non_negative(self):
        for ws in XGBoostSpec().warm_starts:
            assert ws["gamma"] >= 0.0


# ── suggest_params ────────────────────────────────────────────────────────────


class TestXGBoostSpecSuggestParams:
    def test_suggest_params_returns_all_keys(self):
        import optuna

        spec = XGBoostSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert set(params.keys()) == set(spec.search_space.keys())

    def test_suggest_params_types(self):
        import optuna

        spec = XGBoostSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert isinstance(params["n_estimators"], int)
        assert isinstance(params["max_depth"], int)
        assert isinstance(params["learning_rate"], float)
        assert isinstance(params["subsample"], float)
        assert isinstance(params["colsample_bytree"], float)
        assert isinstance(params["min_child_weight"], int)
        assert isinstance(params["reg_alpha"], float)
        assert isinstance(params["reg_lambda"], float)
        assert isinstance(params["gamma"], float)


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestXGBoostSpecEvaluate:
    def test_evaluate_returns_finite_float(self, long_series):
        spec = XGBoostSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_evaluate_score_below_penalty(self, long_series):
        spec = XGBoostSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert score < OPTIMIZER_PENALTY

    def test_evaluate_soft_failure_bad_params(self, long_series):
        spec = XGBoostSpec()
        # max_depth=-100 causes XGBRegressor to raise XGBoostError (below 0)
        bad_params = {
            "n_estimators": 100,
            "max_depth": -100,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 1e-8,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
        score = spec.evaluate(long_series, bad_params, combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_returns_penalty_for_too_short_series(self):
        spec = XGBoostSpec()
        short = pd.Series(
            range(10), index=pd.date_range("2020-01-01", periods=10, freq="MS")
        )
        score = spec.evaluate(short, spec.warm_starts[0], combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_does_not_raise(self, long_series):
        spec = XGBoostSpec()
        try:
            spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        except Exception as exc:
            pytest.fail(f"evaluate() raised unexpectedly: {exc}")

    def test_evaluate_second_warm_start(self, long_series):
        spec = XGBoostSpec()
        score = spec.evaluate(long_series, spec.warm_starts[1], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)


# ── No-leakage ────────────────────────────────────────────────────────────────


class TestXGBoostNoLeakage:
    def test_evaluate_uses_only_training_data(self, long_series):
        """Two evaluations on identical train windows must return the same score."""
        spec = XGBoostSpec()
        params = spec.warm_starts[0]
        score1 = spec.evaluate(long_series, params, combined_metric)
        score2 = spec.evaluate(long_series, params, combined_metric)
        assert score1 == pytest.approx(score2, rel=1e-6)

    def test_build_forecaster_no_future_data_in_features(self, long_series):
        """Forecasted values must all come after the last training date."""
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index[0] > long_series.index[-1]


# ── build_forecaster ──────────────────────────────────────────────────────────


class TestXGBoostSpecBuildForecaster:
    def test_build_forecaster_returns_callable(self):
        spec = XGBoostSpec()
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        assert callable(forecaster)

    def test_forecaster_output_length_12(self, long_series):
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert len(preds) == 12

    def test_forecaster_output_length_custom_horizon(self, long_series):
        spec = XGBoostSpec(forecast_horizon=6)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert len(preds) == 6

    def test_forecaster_returns_series(self, long_series):
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert isinstance(preds, pd.Series)

    def test_forecaster_datetime_index(self, long_series):
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert isinstance(preds.index, pd.DatetimeIndex)

    def test_forecaster_future_dates_after_train(self, long_series):
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index[0] > long_series.index[-1]

    def test_forecaster_monthly_frequency(self, long_series):
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index.freqstr == "MS"

    def test_forecaster_no_nans_in_output(self, long_series):
        spec = XGBoostSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert not preds.isna().any()


# ── Model registry ────────────────────────────────────────────────────────────


class TestXGBoostSpecRegistry:
    def test_registered_in_model_registry(self):
        from boa_forecaster.models import MODEL_REGISTRY

        assert "xgboost" in MODEL_REGISTRY

    def test_get_model_spec_returns_instance(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("xgboost")
        assert isinstance(spec, XGBoostSpec)

    def test_get_model_spec_passes_kwargs(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("xgboost", forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_exported_from_top_level(self):
        from boa_forecaster import XGBoostSpec as XGS

        assert XGS is XGBoostSpec


# ── ImportError when xgboost not available ────────────────────────────────────
# (NOT gated by pytestmark so it runs even without xgboost installed)


class TestXGBoostImportError:
    def test_raises_import_error_when_xgboost_missing(self, monkeypatch):
        import boa_forecaster.models.xgboost as xgb_module

        monkeypatch.setattr(xgb_module, "HAS_XGBOOST", False)
        with pytest.raises(ImportError, match="xgboost is required"):
            XGBoostSpec()
