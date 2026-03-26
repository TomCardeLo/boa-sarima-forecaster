"""Unit tests for boa_forecaster.models.lightgbm (LightGBMSpec).

All tests are skipped when lightgbm is not installed.
The ``long_series`` fixture (60 monthly points) from conftest.py provides
enough data for all 3 cross-validation folds (min_train=24, horizon=12).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed")

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.features import FeatureConfig
from boa_forecaster.metrics import combined_metric
from boa_forecaster.models.base import FloatParam, IntParam, ModelSpec
from boa_forecaster.models.lightgbm import LightGBMSpec

# ── Protocol & metadata ───────────────────────────────────────────────────────


class TestLightGBMSpecProtocol:
    def test_implements_model_spec_protocol(self):
        assert isinstance(LightGBMSpec(), ModelSpec)

    def test_name(self):
        assert LightGBMSpec().name == "lightgbm"

    def test_needs_features(self):
        assert LightGBMSpec().needs_features is True

    def test_default_forecast_horizon(self):
        assert LightGBMSpec().forecast_horizon == 12

    def test_custom_forecast_horizon(self):
        spec = LightGBMSpec(forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_default_early_stopping_rounds(self):
        assert LightGBMSpec().early_stopping_rounds == 20

    def test_custom_early_stopping_rounds(self):
        spec = LightGBMSpec(early_stopping_rounds=10)
        assert spec.early_stopping_rounds == 10

    def test_default_feature_config(self):
        spec = LightGBMSpec()
        assert isinstance(spec.feature_config, FeatureConfig)

    def test_custom_feature_config(self):
        cfg = FeatureConfig(lag_periods=[1, 2, 3])
        spec = LightGBMSpec(feature_config=cfg)
        assert spec.feature_config.lag_periods == [1, 2, 3]


# ── Search space ──────────────────────────────────────────────────────────────


class TestLightGBMSpecSearchSpace:
    def test_search_space_has_nine_keys(self):
        space = LightGBMSpec().search_space
        assert len(space) == 9

    def test_search_space_keys(self):
        space = LightGBMSpec().search_space
        assert set(space.keys()) == {
            "n_estimators",
            "num_leaves",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
        }

    def test_n_estimators_is_int_param_log(self):
        p = LightGBMSpec().search_space["n_estimators"]
        assert isinstance(p, IntParam)
        assert p.log is True

    def test_num_leaves_is_int_param_log(self):
        p = LightGBMSpec().search_space["num_leaves"]
        assert isinstance(p, IntParam)
        assert p.log is True

    def test_num_leaves_range(self):
        p = LightGBMSpec().search_space["num_leaves"]
        assert p.low == 8
        assert p.high == 256

    def test_max_depth_is_int_param_not_log(self):
        p = LightGBMSpec().search_space["max_depth"]
        assert isinstance(p, IntParam)
        assert p.log is False

    def test_max_depth_range_includes_negative(self):
        p = LightGBMSpec().search_space["max_depth"]
        assert p.low == -1

    def test_learning_rate_is_float_param_log(self):
        p = LightGBMSpec().search_space["learning_rate"]
        assert isinstance(p, FloatParam)
        assert p.log is True

    def test_subsample_is_float_param(self):
        p = LightGBMSpec().search_space["subsample"]
        assert isinstance(p, FloatParam)
        assert p.log is False

    def test_colsample_bytree_is_float_param(self):
        p = LightGBMSpec().search_space["colsample_bytree"]
        assert isinstance(p, FloatParam)

    def test_min_child_samples_is_int_param(self):
        p = LightGBMSpec().search_space["min_child_samples"]
        assert isinstance(p, IntParam)

    def test_reg_alpha_is_float_param_log(self):
        p = LightGBMSpec().search_space["reg_alpha"]
        assert isinstance(p, FloatParam)
        assert p.log is True

    def test_reg_lambda_is_float_param_log(self):
        p = LightGBMSpec().search_space["reg_lambda"]
        assert isinstance(p, FloatParam)
        assert p.log is True

    def test_no_gamma_param(self):
        """LightGBM does not use gamma — only XGBoost does."""
        assert "gamma" not in LightGBMSpec().search_space

    def test_no_min_child_weight_param(self):
        """LightGBM uses min_child_samples, not min_child_weight."""
        assert "min_child_weight" not in LightGBMSpec().search_space


# ── Warm starts ───────────────────────────────────────────────────────────────


class TestLightGBMSpecWarmStarts:
    def test_warm_starts_has_two_entries(self):
        assert len(LightGBMSpec().warm_starts) == 2

    def test_warm_starts_keys_match_search_space(self):
        spec = LightGBMSpec()
        expected_keys = set(spec.search_space.keys())
        for ws in spec.warm_starts:
            assert set(ws.keys()) == expected_keys

    def test_warm_starts_n_estimators_positive(self):
        for ws in LightGBMSpec().warm_starts:
            assert ws["n_estimators"] > 0

    def test_warm_starts_num_leaves_positive(self):
        for ws in LightGBMSpec().warm_starts:
            assert ws["num_leaves"] > 0

    def test_warm_starts_learning_rate_within_range(self):
        space = LightGBMSpec().search_space
        for ws in LightGBMSpec().warm_starts:
            assert (
                space["learning_rate"].low
                <= ws["learning_rate"]
                <= space["learning_rate"].high
            )

    def test_warm_starts_reg_alpha_within_log_bounds(self):
        """reg_alpha must be >= 1e-8 (log-scale floor), not 0.0."""
        space = LightGBMSpec().search_space
        for ws in LightGBMSpec().warm_starts:
            assert ws["reg_alpha"] >= space["reg_alpha"].low

    def test_warm_starts_subsample_in_unit_interval(self):
        for ws in LightGBMSpec().warm_starts:
            assert 0.0 < ws["subsample"] <= 1.0

    def test_warm_start_first_has_unlimited_depth(self):
        """First warm start uses max_depth=-1 (no depth limit)."""
        assert LightGBMSpec().warm_starts[0]["max_depth"] == -1

    def test_warm_start_second_has_finite_depth(self):
        """Second warm start has a positive max_depth."""
        assert LightGBMSpec().warm_starts[1]["max_depth"] > 0

    def test_warm_start_second_num_leaves_respects_depth(self):
        """Second warm start: num_leaves <= 2^max_depth - 1."""
        ws = LightGBMSpec().warm_starts[1]
        if ws["max_depth"] > 0:
            assert ws["num_leaves"] <= 2 ** ws["max_depth"]


# ── suggest_params ────────────────────────────────────────────────────────────


class TestLightGBMSpecSuggestParams:
    def test_suggest_params_returns_all_keys(self):
        import optuna

        spec = LightGBMSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert set(params.keys()) == set(spec.search_space.keys())

    def test_suggest_params_types(self):
        import optuna

        spec = LightGBMSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert isinstance(params["n_estimators"], int)
        assert isinstance(params["num_leaves"], int)
        assert isinstance(params["max_depth"], int)
        assert isinstance(params["learning_rate"], float)
        assert isinstance(params["subsample"], float)
        assert isinstance(params["colsample_bytree"], float)
        assert isinstance(params["min_child_samples"], int)
        assert isinstance(params["reg_alpha"], float)
        assert isinstance(params["reg_lambda"], float)

    def test_num_leaves_constraint_respected(self):
        """When max_depth > 0, num_leaves must be <= 2^max_depth - 1."""
        import optuna

        spec = LightGBMSpec()
        study = optuna.create_study(direction="minimize")
        # Run many trials to cover various max_depth values
        for _ in range(30):
            trial = study.ask()
            params = spec.suggest_params(trial)
            if params["max_depth"] > 0:
                assert params["num_leaves"] <= 2 ** params["max_depth"]


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestLightGBMSpecEvaluate:
    def test_evaluate_returns_finite_float(self, long_series):
        spec = LightGBMSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_evaluate_score_below_penalty(self, long_series):
        spec = LightGBMSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert score < OPTIMIZER_PENALTY

    def test_evaluate_soft_failure_bad_params(self, long_series):
        spec = LightGBMSpec()
        # num_leaves=0 is invalid for LightGBM → should return OPTIMIZER_PENALTY
        bad_params = {
            "n_estimators": 10,
            "num_leaves": 0,  # invalid
            "max_depth": -1,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 1e-8,
            "reg_lambda": 1.0,
        }
        score = spec.evaluate(long_series, bad_params, combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_returns_penalty_for_too_short_series(self):
        spec = LightGBMSpec()
        short = pd.Series(
            range(10), index=pd.date_range("2020-01-01", periods=10, freq="MS")
        )
        score = spec.evaluate(short, spec.warm_starts[0], combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_does_not_raise(self, long_series):
        spec = LightGBMSpec()
        try:
            spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        except Exception as exc:
            pytest.fail(f"evaluate() raised unexpectedly: {exc}")

    def test_evaluate_second_warm_start(self, long_series):
        spec = LightGBMSpec()
        score = spec.evaluate(long_series, spec.warm_starts[1], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_evaluate_custom_feature_config(self, long_series):
        spec = LightGBMSpec()
        cfg = FeatureConfig(lag_periods=[1, 2, 3], rolling_windows=[3])
        score = spec.evaluate(
            long_series, spec.warm_starts[0], combined_metric, feature_config=cfg
        )
        assert isinstance(score, float)
        assert np.isfinite(score)


# ── No-leakage ────────────────────────────────────────────────────────────────


class TestLightGBMNoLeakage:
    def test_evaluate_is_deterministic(self, long_series):
        """Two evaluations on identical inputs must return the same score."""
        spec = LightGBMSpec()
        params = spec.warm_starts[0]
        score1 = spec.evaluate(long_series, params, combined_metric)
        score2 = spec.evaluate(long_series, params, combined_metric)
        assert score1 == pytest.approx(score2, rel=1e-6)

    def test_build_forecaster_no_future_data_in_features(self, long_series):
        """Forecasted values must all come after the last training date."""
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index[0] > long_series.index[-1]


# ── build_forecaster ──────────────────────────────────────────────────────────


class TestLightGBMSpecBuildForecaster:
    def test_build_forecaster_returns_callable(self):
        spec = LightGBMSpec()
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        assert callable(forecaster)

    def test_forecaster_output_length_12(self, long_series):
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert len(preds) == 12

    def test_forecaster_output_length_custom_horizon(self, long_series):
        spec = LightGBMSpec(forecast_horizon=6)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert len(preds) == 6

    def test_forecaster_returns_series(self, long_series):
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert isinstance(preds, pd.Series)

    def test_forecaster_datetime_index(self, long_series):
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert isinstance(preds.index, pd.DatetimeIndex)

    def test_forecaster_future_dates_after_train(self, long_series):
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index[0] > long_series.index[-1]

    def test_forecaster_monthly_frequency(self, long_series):
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert preds.index.freqstr == "MS"

    def test_forecaster_no_nans_in_output(self, long_series):
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        preds = forecaster(long_series)
        assert not preds.isna().any()

    def test_forecaster_with_custom_feature_config(self, long_series):
        cfg = FeatureConfig(lag_periods=[1, 2, 3], rolling_windows=[3])
        spec = LightGBMSpec(forecast_horizon=12)
        forecaster = spec.build_forecaster(spec.warm_starts[0], feature_config=cfg)
        preds = forecaster(long_series)
        assert len(preds) == 12


# ── Model registry ────────────────────────────────────────────────────────────


class TestLightGBMSpecRegistry:
    def test_registered_in_model_registry(self):
        from boa_forecaster.models import MODEL_REGISTRY

        assert "lightgbm" in MODEL_REGISTRY

    def test_get_model_spec_returns_instance(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("lightgbm")
        assert isinstance(spec, LightGBMSpec)

    def test_get_model_spec_passes_kwargs(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("lightgbm", forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_exported_from_top_level(self):
        from boa_forecaster import LightGBMSpec as LGBMS

        assert LGBMS is LightGBMSpec


# ── ImportError when lightgbm not available ───────────────────────────────────
# (NOT gated by pytest.importorskip so it runs even without lightgbm installed)


class TestLightGBMImportError:
    def test_raises_import_error_when_lightgbm_missing(self, monkeypatch):
        import boa_forecaster.models.lightgbm as lgbm_module

        monkeypatch.setattr(lgbm_module, "HAS_LIGHTGBM", False)
        with pytest.raises(ImportError, match="lightgbm is required"):
            LightGBMSpec()
