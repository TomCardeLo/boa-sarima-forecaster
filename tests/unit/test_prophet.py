"""Unit tests for boa_forecaster.models.prophet (ProphetSpec).

All tests are skipped when prophet is not installed.

Prophet is SARIMA-shaped (raw series in, full-horizon forecast out) rather than
feature-based, so this test module mirrors ``test_sarima`` more than the
XGBoost / LightGBM pattern.  The ``long_series`` fixture (60 monthly points)
gives Prophet enough data to detect yearly seasonality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prophet", reason="prophet not installed")

from boa_forecaster.config import OPTIMIZER_PENALTY  # noqa: E402
from boa_forecaster.metrics import combined_metric  # noqa: E402
from boa_forecaster.models.base import (  # noqa: E402
    CategoricalParam,
    FloatParam,
    ModelSpec,
)
from boa_forecaster.models.prophet import ProphetSpec  # noqa: E402

# ── Protocol & metadata ───────────────────────────────────────────────────────


class TestProphetSpecProtocol:
    def test_implements_model_spec_protocol(self):
        assert isinstance(ProphetSpec(), ModelSpec)

    def test_name(self):
        assert ProphetSpec().name == "prophet"

    def test_needs_features_is_false(self):
        assert ProphetSpec().needs_features is False

    def test_uses_early_stopping_is_false(self):
        assert ProphetSpec().uses_early_stopping is False

    def test_default_forecast_horizon(self):
        assert ProphetSpec().forecast_horizon == 12

    def test_custom_forecast_horizon(self):
        assert ProphetSpec(forecast_horizon=6).forecast_horizon == 6

    def test_default_freq_is_monthly_start(self):
        assert ProphetSpec().freq == "MS"

    def test_default_seasonality_mode_is_additive(self):
        assert ProphetSpec().seasonality_mode == "additive"

    def test_default_country_holidays_is_none(self):
        assert ProphetSpec().country_holidays is None

    def test_default_growth_is_linear(self):
        assert ProphetSpec().growth == "linear"


# ── Search space ──────────────────────────────────────────────────────────────


class TestProphetSpecSearchSpace:
    def test_search_space_has_four_keys(self):
        assert len(ProphetSpec().search_space) == 4

    def test_search_space_keys(self):
        space = ProphetSpec().search_space
        assert set(space.keys()) == {
            "changepoint_prior_scale",
            "seasonality_prior_scale",
            "holidays_prior_scale",
            "seasonality_mode",
        }

    def test_changepoint_prior_scale_is_float_log(self):
        param = ProphetSpec().search_space["changepoint_prior_scale"]
        assert isinstance(param, FloatParam)
        assert param.log is True
        assert param.low == pytest.approx(0.001)
        assert param.high == pytest.approx(0.5)

    def test_seasonality_prior_scale_is_float_log(self):
        param = ProphetSpec().search_space["seasonality_prior_scale"]
        assert isinstance(param, FloatParam)
        assert param.log is True
        assert param.low == pytest.approx(0.01)
        assert param.high == pytest.approx(10.0)

    def test_holidays_prior_scale_is_float_log(self):
        param = ProphetSpec().search_space["holidays_prior_scale"]
        assert isinstance(param, FloatParam)
        assert param.log is True

    def test_seasonality_mode_is_categorical(self):
        param = ProphetSpec().search_space["seasonality_mode"]
        assert isinstance(param, CategoricalParam)
        assert set(param.choices) == {"additive", "multiplicative"}


# ── Warm starts ───────────────────────────────────────────────────────────────


class TestProphetSpecWarmStarts:
    def test_warm_starts_has_two_entries(self):
        assert len(ProphetSpec().warm_starts) == 2

    def test_warm_starts_keys_match_search_space(self):
        spec = ProphetSpec()
        expected = set(spec.search_space.keys())
        for ws in spec.warm_starts:
            assert set(ws.keys()) == expected

    def test_warm_starts_within_ranges(self):
        spec = ProphetSpec()
        space = spec.search_space
        for ws in spec.warm_starts:
            for key in (
                "changepoint_prior_scale",
                "seasonality_prior_scale",
                "holidays_prior_scale",
            ):
                assert space[key].low <= ws[key] <= space[key].high
            assert ws["seasonality_mode"] in space["seasonality_mode"].choices


# ── suggest_params ────────────────────────────────────────────────────────────


class TestProphetSpecSuggestParams:
    def test_suggest_params_returns_all_keys(self):
        import optuna

        spec = ProphetSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert set(params.keys()) == set(spec.search_space.keys())

    def test_suggest_params_types(self):
        import optuna

        spec = ProphetSpec()
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        params = spec.suggest_params(trial)
        assert isinstance(params["changepoint_prior_scale"], float)
        assert isinstance(params["seasonality_prior_scale"], float)
        assert isinstance(params["holidays_prior_scale"], float)
        assert params["seasonality_mode"] in ("additive", "multiplicative")


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestProphetSpecEvaluate:
    def test_evaluate_returns_finite_float(self, long_series):
        spec = ProphetSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_evaluate_score_below_penalty(self, long_series):
        spec = ProphetSpec()
        score = spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        assert score < OPTIMIZER_PENALTY

    def test_evaluate_soft_failure_bad_params(self, long_series):
        """Negative changepoint_prior_scale is rejected by Prophet."""
        spec = ProphetSpec()
        bad = {
            "changepoint_prior_scale": -1.0,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "seasonality_mode": "additive",
        }
        score = spec.evaluate(long_series, bad, combined_metric)
        assert score == OPTIMIZER_PENALTY

    def test_evaluate_does_not_raise(self, long_series):
        spec = ProphetSpec()
        try:
            spec.evaluate(long_series, spec.warm_starts[0], combined_metric)
        except Exception as exc:
            pytest.fail(f"evaluate() raised unexpectedly: {exc}")

    def test_evaluate_accepts_multiplicative_seasonality(self, long_series):
        spec = ProphetSpec()
        score = spec.evaluate(long_series, spec.warm_starts[1], combined_metric)
        assert np.isfinite(score)
        assert score < OPTIMIZER_PENALTY


# ── build_forecaster ──────────────────────────────────────────────────────────


class TestProphetSpecBuildForecaster:
    def test_build_forecaster_returns_callable(self):
        forecaster = ProphetSpec().build_forecaster(ProphetSpec().warm_starts[0])
        assert callable(forecaster)

    def test_forecaster_output_length_12(self, long_series):
        spec = ProphetSpec(forecast_horizon=12)
        preds = spec.build_forecaster(spec.warm_starts[0])(long_series)
        assert len(preds) == 12

    def test_forecaster_output_length_custom_horizon(self, long_series):
        spec = ProphetSpec(forecast_horizon=6)
        preds = spec.build_forecaster(spec.warm_starts[0])(long_series)
        assert len(preds) == 6

    def test_forecaster_returns_series(self, long_series):
        spec = ProphetSpec(forecast_horizon=12)
        preds = spec.build_forecaster(spec.warm_starts[0])(long_series)
        assert isinstance(preds, pd.Series)

    def test_forecaster_datetime_index(self, long_series):
        spec = ProphetSpec(forecast_horizon=12)
        preds = spec.build_forecaster(spec.warm_starts[0])(long_series)
        assert isinstance(preds.index, pd.DatetimeIndex)

    def test_forecaster_future_dates_after_train(self, long_series):
        spec = ProphetSpec(forecast_horizon=12)
        preds = spec.build_forecaster(spec.warm_starts[0])(long_series)
        assert preds.index[0] > long_series.index[-1]

    def test_forecaster_no_nans_in_output(self, long_series):
        spec = ProphetSpec(forecast_horizon=12)
        preds = spec.build_forecaster(spec.warm_starts[0])(long_series)
        assert not preds.isna().any()

    def test_forecaster_deterministic_on_same_input(self, long_series):
        """Two forecasts on identical train produce identical output (stan seed is fixed)."""
        spec = ProphetSpec(forecast_horizon=6)
        preds1 = spec.build_forecaster(spec.warm_starts[0])(long_series)
        preds2 = spec.build_forecaster(spec.warm_starts[0])(long_series)
        np.testing.assert_allclose(preds1.values, preds2.values, rtol=1e-4)


# ── for_frequency ─────────────────────────────────────────────────────────────


class TestProphetSpecForFrequency:
    def test_for_frequency_monthly_start(self):
        spec = ProphetSpec.for_frequency("MS")
        assert spec.freq == "MS"

    def test_for_frequency_daily(self):
        spec = ProphetSpec.for_frequency("D")
        assert spec.freq == "D"

    def test_for_frequency_hourly_lowercase(self):
        spec = ProphetSpec.for_frequency("h")
        assert spec.freq == "h"

    def test_for_frequency_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown frequency"):
            ProphetSpec.for_frequency("NOT_A_FREQ")

    def test_for_frequency_forwards_overrides(self):
        spec = ProphetSpec.for_frequency("D", forecast_horizon=30)
        assert spec.forecast_horizon == 30
        assert spec.freq == "D"


# ── Model registry & top-level re-export ──────────────────────────────────────


class TestProphetSpecRegistry:
    def test_registered_in_model_registry(self):
        from boa_forecaster.models import MODEL_REGISTRY

        assert "prophet" in MODEL_REGISTRY

    def test_get_model_spec_returns_instance(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("prophet")
        assert isinstance(spec, ProphetSpec)

    def test_get_model_spec_passes_kwargs(self):
        from boa_forecaster.models import get_model_spec

        spec = get_model_spec("prophet", forecast_horizon=6)
        assert spec.forecast_horizon == 6

    def test_exported_from_top_level(self):
        from boa_forecaster import ProphetSpec as PS

        assert PS is ProphetSpec


# ── Regression vs. SARIMA (loose bound) ───────────────────────────────────────


class TestProphetVsSarimaConvergence:
    def test_prophet_converges_on_synthetic(self, synthetic_series):
        """Prophet on a clean trend+noise series should produce a finite, non-penalty score.

        Loose bound: Prophet's in-sample combined_metric must not be more than
        3× SARIMA's — guards against a broken fit without being flaky.
        """
        from boa_forecaster.models.sarima import SARIMASpec

        prophet_spec = ProphetSpec()
        prophet_score = prophet_spec.evaluate(
            synthetic_series, prophet_spec.warm_starts[0], combined_metric
        )
        sarima_spec = SARIMASpec()
        sarima_score = sarima_spec.evaluate(
            synthetic_series, sarima_spec.warm_starts[0], combined_metric
        )
        assert np.isfinite(prophet_score)
        assert np.isfinite(sarima_score)
        assert prophet_score < OPTIMIZER_PENALTY
        # Loose: Prophet shouldn't be catastrophically worse than SARIMA
        assert prophet_score <= sarima_score * 3.0 + 1e-6


# ── ImportError when prophet not available ────────────────────────────────────
# (NOT gated by pytestmark so it runs even without prophet installed)


class TestProphetImportError:
    def test_raises_import_error_when_prophet_missing(self, monkeypatch):
        import boa_forecaster.models.prophet as prophet_module

        monkeypatch.setattr(prophet_module, "HAS_PROPHET", False)
        with pytest.raises(ImportError, match="prophet is required"):
            ProphetSpec()
