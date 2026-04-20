"""Tests for ``BaseMLSpec`` construction helpers (v2.3 E2).

Regression coverage for the silent-bug where ``forecast_horizon`` was
accepted as a kwarg but never threaded into the default ``FeatureConfig``
— users setting ``forecast_horizon=24`` never got ``lag_24`` in the
feature matrix, which is the single most informative feature for a
24-month horizon.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")  # BaseMLSpec subclasses need an ML backend.

from boa_forecaster.features import FeatureConfig  # noqa: E402
from boa_forecaster.models.random_forest import RandomForestSpec  # noqa: E402


class TestForecastHorizonInjection:
    """``BaseMLSpec.__init__`` must auto-inject ``forecast_horizon``.

    Only triggers when ``feature_config is None`` — an explicitly-passed
    ``FeatureConfig`` is the user's choice and must never be overridden.
    """

    def test_forecast_horizon_injects_lag(self) -> None:
        """``forecast_horizon=24`` → ``24`` appears in ``lag_periods``."""
        spec = RandomForestSpec(forecast_horizon=24)
        assert 24 in spec.feature_config.lag_periods
        # Default monthly lags must still be present.
        for expected in (1, 2, 3, 6, 12):
            assert expected in spec.feature_config.lag_periods
        # And the list must remain sorted (determinism for downstream diffs).
        assert spec.feature_config.lag_periods == sorted(
            spec.feature_config.lag_periods
        )

    def test_default_horizon_does_not_duplicate(self) -> None:
        """``forecast_horizon=12`` is already in the default lags — no dup."""
        spec = RandomForestSpec()  # horizon defaults to 12
        assert spec.feature_config.lag_periods.count(12) == 1

    def test_short_horizon_already_present(self) -> None:
        """``forecast_horizon=6`` is already in default lags — no dup."""
        spec = RandomForestSpec(forecast_horizon=6)
        assert spec.feature_config.lag_periods.count(6) == 1

    def test_explicit_feature_config_not_overridden(self) -> None:
        """An explicit ``FeatureConfig`` must **never** be rewritten by __init__."""
        cfg = FeatureConfig(lag_periods=[1, 2])
        spec = RandomForestSpec(feature_config=cfg, forecast_horizon=24)
        # The object identity is preserved — no copy, no mutation.
        assert spec.feature_config is cfg
        assert spec.feature_config.lag_periods == [1, 2]

    def test_daily_horizon_7(self) -> None:
        """Cross-frequency sanity: ``forecast_horizon=7`` adds 7 to the list."""
        spec = RandomForestSpec(forecast_horizon=7)
        assert 7 in spec.feature_config.lag_periods
