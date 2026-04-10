"""Tests for SARIMA complexity constraint enforcement.

Verifies that SARIMASpec.evaluate returns OPTIMIZER_PENALTY when the
non-seasonal or seasonal order constraints are violated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.metrics import combined_metric
from boa_forecaster.models.sarima import SARIMASpec


@pytest.fixture()
def spec() -> SARIMASpec:
    return SARIMASpec()


@pytest.fixture()
def series() -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01", periods=36, freq="MS")
    return pd.Series(rng.uniform(50, 200, size=36), index=idx)


class TestNonSeasonalConstraint:
    """p + q must not exceed MAX_NON_SEASONAL_ORDER (default 4)."""

    def test_violation_returns_penalty(self, spec: SARIMASpec, series: pd.Series):
        params = {"p": 3, "d": 1, "q": 2, "P": 0, "D": 0, "Q": 0}  # p+q=5
        result = spec.evaluate(series, params, combined_metric)
        assert result == OPTIMIZER_PENALTY

    def test_boundary_is_allowed(self, spec: SARIMASpec, series: pd.Series):
        params = {"p": 2, "d": 1, "q": 2, "P": 0, "D": 0, "Q": 0}  # p+q=4
        result = spec.evaluate(series, params, combined_metric)
        assert result != OPTIMIZER_PENALTY


class TestSeasonalConstraint:
    """P + Q must not exceed MAX_SEASONAL_ORDER (default 3)."""

    def test_violation_returns_penalty(self, spec: SARIMASpec, series: pd.Series):
        params = {"p": 1, "d": 1, "q": 1, "P": 2, "D": 1, "Q": 2}  # P+Q=4
        result = spec.evaluate(series, params, combined_metric)
        assert result == OPTIMIZER_PENALTY

    def test_boundary_is_allowed(self, spec: SARIMASpec, series: pd.Series):
        params = {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 2}  # P+Q=3
        result = spec.evaluate(series, params, combined_metric)
        assert result != OPTIMIZER_PENALTY


class TestBothConstraints:
    """Both constraints violated simultaneously."""

    def test_both_violated(self, spec: SARIMASpec, series: pd.Series):
        params = {"p": 3, "d": 0, "q": 3, "P": 2, "D": 0, "Q": 2}
        result = spec.evaluate(series, params, combined_metric)
        assert result == OPTIMIZER_PENALTY

    def test_valid_order_does_not_return_penalty(
        self, spec: SARIMASpec, series: pd.Series
    ):
        params = {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1}
        result = spec.evaluate(series, params, combined_metric)
        assert result != OPTIMIZER_PENALTY
        assert np.isfinite(result)
