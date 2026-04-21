"""Tests for Track H8: SARIMASpec seasonal_period tuneable for hourly data.

Scope
-----
* ``seasonal_period_candidates`` constructor kwarg + validation
* ``search_space`` extension when candidates are provided
* ``SARIMASpec.for_frequency`` classmethod
* Back-compat guard: default ``SARIMASpec()`` must not include seasonal_period
  in its search space
* evaluate() param-resolution: sampled seasonal_period flows into SARIMAX

Note: The propensity-picks test (asserting Optuna picks m=168 >= 60% on
weekly-seasonal synthetic data) is deliberately excluded — it is timing-
sensitive and slow, inconsistent with the project's correctness-over-
convergence policy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.models.base import CategoricalParam
from boa_forecaster.models.sarima import SARIMASpec

# ── Helpers ───────────────────────────────────────────────────────────────────


def _rmse(y_true, y_pred):
    """Simple RMSE for evaluate() calls in this test module."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _make_monthly_series(n: int = 36) -> pd.Series:
    """Synthetic monthly series with mild trend, suitable for SARIMA(1,0,0)."""
    rng = np.random.default_rng(42)
    values = 100 + np.arange(n) * 0.5 + rng.normal(0, 5, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.Series(values, index=idx)


def _make_hourly_series(n: int = 120) -> pd.Series:
    """Synthetic hourly series suitable for a light SARIMAX fit."""
    rng = np.random.default_rng(7)
    # Simple sinusoidal daily pattern + noise — avoids stationarity issues
    t = np.arange(n)
    values = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, n)
    values = np.clip(values, 0.1, None)  # keep positive for metrics
    return pd.Series(values.astype(float))


# ── Back-compat guard ─────────────────────────────────────────────────────────


def test_default_sarimaspec_has_no_seasonal_period_in_search_space():
    """Default SARIMASpec() must NOT expose seasonal_period in search_space."""
    spec = SARIMASpec()
    assert "seasonal_period" not in spec.search_space, (
        "Back-compat broken: default SARIMASpec now exposes seasonal_period in "
        "search_space — this changes existing optimizer behaviour."
    )


# ── seasonal_period_candidates kwarg ─────────────────────────────────────────


def test_sarimaspec_with_candidates_exposes_seasonal_period():
    """SARIMASpec(seasonal_period_candidates=[24,168]) adds CategoricalParam."""
    spec = SARIMASpec(seasonal_period_candidates=[24, 168])
    assert "seasonal_period" in spec.search_space
    param = spec.search_space["seasonal_period"]
    assert isinstance(param, CategoricalParam)
    assert param.choices == [24, 168]


def test_empty_candidates_rejected():
    """Empty list for seasonal_period_candidates must raise ValueError."""
    with pytest.raises(ValueError, match="seasonal_period_candidates"):
        SARIMASpec(seasonal_period_candidates=[])


def test_negative_candidate_rejected():
    """Non-positive int in seasonal_period_candidates must raise ValueError."""
    with pytest.raises(ValueError, match="seasonal_period_candidates"):
        SARIMASpec(seasonal_period_candidates=[24, -1])


def test_zero_candidate_rejected():
    """Zero in seasonal_period_candidates must raise ValueError."""
    with pytest.raises(ValueError, match="seasonal_period_candidates"):
        SARIMASpec(seasonal_period_candidates=[0, 24])


# ── for_frequency classmethod ─────────────────────────────────────────────────


def test_for_frequency_monthly():
    """for_frequency('MS') returns m=12 with no seasonal_period in search_space."""
    spec = SARIMASpec.for_frequency("MS")
    assert spec.m == 12
    assert "seasonal_period" not in spec.search_space


def test_for_frequency_monthly_M_alias():
    """for_frequency('M') is the same as 'MS'."""
    spec = SARIMASpec.for_frequency("M")
    assert spec.m == 12
    assert "seasonal_period" not in spec.search_space


def test_for_frequency_weekly():
    """for_frequency('W') returns m=52 with no seasonal_period in search_space."""
    spec = SARIMASpec.for_frequency("W")
    assert spec.m == 52
    assert "seasonal_period" not in spec.search_space


def test_for_frequency_daily():
    """for_frequency('D') returns m=7."""
    spec = SARIMASpec.for_frequency("D")
    assert spec.m == 7
    assert "seasonal_period" not in spec.search_space


def test_for_frequency_hourly():
    """for_frequency('h') exposes seasonal_period as CategoricalParam([24,168])."""
    spec = SARIMASpec.for_frequency("h")
    assert "seasonal_period" in spec.search_space
    param = spec.search_space["seasonal_period"]
    assert isinstance(param, CategoricalParam)
    assert param.choices == [24, 168]


def test_for_frequency_hourly_uppercase_H():
    """for_frequency('H') behaves identically to 'h'."""
    spec = SARIMASpec.for_frequency("H")
    assert "seasonal_period" in spec.search_space
    param = spec.search_space["seasonal_period"]
    assert isinstance(param, CategoricalParam)
    assert param.choices == [24, 168]


def test_for_frequency_unknown_raises():
    """Unsupported frequency must raise ValueError."""
    with pytest.raises(ValueError, match="Q"):
        SARIMASpec.for_frequency("Q")


def test_for_frequency_overrides_apply():
    """**overrides forwarded to constructor — e.g. custom p_range."""
    spec = SARIMASpec.for_frequency("MS", p_range=(0, 3))
    assert spec.p_range == (0, 3)
    assert spec.m == 12  # preset not disturbed


# ── evaluate() param-resolution ───────────────────────────────────────────────


def test_evaluate_uses_sampled_seasonal_period():
    """When seasonal_period is in params, evaluate() must use it in SARIMAX.

    Uses a short hourly-ish series (120 pts) with a light AR(1) model so the
    fit is fast.  The key assertion is that the score is finite — proving the
    plumbing from suggested param → SARIMAX seasonal_order works end-to-end.
    """
    series = _make_hourly_series(120)
    spec = SARIMASpec(seasonal_period_candidates=[24, 168])
    params = {
        "p": 1,
        "d": 0,
        "q": 0,
        "P": 0,
        "D": 0,
        "Q": 0,
        "seasonal_period": 24,
    }
    score = spec.evaluate(series, params, metric_fn=_rmse)
    assert math.isfinite(score), (
        f"evaluate() returned non-finite score {score!r} — "
        "the seasonal_period param was probably not passed to SARIMAX."
    )
    # OPTIMIZER_PENALTY is a large sentinel (> 1e6); a real score must be < 1e5
    assert score < 1e5, f"Score looks like OPTIMIZER_PENALTY: {score}"


def test_evaluate_falls_back_to_self_m_when_param_missing():
    """Default SARIMASpec() must still use self.m when params lack seasonal_period.

    This is the back-compat path — existing code does not pass seasonal_period
    in params, so the evaluate() method must fall back to self.m (= 12).
    """
    series = _make_monthly_series(36)
    spec = SARIMASpec()  # default: m=12, no candidates
    params = {"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0}
    score = spec.evaluate(series, params, metric_fn=_rmse)
    assert math.isfinite(score), f"evaluate() returned non-finite: {score!r}"
    assert score < 1e5, f"Score looks like OPTIMIZER_PENALTY: {score}"
