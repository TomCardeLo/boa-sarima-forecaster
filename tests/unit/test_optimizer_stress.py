"""Stress tests for :mod:`boa_forecaster.optimizer`.

Guards against wall-time regressions in the end-to-end optimisation pipeline
on realistic-sized series.  A perf regression in SARIMAX fitting, recursive
forecasting, metric computation, or the TPE sampler would push these tests
over budget.

These tests are marked ``slow``; deselect with ``pytest -m "not slow"``
during local iteration.  CI runs them unconditionally.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.optimizer import optimize_model

_N_POINTS = 500
_N_TRIALS = 5
_TIME_BUDGET_S = 30.0


@pytest.fixture
def stress_series() -> pd.Series:
    """Deterministic 500-point monthly series with trend, seasonality, and noise."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=_N_POINTS, freq="MS")
    t = np.arange(_N_POINTS, dtype=float)
    trend = 0.05 * t
    season = 5.0 * np.sin(2 * np.pi * t / 12.0)
    noise = rng.normal(0.0, 1.0, size=_N_POINTS)
    return pd.Series(100.0 + trend + season + noise, index=idx)


@pytest.mark.slow
def test_optimize_model_500_points_5_trials_under_budget(stress_series):
    """``optimize_model`` on 500 points with 5 trials must finish under budget."""
    spec = SARIMASpec()

    start = time.perf_counter()
    result = optimize_model(stress_series, spec, n_calls=_N_TRIALS, seed=0)
    elapsed = time.perf_counter() - start

    assert elapsed < _TIME_BUDGET_S, (
        f"optimize_model took {elapsed:.2f}s on {_N_POINTS}-point series "
        f"with {_N_TRIALS} trials (budget {_TIME_BUDGET_S}s) — "
        "possible perf regression in SARIMAX fit / metric / TPE path."
    )
    assert result.n_trials == _N_TRIALS
    assert result.model_name == "sarima"
    assert result.is_fallback is False
