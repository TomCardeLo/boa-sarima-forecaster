"""Performance benchmark: end-to-end optimize_model on SARIMASpec.

A thin SARIMA optimisation run captures regressions anywhere along the
Optuna / metric / evaluate pipeline.  ``n_calls`` is kept small to keep
the bench under a few seconds per iteration while still exercising the
warm-start + TPE path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.optimizer import optimize_model

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def bench_series() -> pd.Series:
    """96-point monthly series with trend + seasonality."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2017-01-01", periods=96, freq="MS")
    trend = np.linspace(100, 160, 96)
    season = 10 * np.sin(2 * np.pi * np.arange(96) / 12)
    noise = rng.normal(0, 3, 96)
    return pd.Series(trend + season + noise, index=idx)


def test_bench_optimize_sarima_small(benchmark, bench_series):
    """8-call SARIMA optimisation on a realistic 96-point series."""
    benchmark.pedantic(
        optimize_model,
        args=(bench_series, SARIMASpec()),
        kwargs={"n_calls": 8, "n_jobs": 1},
        rounds=3,
        iterations=1,
    )
