"""Performance benchmark: vectorised weighted_moving_stats_series (A2).

Guards against regressions in the O(n) vectorised path added in v2.1.0.
If someone accidentally re-introduces the O(n*window) per-row Python loop,
``--benchmark-compare-fail=mean:20%`` will flag it in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from boa_forecaster.standardization import weighted_moving_stats_series

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def long_float_series() -> np.ndarray:
    rng = np.random.default_rng(42)
    # 2 000 points ≈ ~166 years of monthly data — big enough to expose any
    # per-row Python overhead while staying under ~10 ms per iteration.
    return 100 + rng.normal(0, 5, 2_000)


def test_bench_weighted_moving_stats_series_default(benchmark, long_float_series):
    """Default window/threshold — the hot path for the standardisation step."""
    benchmark(weighted_moving_stats_series, long_float_series)


def test_bench_weighted_moving_stats_series_wide_window(benchmark, long_float_series):
    """Wider window (12) — stresses the rolling-sum kernel."""
    benchmark(weighted_moving_stats_series, long_float_series, window_size=12)
